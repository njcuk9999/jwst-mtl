"""
Created on Sun Jan 26 16:39:05 2020

@author: caroline

TimeSeries objects for simulations of SOSS observations
"""

# General imports.
from copy import deepcopy
from pkg_resources import resource_filename
import os

# General science imports.
import numpy as np
import numpy.random as rdm

# Astronomy imports.
from astropy.io import fits

# Home-brew and intra module imports.
import hxrg

# Plotting.
import matplotlib.pyplot as plt

# Global variables.
GAIN = 1.61
DARKVALUE = 0.0414
FULL_WELL = 72000.

# TODO header section which files and values were used.


class TimeSeries(object):

    def __init__(self, ima_path):
        """Make a TimeSeries object from a series of synthetic images."""

        self.ima_path = ima_path

        hdu_ideal = fits.open(ima_path)  # read in fits file
        header = hdu_ideal[1].header

        self.hdu_ideal = hdu_ideal
        self.data = hdu_ideal[1].data  # image to be altered

        self.nrows = header['NAXIS1']
        self.ncols = header['NAXIS2']
        self.ngroups = header['NAXIS3']  # number of groups per integration
        self.nintegs = header['NAXIS4']  # number of integrations in time series observations
        self.subarray = hdu_ideal[0].header['SUBARRAY']
        self.tgroup = hdu_ideal[0].header['TGROUP']  # TODO Used as exposure time per frame.

        self.modif_str = '_mod'  # string encoding the modifications

        # TODO: Handle paths properly.
        # Here, I hardcoded the path but really we should read it from the config file
        # /genesis/jwst/jwst-mtl-user/jwst-mtl_configpath.txt 
        # NOISE_FILES is the parameter in that file
        self.noisefiles_dir = '/genesis/jwst/jwst-ref-soss/noise_files/' # PATH where reference detector noise files can be found.
        # Same here, we need to pass this or read it from teh config path
        # USER_PATH is the parameter in that file
        self.output_path = '/genesis/jwst/userland-soss/'

    def get_normfactor(self, full_well=FULL_WELL):
        """Determine a re-normalization factor so that the highest pixel value in the simulation
         will match the full well capacity"""

        max_value = np.amax(self.data)
        normfactor = full_well/max_value

        return normfactor

    def apply_normfactor(self, normfactor):
        """Apply an arbitrary re-normalization to the simulations."""

        self.data = self.data*normfactor

        self.modif_str = self.modif_str + '_norm'

    def add_poisson_noise(self):
        """Add Poisson noise to the simulation."""

        # Can be done without loops, but this reduces memory requirements.
        for i in range(self.nintegs):

            ramp = deepcopy(self.data[i])

            # Convert up the ramp samples, to flux between reads.
            ramp[1:] = np.diff(ramp, axis=0)

            # Add the poisson noise.
            ramp = np.where(ramp < 0, 0, ramp)  # Sanity check.
            ramp = rdm.poisson(ramp)

            # Convert back to up the ramp samples.
            ramp = np.cumsum(ramp, axis=0)

            self.data[i] = deepcopy(ramp)

        self.modif_str = self.modif_str + '_poisson_noise'

    def add_non_linearity(self, coef_file=None, gain=GAIN):
        """Add non-linearity on top of the linear integration-long ramp."""

        if coef_file is None:
            coef_file = self.noisefiles_dir+'/jwst_niriss_linearity_0011_bounds_0_60000_npoints_100_deg_5.fits'
            #coef_file = resource_filename('detector', 'files/jwst_niriss_linearity_0011_bounds_0_60000_npoints_100_deg_5.fits')

        # Read the coefficients of the non-linearity function.
        with fits.open(coef_file) as hdu:
            non_linearity = hdu[0].data

        ncoeffs = non_linearity.shape[0]

        # Select the appropriate subarray.
        if self.subarray == 'SUBSTRIP96':
            slc = slice(1792, 1888)
        elif self.subarray == 'SUBSTRIP256':
            slc = slice(1792, 2048)
        elif self.subarray == 'FULL':
            slc = slice(0, 2048)
        else:
            raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')

        non_linearity = non_linearity[:, slc, :]

        # Add non-linearity to each ramp
        for i in range(self.nintegs):
            # select part of the time series lying within this integration
            # (i.e., ith sequence of ngroups groups)
            new_integ = deepcopy(self.data[i, :, :, :])/gain  # [ADU] because the non-linearity correction works on ADU.

            # Iterate over groups
            for g in range(self.ngroups):
                frame = deepcopy(new_integ[g, :, :])

                corr = non_linearity[0, :, :]  # Constant.
                for k in range(1, ncoeffs):  # Higher degree terms.
                    corr = corr + non_linearity[k, :, :]*(frame**k)

                new_integ[g, :, :] = corr

            self.data[i, :, :, :] = deepcopy(new_integ)*gain  # [electrons] convert back to electrons for next steps.

        self.modif_str = self.modif_str + '_nonlin'
    
    def add_detector_noise(self, offset=500., gain=GAIN, pca0_file=None, noise_seed=None, dark_seed=None):
        """Add read-noise, 1/f noise, kTC noise, and alternating column noise
        using the HxRG noise generator.
        """

        # In the current implementation the pca0 file goes unused, but it is a mandatory input of HxRG.
        if pca0_file is None:
            pca0_file = self.noisefiles_dir+'/niriss_pca0.fits'

        if noise_seed is None:
            noise_seed = 7 + int(np.random.uniform() * 4000000000.)

        if dark_seed is None:
            dark_seed = 5 + int(np.random.uniform() * 4000000000.)

        np.random.seed(dark_seed)

        # Define noise parameters.
        # White read-noise.
        rd_noise = 12.95  # [electrons]

        # Correlated pink noise.
        c_pink = 9.6  # [electrons]

        # Uncorrelated pink noise.
        u_pink = 3.2  # [electrons]

        # Alternating column noise.
        acn = 2.0  # [electrons]

        # PCA0 (picture frame) noise.
        pca0_amp = 0.  # Do not use PCA0 component.

        # Bias pattern.
        bias_amp = 0.  # Do not use PCA0 component.
        bias_offset = offset*gain  # [electrons]

        # Dark current.
        dark_current = 0.0  # [electrons/frame] Unused because pca0_amp = 0.

        # Pedestal drifts.
        pedestal = 18.30  # [electrons] Unused because pca0_amp = 0.

        # Define the HXRGN instance (in detector coordinates).
        noisegenerator = hxrg.HXRGNoise(naxis1=self.ncols, naxis2=self.nrows, naxis3=self.ngroups, pca0_file=pca0_file,
                                        x0=0, y0=0, det_size=2048, verbose=False)

        # Iterate over integrations
        for i in range(self.nintegs):

            # Choose a new random seed for this iteration.
            seed1 = noise_seed + 24*int(i)

            # Generate a noise-cube for this integration.
            noisecube = noisegenerator.mknoise(c_pink=c_pink, u_pink=u_pink, bias_amp=bias_amp, bias_offset=bias_offset,
                                               acn=acn, pca0_amp=pca0_amp, rd_noise=rd_noise, pedestal=pedestal,
                                               dark_current=dark_current, dc_seed=dark_seed, noise_seed=seed1,
                                               gain=gain)

            # Ensure the noise-cube has the correct dimensions (when Ngroups = 1).
            if noisecube.ndim == 2:
                noisecube = noisecube[np.newaxis, :, :]

            # Change from detector coordinates to science coordinates.
            noisecube = np.transpose(noisecube, (0, 2, 1))
            noisecube = noisecube[::, ::-1, ::-1]

            # Add the detector noise to the simulation.
            self.data[i] = self.data[i] + noisecube

        self.modif_str = self.modif_str + '_detector'

    def apply_flatfield(self, flatfile=None):
        """Apply the flat field correction to the simulation."""

        if flatfile is None:
            flatfile = self.noisefiles_dir+'/jwst_niriss_flat_0181.fits'

        # Read the flat-field from file (in science coordinates).
        with fits.open(flatfile) as hdu:
            flatfield = hdu[1].data

        # Select the appropriate subarray.
        if self.subarray == 'SUBSTRIP96':
            slc = slice(1792, 1888)
        elif self.subarray == 'SUBSTRIP256':
            slc = slice(1792, 2048)
        elif self.subarray == 'FULL':
            slc = slice(0, 2048)
        else:
            raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')

        subflat = flatfield[slc, :]

        # Apply the flatfield to the simulation.
        self.data = self.data*subflat

        self.modif_str = self.modif_str + '_flat'

    def add_superbias(self, gain=GAIN, biasfile=None):
        """Add the bias level to the simulation."""

        if biasfile is None:
            biasfile = self.noisefiles_dir+'/jwst_niriss_superbias_0137.fits'

        # Read the super bias from file (in science coordinates).
        with fits.open(biasfile) as hdu:
            superbias = hdu[1].data  # [ADU]

        superbias = superbias*gain  # [electrons]

        # Select the appropriate subarray.
        if self.subarray == 'SUBSTRIP96':
            slc = slice(1792, 1888)
        elif self.subarray == 'SUBSTRIP256':
            slc = slice(1792, 2048)
        elif self.subarray == 'FULL':
            slc = slice(0, 2048)
        else:
            raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')

        subbias = superbias[slc, :]

        # Add the bias level to the simulation.
        self.data = self.data + subbias

        self.modif_str = self.modif_str + '_bias'

    def add_simple_dark(self, darkvalue=DARKVALUE):  # TODO dark should be lower in the voids.
        """Add a simple dark current to the simulation.

        .. note::
        Uses 0.0414 electrons/s by default. Taken from Jdox on 04-May-2020, note that the actual dark current
        is lower in the voids.
        """

        # Generate the dark ramps for the simulation.
        # TODO loop over integrations to save memory?
        dark = rdm.poisson(darkvalue*self.tgroup, size=self.data.shape).astype('float32')  # [electrons]
        darkramp = np.cumsum(dark, axis=1)

        # Add the dark ramps to the simulation.
        self.data = self.data + darkramp

        self.modif_str = self.modif_str + '_dark'

    def add_zodiacal_background(self, zodifile=None):
        """Add the zodiacal background signal to the simulation."""

        if zodifile is None:
            zodifile = self.noisefiles_dir+'/background_detectorfield_normalized.fits'

        # Read the background file.
        with fits.open(zodifile) as hdu:
            zodiimage = hdu[0].data  # [electrons/s]

        # Select the appropriate subarray.
        if self.subarray == 'SUBSTRIP96':
            slc = slice(1792, 1888)
        elif self.subarray == 'SUBSTRIP256':
            slc = slice(1792, 2048)
        elif self.subarray == 'FULL':
            slc = slice(0, 2048)
        else:
            raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')

        subzodi = zodiimage[slc, :]

        # Scale to the exposure time, and match shape to integrations.
        subzodi = subzodi*self.tgroup  # [electrons]
        subzodi = np.tile(subzodi[np.newaxis, :, :], (self.ngroups, 1, 1))

        for i in range(self.nintegs):

            # Add poisson noise, and convert to up the ramp samples.
            zodiramp = rdm.poisson(subzodi)
            zodiramp = np.cumsum(zodiramp, axis=0)

            # Add the background ramps to the simulation.
            self.data[i] = self.data[i] + zodiramp

        self.modif_str = self.modif_str + '_zodibackg'

    def write_to_fits(self, filename=None, gain=GAIN):
        """Write to a .fits file the new header and data."""

        hdu_new = self.hdu_ideal
        hdu_new[1].data = (self.data/gain).astype('uint16')  # Convert to ADU in 16 bit integers.

        if filename is None:
            dir_and_filename, suffix = os.path.splitext(self.ima_path)
            #filename = self.output_path +os.path.basename(self.ima_path) + self.modif_str + '.fits'
            filename = dir_and_filename + self.modif_str + '.fits'
            hdu_new.writeto(filename, overwrite=True)
        elif filename is True:
            hdu_new.writeto(filename, overwrite=True)

        print('Writing to file: ' + filename)

    def plot_image(self, i_group=0, i_integ=0, log=False, reverse_y=True, save=False, filename=None):
        """Plot the detector image for a chosen frame."""

        img = self.data[i_integ, i_group, :, :]

        fig, ax = plt.subplots(1, 1, figsize=(8, 3))

        if log:
            im = ax.imshow(np.log10(img))
            ax.set_title('log10 Group {}; Integ {}'.format(i_group, i_integ))
        else:
            im = ax.imshow(img)
            ax.set_title('Group {}; Integ {}'.format(i_group, i_integ))

        if reverse_y:
            ax.invert_yaxis()

        fig.colorbar(im, ax=ax, orientation='horizontal')
        plt.tight_layout()

        # option to save the image
        if save:
            if filename is None:
                filename = 'image_G{}_I{}.png'.format(i_group, i_integ)
            fig.savefig(filename)

    def plot_pixel(self, i_row=1380, i_col=55, plot_on_im=True, save=False, filename=None):
        """Plot the flux in a given pixel as a function of Frame Number."""

        # to distinguish integrations and groups in plotting
        colors = ['b', 'orange', 'g', 'red']
        markers = ['o', '^', '*']
        count = 0

        if plot_on_im:  # if True, plot location of pixel on the first image
            fig, (ax2, ax) = plt.subplots(2, 1, figsize=(7, 5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 3))

        for i in range(self.nintegs):
            for j in range(self.ngroups):
                ax.plot(count, self.data[i, j, i_col, i_row], marker=markers[j % 3], color=colors[i % 4], ls='')
                count = count + 1

        ax.set_xlabel('Frames')
        ax.set_ylabel('Pixel count')

        ax.set_title('Row {}; Column {}'.format(i_row, i_col))

        # ---- In addition, plot location of pixel on image --- #
        if plot_on_im:
            img = self.data[0, 0, :, :]
            ax2.imshow(img)
            ax2.plot(i_row, i_col, marker='x', color='r')
            ax2.invert_yaxis()
            ax2.set_title('Group {}; Integ {}'.format(0, 0))

        # option to save the image
        if save:
            if filename is None:
                filename = 'pixel_{}_{}.png'.format(i_row, i_col)
            fig.savefig(filename)


def main():

    return


if __name__ == '__main__':
    main()
