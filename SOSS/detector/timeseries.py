
"""
Created on Sun Jan 26 16:39:05 2020

@author: caroline

TimeSeries objects for simulations of SOSS observations
"""

# Import required modules

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from copy import deepcopy
import numpy.random as rdm
import hxrg
from pkg_resources import resource_filename


class TimeSeries(object):

    def __init__(self, ima_path):
        """
        Make a TimeSeries object from a series of synthetic images
        """

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
        self.tgroup = hdu_ideal[0].header['TGROUP']  # TODO Uses as exposure time per frame.

        self.modif_str = '_mod'  # string encoding the modifications

    def add_poisson_noise(self):
        """
        Adds Poisson noise to each group, looping over integrations
        """

        for i in range(self.nintegs):
            # for each integration, re-initialize Poisson Noise
            # to remember the previous frame for Poisson noise calculations

            # counts in previous image for noise-free case
            prev_frame_theo = None
            prev_frame_noisy = None

            for g in range(self.ngroups):
                frame = deepcopy(self.data[i, g, :, :])
                frame[np.where(frame < 0.)] = 0.  # sanity check

                if prev_frame_theo is not None:
                    diff_theo = frame-prev_frame_theo
                    noisy_frame = prev_frame_noisy + rdm.poisson(diff_theo)

                else:
                    noisy_frame = rdm.poisson(frame)

                # store for next step
                prev_frame_theo = deepcopy(frame)
                prev_frame_noisy = noisy_frame

                self.data[i, g, :, :] = deepcopy(noisy_frame)

        self.modif_str = self.modif_str + '_poisson_noise'

    def add_non_linearity(self, non_linearity):
        """
        Add non-linearity on top of the linear integration-long ramp
        non_linearity: array of polynomial coefficients
        offset: removed prior to correction and put back after
        """

        # Add non-linearity to each ramp
        for i in range(self.nintegs):
            # select part of the time series lying within this integration
            # (i.e., ith sequence of ngroups groups)
            integ = self.data[i, :, :, :]

            # Apply offset before introducing non-linearity
            new_integ = deepcopy(integ)

            # Iterate over groups
            for g in range(self.ngroups):
                frame = deepcopy(new_integ[g, :, :])

                ncoeffs = non_linearity.shape[0]

                corr = non_linearity[ncoeffs-1, :, :]
                for k in range(1, ncoeffs):
                    corr = corr + non_linearity[-k-1, :, :] * frame**k

                new_integ[g, :, :] = corr

            self.data[i, :, :, :] = deepcopy(new_integ)

        self.modif_str = self.modif_str+'_nonlin'
    
    def add_detector_noise(self, offset=500., gain=1.61, pca0_file=None, noise_seed=None, dark_seed=None):
        """
        Add read-noise, 1/f noise, kTC noise, and alternating column noise using the HxRG noise generator.
        """

        # In the current implementation the pca0 file goes unused, but it is a mandatory input of HxRG.
        if pca0_file is None:
            pca0_file = resource_filename('detector', 'files/niriss_pca0.fits')

        if noise_seed is None:
            noise_seed = 7 + int(np.random.uniform() * 4000000000.)

        if dark_seed is None:
            dark_seed = 5 + int(np.random.uniform() * 4000000000.)

        np.random.seed(dark_seed)

        # Make empty data array
        detector_noise = np.zeros([self.nintegs, self.ngroups, self.ncols, self.nrows], dtype=np.float32)

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
        bias_offset = offset * gain  # [electrons]

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
            seed1 = noise_seed + 24 * int(i)

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

            # Add the final noise array.
            detector_noise[i, :, :, :] = np.copy(noisecube)

        self.data = self.data + detector_noise

        self.modif_str = self.modif_str + '_detector'

    def apply_flatfield(self, flatfile=None):
        """
        Apply the flat field correction to the simulation.
        """

        if flatfile is None:
            flatfile = resource_filename('detector', 'files/jwst_niriss_flat_0181.fits')

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

        # Apply to the simulation.
        self.data = self.data * subflat

        self.modif_str = self.modif_str + '_flat'

    def add_superbias(self, gain=1.61, biasfile=None):
        """
        Add the bias level to the simulation.
        """

        if biasfile is None:
            biasfile = resource_filename('detector', 'files/jwst_niriss_superbias_0137.fits')

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

    def add_simple_dark(self, darkvalue=0.0414):  # TODO dark should be lower in the voids.
        """
        Add a simple dark current to the simulation.

        - Uses 0.0414 electrons/s by default. Taken from Jdox on 04-May-2020, note that the actual dark current is lower
        in the voids.

        """

        # Generate the dark ramps for the simulation.
        dark = rdm.poisson(darkvalue*self.tgroup, size=self.data.shape)  # [electrons]
        darkramp = np.cumsum(dark, axis=1)

        # Add the dark ramps to the simulation.
        self.data = self.data + darkramp

        self.modif_str = self.modif_str + '_dark'

    def write_to_fits(self, filename=None):
        """
        Write to a fits file the new header and data
        """

        hdu_new = self.hdu_ideal
        hdu_new[1].data = self.data

        if filename is None:
            filename = self.ima_path[:-5]+self.modif_str+'.fits'
            hdu_new.writeto(filename, overwrite=True)

        print('Writing to file: '+filename)

    def plot_image(self, i_group=0, i_integ=0, log=False, reverse_y=True,
                   save=False, filename=None):
        """
        Plot the detector image for a chosen frame
        """

        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        img = self.data[i_integ, i_group, :, :]
        if log:
            im = ax.imshow(np.log10(img))
            if reverse_y:
                ax.invert_yaxis()
            ax.set_title('log10 Group '+str(i_group)+'; Integ '+str(i_integ))
        else:
            im = ax.imshow(img)
            if reverse_y:
                ax.invert_yaxis()
            ax.set_title('Group '+str(i_group)+'; Integ '+str(i_integ))

        fig.colorbar(im, ax=ax, orientation='horizontal')
        plt.tight_layout()

        # option to save the image
        if save:
            if filename is None:
                filename = 'image_G'+str(i_group)+'_I'+str(i_integ)+'.png'
            fig.savefig(filename)

    def plot_pixel(self, i_row=1380, i_col=55,
                   plot_on_im=True, save=False, filename=None):
        """
        Plot the flux in a given pixel as a function of Frame #
        """

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
                ax.plot(count, self.data[i, j, i_col, i_row],
                        marker=markers[j % 3], color=colors[i % 4], ls='')
                count = count + 1

        ax.set_xlabel('Frames')
        ax.set_ylabel('Pixel count')

        ax.set_title('Row '+str(i_row)+'; Column '+str(i_col))

        # ---- In addition, plot location of pixel on image --- #
        if plot_on_im:
            img = self.data[0, 0, :, :]
            ax2.imshow(img)
            ax2.plot(i_row, i_col, marker='x', color='r')
            ax2.invert_yaxis()
            ax2.set_title('Group '+str(0)+'; Integ '+str(0))

        # option to save the image
        if save:
            if filename is None:
                filename = 'pixel_'+str(i_row)+'_'+str(i_col)+'.png'
            fig.savefig(filename)


def main():

    return


if __name__ == '__main__':
    main()
