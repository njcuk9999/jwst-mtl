"""
Created on Sun Jan 26 16:39:05 2020

@author: caroline

TimeSeries objects for simulations of SOSS observations
"""
import sys

# General imports.
from copy import deepcopy
from pkg_resources import resource_filename
import os

# General science imports.
import numpy as np
import numpy.random as rdm

# Astronomy imports.
from astropy.io import fits
import astropy.io.fits as pyfits


# Home-brew and intra module imports.
#from . import hxrg
import hxrg

#Cosmic rays import
import addCRs2Exposure


# Plotting.
import matplotlib.pyplot as plt

# To download reference files from CRDS
from astropy.utils.data import download_file
import shutil
from time import sleep


# TODO header section which files and values were used.


def download_ref_files(noisefiles_path, fitsname,
                       crds_http='https://jwst-crds.stsci.edu/unchecked_get/references/jwst/'):
                       #crds_http='https://jwst-crds.stsci.edu/browse/'):
    """One by one, check that the ref files are already in the noisefile_path and download if necessary"""

    local_name = os.path.join(noisefiles_path, fitsname)
    http_name = os.path.join(crds_http, fitsname)
    if os.path.isfile(local_name) is False:
        print('Attempting to download from CRDS ',http_name)
        while True:
            try:
                file_path = download_file(http_name)
                break
            except OSError:
                message = 'HTTPError: HTTP Error 503: Service Temporarily Unavailable'
                print('Can not connect to CRDS to download', http_name, ' trying every second.')
                sleep(1)
        shutil.copy(file_path, local_name)
        os.remove(file_path)
        # change file access rights so anybody can edit
        os.chmod(local_name, 0o666)


class TimeSeries(object):

    def __init__(self, ima_path, noisefiles_path, gain=1.6221, dark_value=0.0414, full_well=72000, ):
        """Make a TimeSeries object from a series of synthetic images."""

        self.ima_path = ima_path

        hdu_ideal = fits.open(ima_path)  # read in fits file
        header = hdu_ideal[1].header

        self.hdu_ideal = hdu_ideal
        self.data = np.array(hdu_ideal[1].data, dtype=np.float64)  # image to be altered

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
        #self.noisefiles_dir = '/genesis/jwst/jwst-ref-soss/noise_files/' # PATH where reference detector noise files can be found.
        self.noisefiles_dir = noisefiles_path
        # Same here, we need to pass this or read it from teh config path
        # USER_PATH is the parameter in that file
        self.output_path = '/genesis/jwst/userland-soss/'

        # Gain, dark, full well
        self.gain = gain # 1.6221 obtained from CRDS reference file pre-Commissioning
        self.dark_value = dark_value
        self.full_well = full_well

        # Reference files defaults
        self.darkdir_ss256 = '/genesis/jwst/jwst-ref-soss/darks_SS256/'
        self.darkdir_ss96 = '/genesis/jwst/jwst-ref-soss/darks_SS96/'
        self.darkdir_full = '/genesis/jwst/jwst-ref-soss/darks_FULL/'

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

    def add_non_linearity(self, coef_file=None):
        """Add non-linearity on top of the linear integration-long ramp."""

        if coef_file is None:
            coef_file = 'jwst_niriss_linearity_0011_bounds_0_60000_npoints_100_deg_5.fits'

        print('\tUsing {:} as the non-linearity coefficients reference file'.format(coef_file))

        # Read the coefficients of the non-linearity function.
        with fits.open(self.noisefiles_dir+coef_file) as hdu:
            non_linearity = hdu[0].data

        ncoeffs = non_linearity.shape[0]

        # Select the appropriate subarray.
        if self.subarray == 'SUBSTRIP96':
            slc = slice(1802, 1898)
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
            new_integ = deepcopy(self.data[i, :, :, :])/self.gain  # [ADU] because the non-linearity correction works on ADU.

            # Iterate over groups
            for g in range(self.ngroups):
                frame = deepcopy(new_integ[g, :, :])

                corr = non_linearity[0, :, :]  # Constant.
                for k in range(1, ncoeffs):  # Higher degree terms.
                    corr = corr + non_linearity[k, :, :]*(frame**k)

                new_integ[g, :, :] = corr

            self.data[i, :, :, :] = deepcopy(new_integ)*self.gain  # [electrons] convert back to electrons for next steps.

        self.modif_str = self.modif_str + '_nonlin'

    def add_cv3_dark(self):
        """Use the pool of CV3 darks (typically 25 files with nint=3 ngroup=50
        to add detector noise to our simulations.
        """

        # Select the appropriate subarray.
        if self.subarray == 'SUBSTRIP96':
            ls = self.subprocess.getoutput('ls -1 '+self.darkdir_ss96+' *.fits')
            darklist = np.array(ls.split('\n'))
        elif self.subarray == 'SUBSTRIP256':
            ls = self.subprocess.getoutput('ls -1 ' + self.darkdir_ss256 + ' *.fits')
            darklist = np.array(ls.split('\n'))
        elif self.subarray == 'FULL':
            #TODO: copy CV3 full frame darks to the genesis darks directory
            print('FULL frame darks files missing. Ask Loic to copy them from CV3.')
            sys.exit()
            #ls = subprocess.getoutput('ls -1 '+self.darkdir_full+' *.fits')
            #darklist = np.array(ls.split('\n'))
        else:
            raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')

        # Randomly pick files and extract the necessary number of groups

        # TODO: Complete the add_cv3_dark function to implement capability of inserting real CV3 darks


    def add_readout_noise(self, rms=13.8903):
        """Add white readout noise"""
        """This is single-read readout noise rms """

        #TODO: handle FULL or GENERIC case - actually dont`read the ref file readnoise
        # rms=13.8903 dans le cas SS256 ou SS96
        # rms=10.34 dans le cas FULL
        #TODO: do same for gain: 1.6221
        # 20220120 - Antoine found that using rms=13.8903 produced
        # exactly a factor sqrt(2) higher readout noise in the final rateints images
        # TODO: use rms = 13.8903 / sqrt(2)

        mynoise = np.random.standard_normal(np.size(self.data)) * rms / np.sqrt(2)
        #print(np.std(mynoise))
        mynoise = np.reshape(mynoise, (self.nintegs, self.ngroups, self.ncols, self.nrows))
        #print(np.shape(mynoise))

        self.data += np.copy(mynoise)

        self.modif_str = self.modif_str + '_readnoise'

    def add_1overf_noise(self, c_pink = 9.6, alpha = -1):
        """Extracted from HxRGNoise """

        # Correlated 1/f noise (pink noise)
        # c_pink = 9.6  # [electrons]
        # alpha = -1  # Hard code for 1/f noise until proven otherwise

        # Select the appropriate subarray.
        if self.subarray == 'SUBSTRIP96':
            cols = 96
            rows = 2048
            cols_over = 12
            rows_over = 2
            amps = 1
        elif self.subarray == 'SUBSTRIP256':
            cols = 256
            rows = 2048
            cols_over = 12
            rows_over = 2
            amps = 1
        elif self.subarray == 'FULL':
            cols = 2048
            rows = 2048
            cols_over = 12
            rows_over = 1
            amps = 4
        else:
            raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')

        # Generate 1/f noise on a per integration basis (so not to run out of memory)
        tt = []
        for nint in range(self.nintegs):

            # naxis1 is in the detector orientation
            #nstep = (cols // amps + cols_over) * (rows + rows_over) * self.ngroups * self.nintegs
            nstep = (cols // amps + cols_over) * (rows + rows_over) * self.ngroups
            # Pad nsteps to a power of 2, which is much faster (JML)
            nstep2 = int(2 ** np.ceil(np.log2(nstep)))

            # Define frequency arrays
            f2 = np.fft.rfftfreq(nstep2)  # ... for 2*nstep elements

            # Define pinkening filters. F2 and p_filter2 are used to generate 1/f noise.
            p_filter2 = np.sqrt(f2 ** alpha)
            p_filter2[0] = 0.
            # Generate seed noise
            mynoise = np.random.standard_normal(nstep2)

            # Save the mean and standard deviation of the first
            # half. These are restored later. We do not subtract the mean
            # here. This happens when we multiply the FFT by the pinkening
            # filter which has no power at f=0.
            the_mean = np.mean(mynoise[:(2*nstep2) // 2])
            the_std = np.std(mynoise[:(2*nstep2) // 2])

            # Apply the pinkening filter.
            thefft = np.fft.rfft(mynoise)
            thefft = np.multiply(thefft, p_filter2)
            result = np.fft.irfft(thefft)
            result = result[:(2*nstep) // 2]  # Keep 1st half of nstep

            # Restore the mean and standard deviation
            result *= the_std / np.std(result)
            result = result - np.mean(result) + the_mean

            tt_1_integ = c_pink * result  # tt_1_integ is a temp. variable

            # tt is the full exposure temp. variable
            tt.append(np.copy(tt_1_integ))

        # CONvert list to np.array
        tt = np.array(tt)

        # Reshape the time series into the subarray (detector coordinate system, not DMS yet)
        if (self.subarray == 'SUBSTRIP96') | (self.subarray == 'SUBSTRIP256'):
            # 3D, with rows and cols overheads
            tt = np.reshape(tt, (self.nintegs, self.ngroups, rows + rows_over, cols + cols_over))
            # Remove rows/cols overheads
            tt = tt[:, :, :rows, :cols]
            # Apply coordinate transform: detector --> DMS
            tt = np.swapaxes(tt, 2, 3)

        elif self.subarray == 'FULL':
            # 3D, with rows and cols overheads
            tt = np.reshape(tt, (self.nintegs, self.ngroups, rows + rows_over, cols // amps + cols_over))
            # Remove rows/cols overheads to produce ninteg x ngroup x 2048 x 512 matrix
            tt = tt[:, :, :rows, :cols // amps]
            # Replicate this amp to the next 3 amps, respecting the readout direction
            # -->|<--|-->|<--
            temp = np.zeros((self.nintegs, self.ngroups, rows, cols))
            fivehundredtwelve = cols // amps
            temp[:, :, :, :fivehundredtwelve] = np.copy(tt)
            temp[:, :, :, fivehundredtwelve:2*fivehundredtwelve] = np.copy(tt[:, :, :, ::-1])
            temp[:, :, :, 2*fivehundredtwelve:3*fivehundredtwelve] = np.copy(tt)
            temp[:, :, :, 3*fivehundredtwelve:4*fivehundredtwelve] = np.copy(tt[:, :, :, ::-1])
            tt = np.copy(temp)
            del temp
            # Apply coordinate transform: detector --> DMS
            tt = np.swapaxes(tt, 2, 3)
        else:
            raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')

        self.data += deepcopy(tt)

        self.modif_str = self.modif_str + '_1overf'

    def apply_flatfield(self, flatfile=None):
        """Apply the flat field correction to the simulation."""

        #TODO: Find the correct flatsss in CRDS
        if flatfile is None:
            if self.subarray == 'SUBSTRIP256': flatfile = 'jwst_niriss_flat_0190.fits'
            elif self.subarray == 'SUBSTRIP96': flatfile = 'jwst_niriss_flat_0190.fits'
            elif self.subarray == 'FULL': flatfile = 'jwst_niriss_flat_0190.fits'
            else:
                raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')
        # Check that the ref file is on local disk and download if required
        print('\tUsing {:} as the flat field reference file'.format(flatfile))
        download_ref_files(self.noisefiles_dir, os.path.basename(flatfile))
        # Complete full path
        flatfile = self.noisefiles_dir+os.path.basename(flatfile)

        # Read the flat-field from file (in science coordinates).
        with fits.open(flatfile) as hdu:
            flatfield = hdu[1].data

        # As of Nov 1 2021, the flat ref files 0190 is a 2048x2048 file, so need to
        # pick the subarray here
        # TODO: Update this once the CRDS have a separate reference file for the different subarrays
        if self.subarray == 'SUBSTRIP256': flatfield = flatfield[2048-256:2048,:]
        elif self.subarray == 'SUBSTRIP96': flatfield = flatfield[2048-106:2048-10,:]

        # Apply the flatfield to the simulation.
        self.data = self.data * flatfield

        # Append that step to the filename.
        self.modif_str = self.modif_str + '_flat'

    def add_superbias(self, biasfile=None):
        """Add the bias level to the simulation."""

        if biasfile is None:
            if self.subarray == 'SUBSTRIP256': biasfile = 'jwst_niriss_superbias_0120.fits'
            elif self.subarray == 'SUBSTRIP96': biasfile = 'jwst_niriss_superbias_0111.fits'
            elif self.subarray == 'FULL': biasfile = 'jwst_niriss_superbias_0150.fits'
            else:
                raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')

        # Check that the ref file is on local disk and download if required
        print('\tUsing {:} as the super bias reference file'.format(biasfile))
        download_ref_files(self.noisefiles_dir, os.path.basename(biasfile))
        # Complete full path
        biasfile = self.noisefiles_dir+os.path.basename(biasfile)

        # Read the flat-field from file (in science coordinates).
        with fits.open(biasfile) as hdu:
            superbias = hdu[1].data #ADU

        # TODO: superbias reference file should have its reference pixels not set to zero
        if False:
            # Fix that by adding an arbitrary level to prevent wrapping bug at 65535.
            if self.subarray == 'SUBSTRIP96':
                superbias[:, 0:4] =+ 10000
                superbias[:, 2044:2048] =+ 10000
            elif self.subarray == 'SUBSTRIP256':
                superbias[:, 0:4] =+ 10000
                superbias[:, 2044:2048] =+ 10000
                superbias[252:256, :] =+ 10000
            elif self.subarray == 'FULL':
                superbias[:, 0:3] =+ 10000
                superbias[:, 2044:2048] =+ 10000
                superbias[2044:2048, :] =+ 10000
                superbias[0:4, :] =+ 10000

        superbias = superbias*self.gain  # [electrons]

        # Add the bias level to the simulation.
        self.data = self.data + superbias

        # Append that step to the filename.
        self.modif_str = self.modif_str + '_bias'

    def add_dark(self, darkfile=None):
        """Add dark current to the simulation.

        .. note::
        Uses 0.0414 electrons/s by default. Taken from Jdox on 04-May-2020, note that the actual dark current
        is lower in the voids.
        """

        # OLD METHOD - scalar value across the detector
        if False:
            # Generate the dark ramps for the simulation.
            # TODO loop over integrations to save memory?
            dark = rdm.poisson(self.dark_value*self.tgroup, size=self.data.shape).astype('float32')  # [electrons]
            darkramp = np.cumsum(dark, axis=1)

        if darkfile is None:
            if self.subarray == 'SUBSTRIP256': darkfile = 'jwst_niriss_dark_0147.fits'
            elif self.subarray == 'SUBSTRIP96': darkfile = 'jwst_niriss_dark_0150.fits'
            elif self.subarray == 'FULL': darkfile = 'jwst_niriss_dark_0145.fits'
            else:
                raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')
        # Check that the ref file is on local disk and download if required
        print('\tUsing {:} as the dark reference file'.format(darkfile))
        download_ref_files(self.noisefiles_dir, os.path.basename(darkfile))
        # Complete full path
        darkfile = self.noisefiles_dir+os.path.basename(darkfile)

        # Read the flat-field from file (in science coordinates).
        with fits.open(darkfile) as hdu:
            darkramp = hdu[1].data #ADU

        # Convert dark current to electrons
        darkramp = darkramp * self.gain  # [electrons]

        # Initialize the dark exposure
        dark_exposure = np.zeros((self.nintegs, self.ngroups, self.ncols, self.nrows))
        # Take one 3D ramp of the dark and copy it to all integrations
        dark_exposure[:][:, :, :] = darkramp[self.ngroups, :, :]

        # Add Poisson noise to the dark exposure
        # Can be done without loops, but this reduces memory requirements.
        for i in range(self.nintegs):

            ramp = deepcopy(dark_exposure[i])

            # Convert up the ramp samples, to flux between reads.
            ramp[1:] = np.diff(ramp, axis=0)

            # Add the poisson noise.
            ramp = np.where(ramp < 0, 0, ramp)  # Sanity check.
            ramp = rdm.poisson(ramp)

            # Convert back to up the ramp samples.
            ramp = np.cumsum(ramp, axis=0)

            dark_exposure[i] = deepcopy(ramp)

        # Add the dark+noise to the simulation.
        self.data = self.data + dark_exposure

        # Free memory
        del dark_exposure

        # Append that step to the filename.
        self.modif_str = self.modif_str + '_dark'

    def add_zodiacal_background(self, zodifile=None):
        """Add the zodiacal background signal to the simulation."""

        if zodifile is None:
            zodifile = self.noisefiles_dir+'background_detectorfield_normalized.fits'

        # Read the background file.
        with fits.open(self.noisefiles_dir+os.path.basename(zodifile)) as hdu:
            zodiimage = hdu[0].data  # [electrons/s]

        # Select the appropriate subarray.
        if self.subarray == 'SUBSTRIP96':
            slc = slice(1802, 1898)
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

        #TODO: Check that the poisson noise for zodi background is done right, see dark for reference
        for i in range(self.nintegs):

            # Add poisson noise, and convert to up the ramp samples.
            zodiramp = rdm.poisson(subzodi)
            zodiramp = np.cumsum(zodiramp, axis=0)

            # Add the background ramps to the simulation.
            self.data[i] = self.data[i] + zodiramp

        self.modif_str = self.modif_str + '_zodibackg'


    def write_to_fits(self, filename=None):
        """Write to a .fits file the new header and data.
        units are converted from electrons back to ADU for this step.
        """
        hdu_new = self.hdu_ideal
        #hdu_new[1].data = (self.data/self.gain).astype('uint16')  # Convert to ADU in 16 bit integers.
        #hdu_new[1].data = (self.data/self.gain).astype('int16')  # Convert to ADU in 16 bit integers.
        hdu_new[1].data = (self.data / self.gain).astype('float32')  # Convert to ADU in 16 bit integers.

        if filename is None:
            print('Forging output noisy file...')
            dir_and_filename, suffix = os.path.splitext(self.ima_path)
            #filename = self.output_path +os.path.basename(self.ima_path) + self.modif_str + '.fits'
            filename = dir_and_filename + self.modif_str + '.fits'

        print('Writing to file: ' + filename)
        hdu_new.writeto(filename, overwrite=True)

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

    def get_normfactor(self):
        """Determine a re-normalization factor so that the highest pixel value in the simulation
         will match the full well capacity"""

        raise Warning('get_normfactor, DEPRECATED FUNCTION - DO NOT USE. Simulations are now flux calibrated.')

        max_value = np.amax(self.data)
        normfactor = self.full_well/max_value

        return normfactor

    def apply_normfactor(self, normfactor):
        """Apply an arbitrary re-normalization to the simulations."""

        raise Warning('apply_normfactor, DEPRECATED FUNCTION - DO NOT USE. Simulations are now flux calibrated.')

        self.data = self.data*normfactor

        self.modif_str = self.modif_str + '_norm'


    def add_detector_noise(self, offset=500., pca0_file=None, noise_seed=None, dark_seed=None):
        """Add read-noise, 1/f noise, kTC noise, and alternating column noise
        using the HxRG noise generator.
        """

        """ DEPRECATED. DO NOT USE ANYMORE. Instead call add_1overf_noise and add_readout_noise"""

        raise Warning('Do not use add_detector_noise anymore. Instead use add_1overf_noise and add_readout_noise.')

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
        bias_offset = offset*self.gain  # [electrons]

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
                                               dark_current=self.dark_current, dc_seed=dark_seed, noise_seed=seed1,
                                               gain=self.gain)

            # Ensure the noise-cube has the correct dimensions (when Ngroups = 1).
            if noisecube.ndim == 2:
                noisecube = noisecube[np.newaxis, :, :]

            # Change from detector coordinates to science coordinates.
            noisecube = np.transpose(noisecube, (0, 2, 1))
            noisecube = noisecube[::, ::-1, ::-1]

            # Add the detector noise to the simulation.
            self.data[i] = self.data[i] + noisecube

        self.modif_str = self.modif_str + '_detector'

    def add_cosmic_rays(self, sun_activity = 'SUNMIN'):

        if((sun_activity  != 'SUNMIN') and (sun_activity != 'SUNMAX') and (sun_activity != 'FLARES')):

            print ('sun_activity = ', sun_activity, ' is not supported... exiting')
            sys.exit()

        filesin = [self.ima_path] #The file on which we wanna run the code

        #np.random.seed(13578) #so we have always the same results

        #addCRs2Exposure.run(InputDir+filesin[0], 'SUNMIN', OutputDir)

        for f in filesin:
            addCRs2Exposure.run(f, sun_activity, self.output_path)



def main():

    return


if __name__ == '__main__':
    main()
