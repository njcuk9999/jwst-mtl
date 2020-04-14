
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

#%%


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

        # number of groups per integration
        self.ngroups = header['NAXIS3']

        # number of integrations in time series observations
        self.nintegs = header['NAXIS4']

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

    def plot_pixel(self, i_row=1380, i_col=55, marker='o', color='b',
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
