#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Apr 05 2:08 2022

@author: MCR

Checkpoint plotting functions to compare SOSS data at different stages of
analysis via different pipelines.
"""

from astropy.io import fits
import matplotlib.backends.backend_pdf
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from jwst import datamodels


class DecontaminationPlot:
    """ Base class to assess the quality of the ATOCA SOSS spectral order
    deconamination.

    Attributes
    ----------
    datacube : array-like
        Data cube for SOSS observational data.
    errorcube : array-like
        Data cube of errors associated with each flux value in datacube.
    o1model : array-like
        Data cube of decontamnted spatial profiles for the first SOSS order.
    o2model : array-like
        Data cube of decontamnted spatial profiles for the second SOSS order.

    Methods
    -------
    make_decontamination_plot
        Plot data - model to assess the quality of the decontamination.
    """

    def __init__(self, datafile, modelfile):
        """ Initializer for DecontaminationPlot.

        Parameters
        ----------
        datafile : str
            Path to fits file containing the SOSS observations. Ideally the
             file passed to Extract1dStep.
        modelfile : str
            Path to fits file containing the decontaminated SOSS profiles as
            output by ATOCA. Generally will have suffix
            '_SossExtractModel.fits'. The soss_modelname parameter must be
            specified in Extract1dStep for ATOCA to produce these
        """
        # The DMS datamodels containers provide a useful way to unpack the fits
        # files produced by the DMS.
        try:
            # Unpack the science observations and associated errors.
            data = datamodels.open(datafile)
            self.datacube = data.data
            self.errorcube = data.err
            # Unpack the decontaminated first and second order profiles
            # produced by ATOCA.
            model = datamodels.open(modelfile)
            self.o1model = model.order1
            self.o2model = model.order2
        # Depending on the DMS version that the user has installed, there is a
        # bug in the ASDF parsing (I think) which prevents datamodels.open
        # from functioning properly. If this occurs, fall back on astopy fits
        # handling to unpack the files
        except AttributeError:
            self.datacube, self.errorcube = _fits_fallback(datafile, [1, 2])
            self.o1model, self.o2model = _fits_fallback(modelfile, [2, 3])

    def make_decontamination_plot(self, integrations=0, savefile=None,
                                  cbar_lims=[-5, 5], **imshow_kwargs):
        """Make a plot showing the results of the decontamination performed by
         ATOCA.

        Parmeters
        ---------
        integrations : int, array-like, str
            integration, or list of integrations to show the decontamination
             results. Pass 'all' to plot decontamination for every integration.
        savefile : str, optional
            To save all plots to a pdf instead of showing them, pass a file
            name here.
        cbar_lims : array-like, optional
            Upper and lower limits for the colourbar.
        **imshow_kwargs : dict
            Extra kwargs for the imshow function
        """

        # Set integrations to plot.
        if integrations == 'all':
            integrations = np.arange(self.datacube.shape[0])
        else:
            integrations = np.atleast_1d(integrations)

        # Initialize the output pdf file if the user wishes to save the plots.
        if savefile is not None:
            outpdf = matplotlib.backends.backend_pdf.PdfPages(savefile)

        # Make the decontamination plot for each specified integration.
        for i in integrations:
            # Get data, errors and models for the ith integration.
            obs = self.datacube[i]
            err = self.errorcube[i]
            err = np.where(err == 0, np.nan, err)  # nan any zero errors
            mod_o1 = self.o1model[i]
            mod_o2 = self.o2model[i]

            # Make the plot.
            fig = plt.figure()
            plt.imshow((obs - mod_o1 - mod_o2) / err, origin='lower',
                       aspect='auto', vmin=cbar_lims[0], vmax=cbar_lims[1],
                       **imshow_kwargs)
            cbar = plt.colorbar()
            cbar.set_label(r'o-m / $\sigma$', rotation=270, labelpad=15,
                           fontsize=14)

            plt.xlabel('Spectral Pixel', fontsize=14)
            plt.ylabel('Spatial Pixel', fontsize=14)
            plt.title('Integration {}'.format(i + 1), fontsize=16)
            # Either show the plot, or save it to the pdf.
            if savefile is not None:
                outpdf.savefig(fig)
                plt.close()
            else:
                plt.show()

        if savefile is not None:
            outpdf.close()


class BackgroundSubPlot:
    def __init__(self, datafile):
        # The DMS datamodels containers provide a useful way to unpack the fits
        # files produced by the DMS.
        try:
            # Unpack the science observations and associated errors.
            data = datamodels.open(datafile)
            self.datacube = data.data
        # Depending on the DMS version that the user has installed, there is a
        # bug in the ASDF parsing (I think) which prevents datamodels.open
        # from functioning properly. If this occurs, fall back on astopy fits
        # handling to unpack the files
        except AttributeError:
            self.datacube = _fits_fallback(datafile, 1)[0]

    def make_backgroundsub_plot(self, bkg_mask, integrations=0, savefile=None,
                                cbar_lims=[-1, 1], **imshow_kwargs):

        # Set integrations to plot.
        if integrations == 'all':
            integrations = np.arange(self.datacube.shape[0])
        else:
            integrations = np.atleast_1d(integrations)

        # Initialize the output pdf file if the user wishes to save the plots.
        if savefile is not None:
            outpdf = matplotlib.backends.backend_pdf.PdfPages(savefile)

        for i in integrations:
            obs = self.datacube[i]
            fig = plt.figure(figsize=(9, 7))
            gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1],
                                   height_ratios=[2, 1])

            ax1 = plt.subplot(gs[0, 0])
            plt.imshow(obs, aspect='auto', origin='lower', vmin=cbar_lims[0],
                       vmax=cbar_lims[1], **imshow_kwargs)
            cbaxes = inset_axes(ax1, width="30%", height="3%", loc=2)
            plt.colorbar(cax=cbaxes, orientation='horizontal')
            ax1.xaxis.set_major_formatter(plt.NullFormatter())
            ax1.set_ylabel('Spatial Pixel', fontsize=14)

            #aa = data.copy()
            ax2 = plt.subplot(gs[1, 0])
            obs[bkg_mask] = np.nan
            ax2.plot(np.nanmedian(obs, axis=0))
            ax2.set_xlabel('Spectral Pixel', fontsize=14)
            ax2.set_ylabel('Background\nColumn Median', fontsize=14)

            ax3 = plt.subplot(gs[0, 1])
            ax3.plot(np.nanmedian(obs, axis=1), np.arange(256))
            ax3.yaxis.set_major_formatter(plt.NullFormatter())
            ax3.tick_params(left=False)
            ax3.set_xlabel('Background\nRow Median', fontsize=14)

            plt.subplots_adjust(wspace=0, hspace=0)

            ax1.set_title('Integration {}'.format(i + 1), fontsize=16)
            # Either show the plot, or save it to the pdf.
            if savefile is not None:
                outpdf.savefig(fig)
                plt.close()
            else:
                plt.show()

        if savefile is not None:
            outpdf.close()


def _fits_fallback(file, extensions):
    """ Depending on the DMS version that the user has installed, there is a
    bug in the ASDF parsing (I think) which prevents datamodels.open from
     functioning properly. If this occurs, fallback on astopy fits handling.
    """

    extensions = np.atleast_1d(extensions)
    # Open and unpack the observed and modeled data via astropy
    observation = fits.open(file)
    unpacked = []
    for extension in extensions:
        unpacked.append(observation[extension].data)

    return unpacked
