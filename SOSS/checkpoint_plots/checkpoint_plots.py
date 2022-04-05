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
import matplotlib.pyplot as plt
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
            result = _fits_fallback(datafile, modelfile)
            self.datacube, self.errorcube, self.o1model, self.o2model = result

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


def _fits_fallback(datafile, modelfile):
    """ Depending on the DMS version that the user has installed, there is a
    bug in the ASDF parsing (I think) which prevents datamodels.open from
     functioning properly. If this occurs, fallback on astopy fits handling.
    """

    # Open and unpack the observed and modeled data via astropy
    observation = fits.open(datafile)
    data = observation[1].data
    error = observation[2].data

    model = fits.open(modelfile)
    o1model = model[2].data
    o2model = model[3].data

    return data, error, o1model, o2model
