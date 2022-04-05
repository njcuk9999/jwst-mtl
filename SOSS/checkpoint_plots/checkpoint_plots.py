#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Apr 05 2:08 2022

@author: MCR

Checkpoint plotting functions to compare SOSS data at different stages of
analysis via different pipelines.
"""

import matplotlib.backends.backend_pdf
import numpy as np
import matplotlib.pyplot as plt
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
        data = datamodels.open(datafile)
        self.datacube = data.data
        self.errorcube = data.err
        model = datamodels.open(modelfile)
        self.o1model = model.order1
        self.o2model = model.order2

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

        if integrations == 'all':
            integrations = np.arange(self.datacube.shape[0])
        else:
            integrations = np.atleast_1d(integrations)

        if savefile is not None:
            outpdf = matplotlib.backends.backend_pdf.PdfPages(savefile)

        for i in integrations:
            obs = self.datacube[i]
            err = self.errorcube[i]
            err = np.where(err == 0, np.nan, err)
            mod_o1 = self.o1model[i]
            mod_o2 = self.o2model[i]

            fig = plt.figure()
            plt.imshow((obs - mod_o1 - mod_o2) / err, origin='lower',
                       aspect='auto',
                       vmin=cbar_lims[0], vmax=cbar_lims[1], **imshow_kwargs)
            cbar = plt.colorbar()
            cbar.set_label(r'o-m / $\sigma$', rotation=270, labelpad=15,
                           fontsize=14)

            plt.xlabel('Spectral Pixel', fontsize=14)
            plt.ylabel('Spatial Pixel', fontsize=14)
            plt.title('Integration {}'.format(i + 1), fontsize=16)
            if savefile is not None:
                outpdf.savefig(fig)
                plt.close()
            else:
                plt.show()

        if savefile is not None:
            outpdf.close()
