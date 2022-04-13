#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-12

@author: cook
"""
from jwst import datamodels
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union

from soss_tfit.core import base
from soss_tfit.core import base_classes
from soss_tfit.core import io

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'science.general.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get parameter dictionary
ParamDict = base_classes.ParamDict
# Define quantities
QUANTITIES = ['WAVELENGTH', 'FLUX', 'FLUX_ERROR']


# =============================================================================
# Define functions
# =============================================================================
class Inputdata:
    """
    Class handling the loading and manipulation of input data
    """
    filename: str
    n_group: int
    t_group: float
    n_int: int
    time_int: float
    # spectrum dictionary spec[order][quantity]
    #       quantities defined by QUANTITIES constant
    spec: Dict[int, Dict[str, np.ndarray]]
    # photo spectra  phot[i][quantity]
    #       quantities are [WAVELENGTH, TIME, FLUX, FLUXERROR, ITIME]
    phot: List[Dict[str, np.ndarray]]
    # order names
    orders: List[int]
    # number of wave length elements in each order
    n_wav: Dict[int, int]

    def __init__(self, params: ParamDict, filename: str, verbose: bool = False):
        # set filename
        self.filename = filename
        # ---------------------------------------------------------------------
        # print progress
        if verbose:
            print('Loading JWST datamodel')
        # load data model from jwst
        data = datamodels.open(filename)
        # ---------------------------------------------------------------------
        # print progress
        if verbose:
            print('Stacking multi spec')
        # set orders
        self.orders = list(params['ORDERS'])
        # convert data into a stack
        self.spec = io.stack_multi_spec(data, self.orders,
                                        quantities=QUANTITIES)
        # storage for number of pixels in spectral dimensions
        self.n_wav = dict()
        # set number of pixels in each order
        for onum in self.orders:
            # get the number of wavelengths
            self.n_wav[onum] = self.spec[onum]['WAVELENGTH'].shape[1]
        # ---------------------------------------------------------------------
        # assign meta data for this observation
        # ---------------------------------------------------------------------
        # Number of groups
        self.n_group = data.meta.exposure.ngroups
        # Reading time [s]
        self.t_group = data.meta.exposure.group_time
        # Number of integrations
        self.n_int = data.meta.exposure.nints
        # integration time
        self.time_int = (self.n_group + 1) * self.t_group
        # can now delete data (no longer needed)
        del data

    def compute_time(self):
        """
        Compute the time for each pixel accounting for readout time

        :return:
        """
        # loop around each order
        for onum in self.orders:
            # get this orders values
            n_wav = self.n_wav[onum]
            time_int = self.time_int
            n_int = self.n_int

            # if n_wav not 2048 must check and update this value,
            #     will it always be 2048?
            pix0 = 0

            time_pix = (pix0 + np.arange(n_wav) - 1024) / 2048 * time_int

            # the 2d array of time, for all integrations and all pixels
            #     (wavelengths)
            # check order here, which end of spectrum is read first?
            time_obs = np.zeros(n_wav) + (np.arange(n_int) * time_int)[:, None]
            time_obs += time_pix[None, :]
            # could choose appropriate time here
            t0 = time_int / 2
            # add to matrix
            time_obs += t0
            # convert to days
            time_obs /= 86400

            # push into the all spec dictionary for this order
            self.spec[onum]['TIME'] = time_obs

    def remove_null_flux(self):
        """
        Remove all exactly zero values and NaN values from all integrations

        :return: None, updates self.spec
        """
        # get the orders
        orders = self.orders
        # get the spectrum storage
        spec = self.spec
        # loop around orders
        for onum in orders:
            # find the bad flux values
            bad_mask = spec[onum]['FLUX'] == 0
            bad_mask |= np.isnan(spec[onum]['FLUX'])
            # mask all values bad for all wavelengths
            bad_mask = bad_mask.all(axis=0)
            # loop around all columns and mask
            for quantity in spec[onum]:
                # get the wave/flux/fluxerror/time
                qvector = spec[onum][quantity]
                # create a new cut down version of quantity
                new_image = np.delete(qvector, bad_mask, axis=1)
                # overwrite original values
                self.spec[onum][quantity] = new_image
            # update the number of pixels and wavelengths
            self.n_wav[onum] = self.spec[onum]['WAVELENGTH'].shape[1]

    def apply_spectral_binning(self, params: ParamDict):
        """
        Apply spectral binning

        :param params: ParamDict, parameter dictionary of constants
        :return:
        """
        # TODO: Must make sure binning is consistent between orders
        # get the number of bins required per order
        binning = params['ORDER_BINS']
        # get the number of integrations
        n_int = self.n_int
        # get the number of wavelength pixels
        n_wav = self.n_wav
        # get the orders
        orders = self.orders
        # ---------------------------------------------------------------------
        # loop around each order
        for onum in orders:
            # get the number of bins for this order
            n_bins = binning[onum]
            # deal with no binning required
            if n_bins is None:
                # get the original wavelength
                wavelength1 = self.spec[onum]['WAVELENGTH']
                # set bin limits
                bin_limits = np.zeros((2, n_wav[onum]))
                # set the bin limits to the wavelength (delta function?)
                for bin_it in range(n_wav[onum]):
                    wave_start = wavelength1[:, bin_it]
                    bin_limits[:, bin_it] = (wave_start, wave_start)
                self.spec[onum]['BIN_LIMITS'] = bin_limits
                # skip binning
                continue
            # calculate the bin size
            bin_size = n_wav[onum] // n_bins
            # set up storage for bin limits
            bin_limits = np.zeros((2, n_bins))
            # get original vectors
            wavelength1 = self.spec[onum]['WAVELENGTH']
            time1 = self.spec[onum]['TIME']
            flux1 = self.spec[onum]['FLUX']
            flux_err1 = self.spec[onum]['FLUX_ERROR']
            # set up binned vectors
            wavelength2 = np.zeros((n_int, n_bins))
            time2 = np.zeros((n_int, n_bins))
            flux2 = np.zeros((n_int, n_bins))
            flux_err2 = np.zeros((n_int, n_bins))
            # loop aroun the bins and fill vectors
            for bin_it in range(n_bins):
                # calculate the start and end point for this bin
                start, end = bin_it * bin_size, (bin_it + 1) * bin_size
                # calculate the binned values (mean)
                bin_wave = np.mean(wavelength1[:, start:end], axis=1)
                bin_time = np.mean(time1[:, start:end], axis=1)
                bin_flux = np.mean(flux1[:, start:end], axis=1)
                # sqrt of the mean squared values
                mean_flux2_err = np.mean(flux_err1[:, start:end]**2, axis=1)
                bin_flux_err = np.sqrt(mean_flux2_err)
                # work out bin limits
                wave_start = wavelength1[0, start]
                wave_end = wavelength1[0, end-1]
                # push into binned vectors
                wavelength2[:, bin_it] = bin_wave
                time2[:, bin_it] = bin_time
                flux2[:, bin_it] = bin_flux
                flux_err2[:, bin_it] = bin_flux_err
                # update the bin limits
                bin_limits[:, bin_it] = (wave_start, wave_end)
            # -----------------------------------------------------------------
            # finally update the spec values
            self.spec[onum]['WAVELENGTH'] = wavelength2
            self.spec[onum]['TIME'] = time2
            self.spec[onum]['FLUX'] = flux2
            self.spec[onum]['FLUX_ERROR'] = flux_err2
            self.spec[onum]['BIN_LIMITS'] = bin_limits
            # update the number of pixels and wavelengths
            self.n_wav[onum] = n_bins

    def normalize_by_out_of_transit_flux(self, params: ParamDict):
        """
        Normalize by the mean of the out-of-transit flux at each wavelength

        :param params: ParamDict, the parameter dictionary of constants

        :return:
        """
        # get the normalized time value before and after transit
        tnorm0 = params['TNORM']['before']
        tnorm1 = params['TNORM']['after']
        # deal with no normalization required
        if tnorm0 is None and tnorm1 is None:
            return
        # get the orders
        orders = self.orders
        # get the spectrum storage
        spec = self.spec

        # loop around orders
        for onum in orders:
            # get time and flux
            time = spec[onum]['TIME']
            flux = spec[onum]['FLUX']
            # get a mask of the orders
            out_mask = np.zeros_like(time, dtype=bool)
            # deal with before transit part
            if tnorm0 is not None:
                out_mask |= time < tnorm0
            # deal with after transit part
            if tnorm1 is not None:
                out_mask |= time > tnorm1
            # all to all integrations
            out_mask = out_mask.all(axis=1)

            # work out mean of out-of-transit flux at each wavelength
            out_flux = np.mean(flux[out_mask, :], axis=0)

            # normalize flux and flux error by out of transit flux
            self.spec[onum]['FLUX'] /= out_flux[None, :]
            self.spec[onum]['FLUX_ERROR'] /= out_flux[None, :]

    def photospectra(self):
        """
        Move the spec data into a photospectra list (self.phot)

        :return: None, updates self.phot
        """
        # create phot list storage
        self.phot = []
        # Integration time [days]
        int_time = (self.n_group - 1) * self.t_group / (60 * 60 * 24)
        # push integration time into an array
        itime = np.full(self.n_int, int_time)
        # loop around order
        for onum in self.orders:
            # get the spectrum
            wave_arr = self.spec[onum]['WAVELENGTH']
            time_arr = self.spec[onum]['TIME']
            flux_arr = self.spec[onum]['FLUX']
            eflux_arr = self.spec[onum]['FLUX_ERROR']
            # loop around wave bandpasses
            for w_it in range(self.n_wav[onum]):
                # define a dictionary to hold the phot element
                phot_it = dict()
                # add the wavelength vector for this bandpass
                phot_it['WAVELENGTH'] = np.array(wave_arr[:, w_it])
                # add the time vector for this bandpass
                phot_it['TIME'] = np.array(time_arr[:, w_it])
                # add the flux vector for this bandpass
                phot_it['FLUX'] = np.array(flux_arr[:, w_it])
                # add the flux error vector for this bandpass
                phot_it['FLUX_ERROR'] = np.array(eflux_arr[:, w_it])
                # add the integration time for this bandpass
                phot_it['ITIME'] = np.array(itime)
                # append to photospectra
                self.phot.append(phot_it)




# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
