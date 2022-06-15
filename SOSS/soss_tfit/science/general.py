#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-12

@author: cook
"""
from astropy.table import Table
from jwst import datamodels
import numpy as np
import os
from scipy.special import erf
from typing import Dict, List, Optional, Tuple

from soss_tfit.core import base
from soss_tfit.core import base_classes

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'science.general.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get parameter dictionary
ParamDict = base_classes.ParamDict
# get FitParam class
FitParam = base_classes.FitParam
# printer
cprint = base_classes.Printer()
# Define quantities
QUANTITIES = ['WAVELENGTH', 'FLUX', 'FLUX_ERROR']
TQUANTITIES = ['WAVELENGTH', 'FLUX', 'FLUX_ERROR', 'TIME']


# =============================================================================
# Define classes
# =============================================================================
class InputData:
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
    phot: Dict[str, np.ndarray]
    # order names
    orders: List[int]
    # number of wave length elements in each order of the spectrum
    n_wav: Dict[int, int]
    # number of photometric bandpasses
    n_phot: int

    def __init__(self, params: ParamDict, filename: str, verbose: bool = False):
        # set filename
        self.filename = filename
        # ---------------------------------------------------------------------
        # print progress
        if verbose:
            cprint('Loading JWST datamodel')
        # load data model from jwst
        data = datamodels.open(filename)
        # print progress
        if verbose:
            cprint('\tLoaded {0}'.format(filename))
        # ---------------------------------------------------------------------
        # print progress
        if verbose:
            cprint('Stacking multi spec')
        # set orders
        self.orders = list(params['ORDERS'])
        # convert data into a stack
        self.spec = stack_multi_spec(data, self.orders, quantities=QUANTITIES)
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

    def simple_spectral_binning(self, params: ParamDict, onum: int
                                ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Provides the start and end indices for a simple spectral binning,
        takes params['ORDER_BINS'] and divides the spectrum into that many
        chunks.

        Removal of orders is NOT done in here and must be done after binning
        is applied.

        :param params: ParamDict, the parameter dictionary of constants
        :param onum: int, the order number

        :return: tuple, 1. int: the number of bins, 2. numpy array: the start
                 indices for each bin, 3. numpy array: the end indices for
                 each bin
        """
        # get the number of bins required per order
        binning = params['ORDER_BINS']
        # get the number of wavelength pixels
        n_wav = self.n_wav
        # get the number of bins for this order
        n_bins = binning[onum]
        # ---------------------------------------------------------------------
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
            # return a single bin, startings at first pixel and going
            #    to last pixel
            n_bins = -1
            starts = np.array([])
            ends = np.array([])
            return n_bins, starts, ends
        # ---------------------------------------------------------------------
        # calculate the bin size
        bin_size = n_wav[onum] // n_bins
        # ---------------------------------------------------------------------
        starts, ends = [], []
        # loop around the bins and fill vectors
        for bin_it in range(n_bins):
            # calculate the start and end point for this bin
            start = bin_it * bin_size
            end = (bin_it + 1) * bin_size

            starts.append(start)
            ends.append(end)
        # ---------------------------------------------------------------------
        # cast into arrays
        starts = np.array(starts)
        ends = np.array(ends)
        # return these
        return n_bins, starts, ends

    def const_r_spectral_binning(self, params: ParamDict, onum: int
                                ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Provides the start and end indices for spectral binning at a
        constant resolution - also removes based on wavelength ranges using:
        params['BIN_WAVE_MIN'][onum] and params['BIN_WAVE_MAX'][onum]

        :param params: ParamDict, the parameter dictionary of constants
        :param onum: int, the order number

        :return: tuple, 1. int: the number of bins, 2. numpy array: the start
                 indices for each bin, 3. numpy array: the end indices for
                 each bin
        """

        # get resolution wave min and wave max for this order
        resolution = params['BIN_RESOLUTION'][onum]
        wave_min = params['BIN_WAVE_MIN'][onum]
        wave_max = params['BIN_WAVE_MAX'][onum]

        # get the wave solution for first spectrum
        #   (this assumes all wave solutions are the same)
        wave = self.spec[onum]['WAVELENGTH'][0]

        # test the wavelengths are truely decending
        if np.min(np.diff(wave)) > 0:
            emsg = 'For binning by const r wave sol much be in decending order.'
            raise base_classes.TransitFitExcept(emsg)
        # ---------------------------------------------------------------------
        # find the maximum/minimum indices to keep
        if wave_min is None:
            imax = wave.size - 1
        else:
            imax = np.searchsorted(-wave, -wave_min) - 1
        if wave_max is None:
            imin = 0
        else:
            imin = np.searchsorted(-wave, -wave_max)
        # ---------------------------------------------------------------------
        # storage of bins
        starts, ends = [], []
        # ---------------------------------------------------------------------
        # iterators starting points
        end = imax
        start = imax - 1
        # have to start rtest_previous at a value
        rtest_previous = np.inf
        # ---------------------------------------------------------------------
        # loop round until we reach the end of our indices
        while start > imin:
            # work out the wave diff between start and end iterator
            dwave = wave[start] - wave[end]
            # work out the resolution
            r_test = wave[start: end + 1].mean() / dwave
            # we only stop when resolution meets our requirement
            #  (resolution will start really high)
            if r_test < resolution:
                # if the difference in resolution is greater than previously
                #   calculated we use the pixel after
                if abs(rtest_previous - resolution) < abs(r_test - resolution):
                    start += 1
                # append to starts and end
                starts.append(int(start))
                ends.append(int(end))
                # change start and end position
                end = start -1
                start -= 1
            # update previous
            rtest_previous = float(r_test)
            # move start back
            start -= 1
        # ---------------------------------------------------------------------
        # cast into arrays
        starts = np.array(starts)
        ends = np.array(ends)
        # record the number of bins
        n_bins = starts.size
        # return these
        return n_bins, starts, ends

    def apply_spectral_binning(self, params: ParamDict):
        """
        Apply spectral binning

        :param params: ParamDict, parameter dictionary of constants
        :return:
        """
        # get the binning mode
        bmode = params['BINNING_MODE']
        # get the number of integrations
        n_int = self.n_int
        # get the orders
        orders = self.orders
        # ---------------------------------------------------------------------
        # deal with simple mode
        # ---------------------------------------------------------------------
        # loop around each order
        for onum in orders:
            # get n_bins, starts, ends from binning mode
            if bmode == 'simple':
                bout  = self.simple_spectral_binning(params, onum)
            elif bmode == 'const_R':
                bout = self.const_r_spectral_binning(params, onum)
            else:
                emsg = (f'Binning mode = {bmode} not valid. '
                        f'Please use "simple" or "const_R"')
                raise base_classes.TransitFitExcept(emsg)
            # extract out values from binning function
            #   (number of bins, start indices, end indices)
            n_bins, starts, ends = bout
            # skip binning if n_bins == -1 (this is the skip criteria)
            if n_bins == -1:
                continue
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
            # loop around the bins and fill vectors
            for bin_it in range(n_bins):
                # get the start and end point (in pixels) for this bin
                start, end = starts[bin_it], ends[bin_it]
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
            # -----------------------------------------------------------------
            # for simple bining we then remove bins
            if bmode == 'simple':
                self.remove_simple_bins(params)

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

    def remove_simple_bins(self, params: ParamDict):
        """
        Remove bin indices from data

        :param params: ParamDict, parameter dictionary of constants
        :return:
        """
        # get remove bins
        remove_bins = params['REMOVE_BINS']
        # loop around each order
        for order in self.orders:
            # if order is not in remove bins then skip
            if order not in remove_bins:
                continue
            # just skip anything that isn't a int or list
            if not isinstance(remove_bins[order], (int, list)):
                continue
            # just in case we have a single value, and it isn't in list format
            if isinstance(remove_bins[order], int):
                remove_bins[order] = [remove_bins[order]]
            # get remove bins for this order
            try:
                remove_bins_order = np.array(remove_bins[order], dtype=int)
            except Exception as e:
                emsg = ('global_params.remove_bins[order] must be a list of'
                        ' integers or None')
                raise base_classes.TransitFitExcept(emsg)
            # get the removed bins in string list form
            rb_map = map(lambda x: str(x), remove_bins[order])
            list_remove_bins_order = ','.join(list(rb_map))
            # print which bins we are removing
            cprint(f'\tOrder {order} removing bins: {list_remove_bins_order}')
            # create list of bins
            bins = np.arange(self.n_wav[order])
            # create mask of remove bins
            remove_mask = np.in1d(bins, remove_bins_order)
            # loop around each quantity
            for quantity in self.spec[order]:
                # keep non removed bins
                masked_quantity = self.spec[order][quantity][:, ~remove_mask]
                # udpate spec quantity
                self.spec[order][quantity] = masked_quantity
            # update nwave
            self.n_wav[order] = self.spec[order]['WAVELENGTH'].shape[1]

    def photospectra(self):
        """
        Move the spec data into a photospectra list (self.phot)

        :return: None, updates self.phot
        """
        # create phot dictionary storage
        self.phot = dict()
        # set the number of band passes to zero
        self.n_phot = 0
        # Integration time [days]
        int_time = (self.n_group - 1) * self.t_group / (60 * 60 * 24)
        # loop around order
        order0 = self.orders[0]
        # fill phot with first order
        for quantity in TQUANTITIES:
            self.phot[quantity] = self.spec[order0][quantity].T
        # keep track of the orders
        self.phot['ORDERS'] = np.full_like(self.phot['WAVELENGTH'], order0)
        # loop around all other order
        for onum in self.orders[1:]:
            # loop around all transit quantities
            for quantity in TQUANTITIES:
                # get this orders spectrum for this quantity
                spec_quant = self.spec[onum][quantity]
                # concatenate with previous orders
                self.phot[quantity] = np.concatenate([self.phot[quantity],
                                                      spec_quant.T])
            # keep track of the orders
            order_quant = np.full_like(self.spec[onum]['WAVELENGTH'].T, onum)
            self.phot['ORDERS'] = np.concatenate([self.phot['ORDERS'],
                                                 order_quant])
        # update number of photometric bandpasses
        self.n_phot = self.phot['WAVELENGTH'].shape[0]
        # add itime
        self.phot['ITIME'] = np.full_like(self.phot['WAVELENGTH'], int_time)

    def update_zeropoint(self, params: ParamDict) -> ParamDict:
        """
        Update the zeropoint value with the median of each band pass

        :param params: ParamDict, the parameter dictionary of constants

        :return:
        """
        # get the median flux value of each photometric bandpass
        zpt = np.median(self.phot['FLUX'], axis=1)
        # deal with zeropoint not being a FitParam class
        if not isinstance(params['ZEROPOINT'], FitParam):
            emsg = 'Param Error: Zeropoint must be a "FitParam" class'
            raise base_classes.TransitFitExcept(emsg)
        # update the zeropoints value
        params['ZEROPOINT'].value = zpt
        # return the parameter dictionary
        return params


# =============================================================================
# Define functions
# =============================================================================
def stack_multi_spec(multi_spec, use_orders: List[int],
                     quantities: List[str]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Convert a jwst multi_spec into a dictionary

    all_spec[order][quantity]

    where order is [1, 2, 3] etc
    quantity is ['WAVELENGTH', 'FLUX', 'FLUX_ERROR', 'TIME', 'BIN_LIMITS']

    :param multi_spec:
    :param use_orders:
    :param quantities:
    :return:
    """
    # define a container for the stack
    all_spec = dict()
    # -------------------------------------------------------------------------
    # loop around orders
    for sp_ord in use_orders:
        # set up dictionary per order for each quantity
        all_spec[sp_ord] = dict()
        # loop around quantities
        for quantity in quantities:
            all_spec[sp_ord][quantity] = []
    # -------------------------------------------------------------------------
    # loop around orders and get data
    for spec in multi_spec.spec:
        # get the spectral order number
        sp_ord = spec.spectral_order
        # skip orders not wanted
        if sp_ord not in use_orders:
            continue
        # load values for each quantity
        for quantity in quantities:
            # only add quantities
            if quantity in spec.spec_table.columns.names:
                all_spec[sp_ord][quantity].append(spec.spec_table[quantity])
    # -------------------------------------------------------------------------
    # convert list to array
    for sp_ord in all_spec:
        for key in all_spec[sp_ord]:
            all_spec[sp_ord][key] = np.array(all_spec[sp_ord][key])
    # -------------------------------------------------------------------------
    # return the all spectrum dictionary
    return all_spec


def sigma_percentiles(sigma: int = 1):
    """
    Produce the percentiles for: 50 - sigma, 50, sigma + 50
    """
    # get the sigma value (using the error function)
    sig1 = erf(sigma / np.sqrt(2.0))
    # get the upper bound 1 sigma as a percentile
    p1 = (1 - (1-sig1)/2) * 100
    # get thelower bound 1 sigma as a percentile
    p0 = 100 - p1
    # return 50 - sigma, 50, sigma + 50
    return p0, 50, p1


def load_model(params: ParamDict, data: InputData,
               keys: Optional[List[str]] = None,
               ) -> Tuple[Optional[Dict[int, Table]], Optional[Table]]:
    """
    Load the model and apply binning for each order

    :param params: ParamDict, paramter dictionary of constants
    :param data: InputData class
    :param keys: List[str], the list of keys to get from model for yaxis

    :return: tuple, 1. the binned model for each order dict[order][Table],
                    2. the unbinned model Table

                    each Table has columns "wave" and "keys"
    """
    # get model file
    model_file = params['MODELPATH']
    # deal with no keys defined
    if keys is None:
        keys = ['RPRS']
    # -------------------------------------------------------------------------
    # deal with a null value for model file
    if model_file in [None, 'None', 'Null', '']:
        return None, None
    # deal with model file not existing
    if not os.path.exists(model_file):
        cprint(f'Model Warning: {model_file} does not exist')
        return None, None
    # -------------------------------------------------------------------------
    # load the model
    fullmodel = Table.read(model_file, format='csv', comment='#')
    # -------------------------------------------------------------------------
    # bin the model to match retrieved spectrum
    models = dict()
    # Make a table for each order
    for order in data.orders:
        models[order] = Table()
    # loop around keys
    for key in keys:
        # ---------------------------------------------------------------------
        # get the RP/RS for this model
        if key == 'RPRS':
            fullmodel[key] = np.sqrt(fullmodel['dppm'] / 1.0e6)
        # ---------------------------------------------------------------------
        # loop over orders
        for order in data.orders:
            # set up arrays for this order
            wave = np.zeros(data.n_wav[order])
            yvalue = np.zeros(data.n_wav[order])
            # loop around band pass
            for phot_it in range(data.n_wav[order]):
                # get limits of wave bin for this bandpass
                lam2, lam1 = data.spec[order]['BIN_LIMITS'][:, phot_it]
                # find the index at which these thresholds are crossed
                ilam1 = np.searchsorted(fullmodel['wave'], lam1)
                ilam2 = np.searchsorted(fullmodel['wave'], lam2)
                # work out the mean RPRS value and wavevalue for this new bin
                wave[phot_it] = fullmodel['wave'][ilam1: ilam2].mean()
                yvalue[phot_it] = fullmodel[key][ilam1: ilam2].mean()
            # push these into output models
            models[order]['wave'] = wave
            models[order][key] = yvalue
    # -------------------------------------------------------------------------
    # return model
    return models, fullmodel


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
