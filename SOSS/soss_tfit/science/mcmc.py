#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-06

@author: cook
"""
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

import transitfit5 as transit_fit

from soss_tfit.core import base
from soss_tfit.core import base_classes
from soss_tfit.science import general

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
# get data class
InputData = general.InputData
# Out of bounds value
BADLPR = -np.inf
# transit fit (fortran code) stellar parameters in correct order
TPS_ORDERED = ['RHO_STAR', 'LD1', 'LD2', 'LD3', 'LD4', 'DILUTION', None,
               'ZEROPOINT']
# transit fit (fortran code) planet fit parameters (one per planet in this
#    order)
TPP_ORDERED = ['T0', 'PERIOD', 'B', 'RPRS', 'SQRT_E_COSW', 'SQRT_E_SINW',
               None, 'ECLIPSE_DEPTH', 'ELLIPSOIDAL', 'PHASECURVE']

# transit fit hyper parameters
TPH_ORDERED = ['ERROR_SCALE', 'AMPLITUDE_SCALE', 'LENGTH_SCALE']
# transit fit additional arguments
TP_KWARGS = ['NTT', 'T_OBS', 'OMC']

# These are the attributes of TransitFit that have the same length as x0
X_ATTRIBUTES = ['x0', 'beta', 'x0pos']


# =============================================================================
# Define fit class passed to mcmc code
# =============================================================================
class TransitFit:
    """
    One could just set all the correct values manually in this and
    expect it to work later
    """
    # the number of integrations we have
    n_int: int
    # the number of parameters we have
    n_param: int
    # the number of photometric bandpasses we have
    n_phot: int
    # numpy array [n_param, n_phot] the initial value of each parameter
    p0: np.ndarray
    # numpy array [n_param] whether we are fitting each parameter [Bool]
    fmask: np.ndarray
    # numpy array [n_param] whether chromatic fit used for each parameter [Bool]
    wmask: np.ndarray
    # priors [n_param]
    prior: Dict[str, Any]
    # name of each parameter
    pnames: List[str]
    # additional arguments passed to mcmc
    pkwargs: Dict[str, Any]
    # the position in the flattened x array [n_param, n_phot]
    p0pos: np.ndarray
    # -------------------------------------------------------------------------
    # the data:
    # -------------------------------------------------------------------------
    #     [0][WAVELENGTH][n_phot, n_int]
    phot: List[Dict[str, np.ndarray]]
    # the wavelength array [n_phot, n_int]
    wavelength: np.ndarray
    # the time array [n_phot, n_int]
    time: np.ndarray
    # the integration time array [n_phot, n_int]
    itime: np.ndarray
    # the flux array [n_phot, n_int]
    flux: np.ndarray
    # the flux error array [n_phot, n_int]
    fluxerr: np.ndarray
    # the order array [n_phot, n_int]
    orders: np.ndarray
    # -------------------------------------------------------------------------
    # parameters that must have shape [n_x]
    # -------------------------------------------------------------------------
    # the initial fitted parameters
    x0: np.ndarray
    # name of the fitted params flattened [n_x]
    xnames: np.ndarray
    # length of fitted params flattened [n_x]
    n_x: int
    # numpy array [n_param]
    beta: np.ndarray
    # the mapping from x0 onto p0 [n_x, 2] each element
    #   is the tuple position in p0
    x0pos: np.ndarray
    # -------------------------------------------------------------------------
    # will be filled out by the mcmc
    # -------------------------------------------------------------------------
    # the current position chosen by the sampler
    n_tmp: int = 0
    # the previous loglikelihood value
    llx: float
    # the rejection [rejected, parameter number]   where rejected = 0 for
    #   accepted and rejected = 1 for rejected
    ac: List[int]

    def __init__(self):
        pass

    def __getstate__(self) -> dict:
        """
        For when we have to pickle the class
        :return:
        """
        # set state to __dict__
        state = dict(self.__dict__)
        # return dictionary state
        return state

    def __setstate__(self, state: dict):
        """
        For when we have to unpickle the class

        :param state: dictionary from pickle
        :return:
        """
        # update dict with state
        self.__dict__.update(state)

    def copy(self) -> 'TransitFit':
        """
        Copy class - deep copying values which need copying
        e.g. p0, x0

        :return:
        """
        new = TransitFit()
        # the number of integrations we have
        new.n_int = self.n_int
        # the number of parameters we have
        new.n_param = self.n_param
        # the number of photometric bandpasses we have
        new.n_phot = self.n_phot
        # numpy array [n_param, n_phot] the initial value of each parameter
        new.p0 = np.array(self.p0)
        # numpy array [n_param] whether we are fitting each parameter [Bool]
        new.fmask = self.fmask
        # numpy array [n_param] whether chromatic fit used for each
        #     parameter [Bool]
        new.wmask = self.wmask
        # priors [n_param]
        new.prior = self.prior
        # name of each parameter
        new.pnames = self.pnames
        # additional arguments passed to mcmc
        new.pkwargs = self.pkwargs
        # the position in the flattened x array [n_param, n_phot]
        new.p0pos = self.p0pos
        # ---------------------------------------------------------------------
        # the data:
        # ---------------------------------------------------------------------
        # the wavelength array [n_phot, n_int]
        new.wavelength = self.wavelength
        # the time array [n_phot, n_int]
        new.time = self.time
        # the integration time array [n_phot, n_int]
        new.itime = self.itime
        # the flux array [n_phot, n_int]
        new.flux = self.flux
        # the flux error array [n_phot, n_int]
        new.fluxerr = self.fluxerr
        # the order array [n_phot, n_int]
        new.orders = self.orders
        # ---------------------------------------------------------------------
        # parameters that must have shape [n_x]
        # ---------------------------------------------------------------------
        # the initial fitted parameters
        new.x0 = np.array(self.x0)
        # name of the fitted params flattened [n_x]
        new.xnames = self.xnames
        # length of fitted params flattened [n_x]
        new.n_x = self.n_x
        # numpy array [n_param]
        new.beta = np.arary(self.beta)
        # the mapping from x0 onto p0 [n_x, 2] each element
        #   is the tuple poisition in p0
        new.x0pos = self.x0pos
        # ---------------------------------------------------------------------
        # will be filled out by the mcmc
        # ---------------------------------------------------------------------
        # the current position chosen by the sampler
        new.n_tmp = 0
        # the previous loglikelihood value
        new.llx = 1.0
        # the rejection [rejected, parameter number]   where rejected = 0 for
        #   accepted and rejected = 1 for rejected
        new.ac = []
        # ---------------------------------------------------------------------
        return new

    def get_fitted_params(self):
        """
        Get the fitted parameters in a flattened way
        Here chromatic values add self.n_phot values and bolometric values add
        a single value

        :return: None, updates self.x0 and self.xnames
        """
        # set up storage
        x0 = []
        xnames = []
        # position of x in p
        p0pos = np.zeros_like(self.p0)
        x0pos = []
        # counter for x0 position
        count_x = 0
        # loop around all parameters
        for it, name in enumerate(self.pnames):
            # if we are not fitting this parameter skip
            if not self.fmask[it]:
                p0pos[it] = np.full(self.n_phot, -1)
                continue
            # if we have a wavelength dependence add all terms (one for each
            #   bandpass)
            if self.wmask[it]:
                x0 += list(self.p0[it])
                xnames += [self.pnames[it]] * self.n_phot
                # add to the positions so we can recover p0 from x0
                p0pos[it] = np.arange(count_x, count_x + self.n_phot)
                # update the x positions
                x0pos += list(zip(np.full(self.n_phot, it), np.arange(self.n_phot)))
                # update x0 position
                count_x += self.n_phot
            # else add the first term (all band passes should have the same
            #   value)
            else:
                x0 += [self.p0[it][0]]
                xnames += [self.pnames[it]]
                # add to the positions so we can recover p0 from x0
                p0pos[it] = np.full(self.n_phot, count_x)
                # update x0 position
                count_x += 1
                # update the x positions
                x0pos += [(it, 0)]
        # set values
        self.x0 = np.array(x0)
        self.xnames = np.array(xnames)
        self.n_x = len(x0)
        self.p0pos = np.array(p0pos, dtype=int)
        self.x0pos = np.array(x0pos)

    def update_p0_from_x0(self):
        """
        Take a set of fitted parameters (x0) and project back into the
        full solution (p0)

        :return:
        """
        # loop around all parameters
        for it in range(self.n_x):
            # find all places where it is valid
            mask = it == self.p0pos
            # update p0 with x0 value
            self.p0[mask] = self.x0[it]

    def update_x0_from_p0(self):
        """
        Take a full solution (p0) and project back into the set of fitted
        parameters (x0)

        :return:
        """
        for it in range(self.n_x):
            # push the value into x
            self.x0[it] = self.p0[tuple(self.x0pos)]

    def generate_gibbs_sample(self, beta: np.ndarray):
        # choose random parameter to vary
        param_it = np.random.randint(0, self.n_x + 1)
        # update the position choosen by the sampler
        self.n_tmp = param_it
        # update the choosen value by a random number drawn from a gaussian
        #   with centre = 0 and fwhm = beta
        self.x0[param_it] += np.random.normal(0.0, beta[param_it])
        # update full solution
        self.update_p0_from_x0()

    def generate_demcmc_sample(self, buffer, corbeta: float):
        # get the length of the buffer
        nbuffer = len(buffer[:, 0])
        # update the position choosen by the sampler
        self.n_tmp = -1
        # get two random numbers
        int1, int2 = np.random.randint(0, nbuffer + 1, size=2)
        # calculate the vector jump
        vector_jump = buffer[int1, :] - buffer[int2, :]
        # apply the vector jump to x0
        self.x0 += vector_jump * corbeta
        # update full solution
        self.update_p0_from_x0()

    def get(self, param_name: str, attribute_name: str):
        # get attribute
        attribute = self.__get_attribute(attribute_name)
        # deal with fitted parameters
        if attribute_name in X_ATTRIBUTES:
            return dict(zip(self.xnames, attribute))[param_name]
        else:
            return dict(zip(self.pnames, attribute))[param_name]

    def view(self, attribute_name: str,
             attribute: Optional[Any] = None):
        """
        View a parameter (i.e. p0, fmask, wmasl, beta, prior)
        with its associated

        :param attribute_name: if parameter is None this must be an attribute of
                     TransitFit, if parameter is defined can any string
                     describing the variable name

        :param attribute: can be unset, but if parameter_name if not in
                          TransitFit must be a vector of length = self.n_param

        examples:
            >> tfit.view('beta', beta[:, 0]) # view the 0th bandpass beta values
            >> tfit.view('p0', tfit.p0[:, 0])  # view the 0th bandpass p0 values
            >> tfit.view('prior')      # view the prior values
            >> tfit.view('fmask')     # view which values are fitted
            >> tfit.view('x0')      # view the flattened fitted parameters

        :return: None, prints to stdout
        """
        # get attribute
        attribute = self.__get_attribute(attribute_name, attribute)
        # deal with fitted parameters
        if attribute_name in X_ATTRIBUTES:
            names = self.xnames
            length = self.n_x
        else:
            names = self.pnames
            length = self.n_param

        # make sure we can view this parameter
        if len(attribute) != length:
            emsg = (f'TransitFit View Error: Cannot view parameter length '
                    f'(on first axis) must be: {length}')
            base_classes.TransitFitExcept(emsg)
        # loop around and print info
        for it in range(length):
            print(f'{names[it]:14s}: {attribute[it]}')

    def __get_attribute(self, attribute_name: str,
                        attribute: Optional[Any] = None) -> Any:
        # report error
        if not hasattr(self, attribute_name) and attribute is None:
            emsg = ('TransitFit View Error: Parameter name not set and '
                    '"parameter" not set. Please define one')
            base_classes.TransitFitExcept(emsg)
        elif attribute is None:
            attribute = getattr(self, attribute_name)

        return attribute

    def assign_beta(self, beta_kwargs: Dict[str, Any]):
        """
        Assign beta from either beta_kwargs[key] = value
        or as a random number: np.random.rand(size) * 1.0e-5

        note the value can either be a float or a np.ndarray of length the
           number of band passes

        :param beta_kwargs: dictionary, key value pairs to change beta values
                            for a specific parameter
        :return:
        """
        # beta factor
        bfactor = 1.0e-5
        # start of with all values at zero (all fixed parameters have beta=0)
        self.beta = np.zeros(self.n_x)
        # loop through and find fitted parameters
        for it in range(self.n_x):
            # override beta values
            if self.xnames[it] in beta_kwargs:
                # set beta value
                self.beta[it] = beta_kwargs[self.xnames[it]]
                # skip step
                continue
            # else set beta as a random number (multiplied by the beta factor)
            else:
                self.beta[it] = np.random.rand() * bfactor


# =============================================================================
# Define prior functions
#     Must have arguments value, **kwargs
#     Must return True if passed, False otherwise
# =============================================================================
def tophat_prior(value: float, minimum: float, maximum: float) -> bool:
    """
    Simple tophat proir
    :param value:
    :param minimum:
    :param maximum:
    :return:
    """
    # test whether condition is less than the minimum of the prior
    if value < minimum:
        return False
    # test whether condition is greater than the maximum of the prior
    if value > maximum:
        return False
    # if we get to here we have passed this priors conditions
    return True


# =============================================================================
# Define functions
# =============================================================================
def setup_params_mcmc(params: ParamDict, data: InputData) -> TransitFit:
    """
    Using params and data load everything in to the Transit Fit data class

    :param params: ParamDict, parameter dictionary of constants
    :param data: InputData, input data class

    :return: transit fit data class
    """
    # set function name
    func_name = __NAME__ + '.get_starting_params()'
    # get number of wavelength bandpasses
    n_phot = data.n_phot
    # get the number of planets to add
    n_planets = params['NPLANETS']
    # get the number of parameters we are adding
    n_param = len(TPS_ORDERED) + len(TPP_ORDERED) + len(TPH_ORDERED)
    # -------------------------------------------------------------------------
    # set up storage
    transit_p0 = np.zeros([n_param, n_phot])
    transit_fmask = np.zeros(n_param, dtype=bool)
    transit_wmask = np.zeros(n_param, dtype=bool)
    transit_prior = []
    transit_pnames = []
    # -------------------------------------------------------------------------
    # get the transit fit stellar params
    # -------------------------------------------------------------------------
    pnum = 0
    # loop around stellar parameters
    for key in TPS_ORDERED:
        p0, fmask, wmask, prior, pname = __assign_pvalue(key, params, n_phot,
                                                         func_name)
        # add values to storage
        transit_p0[pnum] = p0
        transit_fmask[pnum] = fmask
        transit_wmask[pnum] = wmask
        transit_prior.append(prior)
        transit_pnames.append(pname)
        # add to the parameter number
        pnum += 1
    # -------------------------------------------------------------------------
    # get the transit fit planet params
    # -------------------------------------------------------------------------
    # loop around planets
    for nplanet in range(1, n_planets + 1):
        # need the planet key
        pkey = f'PLANET{nplanet}'
        # check that we have this planet
        if pkey not in params:
            emsg = (f'ParamError: planet {nplanet} not found. '
                    f'Please add or adjust "NPLANETS" parameter.')
            raise base_classes.TransitFitExcept(emsg)
        # get planet dictionary
        planetdict = params[pkey]
        # loop around planetN parameters
        for key in TPP_ORDERED:
            p0, fmask, wmask, prior, pname = __assign_pvalue(key, planetdict,
                                                             n_phot, func_name)
            # add values to storage
            transit_p0[pnum] = p0
            transit_fmask[pnum] = fmask
            transit_wmask[pnum] = wmask
            transit_prior.append(prior)
            transit_pnames.append(f'{pname}{nplanet}')
            # add to the parameter number
            pnum += 1
    # -------------------------------------------------------------------------
    # get the transit fit hyper params
    # -------------------------------------------------------------------------
    # loop around hyper parameters
    for key in TPH_ORDERED:
        p0, fmask, wmask, prior, pname = __assign_pvalue(key, params, n_phot,
                                                         func_name)
        # hyper parameters are set to zero if not in fit mode
        if not fmask:
            transit_p0[pnum] = np.zeros(n_phot)
        else:
            transit_p0[pnum] = p0
        # add other values to storage
        transit_fmask[pnum] = fmask
        transit_wmask[pnum] = wmask
        transit_prior.append(prior)
        transit_pnames.append(pname)
        # add to the parameter number
        pnum += 1
    # -------------------------------------------------------------------------
    # create a class for the fit
    # -------------------------------------------------------------------------
    tfit = TransitFit()
    # add the data to the transit fit
    tfit.wavelength = data.phot['WAVELENGTH']
    tfit.time = data.phot['TIME']
    tfit.itime = data.phot['ITIME']
    tfit.flux = data.phot['FLUX']
    tfit.fluxerr = data.phot['FLUX_ERROR']
    tfit.orders = data.phot['ORDERS']
    # add the parameters
    tfit.p0 = transit_p0
    tfit.fmask = transit_fmask
    tfit.wmask = transit_wmask
    tfit.prior = transit_prior
    tfit.pnames = np.array(transit_pnames)
    # set the number of params
    tfit.n_param = len(transit_p0)
    # set the number of photometric bandpasses
    tfit.n_phot = int(data.n_phot)
    # set the number of integrations
    tfit.n_int = int(data.n_int)
    # -------------------------------------------------------------------------
    # set additional arguments
    # -------------------------------------------------------------------------
    tfit.pkwargs = dict()
    # loop through transit param additional keyword arguments
    for key in TP_KWARGS:
        # if this key exists in params set it
        if key in params:
            tfit.pkwargs[key] = params[key]
    # -------------------------------------------------------------------------
    # update fitted parameters
    # -------------------------------------------------------------------------
    # set x0 and xnames
    tfit.get_fitted_params()
    # start of beta
    tfit.beta = np.zeros_like(tfit.x0)
    # -------------------------------------------------------------------------
    # return the transit fit class
    # -------------------------------------------------------------------------
    return tfit


def lnpriors(tfit) -> float:
    """
    Log prior function - basically runs the prior function for
    each parameter and each bandpass

    If any fail returns immediately

    :param tfit: the transit fit class

        tfit must have attributes:
            n_phot: int, the number of band passes
            n_param: int, the number of parameters (fitted and fixed)
            p0: np.ndarray - the trial solution [n_param, n_phot]
            pnames: np.ndarray [n_param, n_phot] {STRING} - parameter names
            fmask: np.ndarray [n_param] {BOOL} - whether parameter is to be fit

    :return: float, either 1 (if good) or -np.inf (if bad)
    """
    # set initial value of loglikelihood
    logl = 1.0
    # the number of band passes
    n_phot = tfit.n_phot
    # the number of parameters
    n_param = tfit.n_param
    # name of parameters
    names = tfit.pnames
    # whether parameters are fitted or fixed
    fmask = tfit.fmask
    # trial solution
    sol = np.array(tfit.p0)
    # get priors
    priors = tfit.get(names, 'prior')
    # -------------------------------------------------------------------------
    # loop around all parameters and test priors
    for param_it in range(n_param):
        # skip fixed terms (we don't care about the priors for these parameters)
        if not fmask[param_it]:
            continue
        # get this parameters priors
        prior = priors[param_it]
        # loop around band passes
        for phot_it in range(n_phot):
            # get prior function - default prior is a tophat function
            func = prior.get('func', tophat_prior)
            # function returns True if pass prior condition
            if not func(sol[:, phot_it], **prior):
                return BADLPR
    # if we have got to here we return the good loglikelihood (all priors have
    #    passed)
    return logl


def lnprob(tfit: TransitFit) -> float:
    """
    The loglikelihood function

    :param tfit: the transit fit class

        tfit must have attributes:
            time: np.ndarray
            itime: np.ndarray
            flux: np.ndarray
            fluxerr: np.ndarray
            n_phot: int, the number of band passes
            n_param: int, the number of parameters (fitted and fixed)
            ntt:
            tobs:
            omc:
            p0: np.ndarray - the trial solution [n_param, n_phot]
            pnames: np.ndarray [n_param, n_phot] {STRING} - parameter names

    :return:
    """
    # -------------------------------------------------------------------------
    # lets get the values out of tfit (shallow copy)
    # -------------------------------------------------------------------------
    # the data
    time = tfit.time
    itime = tfit.itime
    flux = tfit.flux
    fluxerr = tfit.fluxerr
    # -------------------------------------------------------------------------
    # other parameters
    n_phot = tfit.n_phot
    ntt = tfit.pkwargs['NTT']
    tobs = tfit.pkwargs['T_OBS']
    omc = tfit.pkwargs['OMC']
    # trial solution
    sol = np.array(tfit.p0)
    # -------------------------------------------------------------------------
    # the hyper parameters
    # photometric error scale (DSC) for this bandpass
    dscale = tfit.get('DSC', 'p0')
    # GP Kernel Amplitude (ASC) for this bandpass
    ascale = tfit.get('ASC', 'p0')
    # GP length scale (LSC) for this bandpass
    lscale = tfit.get('LSC', 'p0')
    # deal with the fitting of hyper parameters
    fit_error_scale = tfit.get('DSC', 'fmask')
    fit_amplitude_scale = tfit.get('DSC', 'fmask')
    fit_length_scale = tfit.get('DSC', 'fmask')
    # -------------------------------------------------------------------------
    # Select model type: 1 = GP model, 2 = uncorrelated noise model
    model_type = int(fit_amplitude_scale | fit_length_scale)

    # -------------------------------------------------------------------------
    # Step 1: Prior calculation
    # -------------------------------------------------------------------------
    # check priors
    logl = lnpriors(tfit)
    # if out of limits return here
    if not (logl > BADLPR):
        return BADLPR
    # -------------------------------------------------------------------------
    # Step 2: Calculate loglikelihood per band pass
    # -------------------------------------------------------------------------
    # QUESTION: Can we parallelize this?
    # QUESTION: If one phot_it is found to be inf, we can skip rest?
    # loop around band passes
    for phot_it in range(n_phot):
        # check dscale, ascale and lscale hyper parameters
        #    (they must be positive)
        if (dscale[phot_it] <= 0.0) and fit_error_scale:
            return BADLPR
        if (ascale[phot_it] <= 0.0) and fit_amplitude_scale:
            return BADLPR
        if (lscale[phot_it] <= 0.0) and fit_length_scale:
            return BADLPR
        # get transit for current parameters
        tkwargs = dict(sol=sol[:, phot_it], time=time[phot_it],
                       itime=itime[phot_it], ntt=ntt, tobs=tobs, omc=omc)
        # get and plot the model
        model = transit_fit.transitmodel(**tkwargs)
        # check for NaNs -- we don't want these.
        if np.isfinite(model.sum()):
            # Question: What about the GP model, currently it does not update
            #           logl
            # non-correlated noise-model
            if model_type == 0:
                sum1 = np.sum(np.log(fluxerr[phot_it] ** 2 * dscale ** 2))
                sqrdiff = (flux[phot_it] - model)**2
                sum2 = np.sum(sqrdiff / (fluxerr**2 * dscale[phot_it]**2))
                logl += -0.5 * (sum1 + sum2)
        # else we return our bad log likelihood
        else:
            return BADLPR
    # if we have got to here we return the good loglikelihood (sum of each
    #   bandpass)
    return logl


def mhg_mcmc(tfit: TransitFit, loglikelihood: Any, beta: np.ndarray,
             buffer: Optional[np.ndarray] = None,
             corbeta: Optional[float] = None) -> TransitFit:
    """
    A Metropolis-Hastings MCMC with Gibbs sampler

    :param tfit: the Transit fit class
    :param loglikelihood: The loglikelihodd function here
                          loglikelihood must have a single argument of type
                          Transit fit class
    :param beta: ibb's factor : characteristic step size for each parameter
    :param buffer: Not used for mhg_mcmc (used in de_mhg_mcmc)
    :param corbeta: Not used for mhg_mcmc (used in de_mhg_mcmc)

    :return:
    """
    # we do not use bugger and corbeta here (used for de_mhg_mcmc)
    _ = buffer, corbeta
    # copy fit
    tfit0 = tfit.copy()
    # -------------------------------------------------------------------------
    # Step 1: Generate trial state
    # -------------------------------------------------------------------------
    # Generate trial state with Gibbs sampler
    tfit.generate_gibbs_sample(beta)
    # -------------------------------------------------------------------------
    # Step 2: Compute log(p(x'|d))=log(p(x'))+log(p(d|x'))
    # -------------------------------------------------------------------------
    llxt = loglikelihood(tfit)
    # -------------------------------------------------------------------------
    # Step 3 Compute the acceptance probability
    # -------------------------------------------------------------------------
    alpha = min(np.exp(llxt - tfit.llx), 1.0)
    # -------------------------------------------------------------------------
    # Step 4 Accept or reject trial
    # -------------------------------------------------------------------------
    # generate a random number
    test = np.random.rand()
    # if test is less than our acceptance level accept new trial
    if test <= alpha:
        tfit.x0 = np.array(tfit.x0)
        tfit.llx = float(llxt)
        tfit.ac = [0, tfit.n_tmp]
        # update full solution
        tfit.update_p0_from_x0()
        tfit.p0 = np.array(tfit.p0)
    # else we reject and start from previous point (tfit0)
    else:
        tfit.x0 = np.array(tfit0.x0)
        tfit.p0 = np.array(tfit0.p0)
        tfit.ac = [1, tfit.n_tmp]
    # return tfit instance
    return tfit


def de_mhg_mcmc(tfit: TransitFit, loglikelihood: Any, beta: np.ndarray,
                buffer: Optional[np.ndarray] = None,
                corbeta: Optional[float] = None) -> TransitFit:
    """
    A Metropolis-Hastings MCMC with Gibbs sampler

    :param tfit: the Transit fit class
    :param loglikelihood: The loglikelihodd function here
                          loglikelihood must have a single argument of type
                          Transit fit class
    :param beta: ibb's factor : characteristic step size for each parameter
    :param buffer: np.ndarray, previous chains used as a buffer state to
                   calculate a vector jump
    :param corbeta: float, scale factor correction to the vector jump

    :return:
    """
    # deal with no buffer or corbeta
    if buffer is None and corbeta is None:
        emsg = 'buffer and corbeta must not be None for de_mhg_mcmc()'
        raise base_classes.TransitFitExcept(emsg)
    # copy fit
    tfit0 = tfit.copy()
    # draw a random number to decide which sampler to use
    rsamp = np.random.rand()
    # -------------------------------------------------------------------------
    # Step 1: Generate trial state
    # -------------------------------------------------------------------------
    # if rsamp is less than 0.5 use a Gibbs sampler
    if rsamp < 0.5:
        # Generate trial state with Gibbs sampler
        tfit.generate_gibbs_sample(beta)
    # else we use our deMCMC sampler
    else:
        tfit.generate_demcmc_sample(buffer, corbeta)
    # -------------------------------------------------------------------------
    # Step 2: Compute log(p(x'|d))=log(p(x'))+log(p(d|x'))
    # -------------------------------------------------------------------------
    llxt = loglikelihood(tfit)
    # -------------------------------------------------------------------------
    # Step 3 Compute the acceptance probability
    # -------------------------------------------------------------------------
    alpha = min(np.exp(llxt - tfit.llx), 1.0)
    # -------------------------------------------------------------------------
    # Step 4 Accept or reject trial
    # -------------------------------------------------------------------------
    # generate a random number
    test = np.random.rand()
    # if test is less than our acceptance level accept new trial
    if test <= alpha:
        tfit.x0 = np.array(tfit.x0)
        tfit.llx = float(llxt)
        tfit.ac = [0, tfit.n_tmp]
        # update full solution
        tfit.update_p0_from_x0()
        tfit.p0 = np.array(tfit.p0)
    # else we reject and start from previous point (tfit0)
    else:
        tfit.x0 = np.array(tfit0.x0)
        tfit.p0 = np.array(tfit0.p0)
        tfit.ac = [1, tfit.n_tmp]
    # return tfit instance
    return tfit


# TODO: Fill out
def beta_rescale(params: ParamDict, tfit: TransitFit) -> np.ndarray:

    # get alow, ahigh define the acceptance rate range we want
    alow = params['BETA_ALOW']
    ahigh = params['BETA_AHIGH']
    # parameter controling how fast corscale changes - from Gregory 2011.
    delta = params['BETA_DELTA']

    # TODO: Fill out code
    corscale = np.zeros_like(tfit.n_param)

    return corscale


def genchain(tfit: TransitFit, niter: int, beta: np.ndarray,
             mcmcfunc, loglikelihood,
             buffer: Optional = None, corbeta: float = 1.0,
             progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Markov Chain

    :param tfit: Transit fit class of parameters
    :param niter: int, the number of steps for each chain to run through
    :param betacor: np.ndarray, the Gibb's factor : characteristic step size
                    for each parameter
    :param mcmcfunc: MCMC function
            - arguments: tfit: TransitFit,
                         loglikelihood: Any,
                         beta: np.ndarray,
                         buffer: np.ndarray, corbeta: float
    :param loglikelihood: log likelihood function
        - arguments: tfit: TransitFit
    :param progress: bool, if True uses tqdm to print progress of the
                     MCMC chain

    :return: tuple, 1. chains (numpy array [n_iter, n_x])
                    2. rejects (numpy array [n_iter, 2]
                       where reject=(rejected 1 or 0, parameter num changed)
    """
    # deal with no buffer set
    if buffer is None:
        buffer = []
    # -------------------------------------------------------------------------
    # Initialize list to hold chain values and set first value to the initial
    #    solution
    chains = [np.array(tfit.x0)]
    # Track our acceptance rate - and set the first one to (0, 0)
    #    note reject=(rejected 1 or 0, parameter changed)
    rejections = [(0, 0)]
    # -------------------------------------------------------------------------
    # pre-compute the first log-likelihood
    tfit.llx = loglikelihood(tfit)
    # -------------------------------------------------------------------------
    # now to the full loop of niterations
    if progress:
        # loop around iterations
        for _ in tqdm(range(0, niter)):
            # run the mcmc function
            tfit = mcmcfunc(tfit, loglikelihood, beta, buffer, corbeta)
            # append results to chains and rejections lists
            chains.append(np.array(tfit.p0))
            rejections.append(np.array(tfit.ac))
    else:
        # loop around iterations
        for _ in range(0, niter):
            # run the mcmc function
            tfit = mcmcfunc(tfit, loglikelihood, beta, buffer, corbeta)
            # append results to chains and rejections lists
            chains.append(np.array(tfit.p0))
            rejections.append(np.array(tfit.ac))
    # -------------------------------------------------------------------------
    # convert lists to arrays
    chains = np.array(chains)
    rejections = np.array(rejections)
    # -------------------------------------------------------------------------
    # return the chains and rejections
    return chains, rejections


def calculate_acceptance_rates(rejections: np.ndarray,
                               burnin: int) -> Dict[int, float]:
    """
    Calculate Acceptance Rates

    :param rejections: np.ndarray [n_chains, 2]
                       where the second axis is (rejected, param_number)
                       rejected: 0 when accepted, 1 when rejected
                       param_number is either [-1, 0 to n_param]
                       when param_number is -1 we have deMCMC fit

    :param burnin: int, the number of chains the burn at the start

    :return: acceptance dictionary, keys = [-1, 0 to n_param]
             values = the acceptance for each key
    """
    # get the number of chains
    nchain = len(rejections[:, 0])
    # get the number of chains minus the burnin
    nchainb = nchain - burnin
    # acceptance is the fraction of chains accepted compared to total number
    #  (when burn in is considered)
    gaccept = (nchainb - np.sum(rejections[burnin:, 0])) / nchainb
    # print the global acceptance rate
    print(f'Global Acceptance Rate: {gaccept:.3f}')
    # -------------------------------------------------------------------------
    # storage of values
    acceptance_dict = dict()
    # -------------------------------------------------------------------------
    # deMCMC number of proposals
    de_nprop = 0
    # deMCMC acceptance rate
    de_accept_rate = 0
    # -------------------------------------------------------------------------
    # loop around parameter number
    # Question: Why can't we use the max number of parameters here?
    #           Is it because all parameters might not be selected by
    #           the Gibbs sampler?
    for p_num in range(max(rejections[burnin:, 1]) + 1):
        # deMCMC number of proposals
        de_nprop = 0
        # deMCMC acceptance rate
        de_accept_rate = 0
        # number of proposals
        n_prop = 0
        # acceptance rate
        accept_rate = 0
        # loop around each chain
        for chain_it in range(burnin, nchain):
            # if the rejection comes from this parameter
            if rejections[chain_it, 1] == p_num:
                # add one to the number of proposals
                n_prop +=  1
                # add to the acceptance rate (0 = accept, 1 = reject)
                accept_rate += rejections[chain_it, 0]
            # if we are in deMCMC mode (n_param = -1) add to de variables
            elif rejections[chain_it, 1] == -1:
                # add one to the de number of proposals
                de_nprop += 1
                # add to the de acceptance rate (0 = accept, 1 = reject)
                de_accept_rate += rejections[chain_it, 0]
        # print the acceptance rate for this
        acceptance = (n_prop - accept_rate) / (n_prop + 1)
        print(f'Param {p_num}: Acceptance Rate {acceptance:.3f}')
        # store for later use
        acceptance_dict[p_num] = acceptance

    # Question: This is only calculated for the last loop (as denprop reset
    #           inside loop) is this what we want?
    # if we have deMCMC results, report the acceptance rate.
    if de_nprop > 0:
        de_acceptance = (de_nprop - de_accept_rate) / de_nprop
        print(f'deMCMC: Acceptance Rate {de_acceptance:.3f}')
        # store for later use
        acceptance_dict[-1] = de_acceptance
    # return the acceptance dictionary
    return acceptance_dict


def gelman_rubin_convergence(chains: Dict[int, np.ndarray],
                             burnin: int, npt: int) -> np.ndarray:
    """
    Estimating PSRF

    See pdf doc BrooksGelman for info

    :param chains: dictionary of chains, each one being a np.ndarray
                   of shape [n_params, n_phot]
    :param burnin: int, the number of chains to burn
    :param npt: int, the number of parameters

    :return: numpy array [n_params], the gelman rubin convergence for each
             parameter
    """
    # get the number of walkers (c.f. number of chains)
    n_walkers = len(chains)
    # assume all chains have the same size
    n_chain = chains[0].shape[0] - burnin
    # get the number of parameters
    n_param = chains[0].shape[1]
    # -------------------------------------------------------------------------
    # allocate an array to hold mean calculations
    pmean = np.zeros((n_walkers, n_param))
    # allocate an array to hold variance calculations
    pvar = np.zeros((n_walkers, n_param))
    # -------------------------------------------------------------------------
    # loop over each walker
    for walker in range(n_walkers):
        # Generate means for each parameter in each chain
        # TODO: check axis
        pmean[walker] = np.mean(chains[walker][burnin:], axis=1)
        # Generate variance for each parameter in each chain
        # TODO: check axis
        pvar[walker] = np.var(chains[walker][burnin:], axis=1)
    # -------------------------------------------------------------------------
    # calculate the posterior mean for each parameter
    # TODO: check axis
    posteriormean = np.mean(pmean, axis=1)
    # -------------------------------------------------------------------------
    # Calculate between chains variance
    bvar = np.sum((pmean - posteriormean) ** 2, axis=0)
    bvar = bvar * n_chain / (n_walkers - 1.0)
    # -------------------------------------------------------------------------
    # Calculate within chain variance
    wvar = np.sum(pvar, axis=0)
    wvar = wvar / n_walkers
    # -------------------------------------------------------------------------
    # Calculate the pooled variance
    part1 = (wvar / n_chain)
    part2 = bvar * (n_walkers + 1) / (n_walkers * n_chain)
    vvar = (n_chain - 1) * part1 * part2
    # -------------------------------------------------------------------------
    # degrees of freedom
    dof = npt - 1
    # -------------------------------------------------------------------------
    # dof ratio for rc and ru
    dofr = (dof + 3.0) / (dof + 1.0)
    # -------------------------------------------------------------------------
    # PSRF from Brooks and Gelman (1997)
    rc = np.sqrt(dofr * vvar / wvar)
    # -------------------------------------------------------------------------
    # Calculate Ru
    # part1 = (n_chain - 1.0)/n_cahin
    # part2 = qa * (n_walkers + 1)/n_walkers
    # ru = np.sqrt((dofr * part1 * wvar) + part2)
    # -------------------------------------------------------------------------
    # return rc
    return rc


# =============================================================================
# MCMC Sampler
# =============================================================================
class Sampler:
    wchains: Dict[int, np.ndarray]
    wrejects: Dict[int, np.ndarray]

    def __init__(self, params: ParamDict, tfit: TransitFit, mode='full'):
        self.params = params
        self.tfit = tfit.copy()
        self.mode = mode
        self.wchains = dict()
        self.wrejects = dict()
        # set up storage of chainns
        for nwalker in range(self.params['WALKERS']):
            self.wchains[nwalker] = np.array([])
            self.wrejects[nwalker] = np.array([])

    def run_mcmc(self, corscale: np.ndarray, loglikelihood, mcmcfunc,
                 trial: Optional['Sampler'] = None):
        """
        Run the MCMC using a correction scale of beta previously
        calculated by betarescale)

        :param corscale: numpy array [n_param]
        :param loglikelihood: log likelihood function
                - arguments: tfit: TransitFit
        :param mcmcfunc: MCMC function
                - arguments: tfit: TransitFit,
                             loglikelihood: Any,
                             beta: np.ndarray,
                             buffer: np.ndarray, corbeta: float
        :param trial: optional, Sampler class, a previous sampler to start
                      the chains from
        :return:
        """
        # ---------------------------------------------------------------------
        # get parameters from params
        # ---------------------------------------------------------------------
        # get the maximum number of loops for this mode
        nloopsmax = self.params['NLOOPMAX'][self.mode]
        # get the number of steps for the MCMC for this mode
        nsteps = self.params['NSTEPS'][self.mode]
        # set number of walkers
        nwalkers = self.params['WALKERS']
        # set the burnin parameter
        burninf = self.params['BURNINF'][self.mode]
        # convergence criteria for buffer
        buf_converge_crit = self.params['BUFFER_COVERGE_CRIT']
        # the number of steps we add on next loop (if convergence not met)
        nsteps_inc = self.params['NSTEPS_INC']
        # correction to beta term for deMCMC
        corbeta = self.params['CORBETA'][self.mode]
        # ---------------------------------------------------------------------
        # deal with having a trial sampler
        sampler = trial
        # ---------------------------------------------------------------------
        # loop around
        # ---------------------------------------------------------------------
        # start loop counter
        nloop = 0
        # set the constant genchain parameters
        gkwargs = dict(niter=nsteps, betacor=self.tfit.beta * corscale,
                       loglikelihood=loglikelihood, mcmcfunc=mcmcfunc,
                       corbeta=corbeta, progress=True)
        # -----------------------------------------------------------------
        # Loop around iterations until we break (convergence met) or max
        #     number of loops exceeded
        # -----------------------------------------------------------------
        while nloop < nloopsmax:
            # print progress
            print(f'Loop {nloop+1}')
            # -----------------------------------------------------------------
            # loop around walkers
            # -----------------------------------------------------------------
            # get the chains
            # TODO: parallize this
            for nwalker in range(nwalkers):
                # copy tfit
                htfit = self.tfit.copy()
                # get the buffer from previous chain and update tfit from
                #   previous chain.
                # Previous chain can be from another sampler or from a
                #   previous iteration of the NLOOP while loop
                # Also updates the starting x0 and p0 for htfit
                buffer, htfit = start_from_previous_chains(self, htfit, sampler)
                # get chains and rejects for this walker
                hchains, hrejects = genchain(htfit, buffer=buffer, **gkwargs)
                # push chains into walker storage
                wcs = np.concatenate([self.wchains[nwalker], hchains])
                self.wchains[nwalker] = wcs
                # push rejects into walker storage
                rcs = np.concatenate([self.wrejects[nwalker], hrejects])
                self.wrejects[nwalker] = rcs

            # -----------------------------------------------------------------
            # Calculate the Gelman-Rubin Convergence
            # -----------------------------------------------------------------
            # get number of chains to burn (using burn in fraction)
            burnin = int(self.wchains[0].shape[0] * burninf)
            # calculate the rc factor
            grtest = gelman_rubin_convergence(self.wchains, burnin=burnin,
                                              npt=self.tfit.n_phot)
            # print the factors
            print('Gelman-Rubin Convergence. Rc param:')
            for param_it in range(self.tfit.n_param):
                # print Rc parameter
                pargs = [param_it, self.tfit.pnames[param_it],
                         grtest[param_it]]
                print(f'\t{0:3d} {1:3s}: {2:.4f}'.format(*pargs))
            # update the full chains
            self.chain, self.reject = join_chains(self, burnin)

            # -----------------------------------------------------------------
            # Calculate acceptance rate
            # -----------------------------------------------------------------
            if self.mode == 'trial':
                rejects = self.wrejects[0]
                burnin_full = int(self.wchains[0].shape[0] * burninf)
            else:
                rejects = self.reject
                burnin_full = int(self.chain.shape[0] * burninf)

            # calculate acceptance for chain1
            self.acc_dict = calculate_acceptance_rates(rejects,
                                                       burnin=burnin_full)

            # -----------------------------------------------------------------
            # Test criteria for success
            # -----------------------------------------------------------------
            # criteria for accepting mcmc chains
            #   all parameters grtest greater than or equal to
            #   buf_converge_crit
            # Question: Is this the same thing?
            #  previously sum(grtest[accept_mask] / grtest[accept_mask])
            if np.sum(grtest < buf_converge_crit) == len(grtest):
                break
            else:
                # add to the number of steps (for next loop)
                nsteps += nsteps_inc
                # add to the loop iteration
                nloop += 1
                # deal with chain update for next loop
                if self.mode == 'full':
                    sampler = self

    def posterior_print(self):

        # calulate medians
        medians = np.median(self.chain, axis=0)

        print('Posterior median values')
        # loop around parameters
        for x_it in range(self.tfit.n_x):
            # print argument
            pargs = [x_it, self.tfit.xnames[x_it], medians[x_it]]
            print('\t{0} {1: 3s}: {2:.5f}'.format(*pargs))



def start_from_previous_chains(current: Sampler, tfit: TransitFit,
                               previous: Sampler
                               ) -> Tuple[Optional[np.ndarray], TransitFit]:
    """
    Start from a previous chain (be it a previous sampler (i.e. trial) or
    from the current chain)

    :param tfit: Transit fit parameter container
    :param sampler: Sampler class

    :return: tuple, 1. the buffer (previous chains burnt in and combined)
             2. the update tfit (x0 and p0) using most recent chain
             (chains[-1])
    """
    # if in trial mode we don't do this
    if current.mode == 'trial':
        # set the buffer to None and return tfit as it is
        return None, tfit
    # set the burnin parameter
    burninf = current.params['BURNINF'][current.mode]
    # get the burn in from trial shape
    burnin = int(previous.wchains[0].shape[0] * burninf)
    # get buffer (by joining chains)
    buffer, _ = join_chains(previous, burnin)
    # loop around walkers in the trial
    for walker in previous.wchains:
        # get start point for this chain (the last chain
        #    from trial sampler)
        # Question: do you mean to start x1, x2 and x3 from chain1?
        #           if so change [0] to [walker]
        update_x0_p0_from_chain(tfit, previous.wchains[0], -1)

    # return the buffer (
    return buffer, tfit


def join_chains(sampler: Sampler, burnin: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Join a set of sampler chains and rejects (using a burnin)

    :param sampler: Sampler class
    :param burnin: int, the number of chains to burn

    :return: tuple 1. the combined chain arrays, 2. the combined reject arrays
    """
    chains, rejects = [], []
    # loop around chains
    for walker in sampler.wchains:
        # get the chains without the burnin chains
        walker_chain = sampler.wchains[walker][burnin:]
        # get the rejects
        walker_reject = sampler.wrejects[walker][burnin:]
        # add walker to chains
        chains = np.concatenate([chains, walker_chain])
        # add walker to rejects
        rejects = np.concatenate([rejects, walker_reject])
    # return the combined chains and rejects
    return chains, rejects


def update_x0_p0_from_chain(tfit: TransitFit, chain: np.ndarray,
                            chain_num: int) -> TransitFit:
    """
    Update the x0 and p0 from a specific chain

    :param tfit: Transit fit parameter container
    :param chain: np.ndarray, the chain [n_steps, x_n]
    :param chain_num: int, the position in chain to get (positive to count
                      from start, negative to count from end)

    :return: the updated Transit fit parameter container
    """
    # set x0 to the last chain position
    tfit.x0 = chain[0][chain_num, :]
    # update solution
    tfit.update_p0_from_x0()
    # return updated tfit
    return tfit


# =============================================================================
# Define worker functions
# =============================================================================
def __assign_pvalue(key, pdict, n_phot, func_name):
    # deal with key = None
    if key is None:
        p0 = np.zeros(n_phot)
        fmask = False
        wmask = False
        prior = None
        pnames = 'Undefined'
        return p0, fmask, wmask, prior, pnames
    # deal with key not in params (bad)
    if key not in pdict:
        emsg = f'ParamError: key {key} must be set. func={func_name}'
        raise base_classes.TransitFitExcept(emsg)
    # get the value for this key
    fitparam = pdict[key]
    # only consider FitParams classes
    if not isinstance(fitparam, FitParam):
        emsg = (f'ParamError: key {key} must be a FitParam. '
                f'\n\tGot {key}={fitparam}'
                f'\n\tfunc={func_name}')
        raise base_classes.TransitFitExcept(emsg)
    # get the properties from the fit param class
    p0, fmask, wmask, prior, pnames = fitparam.get_value(n_phot)

    return p0, fmask, wmask, prior, pnames


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
