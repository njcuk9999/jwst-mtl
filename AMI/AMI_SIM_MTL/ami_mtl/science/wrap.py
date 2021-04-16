#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 

@author: 
"""
import amical
import numpy as np
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
from mirage.apt import read_apt_xml
import yaml

from ami_mtl.core.core import constant_functions
from ami_mtl.core.core import exceptions
from ami_mtl.core.core import general
from ami_mtl.core.core import param_functions
from ami_mtl.science import etienne
from ami_mtl.io import drs_file

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'apero.core.core.wrap.py'
__DESCRIPTION__ = 'module of wrapper classes and functions'
# get parameter dictionary
ParamDict = param_functions.ParamDict
# get Observation Exception
DrsException = exceptions.DrsException
ObservationException = exceptions.ObservationException
# get general functions
display_func = general.display_func


# =============================================================================
# Define classes
# =============================================================================
class Simulation:

    classname: str = 'Simulation'

    def __init__(self, params: ParamDict, properties: Dict[str, Any]):
        """
        Basic simulation class (what is passed around)

        :param properties: simulation dictionary (from yaml file)
        """
        # set name
        self.name = general.clean_name(properties.get('name', None))
        # set parameters
        self.params = params.copy()
        # ---------------------------------------------------------------------
        # get xml path
        self.xmlpath = properties.get('xmlpath', None)
        # get xml settings
        self.params = _load_xml(params, self.xmlpath)
        # ---------------------------------------------------------------------
        # get filters to use
        raw_filters = properties.get('filters', [])
        # only keep filters in all filters
        self.use_filters = []
        # loop around all filters
        for _filter in raw_filters:
            # make sure filter is allowed
            if _filter in params['ALL_FILTERS']:
                self.use_filters.append(_filter)
        # ---------------------------------------------------------------------
        # get target
        raw_target = properties.get('target', None)
        if raw_target is not None:
            self.target = Target(params, raw_target)
        # ---------------------------------------------------------------------
        # get calibrators
        self.calibrators = []
        self._get_calibrators(properties)
        # ---------------------------------------------------------------------

    def __str__(self) -> str:
        """
        String representation of class

        :return:
        """
        return '{0}[{1}]'.format(self.classname, self.name)

    def __repr__(self) -> str:
        """
        String representation of class

        :return:
        """
        return self.__str__()

    def _get_calibrators(self, properties: Dict[str, Any]):
        """
        Get the calibrators (as instances) for this simulation

        :param properties: simulation dictionary (from yaml file)

        :return:
        """
        # get keys
        keys = list(properties.keys())
        # loop around keys and load companions
        for key in keys:
            # companions should start with "companion"
            if key.startswith('calibrator'):
                # load companion
                calibrator = Calibrator(self.params, properties[key])
                # load magnitudes for companion
                calibrator.get_magnitudes()
                # add to list
                self.calibrators.append(calibrator)


class Observation:

    classname: str = 'Observation'

    def __init__(self, params: ParamDict, properties: Dict[str, Any],
                 mag_key: str = 'magnitude'):
        """
        Basic observation class

        :param params: ParamDict, the parameter dictionary of constants
        :param properties: observation dictionary (from yaml file)
        """
        # set function name
        func_name = __NAME__ + 'Observation.__init__()'
        # set params
        self.params = params.copy()
        self.properties = properties
        # set name
        self.name = properties.get('name', None)
        # deal with no name
        if self.name is None:
            emsg = ('ObservationError: Name must be set for class '
                    '{0}'.format(self.__str__()))
            emsg += '\n input properties: '
            emsg += '\n\t'.join(self._str_properties(properties))
            params.log.error(emsg, exception=ObservationException,
                             funcname=func_name)
        # clean name
        self.name = general.clean_name(self.name)
        # ---------------------------------------------------------------------
        # set raw magnitudes
        self.raw_magnitudes = properties.get(mag_key, None)
        # store magnitudes (per filter)
        self.magnitudes = dict()
        # store the filters to use (based on APT file)
        self.filters = []
        # store the sub array
        self.subarrays = dict()
        # store extracted flux (calculated from magnitudes)
        self.ext_fluxes = dict()
        # store total exposure time (calculated from nints * ngroups * tframe)
        self.tot_exp = dict()
        # store nints (per filter)
        self.num_integrations = dict()
        # store ngroups (per filter)
        self.num_groups = dict()

    def link_to_xml(self):
        """
        Link an observation to an xml file (name must be in xml file)
        Note: name is cleaned in xml and from input name

        :return: None, updates self.params
        """
        # set function name
        func_name = display_func('link_to_xml', __NAME__, self.classname)
        # ---------------------------------------------------------------------
        # make sure observation is in apt-targets dictionary
        apt_targets = dict(self.params['APT-TARGETS'])
        # get target_names
        target_names = list(apt_targets.keys())
        # if name is not in target we need to raise an error
        if self.name not in target_names:
            # get xml file
            xmlfile = apt_targets[target_names[0]]['XML-FILE']
            # log error
            emsg = 'ObservationError: Name must be in xml file.'
            emsg += '\n xml file: {0}'.format(xmlfile)
            emsg += '\n input properties: '
            emsg += '\n\t'.join(self._str_properties(self.properties))
            self.params.log.error(emsg, exception=ObservationException,
                                  funcname=func_name)
        # ---------------------------------------------------------------------
        # get xml info for target
        xml_target = apt_targets[self.name]
        xml_filename = xml_target['XML-FILE']
        # set parameters for target
        for key in self.params:
            # get instance
            instance = self.params.instances[key]
            # deal with no apt value (skip)
            if instance is None or instance.apt is None:
                continue
            # get xml value
            values = xml_target[instance.apt]
            # -----------------------------------------------------------------
            # deal with sub array being None
            if key == 'APT-TARGET-SUBARRAYS':
                values = self._deal_with_subarray(values, func_name)
            # check filters are valid
            if key == 'APT-TARGET-FILTERS':
                values = self._deal_with_filters(values, func_name)
            # -----------------------------------------------------------------
            # else add to params (if in apt file)
            self.params[key] = values
            sargs = [xml_filename, self.name, instance.apt]
            self.params.set_source(key, '{0}[{1}].{2}'.format(*sargs))

    def _str_properties(self, properties: Dict[str, Any]) -> list:
        """
        Take a dictionary and produce a string representation for each
        key

        :param properties: some dictionary

        :return: list of strings (one element for each key in dictionary)
        """
        # storage of out list
        out_list = []
        # loop around properties
        for prop in properties:
            # get the value for the property
            value = properties[prop]

            if isinstance(value, (int, float, str, bool)):
                out_list.append('{0}: {1}'.format(prop, value))
            # don't show sub-dictionaries
            if isinstance(value, dict):
                out_list.append('{0}: Dict'.format(prop))
        # return list
        return out_list

    def _deal_with_subarray(self, in_values: List[str],
                            func_name: str) -> List[str]:
        """
        Deal specifically with sub array value from APT

        :param in_values: list of str, the values for sub array from APT
        :param func_name: str, the function name
        :return:
        """
        # storage of outputs
        out_values = []
        # loop around input values
        for in_value in in_values:
            # assume source is APT originally
            subarray_source = 'APT'
            # check for None value
            if in_value in ['None', None, '']:
                out_value = self.params['DEFAULT_SUBARRAY']
                subarray_source = self.params.sources['DEFAULT_SUBARRAY']
            else:
                out_value = str(in_value)
            # check that value is now valid (in SUBARRAYS)
            if out_value not in self.params['SUBARRAYS']:
                emsg = 'ObservationError: XML Subarray value invalid'
                emsg += '\n\t Subarray = "{0}" (source={1})'
                emsg = emsg.format([out_value, subarray_source])
                self.params.log.error(emsg, exception=ObservationException,
                                      func_name=func_name)
            # add to outputs
            out_values.append(out_value)
        # return value
        return out_values

    def _deal_with_filters(self, in_values: List[str],
                           func_name: str) -> List[str]:
        """
        Deal specifically with sub array value from APT

        :param in_values: list of str, the values for sub array from APT
        :param func_name: str, the function name
        :return:
        """
        # loop around input values
        for in_value in in_values:
            # check that value is now valid (in SUBARRAYS)
            if in_value not in self.params['ALL_FILTERS']:
                emsg = 'ObservationError: XML Filters value invalid'
                emsg += '\n\t Filters = "{0}" (source={1})'
                emsg = emsg.format(in_value, 'APT')
                self.params.log.error(emsg, exception=ObservationException,
                                      func_name=func_name)
        # return value
        return in_values

    def __str__(self) -> str:
        """
        String representation of class

        :return:
        """
        return '{0}[{1}]'.format(self.classname, self.name)

    def __repr__(self) -> str:
        """
        String representation of class

        :return:
        """
        return self.__str__()

    def get_magnitudes(self):
        """
        Get magnitudes from properties (if set) for all filters in
        params['ALL_FILTERS']

        :return:
        """
        if self.raw_magnitudes is not None:
            # loop around all allowed filters
            for mag in list(self.params['ALL_FILTERS']):
                # make sure mag is in raw magnitudes
                if mag in self.raw_magnitudes:
                    self.magnitudes[mag] = float(self.raw_magnitudes[mag])
                else:
                    self.magnitudes[mag] = np.nan

    def get_observational_params(self):
        """
        observational parameters are per filter

        :return:
        """
        # get all filters that can be used (due to APT file setup)
        self.filters = list(self.params['APT-TARGET-FILTERS'])

        # get parameters from params
        nints = list(self.params['APT-TARGET-NINT'])
        ngroups = list(self.params['APT-TARGET-NINT'])
        nsubarrays = list(self.params['APT-TARGET-SUBARRAYS'])
        zeropoints = dict(self.params['ZEROPOINTS'])
        tframes = dict(self.params['T_FRAMES'])
        # ----------------------------------------------------------------------
        # check we have magniutdes for each filter
        for _filter in self.filters:
            # raise error if filter not present in magnitudes
            if _filter not in self.magnitudes:
                emsg = 'Filter {0} missing. Please run X.get_magnitudes()'
                self.params.log.error(emsg.format(_filter))
        # ----------------------------------------------------------------------
        # loop around filters and set values
        for it, _filter in enumerate(self.filters):
            # get this iterations values
            nint = int(nints[it])
            ngroup = int(ngroups[it])
            subarray = str(nsubarrays[it])
            # ------------------------------------------------------------------
            # get magnitude for this target
            mag = self.magnitudes[_filter]
            # get zeropoint for this target
            zeropoint = zeropoints[_filter]
            # get sub array tframe for this sub array
            tframe = tframes[subarray]
            # work out the extracted flux
            ext_flux = self._get_flux_rate(mag, zeropoint)
            # work out the total exposure time
            tot_exp = self._get_total_exp_time(nint, ngroup, tframe)
            # -----------------------------------------------------------------
            # add to storage
            # -----------------------------------------------------------------
            # store extracted flux (calculated from magnitudes)
            self.ext_fluxes[_filter] = ext_flux
            # store total exposure time
            self.tot_exp[_filter] = tot_exp
            # store nints (per filter)
            self.num_integrations[_filter] = nint
            # store ngroups (per filter)
            self.num_groups[_filter] = ngroup

    def get_contrast(self, companion: 'Companion', _filter: str) -> float:
        # make sure we have filter in self
        if _filter not in self.magnitudes:
            emsg = 'ObservationError: Cannot get contrast for target {0}'
            emsg += '\n\tTarget {1} does not have magnitude "{2}"'
            eargs = [self.name, self.name, _filter]
            self.params.log(emsg.format(*eargs))
        # get target magnitude
        target_mag = self.magnitudes[_filter]
        # make sure we have filter in companion
        if _filter not in companion.magnitudes:
            emsg = 'ObservationError: Cannot get contrast for target {0}'
            emsg += '\n\tCompanion {1} does not have magnitude "{2}"'
            eargs = [self.name, companion.name, _filter]
            self.params.log(emsg.format(*eargs))
        # get companion magnitude
        companion_mag = companion.magnitudes[_filter]
        # get delta magnitude
        delta_mag = companion_mag - target_mag
        # return contrast
        return 10 ** (-0.4 * delta_mag)

    def _get_flux_rate(self, magnitude: float, zeropoint: float) -> float:
        """
        Get the flux rate based on a magnitude (in e-/second)

        :param magnitude: float, magnitude for a specific filter
        :param zeropoint: float, the zeropoint magnitude for a specific filter

        :return: the flux rate [e-/second]
        """
        return 10 ** (0.4 * (zeropoint - magnitude))

    def _get_total_exp_time(self, nint: int, ngroup: int,
                            tframe: float) -> float:
        """
        Get the total exposure time (based on number of integrations,
        number of groups and tframe time based on sub array)

        :param nint: int, the number of integrations
        :param ngroup: int, the number of groups
        :param tframe: float, the frame time for a specific sub array [seconds]

        :return: float, the total exposure time [seconds]
        """
        return nint * (ngroup - 1) * tframe


class Target(Observation):

    classname: str = 'Target'

    def __init__(self, params: ParamDict, properties: Dict[str, Any]):
        """
        Basic target class

        :param params: ParamDict, the parameter dictionary of constants
        :param properties: target dictionary (from yaml file)
        """
        super().__init__(params, properties)
        # link to xml file
        self.link_to_xml()
        # load magnitudes
        self.get_magnitudes()
        # get flux and exposure times (using APT info from link_to_xml)
        self.get_observational_params()
        # deal with companions
        self.companions = []
        self.get_companion(properties)

    def get_companion(self, properties: Dict[str, Any]):
        """
        Get all companions and start a companion instance for each

        :param properties: target dictionary (from yaml file)

        :return:
        """
        # get keys
        keys = list(properties.keys())
        # loop around keys and load companions
        for key in keys:
            # companions should start with "companion"
            if key.startswith('companion'):
                # load companion
                companion = Companion(self.params, properties[key])
                # load magnitudes for companion
                companion.get_magnitudes()
                # add to list
                self.companions.append(companion)


class Calibrator(Observation):

    classname: str = 'Calibrator'

    def __init__(self, params: ParamDict, properties: Dict[str, Any]):
        """
        Basic calibrator class

        :param params: ParamDict, the parameter dictionary of constants
        :param properties: calibrator dictionary (from yaml file)
        """
        super().__init__(params, properties)
        # link to xml file
        self.link_to_xml()
        # load magnitudes
        self.get_magnitudes()
        # get flux and exposure times (using APT info from link_to_xml)
        self.get_observational_params()


class Companion(Observation):

    classname: str = 'Companion'

    def __init__(self, params: ParamDict, properties: Dict[str, Any]):
        """
        Basic companion class

        :param params: ParamDict, the parameter dictionary of constants
        :param properties:
        """
        super().__init__(params, properties, mag_key='mag')
        # get name
        self.name = general.clean_name(properties.get('name', 'Unknown'))
        # get kind (currently only support "planet")
        self.kind = properties.get('kind', 'planet')
        # get separation in arc seconds (used for kind="planet")
        self.separation = properties.get('separation', np.nan)
        # get position angle in degrees (used for kind="planet")
        self.position_angle = properties.get('pa', np.nan)
        # get magnitudes
        self.magnitudes = dict()
        self.get_magnitudes()


class XMLReader(read_apt_xml.ReadAPTXML):

    def read_xml_silent(self, infile: Union[Path, str]) -> dict:
        """
        Suppresses print outs

        :param infile: str
        :return:
        """
        # suppress all print outs
        sys.stdout = open(os.devnull, 'w')
        # run read xml
        table = self.read_xml(str(infile), False)
        # restore all print outs
        sys.stdout = sys.__stdout__
        # return table
        return table


# =============================================================================
# Define general functions
# =============================================================================
def load_simulations(params: ParamDict, config_file: Union[str, Path]):
    """
    Load simulation

    :param params:
    :param config_file:
    :return:
    """
    # load config file
    with open(config_file) as yfile:
        properties = yaml.load(yfile, Loader=yaml.FullLoader)
    # get keys
    keys = list(properties.keys())
    # get deep copied params (updated using yaml properties)
    params = _update_params(params, properties, config_file)
    # storage of simulation instances
    simulations = []
    # loop around keys and load companions
    for key in keys:
        # companions should start with "companion"
        if key.startswith('Simulation'):
            # load companion
            try:
                simulation = Simulation(params, properties[key])
                # add to list
                simulations.append(simulation)
            except DrsException as e:
                name = general.clean_name(properties[key].get('name', None))
                wmsg = 'Cannot process simulation: {0}'.format(name)
                wmsg += '\n\t Error {0}'.format(type(e))
                wmsg += '\n\t Skipping simulation'
                params.log.warning(wmsg)
    # return simulations
    return simulations


# =============================================================================
# Define simulation functions
# =============================================================================
def sim_module(simulations: List[Simulation]):
    """
    Loop around simulations and dispatch to simulation codes

    :param simulations: list of simulation instances
    :return:
    """
    # loop around simulations
    for simulation in simulations:
        # simulate using AMISIM
        if simulation.params['AMISIM-USE']:
            # simulate target
            run_ami_sim(simulation.name, simulation.target)
            # simulate calibrators
            for calibrator in simulation.calibrators:
                # simulate calibrator
                run_ami_sim(simulation.name, calibrator)
        # simulate using Mirage
        if simulation.params['MIRAGE-USE']:
            # simulate target
            run_mirage(simulation.target)
            # simulate calibrators
            for calibrator in simulation.calibrators:
                run_mirage(calibrator)


def run_ami_sim(simname: str, observation: Observation):
    """
    Run the AMI SIM module on a specific observation

    :param simname: the name of this simulation
    :param observation: the observation to run through AMI-SIM
    :return:
    """

    # set function name
    func_name = display_func('run_ami_sim', __NAME__)
    # get params
    params = observation.params
    # Process update
    msg = 'Processing Simulation: {0} Observation: {1}'
    margs = [simname, observation.name]
    params.log.info(msg.format(*margs))
    # loop around all filters to use
    for _filter in observation.filters:
        # construct file path
        path = Path(str(params.get('AMISIM-PATH', params['DIRECTORY'])))
        # construct filename
        oargs = [simname, observation.name, _filter]
        filename = 'SKY_SCENE_{0}_{1}_{2}.fits'.format(*oargs)
        # construct abs path to file
        scenepath = str(path.joinpath(filename))
        # get file path as a string
        target_dir = str(path)
        # update params for observation
        akey = 'AMI-SIM-SCENE-{0}'.format(_filter)
        observation.params[akey] = scenepath
        observation.params.set_source(akey, func_name)
        # ---------------------------------------------------------------------
        # step 1: make primary on image
        # ---------------------------------------------------------------------
        # get properties for simple scene
        pkwargs = dict()
        pkwargs['fov_pixels'] = params['FOV_PIXELS']
        pkwargs['oversample'] = params['OVERSAMPLE_FACTOR']
        pkwargs['pix_scale'] = params['PIX_SCALE']
        pkwargs['ext_flux'] = observation.ext_fluxes[_filter]
        pkwargs['tot_exp'] = observation.tot_exp[_filter]
        # add the target at the center of the image
        image, hdict = etienne.ami_sim_observation(**pkwargs)
        # add filter to hdict
        hdict['FILTER'] = (_filter, 'Input filter used')
        hdict['TNAME'] = (observation.name)
        # get combined count_rate
        count_rate = float(hdict['COUNT0'][0])
        # ---------------------------------------------------------------------
        # step 2: add companion(s)
        # ---------------------------------------------------------------------
        # only do this for targets (calibrators do not have companions by
        #     definition)
        if isinstance(observation, Target):
            # loop around all companions
            for it, companion in enumerate(observation.companions):
                # deal with planet companions
                if companion.kind == 'planet':
                    # get companion properties
                    ckwargs = dict()
                    ckwargs['params'] = params
                    ckwargs['image'] = image
                    ckwargs['hdict'] = hdict
                    ckwargs['num'] = it + 1
                    ckwargs['position_angle'] = companion.position_angle
                    ckwargs['separation'] = companion.separation
                    # get the constrast between observation and companion
                    contrast = observation.get_contrast(companion, _filter)
                    ckwargs['contrast'] = contrast
                    # add companion
                    image, hdict = etienne.ami_sim_add_companion(**ckwargs)
        # ---------------------------------------------------------------------
        # step 3: save image to disk
        # ---------------------------------------------------------------------
        etienne.ami_sim_save_scene(params, scenepath, image, hdict)
        # ---------------------------------------------------------------------
        # step 4: Deal with psf
        # ---------------------------------------------------------------------
        # get psf properties
        psfkwargs = dict()
        # get psf path for this filter
        psfkwargs['path'] = params['PSF_{0}_PATH'.format(_filter)]
        # get whether we want to recomputer psf
        psfkwargs['recompute'] = params['PSF_{0}_RECOMPUTE'.format(_filter)]
        # get other properties
        psfkwargs['fov_pixels'] = params['FOV_PIXELS']
        psfkwargs['oversample'] = params['OVERSAMPLE_FACTOR']
        psfkwargs['_filter'] = _filter
        # deal with psf
        psf_filename = etienne.ami_sim_get_psf(params, **psfkwargs)
        # update params for observation
        pkey = 'PSF_{0}_PATH'.format(_filter)
        observation.params[pkey] = psf_filename
        observation.params.set_source(pkey, func_name)
        # ---------------------------------------------------------------------
        # step 5: run ami-sim for observation
        # ---------------------------------------------------------------------
        # get parameters from observation
        target_name = observation.name
        nint = observation.num_integrations[_filter]
        ngroups = observation.num_groups[_filter]


        simfile = etienne.ami_sim_run_code(params, target_dir, _filter,
                                           psf_filename, scenepath, count_rate,
                                           simname, target_name, nint, ngroups)
        # update param for sim file
        okey = define_ami_sim_outkey(_filter)
        observation.params[okey] = simfile
        observation.params.set_source(okey, func_name)
        # ---------------------------------------------------------------------

def define_ami_sim_outkey(_filter: str):
    return 'AMI-SIM-OUT_{0}'.format(_filter)


def run_mirage(observation: Observation):
    pass


def define_mirage_sim_outkey(_filter: str):
    return 'MIRAGE_DMS_SIM-OUT_{0}'.format(_filter)


# =============================================================================
# Define DMS functions
# =============================================================================


# =============================================================================
# Define AMICAL extract functions
# =============================================================================
def amical_extraction(simulations: List[Simulation]):
    # loop around simulations
    for simulation in simulations:
        # simulate using AMISIM
        if simulation.params['AMISIM-USE']:
            # simulate using AMISIM
            if simulation.params['AMICAL-EXT-USE']:
                run_amical_extraction(simulation.name, simulation.target,
                                      simulation.calibrators, mode='amisim')
        # simulate using Mirage
        if simulation.params['MIRAGE-USE']:
            # simulate using AMISIM
            if simulation.params['AMICAL-EXT-USE']:
                run_amical_extraction(simulation.name, simulation.target,
                                      simulation.calibrators, mode='mirage')


def run_amical_extraction(simname: str, observation: Observation,
                          calibrators: List[Observation], mode: str):


    # push parameters into a dictionary for ami
    params_ami = dict()
    # instrument name
    params_ami['instrum'] = 'NIRISS'
    # Name of the mask
    params_ami['maskname'] = observation.params['AMICAL_EXT_MASK_NAME']
    # Use the multiple triangle technique to compute the bispectrum
    params_ami['bs_multi_tri'] = observation.params['AMICAL_EXT_BS_MULTI_TRI']
    # 3 methods are used to sample to u-v space:
    # - 'fft' uses fft between individual holes to compute the expected
    #         splodge position;
    # - 'square' compute the splodge in a square using the expected fraction
    #            of pixel to determine its weight;
    # - 'gauss' considers a gaussian splodge (with a gaussian weight) to get
    #           the same splodge side for each n(n-1)/2 baselines
    params_ami['peakmethod'] = observation.params['AMICAL_EXT_PEAK_METHOD']
    # NO DESC: Define the hole diameter
    params_ami['hole_diam'] = observation.params['AMICAL_EXT_HOLE_DIAMETER']
    # NO DESC: Define the cut off
    params_ami['cutoff'] = observation.params['AMICAL_EXT_CUTOFF']
    # Relative size of the splodge used to compute multiple triangle indices
    #    and the fwhm of the 'gauss' technique
    params_ami['fw_splodge'] = observation.params['AMICAL_EXT_FW_SPLODGE']
    # If True, the uncertainties are computed using the std of the overall
    #         cvis or bs array. Otherwise, the uncertainties are computed using
    #         covariance matrice
    params_ami['naive_err'] = observation.params['AMICAL_EXT_NATIVE_ERR']
    # Number of elements to sample the spectral filters (default: 3)
    params_ami['n_wl'] = observation.params['AMICAL_EXT_N_WL']
    # Number of separated blocks use to split the data cube and get more
    #         accurate uncertainties (default: 0, n_blocks = n_ps)
    params_ami['n_blocks'] = observation.params['AMICAL_EXT_N_BLOCKS']
    # Angle [deg] to rotate the mask compare to the detector (if the mask is not
    #         perfectly aligned with the detector, e.g.: VLT/VISIR)
    params_ami['theta_detctor'] = observation.params['AMICAL_EXT_THETA_DET']
    # Only used for IFU data (e.g.: IFS/SPHERE), select the desired spectral channel
    #         to retrieve the appropriate wavelength and mask positions
    params_ami['scaling_uv'] = observation.params['AMICAL_EXT_SCALING_UV']
    # NO DESC: Define i_wl
    params_ami['i_wl'] = observation.params['AMICAL_EXT_I_WL']
    # If True, the squared visibilities are unbiased using the Fourier base
    params_ami['unbias_v2'] = observation.params['AMICAL_EXT_UNBIAS_V2']
    # NO DESC: Define whether to compute CP cov
    params_ami['compute_cp_cov'] = observation.params['AMICAL_EXT_COMP_CP_COV']
    # NO DESC: Define whether to do the expert plot
    params_ami['expert_plot'] = observation.params['AMICAL_EXT_EXPERT_PLOT']
    # If True, print useful information during the process
    params_ami['verbose'] = observation.params['']
    # If True, display all figures
    params_ami['display'] = observation.params['AMICAL_EXT_DISPLAY_PLOT']
    # ---------------------------------------------------------------------
    # loop around filters
    # ---------------------------------------------------------------------
    # loop around all filters to use
    for _filter in observation.filters:
        # update filter in params for ami
        params_ami['filtname'] = _filter
        # get input key
        if mode == 'amisim':
            key = define_ami_sim_outkey(_filter)
        else:
            key = define_mirage_sim_outkey(_filter)
        # ---------------------------------------------------------------------
        # get input data
        # ---------------------------------------------------------------------
        # get target input filename
        target_inname = str(observation.params[key])
        # ---------------------------------------------------------------------
        # get calibrators input filename
        calibator_innames = []
        # loop around calibrators
        for calibrator in calibrators:
            calibator_innames.append(str(calibrator.params[key]))
        # ---------------------------------------------------------------------
        # construct output filename
        # ---------------------------------------------------------------------
        # get the target output filename
        target_outname = define_amical_ext_outname(simname, observation.name,
                                                   mode, _filter)
        # get the calibrators output filenames
        calibator_outnames = []
        for calibrator in calibrators:
            # construct the outname
            calibrator_outname = define_amical_ext_outname(simname,
                                                           calibrator.name,
                                                           mode, _filter)
            # add to calibrators list
            calibator_outnames.append(calibrator_outname)
        # ---------------------------------------------------------------------
        # read cubes
        # ---------------------------------------------------------------------
        target_cube = drs_file.read_fits(observation.params, target_inname,
                                         get_header=False)
        calibrator_cubes = []
        # loop around calibrators
        for c_it, calibrator in enumerate(calibrators):
            calibrator_cube = drs_file.read_fits(calibrator.params,
                                                 calibator_innames[c_it],
                                                 get_header=False)
            # append to list
            calibrator_cubes.append(calibrator_cube)
        # ---------------------------------------------------------------------
        # Extract raw complex observables for the target and the calibrator:
        # It's the core of the pipeline (amical/mf_pipeline/bispect.py)
        # ---------------------------------------------------------------------
        # extract the target
        target_bs = amical.extract_bs(target_cube, target_inname,
                                      targetname=target_outname,
                                      **params_ami)

        # extract the calibrators
        calibrators_bs = []
        for c_it, calibrator in enumerate(calibrators):
            # extract the calibrator
            calib_bs = amical.extract_bs(calibrator_cubes[c_it],
                                         calibator_innames[c_it],
                                         targetname=calibator_outnames[c_it],
                                         **params_ami)
            # append calibrator to list
            calibrators_bs.append(calib_bs)
        # ---------------------------------------------------------------------
        # Calibrate the raw data to get calibrated V2 and CP.
        # bs_c can be a single calibrator result or a list of calibrators.
        # (see amical/calibration.py for details).
        # ---------------------------------------------------------------------
        cal_target = amical.calibrate(target_bs, calibrators_bs)
        # ---------------------------------------------------------------------
        # Deal with plotting
        # ---------------------------------------------------------------------
        # display the plots
        amical.show(cal_target)
        if observation.params['AMICAL_EXT_DISPLAY_PLOT']:
            plt.show(block=True)
            plt.close()
        # ---------------------------------------------------------------------
        # Write calibrated output
        # ---------------------------------------------------------------------
        # create calibrated output filename
        cal_target_outfilename = define_amical_ext_calname(simname,
                                                           observation.name,
                                                           mode, _filter)
        # save the results as oifits
        _ = amical.save(cal_target, oifits_file=cal_target_outfilename,
                        fake_obj=observation.params['AMICAL_EXT_FAKE_OBJ'])
        # push outfile into observation params
        outkey = define_amical_ext_outkey(_filter)
        observation.params[outkey] = cal_target_outfilename


def define_amical_ext_outname(simname: str, targetname: str, mode: str,
                              _filter: str):
    pargs = [simname, targetname, mode, _filter]
    return '{0}_{1}_{2}_NIRISS_{3}'.format(*pargs)


def define_amical_ext_calname(simname: str, targetname: str, mode: str,
                              _filter: str):

    pargs = [simname, targetname, mode, _filter]
    return '{0}_{1}_{2}_NIRISS_{3}.oifits'.format(*pargs)


def define_amical_ext_outkey(_filter: str):
    return 'AMICAL-EXT-OUT_{0}'.format(_filter)


# =============================================================================
# Define AMICAL analysis functions
# =============================================================================


# =============================================================================
# Define worker functions
# =============================================================================
def _update_params(params: ParamDict, properties: Dict[str, Any],
                   config_file: str) -> ParamDict:
    """
    Update params with info from the yaml file

    :param params: ParamDict, the parameter dictionary of constants
    :param properties: dict, the yaml file
    :param config_file: str, the path to the config file (for source)

    :return:
    """
    # get parameters from properties
    parameters = properties.get('parameters', None)
    # deal with no parameters
    if parameters is None:
        return params.copy()
    # get instances from params
    instances = params.instances
    # store keys that have paths
    keys = dict()
    # look for objects with paths
    for key in instances:
        # get instance for this key
        instance = instances[key]
        # make sure instance is a constant class
        if not isinstance(instance, constant_functions.Constant):
            continue
        # find constants that have yaml paths
        if instance.path is not None:
            # store keys to update
            keys[key] = instance.path
    # update params
    for key in keys:
        # get subkeys as a list
        tkeys = keys[key].split('.')
        value = None
        # loop around sub keys
        for rkey in tkeys:
            # value starts at None then changes to the sub dictionary
            if value is None:
                # must have rkey in properties
                if rkey in parameters:
                    value = parameters[rkey]
            # only modify value if value is a dictionary
            if isinstance(value, dict):
                if rkey in value:
                    value = value[rkey]
        if isinstance(value, dict):
            value = None
        # only update value if not None
        if value is not None:
            # set param and source
            params[key] = value
            params.sources[key] = '{0}.{1}'.format(config_file, keys[key])
    # finally return params
    return params.copy()


def _load_xml(params: ParamDict,
              filename: Union[str, None] = None) -> ParamDict:
    """
    Load an xml into "APT-TARGETS" in the parameter dictionary

    :param params: ParamDict, the parameter dictionary of constants
    :param filename: str, the filename of the xml file to open

    :return:
    """
    # set function name
    func_name = display_func('_load_xml', __NAME__)
    # -------------------------------------------------------------------------
    # deal with filename
    if filename is None:
        # log error
        msg = 'XML-Error: Filename must be defined'
        params.log.error(msg)
    else:
        filename = Path(filename)
    # -------------------------------------------------------------------------
    # deal with file not existing
    if not filename.exists():
        # log error
        msg = 'XML-Error: Filename {0} does not exist'
        params.log.error(msg)
    # -------------------------------------------------------------------------
    # get target name column
    target_name_col = params.instances['APT-TARGET-NAME'].apt
    # log reading xml
    params.log.info('Reading XML: {0}'.format(filename))
    # load xml file as dictionary of keys
    xml = XMLReader()
    table = xml.read_xml_silent(filename)
    # get target names (cleaned)
    target_names = list(map(general.clean_name, table[target_name_col]))
    # storage of xml targets
    xml_targets = dict()
    # -------------------------------------------------------------------------
    # loop around all targets
    for it in range(len(target_names)):
        # get name for this iteration
        name = target_names[it]
        # each entry should be a dictionary
        if name not in xml_targets:
            xml_targets[name] = dict()
        # loop around entrys
        for entry in list(table.keys()):
            # deal with appending a target entry
            if entry in xml_targets[name]:
                xml_targets[name][entry].append(table[entry][it])
            # deal with
            else:
                xml_targets[name][entry] = [table[entry][it]]
        # add xml file path to parameters
        xml_targets[name]['XML-FILE'] = filename
    # -------------------------------------------------------------------------
    # finally add to params
    params['APT-TARGETS'] = xml_targets
    params.set_source('APT-TARGETS', func_name)
    # -------------------------------------------------------------------------
    # return params
    return params


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
