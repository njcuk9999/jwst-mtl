#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on

@author:
"""
import numpy as np
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from mirage.apt import read_apt_xml
import yaml

# try to import other modules
try:
    import amical
except ImportError:
    amical = None

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
        self.xmlpath = properties.get('apt_xmlpath', None)
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
                # load calibrator
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

        # ---------------------------------------------------------------------
        # save raw parameters from xml
        raw_xml_data = dict()
        # get number of observations
        n_robs = len(xml_target[self.params.instances['APT-TARGET-NAME'].apt])

        valids = np.ones(n_robs, dtype=bool)
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
                values, valid = self._deal_with_subarray(values, instance.apt,
                                                         func_name)
            # check filters are valid
            elif key == 'APT-TARGET-FILTERS':
                values, valid = self._deal_with_filters(values, instance.apt,
                                                        func_name)
            elif key in ['APT-TARGET-NGROUP', 'APT-TARGET-NINT']:
                values, valid = self._deal_with_ints(values, key, instance.apt,
                                                     func_name)
            else:
                valid = [True] * len(values)
            # -----------------------------------------------------------------
            # update the valid key
            valids &= np.array(valid)
            # add to raw xml data
            raw_xml_data[key] = values
        # ---------------------------------------------------------------------
        # deal with no valid data in xml
        if np.sum(valids) == 0:
            emsg = 'ObservationError: No valid XML values'
            self.params.log.error(emsg, exceptions=ObservationException,
                                  funcname=func_name)
        # ---------------------------------------------------------------------
        # need to deal with validation (removing observations with bad values)
        for key in raw_xml_data:
            # get instance
            instance = self.params.instances[key]
            # storage of valid values
            values = []
            # loop around raw number of observation entries
            for item in range(n_robs):
                # only add those which are valid
                if valids[item]:
                    values.append(raw_xml_data[key][item])
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

    def _deal_with_subarray(self, in_values: List[str], key: str,
                            func_name: str) -> Tuple[List[str], List[bool]]:
        """
        Deal specifically with sub array value from APT

        :param in_values: list of str, the values for sub array from APT
        :param func_name: str, the function name
        :return:
        """
        # storage of outputs
        out_values = []
        out_valid = []
        # assume source is APT originally
        subarray_source = 'APT'
        # loop around input values
        for in_value in in_values:
            # check for None value
            if in_value in ['None', None, '']:
                out_value = self.params['DEFAULT_SUBARRAY']
                subarray_source = self.params.sources['DEFAULT_SUBARRAY']
            else:
                out_value = str(in_value)
            # check that value is now valid (in SUBARRAYS)
            if out_value not in self.params['SUBARRAYS']:
                out_valid.append(False)
            else:
                out_valid.append(True)
            # add to outputs
            out_values.append(out_value)
        # deal with no sub-arrays
        if len(out_values) == 0:
            emsg = 'ObservationError: No valid XML Subarray values'
            emsg += '\n\t Values = "{0}" (source={1}.{2})'
            emsg = emsg.format(''.join(in_values), subarray_source, key)
            self.params.log.error(emsg, exceptions=ObservationException,
                                  funcname=func_name)
        # return value
        return out_values, out_valid

    def _deal_with_filters(self, in_values: List[str], key: str,
                           func_name: str) -> Tuple[List[str], List[bool]]:
        """
        Deal specifically with sub array value from APT

        :param in_values: list of str, the values for sub array from APT
        :param func_name: str, the function name
        :return:
        """
        # storage of outputs
        out_values = []
        out_valid = []
        # loop around input values
        for in_value in in_values:
            # check that value is now valid (in SUBARRAYS)
            if in_value not in self.params['ALL_FILTERS']:
                out_valid.append(False)
            else:
                out_valid.append(True)
            # add to outputs
            out_values.append(in_value)
        # deal with no sub-arrays
        if len(out_values) == 0:
            emsg = 'ObservationError: No valid XML Filters values'
            emsg += '\n\t Values = "{0}" (source=APT.{1})'
            emsg = emsg.format(''.join(in_values), key)
            self.params.log.error(emsg, exceptions=ObservationException,
                                  funcname=func_name)
        # return value
        return out_values, out_valid

    def _deal_with_ints(self, in_values: List[str], name: str, key: str,
                        func_name: str) -> Tuple[List[int], List[bool]]:
        # storage of outputs
        out_values = []
        out_valid = []
        # loop around input values
        for in_value in in_values:

            try:
                out_values.append(int(in_value))
                out_valid.append(True)
            except Exception as _:
                out_values.append(-1)
                out_valid.append(False)
        # deal with no sub-arrays
        if len(out_values) == 0:
            emsg = 'ObservationError: No valid XML {0} values'
            emsg += '\n\t Values = "{1}" (source=APT.{2})'
            emsg = emsg.format(name, ''.join(in_values), key)
            self.params.log.error(emsg, exceptions=ObservationException,
                                  funcname=func_name)
        # return value
        return out_values, out_valid


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
        ngroups = list(self.params['APT-TARGET-NGROUP'])
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
            self.params.log.error(emsg.format(*eargs))
        # get target magnitude
        target_mag = self.magnitudes[_filter]
        # make sure we have filter in companion
        if _filter not in companion.magnitudes:
            emsg = 'ObservationError: Cannot get contrast for target {0}'
            emsg += '\n\tCompanion {1} does not have magnitude "{2}"'
            eargs = [self.name, companion.name, _filter]
            self.params.log.error(emsg.format(*eargs))
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
        # get active
        self.active = properties.get('active', True)
        # get magnitudes
        self.magnitudes = dict()
        self.get_magnitudes()
        # set companion type storage
        self.planet = None
        self.disk = None
        self.bar = None
        # fill companion type
        if self.kind == 'planet':
            self.planet = CompanionType('planet', properties)
        elif self.kind == 'disk':
            self.disk = CompanionType('disk', properties)
        elif self.kind == 'bar':
            self.bar = CompanionType('bar', properties)


class CompanionType:
    def __init__(self, kind, properties):
        # default values for all parameters (used or not)
        self.kind = kind
        self.separation = None
        self.position_angle = None
        self.plot = False
        self.roll = None
        self.inclination = None
        self.width = None
        self.radius = None
        self.exponent = None
        # fill companion type
        if kind == 'planet':
            self.planet(properties)
        elif kind == 'disk':
            self.disk(properties)
        elif kind == 'bar':
            self.bar(properties)

    def __str__(self) -> str:
        """
        Return string representation of the CompanionType class

        :return: str, the string representation
        """
        if self.kind == 'planet':
            msg = 'CompanionType[planet]'
            msg += '\n\tseparation: {0}'.format(self.separation)
            msg += '\n\tpa: {0}'.format(self.position_angle)
            msg += '\n\tplot: {0}'.format(self.plot)
        elif self.kind == 'disk':
            msg = 'CompanionType[disk]'
            msg += '\n\troll: {0}'.format(self.roll)
            msg += '\n\tinclination: {0}'.format(self.inclination)
            msg += '\n\twidth: {0}'.format(self.width)
            msg += '\n\tradius: {0}'.format(self.radius)
            msg += '\n\texponent: {0}'.format(self.exponent)
            msg += '\n\tplot: {0}'.format(self.plot)
        elif self.kind == 'bar':
            msg = 'CompanionType[bar]'
            msg += '\n\troll: {0}'.format(self.roll)
            msg += '\n\twidth: {0}'.format(self.width)
            msg += '\n\tradius: {0}'.format(self.radius)
            msg += '\n\texponent: {0}'.format(self.exponent)
            msg += '\n\tplot: {0}'.format(self.plot)
        else:
            msg = ''
        # return message
        return msg

    def __repr__(self) -> str:
        return self.__str__()

    def planet(self, properties):
        """
        Get and check planet settings

        :param properties: ParamDict, the parameter dictionary of properties

        :return: None
        """
        # get planet
        planetprops = properties.get('planet', None)
        # deal with no planet
        if planetprops is None:
            emsg = 'For companion kind "planet" must define planet properties'
            emsg += '\n\tplanet:'
            emsg += '\n\t\tseparation: 0.1 (required)'
            emsg += '\n\t\tpa: 0.0 (optional)'
            exceptions.DrsException(emsg, kind='error')
            return
        # get separation in arc seconds (used for kind="planet")
        self.separation = planetprops.get('separation', None)
        # deal with no separation
        if self.separation is None:
            emsg = 'For companion kind "planet" must define separation'
            exceptions.DrsException(emsg, kind='error')
        # get position angle in degrees (used for kind="planet")
        self.position_angle = planetprops.get('pa', 0.0)
        # get plot
        self.plot = planetprops.get('plot', False)

    def disk(self, properties):
        """
        Get and check disk settings

        :param properties: ParamDict, the parameter dictionary of properties

        :return: None
        """
        # get disk
        diskprops = properties.get('disk', None)
        # deal with no planet
        if diskprops is None:
            emsg = 'For companion kind "disk" must define disk properties'
            emsg += '\n\tdisk:'
            emsg += '\n\t\troll: 0.0 (optional)'
            emsg += '\n\t\tinclination: 0.0 (optional)'
            emsg += '\n\t\twidth: 0.1 (required)'
            emsg += '\n\t\tradius: 0.1 (required)'
            emsg += '\n\t\texponent: 2.0 (optional)'
            exceptions.DrsException(emsg, kind='error')
            return
        # in degrees -> rotation on the sky plane
        self.roll = diskprops.get('roll', 0.0)
        # in degrees -> tilt toward the line of sight
        self.inclination = diskprops.get('inclination', 0.0)
        # e-width of annulus in arcsec for disk
        self.width = diskprops.get('width', None)
        # deal with no width
        if self.width is None:
            emsg = 'For companion kind "disk" must define width'
            exceptions.DrsException(emsg, kind='error')
        # long axis radius in arcsec for disk
        self.radius = diskprops.get('radius', None)
        # deal with no radius
        if self.radius is None:
            emsg = 'For companion kind "disk" must define radius'
            exceptions.DrsException(emsg, kind='error')
        # gaussian exponent
        self.exponent = diskprops.get('exponent', 2.0)
        # get plot
        self.plot = diskprops.get('plot', False)

    def bar(self, properties):
        """
        Get and check bar settings

        :param properties: ParamDict, the parameter dictionary of properties

        :return: None
        """
        # get bar
        barprops = properties.get('bar', None)
        # deal with no planet
        if barprops is None:
            emsg = 'For companion kind "bar" must define bar properties'
            emsg += '\n\tbar:'
            emsg += '\n\t\troll: 0.0 (optional)'
            emsg += '\n\t\twidth: 0.1 (required)'
            emsg += '\n\t\tradius: 0.1 (required)'
            emsg += '\n\t\texponent: 2.0 (optional)'
            exceptions.DrsException(emsg, kind='error')
            return
        # in degrees -> rotation on the sky plane
        self.roll = barprops.get('roll', 0.0)
        # thickness of bar
        self.width = barprops.get('width', None)
        # deal with no width
        if self.width is None:
            emsg = 'For companion kind "bar" must define width'
            exceptions.DrsException(emsg, kind='error')
        # long axis of the bar
        self.radius = barprops.get('radius', None)
        # deal with no radius
        if self.radius is None:
            emsg = 'For companion kind "bar" must define radius'
            exceptions.DrsException(emsg, kind='error')
        # gaussian exponent
        self.exponent = barprops.get('exponent', 2.0)
        # get plot
        self.plot = barprops.get('plot', False)


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
    # deal with no config file
    if config_file in [None, 'None', '', 'Null']:
        params.log.error('Config file cannot be None')
        return
    elif not os.path.exists(str(config_file)):
        params.log.error('Config file {0} does not exists'.format(config_file))
        return

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
        # simulate target
        run_ami_sim(simulation.name, simulation.target)
        # simulate calibrators
        for calibrator in simulation.calibrators:
            # simulate calibrator
            run_ami_sim(simulation.name, calibrator)
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
    # get specific properties from observation
    use_amisim = observation.params['AMISIM-USE']
    create_scene = observation.params['AMISIM-CREATE_SCENE']
    create_sim = observation.params['AMISIM-CREATE_SIM']
    # Process update
    msg = 'Processing Simulation: {0} Observation: {1}'
    margs = [simname, observation.name]
    params.log.info(msg.format(*margs))
    # -----------------------------------------------------------------
    # Big loop around filters
    # -----------------------------------------------------------------
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
        # -----------------------------------------------------------------
        # step 1: make primary on image
        # -----------------------------------------------------------------
        if use_amisim and create_scene:
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
        else:
            image, hdict = None, None
            count_rate = 0
        # -------------------------------------------------------------
        # step 2: add companion(s)
        # -------------------------------------------------------------
        if use_amisim and create_scene:
            # only do this for targets (calibrators do not have companions
            #     by definition)
            if isinstance(observation, Target):
                # loop around all companions
                for it, companion in enumerate(observation.companions):
                    # deal with active skip
                    if not companion.active:
                        continue
                    # deal with planet companions
                    if companion.kind == 'planet':
                        # deal with no planet added
                        if companion.planet is None:
                            continue
                        else:
                            planet = companion.planet
                        # get companion properties
                        ckwargs = dict()
                        ckwargs['params'] = params
                        ckwargs['image'] = image
                        ckwargs['hdict'] = hdict
                        ckwargs['num'] = it + 1
                        ckwargs['position_angle'] = planet.position_angle
                        ckwargs['separation'] = planet.separation
                        ckwargs['plot'] = planet.plot
                        # get the contrast between observation and
                        #    companion
                        contrast = observation.get_contrast(companion,
                                                            _filter)
                        ckwargs['contrast'] = contrast
                        # add companion
                        cout = etienne.ami_sim_add_companion(**ckwargs)
                        image, hdict = cout
                    elif companion.kind == 'disk':
                        # deal with no disk added
                        if companion.disk is None:
                            continue
                        else:
                            disk = companion.disk
                        # get companion properties
                        ckwargs = dict()
                        ckwargs['image'] = image
                        ckwargs['hdict'] = hdict
                        ckwargs['num'] = it + 1
                        ckwargs['kind'] = 'disk'
                        ckwargs['roll'] = disk.roll
                        ckwargs['inclination'] = disk.inclination
                        ckwargs['width'] = disk.width
                        ckwargs['radius'] = disk.radius
                        ckwargs['exponent'] = disk.exponent
                        ckwargs['plot'] = disk.plot
                        # get the contrast between observation and
                        #    companion
                        contrast = observation.get_contrast(companion,
                                                            _filter)
                        ckwargs['contrast'] = contrast
                        # add the disk to the image
                        cout = etienne.ami_sim_add_disk(**ckwargs)
                        image, hdict = cout
                    if companion.kind == 'bar':
                        # deal with no bar added
                        if companion.bar is None:
                            continue
                        else:
                            bar = companion.bar
                        # get companion properties
                        ckwargs = dict()
                        ckwargs['image'] = image
                        ckwargs['hdict'] = hdict
                        ckwargs['num'] = it + 1
                        ckwargs['kind'] = 'bar'
                        ckwargs['roll'] = bar.roll
                        ckwargs['width'] = bar.width
                        ckwargs['radius'] = bar.radius
                        ckwargs['exponent'] = bar.exponent
                        ckwargs['plot'] = bar.plot
                        # get the contrast between observation and
                        #    companion
                        contrast = observation.get_contrast(companion,
                                                            _filter)
                        ckwargs['contrast'] = contrast
                        # add the disk to the image
                        cout = etienne.ami_sim_add_disk(**ckwargs)
                        image, hdict = cout
        else:
            image, hdict = None, None
        # -------------------------------------------------------------
        # step 3: save image to disk
        # -------------------------------------------------------------
        if use_amisim and create_scene:
            etienne.ami_sim_save_scene(params, scenepath, image, hdict)

        # -----------------------------------------------------------------
        # step 4: Deal with psf
        #     - need to generate filename for ami sim output
        # -----------------------------------------------------------------
        # get psf properties
        psfkwargs = dict()
        # get psf path for this filter
        psfkwargs['path'] = params['PSF_{0}_PATH'.format(_filter)]
        # get whether we want to recomputer psf
        if use_amisim:
            rkey = 'PSF_{0}_RECOMPUTE'.format(_filter)
            psfkwargs['recompute'] = params[rkey]
        else:
            psfkwargs['recompute'] = False
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
        # -----------------------------------------------------------------
        # step 5: run ami-sim for observation
        # -----------------------------------------------------------------
        # get parameters from observation
        target_name = observation.name
        nint = observation.num_integrations[_filter]
        ngroups = observation.num_groups[_filter]
        # run ami-sim
        if use_amisim and create_sim:
            # TODO: Warn if ami_sim pixscl is different from the pkwargs one
            tag = etienne.ami_sim_run_code(params, target_dir, _filter,
                                           psf_filename, scenepath, count_rate,
                                           simname, target_name, nint, ngroups)
        # else just ge tthe tag for output filename
        else:
            # get parameters from observation
            target_name = observation.name
            # construct tag
            tag = '{0}_{1}'.format(simname, target_name)

        # -----------------------------------------------------------------
        # Construct names of ami-sim outputs
        # -----------------------------------------------------------------
        # get psf basename
        psf_basename = os.path.basename(psf_filename).split('.fits')[0]
        scene_basename = os.path.basename(scenepath).split('.fits')[0]
        # construct ami-sim out filename
        oargs = [scene_basename, psf_basename, tag]
        simfile = 't_{0}__{1}_{2}_00.fits'.format(*oargs)
        simpath = os.path.join(target_dir, _filter)
        # ---------------------------------------------------------------------
        # step 6: define ami sim output (may be required even if AMI-SIM is not
        #         run
        # ---------------------------------------------------------------------
        # update param for sim file
        okey = define_ami_sim_outkey(_filter)

        if params[okey] in [None, 'None', 'Null']:
            params[okey] = os.path.join(simpath, simfile)
            params.set_source(okey, func_name)
        else:
            margs = [okey, params[okey]]
            params.log.info('Setting {0} from yaml. File = {1}'.format(*margs))
        # ---------------------------------------------------------------------


def define_ami_sim_outkey(_filter: str):
    """
    Define the AMI SIM output key in the parameter dictionary

    :param _filter: str, the filter used
    :return:
    """
    return 'AMI-SIM-OUT_{0}'.format(_filter)


def run_mirage(observation: Observation):
    """
    Run the Mirage module on a specific observation

    :param simname: the name of this simulation
    :param observation: the observation to run through Mirage
    :return:
    """
    pass


def define_mirage_sim_outkey(_filter: str):
    """
    Define the Miarge output key in the parameter dictionary

    :param _filter: str, the filter used
    :return:
    """
    return 'MIRAGE_DMS_SIM-OUT_{0}'.format(_filter)


# =============================================================================
# Define DMS functions
# =============================================================================


# =============================================================================
# Define AMICAL extract functions
# =============================================================================
def amical_extraction(simulations: List[Simulation]):
    """
    Loop around simulations and dispatch to AMICAL extraction

    :param simulations: list of simulation instances
    :return:
    """
    # loop around simulations
    for simulation in simulations:
        # simulate using AMISIM
        cond1 = simulation.params['AMICAL-EXT-USE']
        cond2 = simulation.params['AMICAL-INPUT-AMISIM']
        cond3 = simulation.params['AMICAL-INPUT-MIRAGE']
        if cond1 and cond2:
            run_amical_extraction(simulation.name, simulation.target,
                                  simulation.calibrators, mode='amisim')
        # simulate using AMISIM
        if cond1 and cond3:
            run_amical_extraction(simulation.name, simulation.target,
                                  simulation.calibrators, mode='mirage')


def run_amical_extraction(simname: str, observation: Observation,
                          calibrators: List[Observation], mode: str):
    """
    Run the AMI SIM module on a specific observation

    :param simname: the name of this simulation
    :param observation: the observation to run through AMI-SIM
    :param calibrators: the calibrator observations
    :param mode: str, either "amisim" or "mirage"

    :return:
    """
    # get params from observation
    params = observation.params
    # push parameters into a dictionary for ami
    params_ami = dict()
    # instrument name
    params_ami['instrum'] = 'NIRISS'
    # Name of the mask
    params_ami['maskname'] = params['AMICAL_EXT_MASK_NAME']
    # Use the multiple triangle technique to compute the bispectrum
    params_ami['bs_multi_tri'] = params['AMICAL_EXT_BS_MULTI_TRI']
    # 3 methods are used to sample to u-v space:
    # - 'fft' uses fft between individual holes to compute the expected
    #         splodge position;
    # - 'square' compute the splodge in a square using the expected fraction
    #            of pixel to determine its weight;
    # - 'gauss' considers a gaussian splodge (with a gaussian weight) to get
    #           the same splodge side for each n(n-1)/2 baselines
    params_ami['peakmethod'] = params['AMICAL_EXT_PEAK_METHOD']
    # NO DESC: Define the hole diameter
    params_ami['hole_diam'] = params['AMICAL_EXT_HOLE_DIAMETER']
    # NO DESC: Define the cut off
    params_ami['cutoff'] = params['AMICAL_EXT_CUTOFF']
    # Relative size of the splodge used to compute multiple triangle indices
    #    and the fwhm of the 'gauss' technique
    params_ami['fw_splodge'] = params['AMICAL_EXT_FW_SPLODGE']
    # If True, the uncertainties are computed using the std of the overall
    #         cvis or bs array. Otherwise, the uncertainties are computed using
    #         covariance matrice
    params_ami['naive_err'] = params['AMICAL_EXT_NATIVE_ERR']
    # Number of elements to sample the spectral filters (default: 3)
    params_ami['n_wl'] = params['AMICAL_EXT_N_WL']
    # Number of separated blocks use to split the data cube and get more
    #         accurate uncertainties (default: 0, n_blocks = n_ps)
    params_ami['n_blocks'] = params['AMICAL_EXT_N_BLOCKS']
    # Angle [deg] to rotate the mask compare to the detector (if the mask is not
    #         perfectly aligned with the detector, e.g.: VLT/VISIR)
    params_ami['theta_detector'] = params['AMICAL_EXT_THETA_DET']
    # Only used for IFU data (e.g.: IFS/SPHERE), select the desired spectral
    #     channel to retrieve the appropriate wavelength and mask positions
    params_ami['scaling_uv'] = params['AMICAL_EXT_SCALING_UV']
    # NO DESC: Define i_wl
    params_ami['i_wl'] = params['AMICAL_EXT_I_WL']
    # If True, the squared visibilities are unbiased using the Fourier base
    params_ami['unbias_v2'] = params['AMICAL_EXT_UNBIAS_V2']
    # NO DESC: Define whether to compute CP cov
    params_ami['compute_cp_cov'] = params['AMICAL_EXT_COMP_CP_COV']
    # NO DESC: Define whether to do the expert plot
    params_ami['expert_plot'] = params['AMICAL_EXT_EXPERT_PLOT']
    # If True, print useful information during the process
    params_ami['verbose'] = params['AMICAL_EXT_VERBOSE']
    # If True, display all figures
    params_ami['display'] = params['AMICAL_EXT_DISPLAY_PLOT']
    # ---------------------------------------------------------------------
    # loop around filters
    # ---------------------------------------------------------------------
    # loop around all filters to use
    for _filter in observation.filters:
        # log progress
        params.log.info('AMI-CAL EXTRACTION: Filter = {0}'.format(_filter))
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
        target_inname = str(params[key])
        # ---------------------------------------------------------------------
        # get calibrators input filename
        calibrator_innames = []
        # loop around calibrators
        for calibrator in calibrators:
            # get inname
            cal_inname = calibrator.params[key]
            # mmay be that we do not have a calibrator for this filter
            if cal_inname not in ['', 'Null', 'None', None]:
                calibrator_innames.append(str(calibrator.params[key]))

        # deal with no calibrators
        if len(calibrator_innames) == 0:
            wmsg = 'No calibrators defined for filter = {0}. Skipping'
            wargs = [_filter]
            params.log.warning(wmsg.format(*wargs))
            # skip filter
            continue
        # ---------------------------------------------------------------------
        # construct output filename
        # ---------------------------------------------------------------------
        # get the target output filename
        target_outname = define_amical_ext_outname(simname, observation.name,
                                                   mode, _filter)
        # get the calibrators output filenames
        cal_outnames = []
        for calibrator in calibrators:
            # construct the outname
            calibrator_outname = define_amical_ext_outname(simname,
                                                           calibrator.name,
                                                           mode, _filter)
            # add to calibrators list
            cal_outnames.append(calibrator_outname)
        # ---------------------------------------------------------------------
        # read cubes
        # ---------------------------------------------------------------------
        # log progress
        marg = [target_inname]
        params.log.info('AMI-CAL EXTRACTION: Reading Target {0}'.format(*marg))
        # read target
        target_cube = drs_file.read_fits(params, target_inname,
                                         get_header=False)
        calibrator_cubes = []
        # loop around calibrators
        for c_it, calibrator in enumerate(calibrators):
            # log progress
            marg = [calibrator_innames[c_it]]
            params.log.info('AMI-CAL EXTRACTION: Reading Cal {0}'.format(*marg))
            # raed calibrator
            calibrator_cube = drs_file.read_fits(calibrator.params,
                                                 calibrator_innames[c_it],
                                                 get_header=False)
            # append to list
            calibrator_cubes.append(calibrator_cube)
        # ---------------------------------------------------------------------
        # Extract raw complex observables for the target and the calibrator:
        # It's the core of the pipeline (amical/mf_pipeline/bispect.py)
        # ---------------------------------------------------------------------
        # log progress
        marg = [target_inname]
        params.log.info('AMI-CAL EXTRACTION: Extracting {0}'.format(*marg))
        # extract the target
        with general.ModifyPrintouts(text='AMI-CAL Output', flush=True,
                                     logfile=params['LOGFILE']):
            target_bs = amical.extract_bs(target_cube, target_inname,
                                          targetname=target_outname,
                                          **params_ami)
        # ---------------------------------------------------------------------
        # extract the calibrators
        calibrators_bs = []
        for c_it, calibrator in enumerate(calibrators):
            # log progress
            marg = [calibrator_innames[c_it]]
            params.log.info('AMI-CAL EXTRACTION: Extracting {0}'.format(*marg))
            # extract calibrator - catching printouts
            with general.ModifyPrintouts(text='AMI-CAL Output', flush=True,
                                         logfile=observation.params['LOGFILE']):
                # extract the calibrator
                calib_bs = amical.extract_bs(calibrator_cubes[c_it],
                                             calibrator_innames[c_it],
                                             targetname=cal_outnames[c_it],
                                             **params_ami)
            # append calibrator to list
            calibrators_bs.append(calib_bs)
        # ---------------------------------------------------------------------
        # Calibrate the raw data to get calibrated V2 and CP.
        # bs_c can be a single calibrator result or a list of calibrators.
        # (see amical/calibration.py for details).
        # ---------------------------------------------------------------------
        # log progress
        marg = [target_inname]
        params.log.info('AMI-CAL EXTRACTION: Calibrating {0}'.format(*marg))
        # calibrate target
        with general.ModifyPrintouts(text='AMI-CAL Output', flush=True,
                                     logfile=observation.params['LOGFILE']):
            cal_target = amical.calibrate(target_bs, calibrators_bs)
        # ---------------------------------------------------------------------
        # Deal with plotting
        # ---------------------------------------------------------------------
        # display the plots
        if observation.params['AMICAL_EXT_DISPLAY_PLOT']:
            with general.ModifyPrintouts(text='AMI-CAL Output', flush=True,
                                         logfile=observation.params['LOGFILE']):
                amical.show(cal_target)
                plt.show(block=True)
                plt.close()
        # ---------------------------------------------------------------------
        # Write calibrated output
        # ---------------------------------------------------------------------
        # create calibrated output filename
        cal_target_outfilename = define_amical_ext_calname(simname,
                                                           observation.name,
                                                           mode, _filter)
        cal_target_dir = define_amical_ext_dir(observation.params)

        cal_target_abspath = os.path.join(cal_target_dir,
                                          cal_target_outfilename)
        # save the results as oifits
        _ = amical.save(cal_target, oifits_file=cal_target_outfilename,
                        fake_obj=observation.params['AMICAL_EXT_FAKE_OBJ'],
                        datadir=cal_target_dir)
        # print progress
        margs = [cal_target_abspath]
        params.log.info('AMI-CAL EXTRACTION: Saving to: {0}'.format(*margs))

        # push outfile into observation params
        outkey = define_amical_ext_outkey(_filter, mode)
        observation.params[outkey] = cal_target_abspath


def define_amical_ext_outname(simname: str, targetname: str, mode: str,
                              _filter: str):
    """
    Define the amical extraction output name

    :param simname: str, simulation name
    :param targetname: str, the target name of the observation
    :param mode: str, either 'amisim' or 'mirage'
    :param _filter: str, the filter used
    :return:
    """
    pargs = [simname, targetname, mode, _filter]
    return '{0}_{1}_{2}_NIRISS_{3}'.format(*pargs)


def define_amical_ext_calname(simname: str, targetname: str, mode: str,
                              _filter: str):
    """
    Define the amical calibration extraction output name

    :param simname: str, simulation name
    :param targetname: str, the target name of the observation
    :param mode: str, either 'amisim' or 'mirage'
    :param _filter: str, the filter used
    :return:
    """
    pargs = [simname, targetname, mode, _filter]
    return '{0}_{1}_{2}_NIRISS_{3}.oifits'.format(*pargs)


def define_amical_ext_dir(params: ParamDict) -> str:
    """
    Define the AMICAL extraction directory

    :param params: ParamDict, the parameter dictionary of constants

    :return: str, the path to the AMICAL extraction directory
    """
    if params['AMICAL_EXT_PATH'] is None:
        path = os.path.join(params['DIRECTORY'], 'amical-ext')
    else:
        path = str(params['AMICAL_EXT_PATH'])
    # if directory doesn't exist create it
    if not os.path.exists(path):
        os.mkdir(path)
    # return the path
    return path


def define_amical_ana_dir(params: ParamDict) -> str:
    """
    Define the AMICAL analysis directory

    :param params: ParamDict, the parameter dictionary of constants

    :return: str, the path to the AMICAL extraction directory
    """
    if params['AMICAL_ANA_PATH'] is None:
        path = os.path.join(params['DIRECTORY'], 'amical-analysis')
    else:
        path = str(params['AMICAL_ANA_PATH'])
    # if directory doesn't exist create it
    if not os.path.exists(path):
        os.mkdir(path)
    # return the path
    return path


def define_amical_ext_outkey(_filter: str, mode: str) -> str:
    """
    Define the AMICAL extraction output key in the parameter dictionary

    :param _filter: str, the filter used
    :param mode: str, the sim mode [ami-sim, mirage]

    :return: str, the key for storing amical extraction output file
    """
    return 'AMICAL-EXT-OUT_{0}_{1}'.format(_filter, mode)


# =============================================================================
# Define AMICAL analysis functions
# =============================================================================
def amical_analysis(simulations: List[Simulation]):
    """
    Run the AMICAL analysis

    :param simulations: list of simulation instances

    :return: None
    """
    # loop around simulations
    for simulation in simulations:
        # simulate using AMISIM
        cond1 = simulation.params['AMICAL-EXT-USE']
        cond2 = simulation.params['AMICAL-INPUT-AMISIM']
        cond3 = simulation.params['AMICAL-INPUT-MIRAGE']
        if cond1 and cond2:
            run_ami_analysis(simulation.target, mode='amisim')
        # simulate using AMISIM
        if cond1 and cond3:
            run_ami_analysis(simulation.target, mode='mirage')


def run_ami_analysis(observation: Observation, mode: str):
    """
    Run AMICAL analysis

    based on example_analysis.py (AMICAL)

    @author: Anthony Soulain (University of Sydney)

    -------------------------------------------------------------------------
    AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
    -------------------------------------------------------------------------

    The idea is to provide the users with all the tools to analyze and
    interpret their AMI data in the best possible way. We included with AMICAL
    two additional (and independant) packages to perform this purposes.

    CANDID developed by A. Merand & A. Gallenne
           (https://github.com/amerand/CANDID)

    and

    Pymask developed by B. Pope & A. Cheetham
           (https://github.com/AnthonyCheetham/pymask).

    With AMICAL, we provide some easy interface between these codes and the
    outputs of our extraction pipeline. We give below some example to analyze
    and extract the quantitative values of our simulated binary.

    --------------------------------------------------------------------
    """
    # get params from observation
    params = observation.params
    # whether to use candid and pymask
    use_candid = params.get('AMICAL_ANA_USE_CANDID', False)
    use_pymask = params.get('AMICAL_ANA_USE_PYMASK', False)
    do_plot = params.get('AMICAL_ANA_PLOT', False)
    # storage for parameters for analysis codes
    candid_params = dict()
    pymask_params = dict()
    pymask_param_mcmc = dict()
    pymask_param_cr= dict()
    # ---------------------------------------------------------------------
    # Load candid variables
    # ---------------------------------------------------------------------
    if use_candid:
        # define parameters to load
        names = ['rmin', 'rmax', 'step', 'ncore', 'diam']
        variables = ['AMICAL_CANDID_RMIN', 'AMICAL_CANDID_RMAX',
                     'AMICAL_CANDID_STEP', 'AMICAL_CANDID_NCORE',
                     'AMICAL_CANDID_DIAM']
        # loop around variables and store parameter
        for it in range(len(names)):
            # load parameters
            if params[variables[it]] is not None:
                candid_params[names[it]] = params[variables[it]]
            else:
                emsg = 'AMICAL ERROR: {0} must be set to use candid'
                params.log.error(emsg.format(names[it]))
                return
    # ---------------------------------------------------------------------
    # Load pymask variables
    # ---------------------------------------------------------------------
    if use_pymask:
        # define pymask parameters to load
        names = ['sep_prior', 'pa_prior', 'cr_prior', 'ncore', 'extra_error',
                 'err_scale']
        variables = ['AMICAL_PYMASK_SEP_PRIOR', 'AMICAL_PYMASK_PA_PRIOR',
                     'AMICAL_PYMASK_CR_PRIOR', 'AMICAL_PYMASK_NCORE',
                     'AMICAL_PYMASK_EXTRA_ERR', 'AMICAL_PYMASK_ERR_SCALE']
        # loop around variables and store parameter
        for it in range(len(names)):
            # load parameters
            if params[variables[it]] is not None:
                pymask_params[names[it]] = params[variables[it]]
            else:
                emsg = 'AMICAL ERROR: {0} must be set to use pymask'
                params.log.error(emsg.format(names[it]))
                return
        # ---------------------------------------------------------------------
        # define pymask mcmc parameters to load
        names = ['niters', 'walkers', 'initial_guess', 'burn_in']
        variables = ['AMICAL_PYMASK_MCMC_NITERS', 'AMICAL_PYMASK_MCMC_NWALKERS',
                     'AMICAL_PYMASK_MCMC_IGUESS', 'AMICAL_PYMASK_MCMC_NBURN']
        # loop around variables and store parameter
        for it in range(len(names)):
            # load parameters
            if params[variables[it]] is not None:
                pymask_param_mcmc[names[it]] = params[variables[it]]
            else:
                emsg = 'AMICAL ERROR: {0} must be set to use pymask'
                params.log.error(emsg.format(names[it]))
                return
        # ---------------------------------------------------------------------
        # define pymask cr limit parameters
        names = ['nsim', 'ncore', 'smax' , 'nsep', 'cmax', 'nth', 'ncrat']
        variables = ['AMICAL_PYMASK_CR_NSIM', 'AMICAL_PYMASK_CR_NCORE',
                     'AMICAL_PYMASK_CR_SMAX', 'AMICAL_PYMASK_CR_NSEP',
                     'AMICAL_PYMASK_CR_CMAX', 'AMICAL_PYMASK_CR_NTH',
                     'AMICAL_PYMASK_CR_NCRAT']
        # loop around variables and store parameter
        for it in range(len(names)):
            # load parameters
            if params[variables[it]] is not None:
                pymask_param_cr[names[it]] = params[variables[it]]
            else:
                emsg = 'AMICAL ERROR: {0} must be set to use pymask'
                params.log.error(emsg.format(names[it]))
                return

    # ---------------------------------------------------------------------
    # loop around filters
    # ---------------------------------------------------------------------
    # loop around all filters to use
    for _filter in observation.filters:
        # push outfile into observation params
        inkey = define_amical_ext_outkey(_filter, mode)
        cal_target_abspath = str(observation.params[inkey])
        # run the candid code (if we are using candid)
        if use_candid:
            cout = _amical_run_candid(params, cal_target_abspath, candid_params)
        else:
            cout = []
        # run the pymask code (if we are using pymask)
        if use_pymask:
            pout = _amical_run_pymask(params, cal_target_abspath, pymask_params,
                                      pymask_param_mcmc, pymask_param_cr)
        else:
            pout = []
        # plot analysis plot
        if do_plot:
            _amical_analysis_plot(cout, pout)


def _amical_run_candid(params: ParamDict, filename: str,
                       kwargs: Dict[str, Any]
                       ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run CANDID for AMICAL

    :param params: ParamDict, the parameter dictionary of constants
    :param filename: str, the filename to load AMICAL extractions from
    :param kwargs: dict, parameters to run candid

    :return: tuple, 1. the candid fit dictionary, 2. the contrast limit
             dictionary
    """
    # hide all text within module
    with general.ModifyPrintouts(text='AMI-CAL-CANDID Output', flush=True,
                                 logfile=params['LOGFILE']):
        # fit using candid
        fit_candid = amical.candid_grid(filename, **kwargs,
                                        doNotFit=[])
        # get the contrast ratio limit
        cr_candid = amical.candid_cr_limit(filename, **kwargs,
                                           fitComp=fit_candid['comp'])
    # TODO: save something here?
    return fit_candid, cr_candid


def _amical_run_pymask(params: ParamDict, filename: str,
                       kwargs1: Dict[str, Any], kwargs2: Dict[str, Any],
                       kwargs3: Dict[str, Any]
                       ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Run Pymask for AMICAL

    :param params: ParamDict, the parameter dictionary of constants
    :param filename: str, the filename to load AMICAL extractions from
    :param kwargs1: dict, parameters to run pymask_grid
    :param kwargs2: dict, parameters to run pymask_mcmc
    :param kwargs3: dict, parameters to run pymask_cr_limit

    :return: tuple, 1. the pymask_grid fit dictionary,
                    2. the pymask_mcmc fit dictionary,
                    3. the pymask_cr_limit dictionary
    """
    # hide all text within module
    with general.ModifyPrintouts(text='AMI-CAL-PYMASK Output', flush=True,
                                 logfile=params['LOGFILE']):
        # get the pymask fit with a grid
        fit_pymask1 = amical.pymask_grid(filename, **kwargs1)
        # get the pymask fit with mcmc
        fit_pymask2 = amical.pymask_mcmc(filename, **kwargs1, **kwargs2)
        # get the contrast ratio limit
        cr_pymask = amical.pymask_cr_limit(filename, **kwargs3)
    # TODO: save something here?
    return fit_pymask1, fit_pymask2, cr_pymask


def _amical_analysis_plot(cout: Union[list, tuple], pout: Union[list, tuple]):
    """
    Plot the AMICAL analysis plot

    :param cout: list or tuple, the output of _amical_run_candid
    :param pout: list or tuple, the output of _amical_run_pymask

    :return: None, plots a graph
    """
    # -------------------------------------------------------------------------
    # set up plot
    fig, frame = plt.subplots(ncols=1, nrows=1)
    # -------------------------------------------------------------------------
    # deal with candid
    if len(cout) > 0:
        # get the cr limits for candid
        cr_candid = cout[1]
        # plot
        frame.plot(cr_candid['r'], cr_candid['cr_limit'],
                   label='CANDID', alpha=0.5, lw=3)
    # -------------------------------------------------------------------------
    # deal with pymask
    if len(pout) > 0:
        # get the cr limits for pymask
        cr_pymask = cout[2]
        # plot
        frame.plot(cr_pymask['r'], cr_pymask['cr_limit'],
                   label='Pymask', alpha=0.5, lw=3)
    # -------------------------------------------------------------------------
    # set label
    frame.set(xlabel='Separation [mask]',
              ylabel='$\Delta \mathrm{Mag}_{3\sigma}$')
    # set up legend
    frame.legend(loc=0)
    # add a grid to the plot
    frame.grid()
    # add a tight layout
    plt.tight_layout()
    # show plot
    plt.show(block=True)


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
