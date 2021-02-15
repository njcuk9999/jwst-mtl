#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 

@author: 
"""
import numpy as np
from pathlib import Path
from typing import Any, Dict, Union
import yaml

from ami_mtl.core.core import constant_functions
from ami_mtl.core.core import param_functions
from ami_mtl.core.core import exceptions

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'apero.core.core.wrap.py'
__DESCRIPTION__ = 'module of wrapper classes and functions'
# get parameter dictionary
ParamDict = param_functions.ParamDict
# get Observation Exception
ObservationException = exceptions.ObservationException


# =============================================================================
# Define classes
# =============================================================================
class Simulation:
    def __init__(self, params: ParamDict, properties: Dict[str, Any]):
        """
        Basic simulation class (what is passed around)

        :param properties: simulation dictionary (from yaml file)
        """
        # set name
        self.name = properties.get('name', None)
        # set parameters
        self.params = params
        # ---------------------------------------------------------------------
        # get xml path
        self.xmlpath = properties.get('xmlpath', None)
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
        return 'Simulation[{0}]'.format(self.name)

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
        self.params = params
        # set name
        self.name = properties.get('name', None)
        # deal with no name
        if self.name is None:
            emsg = 'Name must be set for class {0}'.format(self.__str__())
            emsg += '\n input properties: '
            emsg += '\n\t'.join(self._str_properties(properties))
            raise ObservationException(emsg, 'error', None, func_name)
        # set raw magnitudes
        self.raw_magnitudes = properties.get(mag_key, None)
        # get magnitudes
        self.magnitudes = dict()

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
                out_list.append('{0}: Dict')
        # return list
        return out_list

    def __str__(self) -> str:
        """
        String representation of class

        :return:
        """
        return 'Observation[{0}]'.format(self.name)

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


class Target(Observation):
    def __init__(self, params: ParamDict, properties: Dict[str, Any]):
        """
        Basic target class

        :param params: ParamDict, the parameter dictionary of constants
        :param properties: target dictionary (from yaml file)
        """
        super().__init__(params, properties)
        # load magnitudes
        self.get_magnitudes()
        # deal with companions
        self.companions = []
        self.get_companion(properties)

    def __str__(self) -> str:
        """
        String representation of class

        :return:
        """
        return 'Target[{0}]'.format(self.name)

    def __repr__(self) -> str:
        """
        String representation of class

        :return:
        """
        return self.__str__()

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
    def __init__(self, params: ParamDict, properties: Dict[str, Any]):
        """
        Basic calibrator class

        :param params: ParamDict, the parameter dictionary of constants
        :param properties: calibrator dictionary (from yaml file)
        """
        super().__init__(params, properties)
        # load magnitudes
        self.get_magnitudes()

    def __str__(self) -> str:
        """
        String representation of class

        :return:
        """
        return 'Calibrator[{0}]'.format(self.name)

    def __repr__(self) -> str:
        """
        String representation of class

        :return:
        """
        return self.__str__()


class Companion(Observation):
    def __init__(self, params: ParamDict, properties: Dict[str, Any]):
        """
        Basic companion class

        :param params: ParamDict, the parameter dictionary of constants
        :param properties:
        """
        super().__init__(params, properties, mag_key='dmag')
        # get name
        self.name = properties.get('name', 'Unknown')
        # get kind (currently only support "planet")
        self.kind = properties.get('kind', 'planet')
        # get separation in arc seconds (used for kind="planet")
        self.separation = properties.get('separation', np.nan)
        # get position angle in degrees (used for kind="planet")
        self.position_angle = properties.get('pa', np.nan)
        # get magnitudes
        self.magnitudes = dict()
        self.get_magnitudes()

    def __str__(self) -> str:
        """
        String representation of class

        :return:
        """
        return 'Companion[{0}]'.format(self.name)

    def __repr__(self) -> str:
        """
        String representation of class

        :return:
        """
        return self.__str__()


# =============================================================================
# Define functions
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
            simulation = Simulation(params, properties[key])
            # add to list
            simulations.append(simulation)
    # return simulations
    return simulations


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

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
