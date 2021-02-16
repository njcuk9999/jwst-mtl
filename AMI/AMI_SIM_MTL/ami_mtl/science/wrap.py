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
from typing import Any, Dict, List, Union
from mirage import read_apt_xml
import yaml

from ami_mtl.core.core import constant_functions
from ami_mtl.core.core import exceptions
from ami_mtl.core.core import general
from ami_mtl.core.core import param_functions
from ami_mtl.science import etienne

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'apero.core.core.wrap.py'
__DESCRIPTION__ = 'module of wrapper classes and functions'
# get parameter dictionary
ParamDict = param_functions.ParamDict
# get Observation Exception
ObservationException = exceptions.ObservationException
# get general functions
display_func = general.display_func


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
        self.params = params.copy()
        # set name
        self.name = properties.get('name', None)
        # deal with no name
        if self.name is None:
            emsg = 'Name must be set for class {0}'.format(self.__str__())
            emsg += '\n input properties: '
            emsg += '\n\t'.join(self._str_properties(properties))
            raise ObservationException(emsg, 'error', None, func_name)
        # clean name
        self.name = general.clean_name(self.name)
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
            simulation = Simulation(params, properties[key])
            # add to list
            simulations.append(simulation)
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
            run_ami_sim(simulation.name, simulation.use_filters,
                        simulation.target)
            # simulate calibrators
            for calibrator in simulation.calibrators:
                # simulate calibrator
                run_ami_sim(simulation.name, simulation.use_filters,
                            calibrator)
        # simulate using Mirage
        if simulation.params['MIRAGE-USE']:
            # simulate target
            run_mirage(simulation.target)
            # simulate calibrators
            for calibrator in simulation.calibrators:
                run_mirage(calibrator)



def run_ami_sim(simname: str, filters: List[str], observation: Observation):
    """
    Run the AMI SIM module on a specific observation

    :param simname: the name of this simulation
    :param filters: list of filters to use
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
    for _filter in filters:
        # construct file path
        path = Path(str(params.get('AMISIM-PATH', params['DIRECTORY'])))
        # construct filename
        oargs = [simname, observation.name, _filter]
        filename = 'SKY_SCENE_{0}_{1}_{2}.fits'.format(*oargs)
        # construct abs path to file
        scenepath = path.joinpath(filename)
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
        # TODO: Need funciton to go from mag[fitler] --> flux
        #pkwargs['ext_flux'] = observation.blank
        pkwargs['ext_flux'] = 1000001
        # TODO: Need to get this from xml file
        #pkwargs['tot_exp'] = observation.blank
        pkwargs['tot_exp'] = 600
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
                    # TODO: Need function to get from dmag[filter] --> contrast
                    # ckwargs['contrast'] = companion.blank
                    ckwargs['contrast'] = 0.5
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
        simfile = etienne.ami_sim_run_code(params, path, _filter, psf_filename,
                                           scenepath, count_rate, simname,
                                           observation.name)
        # update param for sim file
        okey = 'AMI-SIM-OUT_{0}'.format(_filter)
        observation.params[okey] = simfile
        observation.params.set_source(okey, func_name)
        # ---------------------------------------------------------------------





def run_mirage(observation: Observation):
    pass


# =============================================================================
# Define DMS functions
# =============================================================================


# =============================================================================
# Define AMICAL functions
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
    # set function name
    func_name = display_func('_load_xml', __NAME__)
    # deal with filename
    if filename is None:
        # log error
        msg = 'XML-Error: Filename must be defined'
        params.log.error(msg)
    else:
        filename = Path(filename)
    # deal with file not existing
    if not filename.exists():
        # log error
        msg = 'XML-Error: Filename {0} does not exist'
        params.log.error(msg)

    # load xml file as dictionary of keys
    xml = read_apt_xml.ReadAPTXML()
    table = xml.read_xml(filename)


    # TODO: just load the parameters into individual dictionaries for
    # TODO:    each target - then search for it in Observation
    # # get apt column name for target
    # target_name_col = params.instances('APT-TARGET-NAME').apt
    #
    # # search for name in table using cleaned TargetID
    # table_names = list(map(table[target_name_col], general.clean_name))


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
