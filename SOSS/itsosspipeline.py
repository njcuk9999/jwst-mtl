

# Read PATH config file

# Python imports

# Read Model parameters config file

# Read Planet model, star model, response file, trace parameters

# Resample planet and star on same grid

# Transit model setup

# Read PSF kernels

# Loop on orders to create convolved image

import sys

import numpy as np


class paths():
    # PATHS
    path_home = '/genesis/jwst/'
    path_userland = '/genesis/jwst/jwst-mtl-user/'
    path_tracemodel = path_home+'/jwst-mtl-ref/trace_model/'
    path_starmodelatm = path_home+'/jwst-mtl-ref/star_model_atm/'
    path_planetmodelatm = path_home+'/jwst-mtl-ref/planet_model_atm/'
    path_spectralconvkernels = path_home+'/jwst-mtl-ref/spectral_conv_kernels/'
    path_monochromaticpsfs = path_home+'/jwst-mtl-ref/monochromatic_PSFs/'
    path_fortranlib = path_home+'/jwst-mtl-ref/fortran_lib/'
    path_noisefiles = path_home+'/jwst-mtl-ref/noise_files/'
    # Reference files
    simulationparamfile = path_userland+'simpars_wide.txt'
    tracefile = path_tracemodel+'/NIRISS_GR700_trace_extended.csv'
    responsefile = path_tracemodel+'/NIRISS_Throughput_STScI.fits'


def readpaths(config_paths_filename, pars):
    # Read a configuration file used for the whole SOSS pipeline
    # It gives the path to various files.

    token = open(config_paths_filename,'r')
    linestoken=token.readlines()
    param,value = [],[]
    for x in linestoken:
        tmp = x.replace(' ','') # but the '\n' will remain so rejects lines of len 1 and more
        if len(tmp) > 1:
            if (x[0] != '#'):
                line_parts = x.split('#')
                non_comments = x.split('#')[0].split()
                param_col = non_comments[0]
                value_col = non_comments[1]
                param.append(param_col)
                value.append(value_col)
    token.close()
    param = np.array(param)
    value = np.array(value)

    # Fill the object with the values rad out from the file
    pars.path_home = str(value[param == 'JWST-MTL_PATH'][0])
    pars.path_userland = str(value[param == 'USER_PATH'][0])
    pars.path_tracemodel = str(value[param == 'TRACE_MODEL'][0])
    pars.path_starmodelatm = str(value[param == 'STAR_MODEL_ATM'][0])
    pars.path_planetmodelatm = str(value[param == 'PLANET_MODEL_ATM'][0])
    pars.path_spectralconvkernels = str(value[param == 'SPECTRAL_CONV_KERNELS'][0])
    pars.path_monochromaticpsfs = str(value[param == 'MONOCHROMATIC_PSFS'][0])
    pars.path_fortranlib = str(value[param == 'FORTRAN_LIB'][0])
    pars.path_noisefiles = str(value[param == 'NOISE_FILES'][0])
    # Reference files
    pars.simulationparamfile = str(value[param =='SIMULATION_PARAM'][0])
    pars.tracefile = str(value[param =='TRACE_FILE'][0])
    pars.responsefile = str(value[param =='RESPONSE_FILE'][0])







def flux_calibrate_simulation(parameters):
    # This is a one call function that uses the same parameters as the actual simulation
    # to perform a flux calibration. It actually returns the scale by which each order of
    # the actual simulation should be scaled to make the flux calibrated against a given
    # magnitude through a given observing filter.
    #
    # INPUT:
    # parameters: the dictionnary of parameters that the actual simulation uses
    #             with same format as pars in the runsimu_loic.ipynb example.
    #
    # OUTPUT:
    # returns one scalar float per spectral order considered.


    print()



