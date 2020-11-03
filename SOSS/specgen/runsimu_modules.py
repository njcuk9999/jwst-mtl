

# Read PATH config file

# Python imports

# Read Model parameters config file

# Read Planet model, star model, response file, trace parameters

# Resample planet and star on same grid

# Transit model setup

# Read PSF kernels

# Loop on orders to create convolved image




def read_config_path(config_paths_filename):
    # Read a configuration file used for the whole SOSS pipeline
    # It gives the path to various files.

    import sys
    import numpy as np

    token = open('/genesis/jwst/jwst-mtl-user/jwst-mtl_configpath.txt','r')
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
    # User config file for paths
    print('param and value for the user config file:')
    print(param)
    print(value)
    print()

    JWST_MTL_PATH = str(value[param == 'JWST-MTL_PATH'][0])
    USER_PATH = str(value[param == 'USER_PATH'][0])
    SIMULATION_PARAM = str(value[param =='SIMULATION_PARAM'][0])
    FORTRAN_LIB = str(value[param == 'FORTRAN_LIB'][0])
    MONOCHROMATIC_PSFS = str(value[param == 'MONOCHROMATIC_PSFS'][0])
    PLANET_MODEL_ATM = str(value[param == 'PLANET_MODEL_ATM'][0])
    SPECTRAL_CONV_KERNELS = str(value[param == 'SPECTRAL_CONV_KERNELS'][0])
    STAR_MODEL_ATM = str(value[param == 'STAR_MODEL_ATM'][0])
    TRACE_MODEL = str(value[param == 'TRACE_MODEL'][0])

    # Include the path to code in the system path
    #print(sys.path)
    #print()
    sys.path.insert(0, JWST_MTL_PATH+'/jwst-mtl/SOSS/trace/')
    sys.path.insert(0, JWST_MTL_PATH+'/jwst-mtl/SOSS/specgen/')
    sys.path.insert(0, FORTRAN_LIB)
    sys.path.insert(0, USER_PATH)

def init_imports():
    # Some imports like tracepol need to happen *after* the path to tracepol
    # has been assigned with a call to sys.path.insert()

    import os  # checking status of requested files
    # os.environ["OMP_NUM_THREADS"] = "24"

    import numpy as np  # numpy gives us better array management

    # Will be used to create FITS writer
    from astropy.io import fits  # astropy modules for FITS IO

    import matplotlib  # ploting
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')

    from skimage.transform import downscale_local_mean, resize

    from tqdm.notebook import trange
    from tqdm.notebook import tqdm as tqdm_notebook

    import pyfftw
    import multiprocessing as mp
    # import scipy.signal
    import scipy.fft
    from matplotlib.colors import LogNorm
    ncpu = mp.cpu_count()
    pyfftw.config.NUM_THREADS = ncpu

    # Trace Library
    import tracepol as tp

    # Top level simulation library
    from write_dmsready_fits import write_dmsready_fits

    # Used to monitor progress for multiprocessing pools
    def barupdate(result):
        pbar.update()

    import spgen as spgen #Python Routines for SpecGen Routines and wrappers for fast-Transit-model.


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

