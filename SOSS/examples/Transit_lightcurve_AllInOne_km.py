# TODO: Update all paths
# imports
import trace.tracepol as tp
import numpy as np
import itsosspipeline as soss
import sys
from sys import path
import os
import glob
import specgen.spgen as spgen
import pyfftw
import multiprocessing as mp
import json as json
import scipy.fft
from jwst.pipeline import Detector1Pipeline
import box_kim

sys.path.insert(0, "/genesis/jwst/jwst-ref-soss/fortran_lib/")
sys.path.insert(1, '/genesis/jwst/jwst-ref-soss/fortran_lib/')
sys.path.insert(2, github_path)

# Plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Will be used to create FITS writer
from astropy.io import fits
from astropy.io import ascii

# To bin the oversampled images down to its native pixel size
from skimage.transform import downscale_local_mean, resize
# To produce progress bars when generating simulations
from tqdm.notebook import trange
from tqdm.notebook import tqdm as tqdm_notebook

# Header and FITS writing function
# Detector noise script
from detector import detector_Frost as detector
# normalization code
import specgen.synthesizeMagnitude as smag


# os.environ["OMP_NUM_THREADS"] = "24"

ncpu = mp.cpu_count()
pyfftw.config.NUM_THREADS = ncpu


# Matplotlib defaults
plt.rc('figure', figsize=(8,4.5))
plt.rc('font', size=12)
plt.rc('image', cmap='inferno')
plt.rc('lines', lw=1)


# Location of the jwst-mtl github repo. Overrides the github path in
# 'jwst_config_fpath', during the import phase
github_path = '/home/kmorel/ongenesis/github/jwst-mtl/SOSS/'
# Location of the simulation config file, as well as the output directory
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/PHY3030/'
# Configuration file for the NIRISS Instrument Team SOSS simulation pipeline
jwst_config_fpath = 'jwst-mtl_configpath_kim.txt'

# Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR + jwst_config_fpath)
pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              # Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars)    # Read in parameter file


gain = 1.61  # [é]
radius_pixel = 30  # Radius for box extraction

# Choose whether to generate the noisy dms simulations or
# work with some already generated.
generate_noisy_simu = True

######################################################################
# NOISELESS
if generate_noisy_simu is True:
    # Generate noiseless simu that passed through the dms
    noise_shopping_lists = [[]
                            ]
    # TODO

simu_noiseless, data_noiseless = box_kim.rateints_dms_simulation(WORKING_DIR + 'IDTSOSS_clear_noNoise_rateints.fits')
simulation_noiseless = 'Noiseless, after DMS'

ng = simu_noiseless[0].header['NGROUPS']  # n_groups
t_read = simu_noiseless[0].header['TGROUP']  # Reading time [s]
tint = (ng - 1) * t_read  # Integration time [s]

t1 = 95   # [image]
t2 = 155  # [image]
t3 = 283  # [image]
t4 = 343  # [image]

# Position of trace for box extraction
x, y, w = box_kim.readtrace(os=1)

# Wavelengths array
lam_array = w

# BOX EXTRACTION
fbox_noiseless = np.zeros(shape=(np.shape(data_noiseless)[0], np.shape(data_noiseless)[2]), dtype=float)

for t in range(np.shape(data_noiseless)[0]):  # For each image of the timeseries
    fbox_noiseless[t] = box_kim.flambda_adu(x, data_noiseless[t], y, radius_pixel=radius_pixel)  # [adu/s]

f_array_noiseless = np.nan_to_num(fbox_noiseless)

# Time array
time = np.arange(f_array_noiseless.shape[0])
time_min = time * tint / 60.  # [min]

# WHITE LIGHT CURVE
f_white_noiseless = np.sum(f_array_noiseless, axis=1)
# Normalize white light curve
f_white_noiseless_norm = box_kim.normalization(f_white_noiseless, t1, t4)

# For each wavelength
f_array_noiseless_norm = np.copy(f_array_noiseless)

for n in range(f_array_noiseless.shape[1]):  # For each wavelength
    f_array_noiseless_norm[:, n] = box_kim.normalization(f_array_noiseless[:, n], t1, t4)
#######################################################################

# This 2D-list determines what noise types are injected into them (list contained in 2nd dimension).
noise_shopping_lists = [['photon']
                       #,['normalize']
                       #,['zodibackg']
                       #,['flatfield']
                       #,['darkframe']
                       ,['nonlinearity']
                       ,['superbias']
                       #,['detector']
                       ]


# LOOP on each noise type
for noise in noise_shopping_lists:
    if generate_noisy_simu is True:
        # TODO

    #######################################################################
    # Load simulation
    simu_filename = ''   # TODO: change name of simulation
    simu_noisy, data_noisy = box_kim.rateints_dms_simulation(simu_filename)
    simulation_noisy = noise  # TODO: To change??
    # Convert data from fits files to float (fits precision is 1e-8)
    data_noisy = data_noisy.astype('float64', copy=False)

    #######################################################################
    # EXTRACTION
    # BOX EXTRACTION
    fbox_noisy = np.zeros(shape=(np.shape(data_noisy)[0], np.shape(data_noisy)[2]), dtype=float)

    for t in range(np.shape(data_noisy)[0]):  # For each image of the timeseries
        fbox_noisy[t] = box_kim.flambda_adu(x, data_noisy[t], y, radius_pixel=radius_pixel)    # [adu/s]
    f_array_noisy = np.nan_to_num(fbox_noisy)

    #######################################################################
    # WHITE LIGHT CURVE
    f_white_noisy = np.sum(f_array_noisy, axis=1)

    # Normalize white light curve
    f_white_noisy_norm = box_kim.normalization(f_white_noisy, t1, t4)


    # For each wavelength
    f_array_noisy_norm = np.copy(f_array_noisy)

    for n in range(f_array_noisy.shape[1]):  # For each wavelength
        f_array_noisy_norm[:, n] = normalization(f_array_noisy[:, n], t1, t4)

    plt.figure()
    plt.plot(time_min, f_white_noisy, '.', markersize=3, color='r', label=simulation_noisy)
    plt.plot(time_min, f_white_noiseless, '.', markersize=3, color='b', label=simulation_noiseless)
    plt.xlabel('Time [min]')
    plt.ylabel('Relative flux')
    plt.legend()
    plt.show()
