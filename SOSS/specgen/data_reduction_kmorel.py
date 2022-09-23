# Example script to use DMS level 1 pipeline
# for data reduction
# By L. Albert
# Few modifications by K. Morel

import sys
sys.path.insert(0, '/genesis/jwst/jwst-ref-soss/fortran_lib/')

# TODO: Update all paths

import glob

import os
# os.environ["OMP_NUM_THREADS"] = "24"
import numpy as np

# Will be used to create FITS writer
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table

# Plotting
import matplotlib
import matplotlib.pyplot as plt
# In a Jupyter notebook, you may add this line
# get_ipython().run_line_magic('matplotlib', 'inline')
# To bin the oversampled images down to its native pixel size
from skimage.transform import downscale_local_mean, resize
# To produce progress bars when generating simulations
from tqdm.notebook import trange

from tqdm.notebook import tqdm as tqdm_notebook

import pyfftw

import multiprocessing as mp

import json as json

import scipy.fft

from matplotlib.colors import LogNorm

# Python Routines for SpecGen Routines and wrappers for fast-Transit-model.
import specgen.spgen as spgen
# Trace Library
import trace.tracepol as tp
# Header and FITS writing function
# Detector noise script
from detector import detector
#import detector as toto

# normalization code
import specgen.synthesizeMagnitude as smag

ncpu = mp.cpu_count()
pyfftw.config.NUM_THREADS = ncpu

import itsosspipeline as soss

from jwst.pipeline import calwebb_detector1

verbose = True

######################################################################
# Read in all paths used to locate reference files and directories
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/master/WASP_96/'
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath.txt')

pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars() #Set up default parameters
simuPars.read_pars(pathPars.simulationparamfile) #read in parameter file

f277webcal = False

# Select which noise source to be corrected
do_superbias = False
do_oneoverf = False
do_nonlinearity = False
do_darkcurrent = False

# Here are de paths of the reference files of the noises used for the simulations.
# One can modify them to use their own reference file(s) for data reduction.
superbias_ref_file = os.path.join(pathPars.path_noisefiles, simuPars.superbias_ref)
linearity_ref_file = os.path.join(pathPars.path_noisefiles, simuPars.nonlin_ref)
dark_ref_file = os.path.join(pathPars.path_noisefiles, simuPars.dark_ref)


print(pathPars.path_planetmodelatm+simuPars.pmodelfile[0])
print(simuPars.pmodeltype[0])


'''
PROCESS THROUGH DMS 
'''
# Process the data through the DMS level 1 pipeline
# here is the option to investigate individual noise sources
# GR700XD+CLEAR - PROCESS THROUGH DMS LEVEL 1

# Define input/output
calwebb_input = os.path.join(pathPars.path_userland, 'IDTSOSS_clear_noisy.fits')
calwebb_output = os.path.join(pathPars.path_userland, 'IDTSOSS_clear_noisy_rateints.fits')

# Step by step DMS processing
result = calwebb_detector1.dq_init_step.DQInitStep.call(calwebb_input,
            output_dir=pathPars.path_userland, output_file='IDTSOSS_clear_noisy', save_results=False)
result = calwebb_detector1.saturation_step.SaturationStep.call(result,
            output_dir=pathPars.path_userland, output_file='IDTSOSS_clear_noisy', save_results=False)

if do_superbias is True: result = calwebb_detector1.superbias_step.SuperBiasStep.call(result,
            output_dir=pathPars.path_userland, output_file='IDTSOSS_clear_noisy', save_results=False,
            override_superbias=superbias_ref_file)
if do_oneoverf is True: result = calwebb_detector1.refpix_step.RefPixStep.call(result,
            output_dir=pathPars.path_userland, output_file='IDTSOSS_clear_noisy', save_results=False)
if do_nonlinearity is True: result = calwebb_detector1.linearity_step.LinearityStep.call(result,
            output_dir=pathPars.path_userland, output_file='IDTSOSS_clear_noisy', save_results=False,
            override_linearity=linearity_ref_file)
if do_darkcurrent is True: result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(result,
            output_dir=pathPars.path_userland, output_file='IDTSOSS_clear_noisy', save_results=False,
            override_dark=dark_ref_file)

_, result = calwebb_detector1.ramp_fit_step.RampFitStep.call(result,
            output_dir=pathPars.path_userland, output_file='IDTSOSS_clear_noisy', save_results=False)
result = calwebb_detector1.gain_scale_step.GainScaleStep.call(result,
            output_dir=pathPars.path_userland, output_file='IDTSOSS_clear_noisy', save_results=False)
result.meta.filetype = 'countrate'
result.write(calwebb_output)


# GR700XD+F277W - PROCESS THROUGH DMS LEVEL 1
if f277webcal is True:
    # Define input on the first iteration of the loop below
    calwebb_input = os.path.join(pathPars.path_userland, 'IDTSOSS_f277_noisy.fits')
    calwebb_output = os.path.join(pathPars.path_userland, 'IDTSOSS_f277_noisy_rateints.fits')

    # Step by step DMS processing
    result = calwebb_detector1.dq_init_step.DQInitStep.call(calwebb_input,
             output_dir=pathPars.path_userland, output_file='IDTSOSS_f277_noisy', save_results=False)
    result = calwebb_detector1.saturation_step.SaturationStep.call(result,
             output_dir=pathPars.path_userland, output_file='IDTSOSS_f277_noisy', save_results=False)

    if do_superbias is True: result = calwebb_detector1.superbias_step.SuperBiasStep.call(result,
             output_dir=pathPars.path_userland, output_file='IDTSOSS_f277_noisy', save_results=False,
             override_superbias=superbias_ref_file)
    if do_oneoverf is True: result = calwebb_detector1.refpix_step.RefPixStep.call(result,
             output_dir=pathPars.path_userland, output_file='IDTSOSS_f277_noisy', save_results=False)
    if do_nonlinearity is True: result = calwebb_detector1.linearity_step.LinearityStep.call(result,
             output_dir=pathPars.path_userland, output_file='IDTSOSS_f277_noisy', save_results=False,
             override_linearity=linearity_ref_file)
    if do_darkcurrent is True: result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(result,
             output_dir=pathPars.path_userland, output_file='IDTSOSS_f277_noisy', save_results=False,
             override_dark=dark_ref_file)

    _, result = calwebb_detector1.ramp_fit_step.RampFitStep.call(result,
             output_dir=pathPars.path_userland, output_file='IDTSOSS_f277_noisy', save_results=False)
    result = calwebb_detector1.gain_scale_step.GainScaleStep.call(result,
             output_dir=pathPars.path_userland, output_file='IDTSOSS_f277_noisy', ave_results=False)
    result.meta.filetype = 'countrate'
    result.write(calwebb_output)