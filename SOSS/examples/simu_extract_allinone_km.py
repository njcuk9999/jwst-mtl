# This script creates a noisy simulation for each noise
# source specified in the simpars.txt file. It extracts
# the flux for each noisy simulation. This is done one
# noise source at a time in order to investigate them.

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
############################ SET UP ##################################
######################################################################
# Location of the jwst-mtl github repo. Overrides the github path in
# 'jwst_config_fpath', during the import phase
github_path = '/home/kmorel/ongenesis/github/jwst-mtl/SOSS/'
# Location of the simulation config file, as well as the output directory
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/PHY3030/WASP_52/'
# Configuration file for the NIRISS Instrument Team SOSS simulation pipeline
jwst_config_fpath = 'jwst-mtl_configpath.txt'

config_paths_filename = os.path.join(WORKING_DIR, jwst_config_fpath)

pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars() #Set up default parameters
simuPars.read_pars(pathPars.simulationparamfile) #read in parameter file

print(pathPars.path_planetmodelatm+simuPars.pmodelfile[0])
print(simuPars.pmodeltype[0])

# Radius for box extraction
radius_pixel = 30   # [pixels]

# Here are de paths of the reference files of the noises used for the simulations.
# One can modify them to use their own reference file(s) for data reduction.
superbias_ref_file = os.path.join(pathPars.path_noisefiles, simuPars.superbias_ref)
linearity_ref_file = os.path.join(pathPars.path_noisefiles, simuPars.nonlin_ref)
dark_ref_file = os.path.join(pathPars.path_noisefiles, simuPars.dark_ref)

# If plots are shown
doShow_plots = True

##########################################################
# Matplotlib defaults
plt.rc('figure', figsize=(10,6))
plt.rc('font', size=16)
plt.rc('image', cmap='inferno')
plt.rc('lines', lw=1)

##########################################################
###################### FUNCTIONS #########################
##########################################################
def no_dms_simulation(file_name, gain=1.61):
    with fits.open(file_name) as hdulist:
        ng = hdulist[0].header['NGROUPS']  # n_groups
        t_read = hdulist[0].header['TGROUP']  # Reading time [s]
        tint = (ng - 1) * t_read  # Integration time [s]

        simu = hdulist
        data = (hdulist[1].data[:, -1] - hdulist[1].data[:, 0]) / tint / gain  # Images of flux [adu/s]

        # Convert data from fits files to float (fits precision is 1e-8)
        data = data.astype('float64', copy=False)
    return simu, data

def rateints_dms_simulation(file_name):
    with fits.open(file_name) as hdulist:
        data_noisy_rateints = hdulist[1].data  # Images of flux [adu/s]
        # delta_noisy = hdulist[2].data            # Errors [adu/s]
        dq = hdulist[3].data  # Data quality
        i = np.where(dq % 2 != 0)  # Odd values of dq = DO NOT USE these pixels
        data_noisy_rateints[i[0], i[1], i[2]] = 0.
        # delta_noisy[i[0], i[1], i[2]] = 0.

        # Convert data from fits files to float (fits precision is 1e-8)
        data_noisy_rateints = data_noisy_rateints.astype('float64', copy=False)

        simu = hdulist
        data = data_noisy_rateints
    return simu, data

def readtrace(os):  # From Loic
    """
    Returns x and y order 1 trace position with corresponding wavelengths.
    os: Oversampling
    """
    trace_filename = '/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv'
    pars = tp.get_tracepars(trace_filename, disable_rotation=False)
    w = np.linspace(0.7, 3.0, 10000)
    x, y, mask = tp.wavelength_to_pix(w, pars, m=1, oversample=os, subarray='SUBSTRIP256')
    x_index = np.arange(2048 * os)
    # np.interp needs ordered x
    ind = np.argsort(x)
    x, y, w = x[ind], y[ind], w[ind]
    wavelength = np.interp(x_index, x, w)
    y_index = np.interp(x_index, x, y)
    return x_index, y_index, wavelength

def flambda_adu(pixels, trace_im, y_trace, radius_pixel=30):
    """
    pixels: Array of pixels
    trace_im: Trace's image [adu/s]
    y_trace: Array for the positions of the center of the trace for each column
    radius_pixel: Radius of extraction box. Default is 30. [pixels]
    return: Extracted flux [adu/s/colonne]
    """
    flux = np.zeros_like(pixels, dtype=float)  # Array for extracted spectrum
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        flux[x_i] = trace_im[int(first), x_i] * (1 - first % int(first)) + np.sum(
            trace_im[int(first) + 1:int(last) + 1, x_i]) + trace_im[int(last) + 1, x_i] * (last % int(last))
    return flux

def normalization(f_lambda, t1, t4):
    """
    Normalize transit light curve by out of transit mean for all wavelengths.
    First dimension is time, second dimension is wavelengths.
    """
    out_transit = np.concatenate((f_lambda[:t1+1], f_lambda[t4:]))
    out_transit_mean = np.mean(out_transit, axis=0)
    return (f_lambda / out_transit_mean)

def relative_difference(data, ref_data):
    """
    data: Data to be compared
    ref_data: Reference data with which to compare
    !data and ref_data must be the same size!
    return: Relative difference
    """
    return ((data - ref_data) / ref_data)

def transit_depth(f_lambda, t1, t2, t3, t4):
    """
    f_lambda: Flux array.
    t1, t2, t3, t4: Characteristic transit times
    :return: Transit light curve.
    """
    # Mean flux value during transit for all wavelengths
    in_transit_mean = np.mean(f_lambda[t2: t3 + 1], axis=0)
    # Mean flux value during out of transit for all wavelengths
    out_transit = np.concatenate((f_lambda[: t1 + 1], f_lambda[t4:]))
    out_transit_mean = np.mean(out_transit, axis=0)
    return out_transit_mean - in_transit_mean  # in_transit_mean / out_transit_mean


###########################################
####### ADD (AND CORRECT FOR) NOISE #######
###########################################
noises = [simuPars.readout, simuPars.zodibackg, simuPars.photon, simuPars.superbias,
                   simuPars.flatfield, simuPars.nonlinearity, simuPars.oneoverf,
                   simuPars.darkcurrent, simuPars.cosmicray]

noises_names = ['readout', 'zodibackg', 'photon', 'superbias', 'flatfield', 'nonlinearity', 'oneoverf',
                'darkcurrent', 'cosmicray']

t = 0
for n in range(len(noises)):
    noises2 = np.full(len(noises), False)
    if noises[n] is True:
        print("Doing {}".format(noises_names[n]))
        noises2[n] = True
        # GR700XD + CLEAR - ADD NOISE
        # Add detector noise to the noiseless data
        detector.add_noise(os.path.join(pathPars.path_userland, 'IDTSOSS_clear.fits'),
                           pathPars.path_noisefiles,
                           outputfilename=os.path.join(pathPars.path_userland,
                                                       'IDTSOSS_clear_noisy_{}.fits'.format(noises_names[n])),
                           dark_ref=simuPars.dark_ref, flat_ref=simuPars.flat_ref,
                           superbias_ref=simuPars.superbias_ref, nlcoeff_ref=simuPars.nlcoeff_ref,
                           zodi_ref=simuPars.zodi_ref,
                           readout=noises2[0], zodibackg=noises2[1],
                           photon=noises2[2], superbias=noises2[3],
                           flatfield=noises2[4], nonlinearity=noises2[5],
                           oneoverf=noises2[6], darkcurrent=noises2[7],
                           cosmicray=noises2[8])


        ############################################
        ### PROCESS THROUGH DMS LEVEL 1 PIPELINE ###
        ############################################

        # Define input/output
        calwebb_input = os.path.join(pathPars.path_userland,
                                     'IDTSOSS_clear_noisy_{}.fits'.format(noises_names[n]))
        calwebb_output = os.path.join(pathPars.path_userland,
                                      'IDTSOSS_clear_noisy_{}_rateints.fits'.format(noises_names[n]))

        # Step by step DMS processing
        result = calwebb_detector1.dq_init_step.DQInitStep.call(calwebb_input, output_dir=pathPars.path_userland,
                    output_file='IDTSOSS_clear_noisy_{}'.format(noises_names[n]), save_results=False)
        result = calwebb_detector1.saturation_step.SaturationStep.call(result, output_dir=pathPars.path_userland,
                    output_file='IDTSOSS_clear_noisy_{}'.format(noises_names[n]), save_results=False)

        if noises_names[n]=='superbias':
            result = calwebb_detector1.superbias_step.SuperBiasStep.call(result, output_dir=pathPars.path_userland,
                        output_file='IDTSOSS_clear_noisy_{}'.format(noises_names[n]), save_results=False,
                        override_superbias=superbias_ref_file)
        if noises_names[n]=='oneoverf':
            result = calwebb_detector1.refpix_step.RefPixStep.call(result, output_dir=pathPars.path_userland,
                        output_file='IDTSOSS_clear_noisy_{}'.format(noises_names[n]), save_results=False)
        if noises_names[n]=='nonlinearity':
            result = calwebb_detector1.linearity_step.LinearityStep.call(result, output_dir=pathPars.path_userland,
                        output_file='IDTSOSS_clear_noisy_{}'.format(noises_names[n]), save_results=False,
                        override_linearity=linearity_ref_file)
        if noises_names[n]=='darkcurrent':
            result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(result, output_dir=pathPars.path_userland,
                        output_file='IDTSOSS_clear_noisy_{}'.format(noises_names[n]), save_results=False,
                        override_dark=dark_ref_file)

        _, result = calwebb_detector1.ramp_fit_step.RampFitStep.call(result, output_dir=pathPars.path_userland,
                        output_file='IDTSOSS_clear_noisy_{}'.format(noises_names[n]), save_results=False)
        result = calwebb_detector1.gain_scale_step.GainScaleStep.call(result, output_dir=pathPars.path_userland,
                        output_file='IDTSOSS_clear_noisy_{}'.format(noises_names[n]), save_results=False)
        result.meta.filetype = 'countrate'
        result.write(calwebb_output)

        if t==0:  # Noiseless
            ### Pass noiseless simulation through DMS
            # Define input/output
            calwebb_input_nonoise = os.path.join(pathPars.path_userland, 'IDTSOSS_clear.fits')
            calwebb_output_nonoise = os.path.join(pathPars.path_userland, 'IDTSOSS_clear_noiseless_rateints.fits')

            # Step by step DMS processing
            result = calwebb_detector1.dq_init_step.DQInitStep.call(calwebb_input_nonoise,
                        output_dir=pathPars.path_userland, output_file='IDTSOSS_clear_noiseless', save_results=False)
            result = calwebb_detector1.saturation_step.SaturationStep.call(result, output_dir=pathPars.path_userland,
                        output_file='IDTSOSS_clear_noiseless', save_results=False)
            _, result = calwebb_detector1.ramp_fit_step.RampFitStep.call(result, output_dir=pathPars.path_userland,
                            output_file='IDTSOSS_clear_noiseless', save_results=False)
            result = calwebb_detector1.gain_scale_step.GainScaleStep.call(result, output_dir=pathPars.path_userland,
                            output_file='IDTSOSS_clear_noiseless', save_results=False)
            result.meta.filetype = 'countrate'
            result.write(calwebb_output_nonoise)
            t+=1


        ######################
        ### BOX EXTRACTION ###
        ######################
        # Simulations files names
        # simu_noiseless_filename = 'IDTSOSS_clear.fits'   # Noiseless
        simu_noiseless_filename = 'IDTSOSS_clear_noiseless_rateints.fits'  # Noiseless
        simu_noisy_filename = 'IDTSOSS_clear_noisy_{}_rateints.fits'.format(noises_names[n])  # Noisy

        ### NOISELESS ###

        # Load simulation
        # simu_noiseless, data_noiseless = no_dms_simulation(WORKING_DIR + simu_noiseless_filename)
        simu_noiseless, data_noiseless = rateints_dms_simulation(WORKING_DIR + simu_noiseless_filename)
        simulation_noiseless = 'noiseless'

        # Index where bad wavelengths start
        bad_wl = -5
        # Remove bad wavelengths
        data_noiseless = data_noiseless[:, :, :bad_wl]

        ng = simu_noiseless[0].header['NGROUPS']  # Number of groups
        t_read = simu_noiseless[0].header['TGROUP']  # Reading time [s]
        tint = (ng - 1) * t_read  # Integration time [s]

        # Characteristic times of transit
        # HAS TO BE MODIFIED FOR EACH MODEL TESTED
        t1, t2, t3, t4 = 53, 74, 110, 128  # [image]

        # Position of trace
        x, y, w = readtrace(os=1)
        x, y, w = x[:bad_wl], y[:bad_wl], w[:bad_wl]

        # BOX EXTRACTION
        print('Extraction: noiseless')
        # To save it:
        fbox_noiseless = np.zeros(shape=(np.shape(data_noiseless)[0], np.shape(data_noiseless)[2]), dtype=float)
        for t in range(np.shape(data_noiseless)[0]):  # For each image of the timeseries
            fbox_noiseless[t] = flambda_adu(x, data_noiseless[t], y, radius_pixel=radius_pixel)  # [adu/s]
        f_array_noiseless = np.nan_to_num(fbox_noiseless)
        # Normalization of flux
        f_array_noiseless_norm = normalization(f_array_noiseless, t1, t4)

        # Time array
        time = np.arange(f_array_noiseless.shape[0])
        time_min = time * tint / 60.  # [min]

        # WHITE LIGHT CURVE
        f_white_noiseless = np.sum(f_array_noiseless, axis=1)
        # Normalize white light curve
        f_white_noiseless_norm = normalization(f_white_noiseless, t1, t4)

        ### NOISY ###

        # Load simulation
        simu_noisy, data_noisy = rateints_dms_simulation(WORKING_DIR + simu_noisy_filename)
        simulation_noisy = noises_names[n]

        # Remove bad wavelengths
        data_noisy = data_noisy[:, :, :bad_wl]

        ng = simu_noisy[0].header['NGROUPS']  # Number of groups
        t_read = simu_noisy[0].header['TGROUP']  # Reading time [s]
        tint = (ng - 1) * t_read  # Integration time [s]

        # BOX EXTRACTION
        print('Extraction: noisy')
        # To save it:
        fbox_noisy = np.zeros(shape=(np.shape(data_noisy)[0], np.shape(data_noisy)[2]), dtype=float)
        for t in range(np.shape(data_noisy)[0]):  # For each image of the timeseries
            fbox_noisy[t] = flambda_adu(x, data_noisy[t], y, radius_pixel=radius_pixel)  # [adu/s]
        f_array_noisy = np.nan_to_num(fbox_noisy)
        # Normalization of flux
        f_array_noisy_norm = normalization(f_array_noisy, t1, t4)

        # Time array
        time = np.arange(f_array_noisy.shape[0])
        time_min = time * tint / 60.  # [min]

        # WHITE LIGHT CURVE
        f_white_noisy = np.sum(f_array_noisy, axis=1)
        # Normalize white light curve
        f_white_noisy_norm = normalization(f_white_noisy, t1, t4)

        ### ANALYSIS ###

        # Graphic of white light curves
        plt.figure()
        plt.plot(time_min, f_white_noisy_norm, '.', markersize=4, color='r', label=simulation_noisy)
        plt.plot(time_min, f_white_noiseless_norm, '.', markersize=4, color='b', label=simulation_noiseless)
        plt.xlabel('Time [min]')
        plt.ylabel('Flux')
        plt.title('White light')
        plt.legend()

        # DISPERSION/PHOTON NOISE RATIO
        # Only the uncontaminated portion of the spectrum is used here
        i_uncont = 1100  # Start of the uncontaminated portion
        # New arrays
        new_w = w[i_uncont:]
        new_f_array_noiseless = f_array_noiseless[:, i_uncont:]
        new_f_array_noisy = f_array_noisy[:, i_uncont:]
        new_f_array_noisy_norm = f_array_noisy_norm[:, i_uncont:]

        gain = 1.61   # Gain [e⁻]
        # To store data:
        photon_noise = np.zeros(new_f_array_noiseless.shape[1], dtype='float')
        dispersion = np.zeros(new_f_array_noisy.shape[1], dtype='float')
        for n in range(new_f_array_noisy.shape[1]):   # for each wavelength
            # Noiseless out of transit spectrum
            out_transit_noiseless = np.concatenate((new_f_array_noiseless[:t1, n], new_f_array_noiseless[t4:, n]))
            # Noisy out of transit spectrum
            out_transit_noisy = np.concatenate((new_f_array_noisy_norm[:t1, n], new_f_array_noisy_norm[t4:, n]))
            # Conversion in electrons (e⁻) for noiseless
            out_transit_noiseless_elec = out_transit_noiseless * tint * gain
            # Photon noise (Poisson) (e⁻)
            photon_noise_elec = np.sqrt(np.mean(out_transit_noiseless_elec))
            # Photon noise (adu/s)
            photon_noise[n] = photon_noise_elec / gain / tint
            # Dispersion in noisy data
            dispersion[n] = np.std(out_transit_noisy)
        ratio = dispersion / photon_noise

        # Plot ratio between dispersion and photon noise vs wavelength
        plt.figure()
        plt.plot(new_w, ratio, '.', color='b', label=simulation_noisy)
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel('Dispersion/Photon noise')
        plt.legend()

        print("Mean dispersion over photon noise ratio =", np.mean(ratio))

        ### RELATIVE DIFFERENCE
        # Between white light curves
        relatDiff_white = relative_difference(f_white_noisy, f_white_noiseless)

        plt.figure()
        plt.plot(time_min, relatDiff_white * 1.0e6, color='b')
        plt.xlabel('Time [min]')
        plt.ylabel('Relative difference [ppm]')
        plt.title('Relative difference between {} and \n{}'.format(simulation_noisy, simulation_noiseless))

if doShow_plots is True: plt.show()


