# Example script to:
# - create a simple simulation
# - calibrate its flux
# - add detector noise
# By L. Albert

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

###############################################################################
# Start of the flow
###############################################################################

# Read in all paths used to locate reference files and directories
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/PHY3030/WASP_52/'
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath.txt')

pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)


# Make the libraries imports. This needs to come after read_config_path
# because the system needs to have the path to our own libraries first.
#soss.init_imports()


# Create and read the simulation parameters
simuPars = spgen.ModelPars() #Set up default parameters
simuPars.read_pars(pathPars.simulationparamfile) #read in parameter file

print(pathPars.path_planetmodelatm+simuPars.pmodelfile[0])
print(simuPars.pmodeltype[0])

# Here one can manually edit the parameters (by changing False for True below)
# but we encourage to rather change the simulation parameter file directly.
simuPars.f277wcal = False

# Save the input parameters to disk
simuPars.save_params(os.path.join(pathPars.path_userland,'IDTSOSS_inputs.txt'))

# For testing, allow skipping steps.
skip_sim = False
skip_addnoise = False


if skip_sim is False:

    # Instrument Throughput (Response)
    throughput = spgen.read_response(pathPars.throughputfile, set_response_to_unity=simuPars.flatthroughput,
                                     set_qy_to_unity=simuPars.flatquantumyield, verbose=verbose)

    # Set up Trace (Position vs. Wavelength)
    tracePars = tp.get_tracepars(pathPars.tracefile)

    # Generate or read the star atmosphere model
    starmodel_angstrom, starmodel_flambda, ld_coeff = soss.starmodel(simuPars, pathPars, tracePars, throughput)

    #plt.figure()
    #plt.plot(starmodel_angstrom, starmodel_flambda)
    #plt.show()
    #sys.exit()

    # Anchor star spectrum on a photometric band magnitude
    starmodel_flambda = smag.anchor_spectrum(starmodel_angstrom/10000., starmodel_flambda, simuPars.filtername,
                                        simuPars.magnitude, pathPars.path_filtertransmission, verbose=True)

    if False:
        plt.figure()
        plt.plot(starmodel_angstrom, starmodel_flambda)
        #plt.plot(throughput.wv, throughput.response[1])
        plt.show()

        t = Table([starmodel_angstrom/10000, starmodel_flambda], names=('micron', 'W/m2/um'))
        ascii.write(t, os.path.join(pathPars.path_userland, 'IDTSOSS_star_model_calibrated.txt'))
        sys.exit()

    # Read Planet Atmosphere Model (wavelength in angstroms and radius_planet/radius_star ratio)
    if False:
        planetmodel_angstrom, planetmodel_rprs = spgen.readplanetmodel(pathPars.path_planetmodelatm+simuPars.pmodelfile[0],
                                                                       simuPars.pmodeltype[0])
    else:
        planetmodel_angstrom, planetmodel_rprs = soss.readplanetmodel(
            os.path.join(pathPars.path_planetmodelatm, simuPars.pmodelfile[0]))

    if True:
        # Generate the time steps array
        tintopen, frametime, nint, timesteps = soss.generate_timesteps(simuPars)
        print('Generated time steps:')
        print('nint={:} nsteps={:} frametime={:}'.format(\
            nint, len(timesteps), frametime))
        print('Generated time steps (in seconds): ', timesteps)

        # Generate a constant trace position offsets
        specpix_offset = 0.0
        spatpix_offset = 0.0 #5.0
        # Generate a time-dependent trace position offsets
        #specpix_offset = np.zeros_like(timesteps)
        #spatpix_offset = np.linspace(0.0, 5.0, np.size(timesteps))
        print('Trace position offsets (as a function of time, or constant):')
        print('specpix_offset = ', specpix_offset)
        print('spatpix_offset = ', spatpix_offset)

        # Generate the Time-Series simulation
        norders = len(simuPars.orderlist)
        dimy, dimx = simuPars.yout, simuPars.xout
        print('norders={:} dimy={:} dimx={:}'.format(norders,dimy,dimx))
        # For each time step, a cube of simulated images is written to disk
        # The cube has all spectral orders in separate slices.
        # The list of such fits cube file names is returned.
        if True:
            imagelist = soss.generate_traces(pathPars.path_userland+'tmp/clear',
                                             pathPars, simuPars, tracePars, throughput,
                                             starmodel_angstrom, starmodel_flambda, ld_coeff,
                                             planetmodel_angstrom, planetmodel_rprs,
                                             timesteps, tintopen, specpix_trace_offset=specpix_offset,
                                             spatpix_trace_offset=spatpix_offset)
        else:
            imagelist = glob.glob(pathPars.path_userland+'tmp/clear*.fits')
            for i in range(np.size(imagelist)):
                imagelist[i] = os.path.join(pathPars.path_userland+'tmp/',imagelist[i])
            print(imagelist)

        # Here, a single out-of-transit simulation is used to establish the
        # normalization scale needed to flux calibrate our simulations.
        expected_counts = smag.expected_flux_calibration(
                            simuPars.filtername, simuPars.magnitude,
                            starmodel_angstrom, starmodel_flambda,
                            simuPars.orderlist, subarray=simuPars.subarray,
                            verbose=False,
                            trace_file=pathPars.tracefile,
                            response_file=pathPars.throughputfile,
                            pathfilter=pathPars.path_filtertransmission,
                            pathvega=pathPars.path_filtertransmission)
        # Measure the actual counts on only the first (out-of-transit) simulated image
        simulated_counts = smag.measure_actual_flux(imagelist[0], xbounds=[0,2048], ybounds=[0,256],
                                noversample=simuPars.noversample)
        # Prints the expected/measured counts
        for i in range(np.size(imagelist)):
            print(i, expected_counts, simulated_counts)
        # Apply flux scaling
        normalization_scale = expected_counts / simulated_counts
        # Handle case where no flux is expected (e.g. order 3 in substrip96 subarray)
        for i in range(len(expected_counts)):
            if expected_counts[i] == 0: normalization_scale[i] = 0
        print('Normalization scales = ',normalization_scale)

        # All simulations are normalized, all orders summed and gathered in a single array
        # with the 3rd dimension being the number of time steps.
        if True:
            data = soss.write_dmsready_fits_init(imagelist, normalization_scale,
                                             simuPars.ngroup, simuPars.nint,
                                             simuPars.frametime, simuPars.granularity)

        # All simulations (e-/sec) are converted to up-the-ramp images.
        if True:
            soss.write_dmsready_fits(data, os.path.join(pathPars.path_userland,'IDTSOSS_clear.fits'),
                                 os=simuPars.noversample, input_frame='dms',
                                 xpadding=simuPars.xpadding, ypadding=simuPars.ypadding,
                                 dark_ref=simuPars.dark_ref, flat_ref=simuPars.flat_ref,
                                 superbias_ref=simuPars.superbias_ref, nlcoeff_ref=simuPars.nlcoeff_ref,
                                 zodi_ref=simuPars.zodi_ref, nonlin_ref=simuPars.nonlin_ref,
                                 readout=simuPars.readout, zodibackg=simuPars.zodibackg,
                                 photon=simuPars.photon, superbias=simuPars.superbias,
                                 flatfield=simuPars.flatfield, nonlinearity=simuPars.nonlinearity,
                                 oneoverf=simuPars.oneoverf, darkcurrent=simuPars.darkcurrent,
                                 cosmicray=simuPars.cosmicray)


    """
    SIMULATE THE F277W CALIBRATION EXPOSURE OBTAINED AFTER THE GR700XD EXPOSURE
    - throughput needs to be rest differently
    - star model: no change
    - planet model: no change
    - determine time steps for F277W short time series
    - Apply normalization_scale * it
    - Convert to up-the-ramp images, store on disk
    - Add detector noise
    """
    if simuPars.f277wcal is True:

        # Get the throughput with the F277W filter in place of the CLEAR
        # throughput_f277 = smag.throughput_withF277W(throughput, pathPars.path_filtertransmission)
        # Instrument Throughput (Response)
        throughput_f277 = spgen.read_response(pathPars.throughputfile, f277w=True,
                                              path_filter_transmission=pathPars.path_filtertransmission,
                                              verbose=verbose)

        # Generate the time steps array
        tintopen, frametime, nint_f277, timesteps_f277 = soss.generate_timesteps(simuPars, f277=True)
        print('F277W calibration generated time steps:')
        print('nint_f277={:} nsteps={:} frametime={:}'.format(\
            nint_f277, len(timesteps_f277), frametime))
        print('F277W calibration generated time steps (in seconds): ', timesteps_f277)

        if True:
            imagelist_f277 = soss.generate_traces(pathPars.path_userland+'tmp/f277',
                                                  pathPars, simuPars, tracePars, throughput_f277,
                                                  starmodel_angstrom, starmodel_flambda, ld_coeff,
                                                  planetmodel_angstrom, planetmodel_rprs,
                                                  timesteps_f277, tintopen)
        else:
            imagelist_f277 = glob.glob(pathPars.path_userland+'tmp/f277*.fits')
            for i in range(np.size(imagelist_f277)):
                imagelist_f277[i] = os.path.join(pathPars.path_userland+'tmp/',imagelist_f277[i])
            print(imagelist_f277)

        # All simulations are normalized, all orders summed and gathered in a single array
        # with the 3rd dimension being the number of time steps.
        data_f277 = soss.write_dmsready_fits_init(imagelist_f277, normalization_scale,
                                                  simuPars.ngroup, simuPars.nintf277,
                                                  simuPars.frametime, simuPars.granularity, verbose=True)

        # All simulations (e-/sec) are converted to up-the-ramp images.
        soss.write_dmsready_fits(data_f277, os.path.join(pathPars.path_userland,'IDTSOSS_f277.fits'),
                                 os=simuPars.noversample, input_frame='dms',
                                 xpadding=simuPars.xpadding, ypadding=simuPars.ypadding, f277=True,
                                 dark_ref=simuPars.dark_ref, flat_ref=simuPars.flat_ref,
                                 superbias_ref=simuPars.superbias_ref, nlcoeff_ref=simuPars.nlcoeff_ref,
                                 zodi_ref=simuPars.zodi_ref, nonlin_ref=simuPars.nonlin_ref,
                                 readout=simuPars.readout, zodibackg=simuPars.zodibackg,
                                 photon=simuPars.photon, superbias=simuPars.superbias,
                                 flatfield=simuPars.flatfield, nonlinearity=simuPars.nonlinearity,
                                 oneoverf=simuPars.oneoverf, darkcurrent=simuPars.darkcurrent,
                                 cosmicray=simuPars.cosmicray)

    print('The end of the noiseless simulation')
    print()


'''
ADD NOISE
'''
if skip_addnoise is False:
    # GR700XD+CLEAR - ADD NOISE
    # Add detector noise to the noiseless data
    detector.add_noise(os.path.join(pathPars.path_userland, 'IDTSOSS_clear.fits'),
                       pathPars.path_noisefiles,
                       outputfilename=os.path.join(pathPars.path_userland, 'IDTSOSS_clear_noisy.fits'),
                       dark_ref=simuPars.dark_ref, flat_ref=simuPars.flat_ref,
                       superbias_ref=simuPars.superbias_ref, nlcoeff_ref=simuPars.nlcoeff_ref,
                       zodi_ref=simuPars.zodi_ref,
                       readout=simuPars.readout, zodibackg=simuPars.zodibackg,
                       photon=simuPars.photon, superbias=simuPars.superbias,
                       flatfield=simuPars.flatfield, nonlinearity=simuPars.nonlinearity,
                       oneoverf=simuPars.oneoverf, darkcurrent=simuPars.darkcurrent,
                       cosmicray=simuPars.cosmicray)

    # F277W - ADD NOISE
    if simuPars.f277wcal is True:
        # Add detector noise to the noiseless data
        detector.add_noise(os.path.join(pathPars.path_userland, 'IDTSOSS_f277.fits'),
                           pathPars.path_noisefiles,
                           outputfilename=os.path.join(pathPars.path_userland, 'IDTSOSS_f277_noisy.fits'),
                           dark_ref=simuPars.dark_ref, flat_ref=simuPars.flat_ref,
                           superbias_ref=simuPars.superbias_ref, nlcoeff_ref=simuPars.nlcoeff_ref,
                           zodi_ref=simuPars.zodi_ref,
                           readout=simuPars.readout, zodibackg=simuPars.zodibackg,
                           photon=simuPars.photon, superbias=simuPars.superbias,
                           flatfield=simuPars.flatfield, nonlinearity=simuPars.nonlinearity,
                           oneoverf=simuPars.oneoverf, darkcurrent=simuPars.darkcurrent,
                           cosmicray=simuPars.cosmicray)