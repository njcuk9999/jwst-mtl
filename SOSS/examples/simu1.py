# Example script to:
# - create a simple simulation
# - calibrate its flux
# - add detector noise
# - run it throught the DMS pipeline
# - extract the spectra

import sys
sys.path.insert(0, '/genesis/jwst/jwst-ref-soss/fortran_lib/')

# TODO: Update all paths

import glob

import os
# os.environ["OMP_NUM_THREADS"] = "24"
import numpy as np

# Will be used to create FITS writer
from astropy.io import fits
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

from jwst.pipeline import Detector1Pipeline

verbose = True

###############################################################################
# Start of the flow
###############################################################################

# Read in all paths used to locate reference files and directories
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/'
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_kim.txt')

pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Make the libraries imports. This needs to come after read_config_path
# because the system needs to have the path to our own libraries first.
#soss.init_imports()

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              #Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars) #read in parameter file

print(pathPars.path_planetmodelatm+simuPars.pmodelfile[0])
print(simuPars.pmodeltype[0])

# Here one can manually edit the parameters but we encourage rather to change
# the simulation parameter file directly.
simuPars.noversample = 2  #example of changing a model parameter
#simuPars.xout = 4000      #spectral axis
#simuPars.yout = 300       #spatial (cross-dispersed axis)
#simuPars.modelfile = 'CONSTANT_FNU'
#simuPars.modelfile = 'BLACKBODY'
#simuPars.modelfile = 't6000g450p000_ldnl.dat'
#simuPars.modelfile = 'lte06900-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011.JWST-RF.simu-SOSS.dat'
simuPars.f277wcal = False
simuPars.flatthroughput = True
simuPars.flatquantumyield = True


# Instrument Throughput (Response)
throughput = spgen.read_response(pathPars.throughputfile, set_response_to_unity=simuPars.flatthroughput,
                                 set_qy_to_unity=simuPars.flatquantumyield, verbose=verbose)

# Set up Trace (Position vs. Wavelength)
tracePars = tp.get_tracepars(pathPars.tracefile)

# Generate or read the star atmosphere model
starmodel_angstrom, starmodel_flambda, ld_coeff = soss.starmodel(simuPars, pathPars)

# Anchor star spectrum on a photometric band magnitude
starmodel_flambda = smag.anchor_spectrum(starmodel_angstrom/10000., starmodel_flambda, simuPars.filter,
                                    simuPars.magnitude, pathPars.path_filtertransmission, verbose=True)

# Read Planet Atmosphere Model (wavelength in angstroms and radius_planet/radius_star ratio)
planetmodel_angstrom, planetmodel_rprs = spgen.readplanetmodel(pathPars.path_planetmodelatm+simuPars.pmodelfile[0],
                                                               simuPars.pmodeltype[0])

# Resample the star and planet models on a common grid
if False:
    resampled_arrays = soss.resample_models(starmodel_angstrom, starmodel_flambda, ld_coeff,
                                        planetmodel_angstrom, planetmodel_rprs, 5000, 55000, resolving_power = 100000)
    bin_starmodel_angstrom,  = resampled_arrays


# Generate the time steps array
tintopen, frametime, nint, timesteps = soss.generate_timesteps(simuPars)
print('Generated time steps:')
print('nint={:} nsteps={:} frametime={:}'.format(\
    nint, len(timesteps), frametime))
print('Generated time steps (in seconds): ', timesteps)

# Generate the Time-Series simulation
norders = len(simuPars.orderlist)
dimy, dimx = simuPars.yout, simuPars.xout
print('norders={:} dimy={:} dimx={:}'.format(norders,dimy,dimx))
# For each time step, a cube of simulated images is written to disk
# The cube has all spectral orders in separate slices.
# The list of such fits cube file names is returned.
if True:
    imagelist = soss.generate_traces(WORKING_DIR + 'tmp/oversampling_{}/clear'.format(simuPars.noversample),
                                     pathPars, simuPars, tracePars, throughput, starmodel_angstrom,
                                     starmodel_flambda, ld_coeff, planetmodel_angstrom, planetmodel_rprs,
                                     timesteps, tintopen)
else:
    SIMUDIR = '/home/kmorel/ongenesis/jwst-user-soss/tmp/'
    imagelist = glob.glob(WORKING_DIR + 'tmp/oversampling_{}/clear*.fits'.format(simuPars.noversample))
    #imagelist = os.listdir(SIMUDIR)
    for i in range(np.size(imagelist)):
        imagelist[i] = os.path.join(SIMUDIR + 'oversampling_{}/'.format(simuPars.noversample),imagelist[i])
    print(imagelist)


# Here, a single out-of-transit simulation is used to establish the
# normalization scale needed to flux calibrate our simulations.
# To override the simpar parameters:
# simuPars.filter = 'J'
# simuPars.magnitude = 8.5
expected_counts = smag.expected_flux_calibration(
                    simuPars.filter, simuPars.magnitude,
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
print('Normalization scales = ',normalization_scale)

# All simulations are normalized, all orders summed and gathered in a single array
# with the 3rd dimension being the number of time steps.
data = soss.write_dmsready_fits_init(imagelist, normalization_scale,
                                     simuPars.ngroup, simuPars.nint,
                                     simuPars.frametime, simuPars.granularity)

# All simulations (e-/sec) are converted to up-the-ramp images.
#soss.write_dmsready_fits(data[:,:,0:256,0:2048], os.path.join(WORKING_DIR,'test_clear.fits'),
                    #os=simuPars.noversample, input_frame='sim')
soss.write_dmsready_fits(data, os.path.join(WORKING_DIR + 'oversampling_{}/'.format(simuPars.noversample),
                                            'test_clear.fits'), os=simuPars.noversample, input_frame='dms',
                         xpadding=simuPars.xpadding, ypadding=simuPars.ypadding)

# Add detector noise to the noiseless data
detector.add_noise(os.path.join(WORKING_DIR + 'oversampling_{}/'.format(simuPars.noversample),'test_clear.fits'),
                   outputfilename=os.path.join(WORKING_DIR + 'oversampling_{}/'.format(simuPars.noversample),
                                               'test_clear_noisy.fits'))

# Process the data through the DMS level 1 pipeline
result = Detector1Pipeline.call(os.path.join(WORKING_DIR + 'oversampling_{}/'.format(simuPars.noversample),
                                             'test_clear_noisy.fits'), output_file='test_clear_noisy',
                                output_dir=WORKING_DIR + 'oversampling_{}/'.format(simuPars.noversample))


"""
SIMULATE THE F277W CALIBRATION EXPOSURE OBTAINED AFTER THE GR700XD EXPOSURE
- throughput needs to be rest differently
- star model: no change
- planet model: no change
- determine time steps for F277W short time series
- Apply normalization_scale * it
- Convert to up-the-ramp images, store on disk
- Add detector noise
- Process the data through DMS
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
        imagelist_f277 = soss.generate_traces(WORKING_DIR+'tmp/f277', pathPars, simuPars, tracePars,
                                              throughput_f277, starmodel_angstrom, starmodel_flambda,
                                              ld_coeff, planetmodel_angstrom, planetmodel_rprs,
                                              timesteps_f277, tintopen)
    else:
        SIMUDIR = '/genesis/jwst/userland-soss/loic_review/tmp/'
        imagelist_f277 = glob.glob(WORKING_DIR + 'tmp/f277*.fits')
        #imagelist_f277 = os.listdir(SIMUDIR)
        for i in range(np.size(imagelist_f277)):
            imagelist_f277[i] = os.path.join(SIMUDIR,imagelist_f277[i])
        print(imagelist_f277)

    # All simulations are normalized, all orders summed and gathered in a single array
    # with the 3rd dimension being the number of time steps.
    data_f277 = soss.write_dmsready_fits_init(imagelist_f277, normalization_scale,
                                              simuPars.ngroup, simuPars.nintf277,
                                              simuPars.frametime, simuPars.granularity, verbose=True)

    # All simulations (e-/sec) are converted to up-the-ramp images.
    soss.write_dmsready_fits(data_f277, os.path.join(WORKING_DIR,'test_f277.fits'),
                        os=simuPars.noversample, input_frame='dms', f277=True)

    # Add detector noise to the noiseless data
    detector.add_noise(os.path.join(WORKING_DIR,'test_f277.fits'),
                       outputfilename=os.path.join(WORKING_DIR, 'test_f277_noisy.fits'))

    # Process the data through the DMS level 1 pipeline
    result = Detector1Pipeline.call(os.path.join(WORKING_DIR, 'test_f277_noisy.fits'),
                                    output_file='test_f277_noisy', output_dir=WORKING_DIR)

print('The end of the simulation')

