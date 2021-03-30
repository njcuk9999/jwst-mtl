# Example script to:
# - create a simple simulation
# - calibrate its flux
# - add detector noise
# - run it throught the DMS pipeline
# - extract the spectra

import sys
sys.path.insert(0, '/genesis/jwst/jwst-ref-soss/fortran_lib/')

# TODO: Update all paths
WORKING_DIR = '/genesis/jwst/jwst-user-soss/loic_review/'

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
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath.txt')
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
#simuPars.noversample = 1  #example of changing a model parameter
#simuPars.xout = 4000      #spectral axis
#simuPars.yout = 300       #spatial (cross-dispersed axis)


if False:
    #result = Detector1Pipeline.call(os.path.join(WORKING_DIR, 'caca_noisy.fits'),
    #                                output_file='toto', output_dir=WORKING_DIR)
    WORKING_DIR = '/genesis/jwst/userland-soss/loic_review/full_20210319/'
    result = Detector1Pipeline.call(os.path.join(WORKING_DIR, 'simu_noisy.fits'),
                                output_file='simu_noisy', output_dir=WORKING_DIR)

    #detector.add_noise([os.path.join(WORKING_DIR, 'test.fits')],
    #                   outputfilename=os.path.join(WORKING_DIR, 'test_noisy.fits'))
    #result = Detector1Pipeline.call(os.path.join(WORKING_DIR, 'test_noisy.fits'))
    #detector.add_noise(os.path.join(WORKING_DIR, 'test.fits')
    #from jwst.pipeline import Detector1Pipeline
    #result = Detector1Pipeline.call(os.path.join(WORKING_DIR, 'test_mod_poisson_noise_zodibackg_flat_dark_nonlin_bias_detector.fits'))
    print('Exiting the test...')
    sys.exit()

# Instrument Throughput (Response)
throughput = spgen.read_response(pathPars.throughputfile, verbose=verbose)

# Set up Trace (Position vs. Wavelength)
tracePars = tp.get_tracepars(pathPars.tracefile)

# Read Stellar Atmosphere Model (wavelength in angstrom and flux in energy/sec/wavelength)
starmodel_angstrom, starmodel_flambda, ld_coeff = spgen.readstarmodel(pathPars.path_starmodelatm+simuPars.modelfile,
                                                                     simuPars.nmodeltype, quiet=False)

# Read Planet Atmosphere Model (wavelength in angstroms and radius_planet/radius_star ratio)
planetmodel_angstrom, planetmodel_rprs = spgen.readplanetmodel(pathPars.path_planetmodelatm+simuPars.pmodelfile[0],
                                                               simuPars.pmodeltype[0])

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
if False:
    imagelist = soss.generate_traces(pathPars, simuPars, tracePars, throughput,
                                   starmodel_angstrom, starmodel_flambda, ld_coeff,
                                   planetmodel_angstrom, planetmodel_rprs,
                                   timesteps, tintopen)
else:
    SIMUDIR = '/genesis/jwst/userland-soss/loic_review/tmp/'
    imagelist = os.listdir(SIMUDIR)
    for i in range(np.size(imagelist)):
        imagelist[i] = os.path.join(SIMUDIR,imagelist[i])
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
# Measure the actual counts on the simulated images
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
data = soss.write_dmsready_fits_init(imagelist, normalization_scale, simuPars)

# All simulations (e-/sec) are converted to up-the-ramp images.
soss.write_dmsready_fits(data[:,:,0:256,0:2048], os.path.join(WORKING_DIR,'test.fits'),
                    os=simuPars.noversample, input_frame='sim')

# Add detector noise to the noiseless data
detector.add_noise(os.path.join(WORKING_DIR,'test.fits'),
                   outputfilename=os.path.join(WORKING_DIR, 'test_noisy.fits'))

# Process the data through the DMS level 1 pipeline
result = Detector1Pipeline.call(os.path.join(WORKING_DIR, 'test_noisy.fits'),
                                output_file='test_noisy', output_dir=WORKING_DIR)

