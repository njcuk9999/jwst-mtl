# Example script to:
# - create a simple simulation
# - calibrate its flux
# - add detector noise
# - run it throught the DMS pipeline
# - extract the spectra




############################# SETUP ##################################
### Only need to change these to 'fully' customize code execution ####
######################################################################

    # Location of the jwst-mtl github repo. Overrides the github path in
    # 'jwst_config_fpath', during the import phase
github_path = '/home/kmorel/ongenesis/github/jwst-mtl/SOSS/'
    # Location of the simulation config file, as well as the output directory
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/PHY3030/'
    # Configuration file for the NIRISS Instrument Team SOSS simulation pipeline
jwst_config_fpath = 'jwst-mtl_configpath_kim.txt'

    # Choose whether to generate the simulation from scratch 
generate_clear_tmp_simu = False     # True
    # Optional override of the amount of integrations in the exposure.
    # By default, the amount is determined by the maximum amount of integrations
    # that can be fit into the observation time (given the detector readout array size).
    # If less integrations are advantageous, then specifying this parameter is the way to go
nIntegrations_override = 300
#nIntegrations_override = None

    # Choose whether to format the generated (clear) simulation in a format
    # processable by the calwebb_detector1 pipeline (DMS)
generate_clear_dmsReady_simu = False    # True

    # Choose whether to add detector noise to the noiseless data
add_noise_to_dmsReady_simu = True

    # Choose whether to inject select noise types in the simulation, instead of the default ones
investigate_noise = True
    # If 'investigate_noise' is True, this 2D-list determines how many different 
    # noisy simulations are generated from clear exposures (length of 1st dimension),
    # as well as what noise types are injected into them (list contained in 2nd dimension).
noise_shopping_lists = [[]
                       ,['photon']
                       #,['normalize']
                       #,['zodibackg']
                       #,['flatfield']
                       #,['darkframe']
                       ,['nonlinearity']
                       ,['superbias']
                       #,['detector']
                       ,['photon', 'nonlinearity', 'superbias']
                       ]
override_noise_files = True
ov_noiz_dir = '/genesis/jwst/jwst-ref-soss/noise_files/'
ov_noiz = { 'flat'  : ov_noiz_dir+'jwst_niriss_flat_0181.fits'               if override_noise_files else None,
            'zodi'  : ov_noiz_dir+'background_detectorfield_normalized.fits' if override_noise_files else None,
            'bias'  : ov_noiz_dir+'jwst_niriss_superbias_0137.fits'          if override_noise_files else None,
            #'liner' : ov_noiz_dir+'jwst_niriss_linearity_0011.fits'          if override_noise_files else None,
            'liner' : ov_noiz_dir+
                       'jwst_niriss_linearity_0011_bounds_0_60000_npoints_100_deg_5.fits' if override_noise_files else None
          }

    # Choose whether to process the data through the DMS level 1 pipeline
run_dms_level_one = True
dms_cfg_files_path = WORKING_DIR+'config_files/'
dms_cfg_files = [None] * len(noise_shopping_lists)  # if None, use the auto-generated one from the
                                                    # 'Generate_calwebb_config_files' notebook

extra_output_dir = ''


    # Some other self-explanatory boolean variables
    # They determine if stuff is printed or plotted at runtime
doPlot = False
doPrint = True
verbose = doPrint






##############################
########## IMPORTS ###########
##############################

import sys
sys.path.insert(1, '/genesis/jwst/jwst-ref-soss/fortran_lib/')

# TODO: Update all paths

import glob

import os
# os.environ["OMP_NUM_THREADS"] = "24"

import numpy as np

# Plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# In a Jupyter notebook, you may add this line
# get_ipython().run_line_magic('matplotlib', 'inline')

# Will be used to create FITS writer
from astropy.io import fits
from astropy.io import ascii

# To bin the oversampled images down to its native pixel size
from skimage.transform import downscale_local_mean, resize

# To produce progress bars when generating simulations
from tqdm.notebook import trange
from tqdm.notebook import tqdm as tqdm_notebook

import pyfftw

import multiprocessing as mp

import json as json

import scipy.fft

sys.path.insert(2, github_path)
# Python Routines for SpecGen Routines and wrappers for fast-Transit-model.
import specgen.spgen as spgen
# Trace Library
import trace.tracepol as tp
# Header and FITS writing function
# Detector noise script
from detector import detector_Frost as detector
# normalization code
import specgen.synthesizeMagnitude as smag
# Loic-reviewed NIRISS-SOSS simulation functions
import itsosspipeline as soss

ncpu = mp.cpu_count()
pyfftw.config.NUM_THREADS = ncpu

from jwst.pipeline import Detector1Pipeline




########################################
########## START OF THE FLOW ###########
########################################
############# SETUP PHASE ##############
########################################

    # Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR, jwst_config_fpath)
#config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_runkim.txt')

if doPrint: print()

pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

    # Make the libraries imports. This needs to come after read_config_path
    # because the system needs to have the path to our own libraries first.
#soss.init_imports()

    # Create and read the simulation parameters
simuPars = spgen.ModelPars()                                     #  Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars) #  Read in parameter file

    # Here one can manually edit the parameters but we encourage rather to change
    # the simulation parameter file directly.
#simuPars.noversample = 4  #example of changing a model parameter
#simuPars.xout = 2048      #spectral axis
#simuPars.yout = 256       #spatial (cross-dispersed axis)
#simuPars.xpadding = 100
#simuPars.ypadding = 100
#simuPars.modelfile = 'BLACKBODY'
#simuPars.modelfile = 't6000g450p000_ldnl.dat'
#simuPars.modelfile = 'lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011.JWST-RF.simu-SOSS.dat'
#simuPars.modelfile = 'lte02300-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011.JWST-RF.simu-SOSS.dat'
#simuPars.modelfile = 'CONSTANT_FLAMBDA'
#simuPars.modelfile = 'CONSTANT_ADU'
#simuPars.f277wcal = True
#simuPars.f277wcal = False
#simuPars.tend = -1.80
#simuPars.tend = -1.97
#simuPars.flatthroughput = False
#simuPars.flatquantumyield = True

    # Save the input parameters to disk
    # TODO: Implement a working version for saving simulation input parameters
#filename = pathPars.path_userland+'tmp/'+'input_parameters.json'
#handle = open(filename, 'w')
#json.dump(simuPars, handle)
#handle.close()



if generate_clear_tmp_simu or generate_clear_dmsReady_simu is True:

    
        # Instrument Throughput (Response)
    throughput = spgen.read_response(pathPars.throughputfile, set_response_to_unity=simuPars.flatthroughput,
                                     set_qy_to_unity=simuPars.flatquantumyield, verbose=verbose)

        # Set up Trace (Position vs. Wavelength)
    tracePars = tp.get_tracepars(pathPars.tracefile)

        # Generate or read the star atmosphere model
    starmodel_angstrom, starmodel_flambda, ld_coeff = soss.starmodel(simuPars, pathPars, tracePars, throughput)

        # Anchor star spectrum on a photometric band magnitude
    starmodel_flambda = smag.anchor_spectrum(starmodel_angstrom/10000., starmodel_flambda, simuPars.filter,
                                        simuPars.magnitude, pathPars.path_filtertransmission, verbose=True)


        # Read Planet Atmosphere Model
        # Caroline Piaulet and Benneke group planet model functionalities
        # Path for the model grid
    path_files = pathPars.path_planetmodelatm+"FwdRuns20210521_0.3_100.0_64_nLay60/"
    # planet_name = 'FwdRuns20210521_0.3_100.0_64_nLay60/HAT_P_1_b'
    planet_name = 'HAT_P_1_b'
        # Get a list of all parameters available
    planet_caselist = soss.get_atmosphere_cases(planet_name, path_files=path_files,
                                                return_caselist=True, print_info=doPrint)
        # select params that you want
    params_dict = soss.make_default_params_dict()
    params_dict["CtoO"] = 0.3
    # print(params_dict)
        # Get path to csv file that contains the model spectrum
    path_csv = soss.get_spec_csv_path(caselist=planet_caselist, params_dict=params_dict,
                                    planet_name=planet_name, path_files=path_files)
    if doPrint:
        t = ascii.read(path_csv)
        print("\nSpectrum file:")
        print(t)
    # Wavelength in angstroms
    planetmodel_angstrom = np.array(t['wave']) * 1e+4
        # Rp/Rstar from depth in ppm
    planetmodel_rprs = np.sqrt(np.array(t['dppm'])/1e+6)


    if doPlot:
        plt.figure(figsize=(18,6))
        plt.subplot(1,3,1); plt.plot(starmodel_angstrom, starmodel_flambda); plt.title('Star')
        plt.subplot(1,3,2); plt.plot(planetmodel_angstrom, planetmodel_rprs); plt.title('Planet')
        plt.subplot(1,3,3); plt.plot(throughput.wv, throughput.response[1]); plt.title('Throughput')
        plt.show()


        # Generate the time steps array
    tintopen, frametime, nint, timesteps = soss.generate_timesteps(simuPars)
        # Override the number of integrations while keeping the same elapsed time
        # specified in simuPars (optional)
    if nIntegrations_override != None:
        timeskip = int( len(timesteps)/nIntegrations_override )
        timesteps = timesteps[::timeskip]
        nint = len(timesteps); simuPars.nint = nint
    if doPrint:
        print('\nGenerated time steps:')
        print('nint={:} nsteps={:} frametime={:}'.format(\
            nint, len(timesteps), frametime))
        print('Generated time steps (in seconds): ', timesteps)
        print()

        # Generate a constant trace position offsets
    specpix_offset = 0.0
    spatpix_offset = 0.0 #5.0
        # Generate a time-dependent trace position offsets
    #specpix_offset = np.zeros_like(timesteps)
    #spatpix_offset = np.linspace(0.0, 5.0, np.size(timesteps))
    if doPrint:
        print('Trace position offsets (as a function of time, or constant):')
        print('specpix_offset = ', specpix_offset)
        print('spatpix_offset = ', spatpix_offset)


if generate_clear_tmp_simu is True:

    ##################################
    ######### RUN SIMULATION #########
    ##################################
    
        # Generate the Time-Series simulation
    norders = len(simuPars.orderlist)
    dimy, dimx = simuPars.yout, simuPars.xout
    print('norders={:} dimy={:} dimx={:}'.format(norders,dimy,dimx))
        # For each time step, a cube of simulated images is written to disk
        # The cube has all spectral orders in separate slices.
        # The list of such fits cube file names is returned.
    imagelist = soss.generate_traces(pathPars.path_userland+'tmp/clear',
                                     pathPars, simuPars, tracePars, throughput,
                                     starmodel_angstrom, starmodel_flambda, ld_coeff,
                                     planetmodel_angstrom, planetmodel_rprs,
                                     timesteps, tintopen, specpix_trace_offset=specpix_offset,
                                     spatpix_trace_offset=spatpix_offset)


    

    
if generate_clear_dmsReady_simu is True:
    
        # If the simulation was not generated at run-time, go get it
    if generate_clear_tmp_simu is False:
        imagelist = glob.glob(pathPars.path_userland+'tmp/clear*.fits')
        for i in range(np.size(imagelist)):
            imagelist[i] = os.path.join(pathPars.path_userland+'tmp/',imagelist[i])

            
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
    soss.write_dmsready_fits(data, os.path.join(pathPars.path_userland,'IDTSOSS_clear.fits'),
                         os=simuPars.noversample, input_frame='dms',
                         xpadding=simuPars.xpadding, ypadding=simuPars.ypadding)

    

    
    


###########################################
####### ADD (AND CORRECT FOR) NOISE #######
###########################################


if investigate_noise is not True: # add all default noise sources, and pass through the default DMS pipeline
                                  # (this default option thing isn't quite tuned correctly yet)
                                  # (just need to make sure the config file input to the DMS routine is correct)
        
        if add_noise_to_dmsReady_simu is True:
            detector.add_noise(os.path.join(pathPars.path_userland,'IDTSOSS_clear.fits'),
                               outputfilename=os.path.join(pathPars.path_userland, 'IDTSOSS_clear_noisy.fits'))
        if run_dms_level_one is True:
            result = Detector1Pipeline.call(os.path.join(pathPars.path_userland, 'IDTSOSS_clear_noisy.fits'),
                                        output_file='IDTSOSS_clear_noisy', output_dir=pathPars.path_userland)


else: # here is the option to investigate select noise sources

    
    noise = { 'photon':False, 'normalize':False, 'zodibackg':False, 'flatfield':False,
             'darkframe':False, 'nonlinearity':False, 'superbias':False, 'detector':False
           }
    noise_to_calwebb_map = { 'photon':[],'normalize':[] , 'zodibackg':[]
                           , 'flatfield':[] , 'darkframe':['dark_current']
                           , 'nonlinearity':['linearity'] , 'superbias':['superbias']
                           , 'detector':['refpix']
                           }
    
    for i,noise_list in enumerate(noise_shopping_lists):
        
            # Basically this whole little section writes out the file names of
            # a) the noisy file to be created
            # b) the config file to be used by the DMS pipeline, based on the noisy components of a)
        _noise = { k:(True if k in noise_list else False) for (k,v) in noise.items() }
        if np.all(list(_noise.values())): # if all noise sources are desired
            err_str = '--allNoise'
            err_calwebb_str = ''
        else:                             # add photon noise + those specfified in err_list
            err_str = ''
            err_calwebb_str = '_STEPS'
            no_steps = True
            for key in _noise.keys():
                if _noise[key] is True:
                    err_str += '--'+key
                    if len(noise_to_calwebb_map[key]) != 0:
                        err_calwebb_str += '--'+key
                        no_steps = False
            if no_steps is True:
                err_calwebb_str += '--NONE'
            if err_str == '':
                n_str = '_noNoise'
            else:
                n_str = '_noisy'
            noisy_file_str = 'IDTSOSS_clear'+n_str+err_str+'.fits'
            if dms_cfg_files[i] == None:
                dms_cfg_files[i] = dms_cfg_files_path+'calwebb_detector1'+err_calwebb_str+'.cfg'
            
        
        if add_noise_to_dmsReady_simu is True:
            print('\nWill be adding noises : '+err_str)
            detector.add_noise(os.path.join(pathPars.path_userland,'IDTSOSS_clear.fits'),
                               normalize=_noise['normalize'], zodibackg=_noise['zodibackg'], zodifile=ov_noiz['zodi'],
                               flatfield=_noise['flatfield'], flatfile=ov_noiz['flat'], darkframe=_noise['darkframe'],
                               nonlinearity=_noise['nonlinearity'], coef_file = ov_noiz['liner'], photon=_noise['photon'],
                               superbias=_noise['superbias'], biasfile=ov_noiz['bias'], detector=_noise['detector'], 
                               outputfilename=os.path.join(pathPars.path_userland, noisy_file_str)
                              )
            
        if run_dms_level_one is True:
            print('\nDMS performed on '+ noisy_file_str)
            print('DMS steps to be performed : ' + err_calwebb_str)
            print('config file is therefore : ' + dms_cfg_files[i])
            result = Detector1Pipeline.call(os.path.join(pathPars.path_userland, noisy_file_str),
                                            output_file=noisy_file_str[:-5], 
                                            output_dir=pathPars.path_userland+extra_output_dir,
                                            config_file=dms_cfg_files[i]
                                           )
            










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
                             xpadding=simuPars.xpadding, ypadding=simuPars.ypadding, f277=True)

    # Add detector noise to the noiseless data
    detector.add_noise(os.path.join(pathPars.path_userland,'IDTSOSS_f277.fits'),
                       outputfilename=os.path.join(pathPars.path_userland, 'IDTSOSS_f277_noisy.fits'))

    # Process the data through the DMS level 1 pipeline
    result = Detector1Pipeline.call(os.path.join(pathPars.path_userland, 'IDTSOSS_f277_noisy.fits'),
                                    output_file='IDTSOSS_f277_noisy', output_dir=pathPars.path_userland)

print('The end of the simulation')

