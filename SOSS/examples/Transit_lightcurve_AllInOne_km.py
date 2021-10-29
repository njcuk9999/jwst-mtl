######################################################################
# Location of the jwst-mtl github repo. Overrides the github path in
# 'jwst_config_fpath', during the import phase
github_path = '/home/kmorel/ongenesis/github/jwst-mtl/SOSS/'
# Location of the simulation config file, as well as the output directory
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/PHY3030/'
# Configuration file for the NIRISS Instrument Team SOSS simulation pipeline
jwst_config_fpath = 'jwst-mtl_configpath_kim.txt'

######################################################################
# IMPORTS
import sys
sys.path.insert(1, '/genesis/jwst/jwst-ref-soss/fortran_lib/')
# TODO: Update all paths
import glob
import os
# os.environ["OMP_NUM_THREADS"] = "24"
from pathlib import Path
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
# from detector import detector_Frost as detector
import detector.detector as detector
# normalization code
import specgen.synthesizeMagnitude as smag
# Loic-reviewed NIRISS-SOSS simulation functions
import itsosspipeline as soss

ncpu = mp.cpu_count()
pyfftw.config.NUM_THREADS = ncpu

from jwst.pipeline import Detector1Pipeline
from example1_helper_funcs import *

import box_kim

sys.path.insert(0, "/genesis/jwst/jwst-ref-soss/fortran_lib/")

######################################################################
# Matplotlib defaults
plt.rc('figure', figsize=(8,4.5))
plt.rc('font', size=12)
plt.rc('image', cmap='inferno')
plt.rc('lines', lw=1)

######################################################################
# CHOOSE whether to generate the noisy dms simulations or
# work with some already generated.
generate_noisy_simu = False

######################################################################
gain = 1.61  # [é]
radius_pixel = 30  # Radius for box extraction

######################################################################
if generate_noisy_simu is True:
    # NOISELESS
    # Generate noiseless simu
    # This one needs to be done before the others in order to compare
    # the noisy ones with the noiseless one for each case.
    noise_shopping_lists = [[]      # empty list means adding no noise (but still converting to ADU)
                            ]

    ######### The 1st step of the simulation process. #########################
        # Outputs images in units of electrons.
        ########################!!!!!!!!######################################
        # tmp clear simulations must already be generated.
        # This means that
        # generate_clear_tmp_simu = False.
        # This code is meant to add noise sources one by one
        # and analyse the results one by one.
        ########################!!!!!!!!######################################
        # Optional override of the amount of integrations in the exposure.
        # By default, the amount is determined by the maximum amount of integrations
        # that can be fit into the observation time (given the detector readout array size)
    nIntegrations_override = 300

    ######### The 2nd step of the simulation process. ##########################
        # Choose whether to format the generated (clear) simulation in a
        # fits format processable by the CALWEBB_DETECTOR1 pipeline (DMS).
        # This step also performs flux calibration via scaling of the images
        # based on expected electron counts.
    generate_clear_dmsReady_simu = True

    ########## The 3rd step of the simulation process ##########################
        # Choose whether to add detector noise to the noiseless data
        # Regardless if noise is really added or not, this step also
        # converts the images from units of electrons to ADUs
    add_noise_to_dmsReady_simu = True

    ######### The 4th step of the simulation process ###########################
        # Choose whether to process the data through the DMS level 1 pipeline
    run_dms_level_one = True

    ######### The 5th step of the simulation process ###########################
        # Inject only certain noise types in the simulation,
        # instead of the default ones:
        # investigate_noise = True

    subarray = 'SUBSTRIP256'  # See example1_investigateNoise.py for other options

    override_noise_files = True
    ov_noiz_dir = '/genesis/jwst/jwst-ref-soss/noise_files/'
    ov_noiz = {'flat': ov_noiz_dir + 'jwst_niriss_flat_0190.fits'               if override_noise_files else None,
               'zodi': ov_noiz_dir + 'background_detectorfield_normalized.fits' if override_noise_files else None,
               'bias': ov_noiz_dir + 'jwst_niriss_superbias_0120.fits'          if override_noise_files else None,
               'liner': ov_noiz_dir + 'jwst_niriss_linearity_0011_bounds_0_60000_npoints_100_deg_5.fits'
                                                                            if override_noise_files else None,
               'dark': ov_noiz_dir + 'jwst_niriss_dark_0147.fits'               if override_noise_files else None,
               }

    dms_cfg_files_path = WORKING_DIR + 'dms_config_files/'
        # CALWEBB_DETECTOR1 config files. If =None, these config files are auto-generated
        # according to the noise input specified in the noise_shopping_lists
    dms_config_files = [None] * len(noise_shopping_lists)
        # CALWEBB steps that are always executed, regardless of step choice in a shopping_list
    calwebb_NIR_TSO_mandatory_steps = ['dq_init', 'saturation', 'ramp_fit']  # 'jump'
        # For each CALWEBB_DETECTOR1 step, specify the reference file to be used (if a file is needed),
        # otherwise the default/best one is chosen (might not be same as one used in simulation)
    override_calwebb_reffiles = True
    user_calwebb_reffiles_dir = '/genesis/jwst/jwst-ref-soss/noise_files/'
    user_calwebb_reffiles = {'superbias': 'jwst_niriss_superbias_0120.fits',
                             'linearity': 'jwst_niriss_linearity_0011.fits',
                             'dark_current': 'jwst_niriss_dark_0147.fits'
                             }
        # These variables are specified by default in the function 'create_calwebb_config_files()',
        # but can be changed here if need be
    calwebb_NIR_steps = ['group_scale', 'dq_init', 'saturation', 'ipc',
                         'superbias', 'refpix', 'linearity', 'dark_current',
                         'jump', 'ramp_fit', 'gain_scale', 'persistence']
        # see function definitions in 'example_helper_funcs.py' for
        # clarifications on the use of this dict
    jwstMTLnoise_to_calwebbSteps_MAP = {'photon': [],
                                        'zodibackg': [],
                                        'flatfield': [],
                                        'darkcurrent': ['dark_current'],
                                        'nonlinearity': ['linearity'],
                                        'superbias': ['superbias'],
                                        'readout': [],
                                        'oneoverf': ['refpix']
                                        }

    # Some other self-explanatory boolean variables
    # They determine if stuff is printed or plotted at runtime
    doPlot = False
    doPrint = True
    verbose = doPrint

    ########################################################################################
    #######################        END OF SETUP PHASE        ###############################
    ########################################################################################


    ########################################
    ########## START OF THE FLOW ###########
    ########################################
    ############# SETUP PHASE ##############
    ########################################

        # Read in all paths used to locate reference files and directories
    config_paths_filename = os.path.join(WORKING_DIR, jwst_config_fpath)
    if doPrint: print()
    pathPars = soss.paths()
    soss.readpaths(config_paths_filename, pathPars)

        # Create and read the simulation parameters
    simuPars = spgen.ModelPars()  # Set up default parameters
    simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars)  # Read in parameter file

    if generate_clear_dmsReady_simu is True:
            # Instrument Throughput (Response)
        throughput = spgen.read_response(pathPars.throughputfile, set_response_to_unity=simuPars.flatthroughput,
                                         set_qy_to_unity=simuPars.flatquantumyield, verbose=verbose)

            # Set up Trace (Position vs. Wavelength)
        tracePars = tp.get_tracepars(pathPars.tracefile)

            # Generate or read the star atmosphere model
        starmodel_angstrom, starmodel_flambda, ld_coeff = soss.starmodel(simuPars, pathPars, tracePars, throughput)

            # Anchor star spectrum on a photometric band magnitude
        starmodel_flambda = smag.anchor_spectrum(starmodel_angstrom / 10000., starmodel_flambda, simuPars.filter,
                                                 simuPars.magnitude, pathPars.path_filtertransmission, verbose=True)

            # Read Planet Atmosphere Model
            # Caroline Piaulet and Benneke group planet model functionalities
            # Path for the model grid
        path_files = pathPars.path_planetmodelatm + "FwdRuns20210521_0.3_100.0_64_nLay60/"
        # planet_name = 'FwdRuns20210521_0.3_100.0_64_nLay60/HAT_P_1_b'
        planet_name = 'HAT_P_1_b'
            # Get a list of all parameters available
        planet_caselist = soss.get_atmosphere_cases(planet_name, path_files=path_files,
                                                    return_caselist=True, print_info=doPrint)
            # Select params that you want
        params_dict = soss.make_default_params_dict()
        params_dict["CtoO"] = 0.3
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
        planetmodel_rprs = np.sqrt(np.array(t['dppm']) / 1e+6)

        if doPlot:
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1); plt.plot(starmodel_angstrom, starmodel_flambda); plt.title('Star')
            plt.subplot(1, 3, 2); plt.plot(planetmodel_angstrom, planetmodel_rprs); plt.title('Planet')
            plt.subplot(1, 3, 3); plt.plot(throughput.wv, throughput.response[1]); plt.title('Throughput')
            plt.show()

            # Generate the time steps array
        tintopen, frametime, nint, timesteps = soss.generate_timesteps(simuPars)
            # Override the number of integrations while keeping the same elapsed time
            # specified in simuPars (optional)
        if nIntegrations_override != None:
            timeskip = int(len(timesteps) / nIntegrations_override)
            timesteps = timesteps[::timeskip]
            nint = len(timesteps); simuPars.nint = nint
        if doPrint:
            print('\nGenerated time steps:')
            print('nint={:} nsteps={:} frametime={:}'.format( \
                nint, len(timesteps), frametime))
            print('Generated time steps (in seconds): ', timesteps)
            print()

            # Generate a constant trace position offsets
        specpix_offset = 0.0
        spatpix_offset = 0.0  # 5.0
            # Generate a time-dependent trace position offsets
        # specpix_offset = np.zeros_like(timesteps)
        # spatpix_offset = np.linspace(0.0, 5.0, np.size(timesteps))
        if doPrint:
            print('Trace position offsets (as a function of time, or constant):')
            print('specpix_offset = ', specpix_offset)
            print('spatpix_offset = ', spatpix_offset)

        ##################################
        ######### RUN SIMULATION #########
        ##################################

            # If the simulation was not generated at run-time, go get it, i.e.
            # generate_clear_tmp_simu is False
        imagelist = glob.glob(pathPars.path_userland + 'tmp/clear*.fits')
        for i in range(np.size(imagelist)):
            imagelist[i] = os.path.join(pathPars.path_userland + 'tmp/', imagelist[i])

            # Here, a single out-of-transit simulation is used to establish the
            # normalization scale needed to flux calibrate our simulations.
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
        simulated_counts = smag.measure_actual_flux(imagelist[0], xbounds=[0, 2048], ybounds=[0, 256],
                                                    noversample=simuPars.noversample)
            # Prints the expected/measured counts
        for i in range(np.size(imagelist)):
            print(i, expected_counts, simulated_counts)
            # Apply flux scaling
        normalization_scale = expected_counts / simulated_counts
        print('Normalization scales = ', normalization_scale)

            # All simulations are normalized, all orders summed and gathered in a single array
            # with the 3rd dimension being the number of time steps.
        data = soss.write_dmsready_fits_init(imagelist, normalization_scale,
                                         simuPars.ngroup, simuPars.nint,
                                         simuPars.frametime, simuPars.granularity)

            # All simulations (e-/sec) are converted to up-the-ramp images.
        soss.write_dmsready_fits(data, os.path.join(pathPars.path_userland, 'IDTSOSS_clear.fits'),
                                 os=simuPars.noversample, input_frame='dms',
                                 xpadding=simuPars.xpadding, ypadding=simuPars.ypadding)

    ###########################################
    ####### ADD (AND CORRECT FOR) NOISE #######
    ###########################################

    # investigate_noise is True:
    jwstMTL_noise = {'photon': False, 'zodibackg': False, 'flatfield': False,
                     'darkcurrent': False, 'nonlinearity': False, 'superbias': False, 'readout': False,
                     'oneoverf': False}
    auto_dms_config_files = create_calwebb_config_files(dms_cfg_files_path, noise_shopping_lists
                                        , calwebb_NIR_TSO_mandatory_steps
                                        , calwebb_NIR_steps=calwebb_NIR_steps
                                        , jwstMTLnoise_to_calwebbSteps_MAP=jwstMTLnoise_to_calwebbSteps_MAP
                                        , override_calwebb_reffiles=override_calwebb_reffiles
                                        , user_calwebb_reffiles_dir=user_calwebb_reffiles_dir
                                        , user_calwebb_reffiles=user_calwebb_reffiles
                                        , doPrint=doPrint
                                        )
    dms_config_files = [(f1 if f2 is None else f2) for (f1, f2)
                        in tuple(np.transpose([auto_dms_config_files, dms_config_files]))
                        ]

    for i, noise_list in enumerate(noise_shopping_lists):
            # This little section writes out the file names of:
            # a) the noisy file to be created
            # b) the config file to be used by the DMS pipeline, based on the noisy components of a)
        noise = {k: (True if k in noise_list else False) for (k, v) in jwstMTL_noise.items()}
        noisy_file_str = create_noisy_filename(noise, jwstMTLnoise_to_calwebbSteps_MAP)

        if add_noise_to_dmsReady_simu is True:
            if doPrint: print('\n\n')
            detector.add_noise(os.path.join(pathPars.path_userland, 'IDTSOSS_clear.fits'),
                               pathPars.path_noisefiles,
                               photon=noise['photon'],
                               zodibackg=noise['zodibackg'], zodi_ref=ov_noiz['zodi'],
                               flatfield=noise['flatfield'], flat_ref=ov_noiz['flat'],
                               darkcurrent=noise['darkcurrent'], dark_ref=ov_noiz['dark'],
                               nonlinearity=noise['nonlinearity'], nlcoeff_ref=ov_noiz['liner'],
                               superbias=noise['superbias'], superbias_ref=ov_noiz['bias'],
                               readout=noise['readout'], oneoverf=noise['oneoverf'],
                               outputfilename=os.path.join(pathPars.path_userland, noisy_file_str)
                               )

        if run_dms_level_one is True:
            print('\nDMS performed on ' + noisy_file_str)
            print('Config file is : ' + dms_config_files[i])
            result = Detector1Pipeline.call(os.path.join(pathPars.path_userland, noisy_file_str),
                                            output_file=noisy_file_str[:-5],
                                            output_dir=pathPars.path_userland,
                                            config_file=dms_config_files[i]
                                            )

    print('The end of the NOISELEE simulation')

    ###########################################
    ####### END OF SIMULATIONS CREATION #######
    ###########################################

##########################################################
# EXTRACTION, Noiseless
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
lam_array = w   # [um]

# BOX EXTRACTION
# To save it:
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

# NORMALIATION OF FLUX FOR EACH WAVELENGTH
# To save it:
f_array_noiseless_norm = np.copy(f_array_noiseless)
for n in range(f_array_noiseless.shape[1]):  # For each wavelength
    f_array_noiseless_norm[:, n] = box_kim.normalization(f_array_noiseless[:, n], t1, t4)

#######################################################################
### Creation of noisy simulations
### one by one (one noise source at a time)
#######################################################################
# This 2D-list determines what noise types are injected into them (list contained in 2nd dimension).
noise_shopping_lists = [['photon']
                       #,['zodibackg']
                       #,['flatfield']
                       ,['darkcurrent']
                       ,['nonlinearity']
                       ,['superbias']
                       ,['readout']
                       ,['oneoverf']
                       #,['photon','nonlinearity','superbias']
                       ]
if generate_noisy_simu is True:
        # CALWEBB_DETECTOR1 config files. If =None, these config files are auto-generated
        # according to the noise input specified in the noise_shopping_lists
    dms_config_files = [None] * len(noise_shopping_lists)

    auto_dms_config_files = create_calwebb_config_files(dms_cfg_files_path, noise_shopping_lists
                                                        , calwebb_NIR_TSO_mandatory_steps
                                                        , calwebb_NIR_steps=calwebb_NIR_steps
                                                        , jwstMTLnoise_to_calwebbSteps_MAP=jwstMTLnoise_to_calwebbSteps_MAP
                                                        , override_calwebb_reffiles=override_calwebb_reffiles
                                                        , user_calwebb_reffiles_dir=user_calwebb_reffiles_dir
                                                        , user_calwebb_reffiles=user_calwebb_reffiles
                                                        , doPrint=doPrint
                                                        )
    dms_config_files = [(f1 if f2 is None else f2) for (f1, f2)
                        in tuple(np.transpose([auto_dms_config_files, dms_config_files]))
                        ]

# LOOP on each noise type
for i,noise_list in enumerate(noise_shopping_lists):

    if generate_noisy_simu is True:
        # if True: Generates noisy simulations before analysing it
        # generate_clear_dmsReady_simu = False
        # Was created upper

        ########################################
        ########## START OF THE FLOW ###########
        ########################################
        ############# SETUP PHASE ##############
        ########################################

        ###########################################
        ####### ADD (AND CORRECT FOR) NOISE #######
        ###########################################

            # This little section writes out the file names of:
            # a) the noisy file to be created
            # b) the config file to be used by the DMS pipeline, based on the noisy components of a)
        noise = {k: (True if k in noise_list else False) for (k, v) in jwstMTL_noise.items()}
        noisy_file_str = create_noisy_filename(noise, jwstMTLnoise_to_calwebbSteps_MAP)

        if add_noise_to_dmsReady_simu is True:
            if doPrint: print('\n\n')
            detector.add_noise(os.path.join(pathPars.path_userland, 'IDTSOSS_clear.fits'), pathPars.path_noisefiles,
                               photon=noise['photon'],
                               zodibackg=noise['zodibackg'], zodi_ref=ov_noiz['zodi'],
                               flatfield=noise['flatfield'], flat_ref=ov_noiz['flat'],
                               darkcurrent=noise['darkcurrent'], dark_ref=ov_noiz['dark'],
                               nonlinearity=noise['nonlinearity'], nlcoeff_ref=ov_noiz['liner'],
                               superbias=noise['superbias'], superbias_ref=ov_noiz['bias'],
                               readout=noise['readout'],
                               oneoverf=noise['oneoverf'],
                               outputfilename=os.path.join(pathPars.path_userland, noisy_file_str)
                               )

        if run_dms_level_one is True:
            print('\nDMS performed on ' + noisy_file_str)
            print('Config file is : ' + dms_config_files[i])
            result = Detector1Pipeline.call(os.path.join(pathPars.path_userland, noisy_file_str),
                                            output_file=noisy_file_str[:-5],
                                            output_dir=pathPars.path_userland,
                                            config_file=dms_config_files[i]
                                            )

        print('The end of the noisy {} simulation'.format(noisy_file_str))

        ###########################################
        ####### END OF SIMULATION CREATION #######
        ###########################################


    #######################################################################
    ### EXTRACTION, Noisy ###
    # Load simulation
    simu_filename = 'IDTSOSS_clear_noisy--' + noise + '.fits'
    simu_noisy, data_noisy = box_kim.rateints_dms_simulation(simu_filename)
    simulation_noisy = noise  # TODO: OKAY?

    # BOX EXTRACTION
    # To save it:
    fbox_noisy = np.zeros(shape=(np.shape(data_noisy)[0], np.shape(data_noisy)[2]), dtype=float)
    for t in range(np.shape(data_noisy)[0]):  # For each image of the timeseries
        fbox_noisy[t] = box_kim.flambda_adu(x, data_noisy[t], y, radius_pixel=radius_pixel)    # [adu/s]
    f_array_noisy = np.nan_to_num(fbox_noisy)

    # WHITE LIGHT CURVE
    f_white_noisy = np.sum(f_array_noisy, axis=1)
    # Normalize white light curve
    f_white_noisy_norm = box_kim.normalization(f_white_noisy, t1, t4)

    # NORMALIZATION OF FLUX FOR EACH WAVELENGTH
    # To save it:
    f_array_noisy_norm = np.copy(f_array_noisy)
    for n in range(f_array_noisy.shape[1]):  # For each wavelength
        f_array_noisy_norm[:, n] = normalization(f_array_noisy[:, n], t1, t4)

    plt.figure()
    plt.plot(time_min, f_white_noisy, '.', markersize=3, color='r', label=simulation_noisy)
    plt.plot(time_min, f_white_noiseless, '.', markersize=3, color='b', label=simulation_noiseless)
    plt.xlabel('Time [min]')
    plt.ylabel('Relative flux')
    plt.title('White light')
    plt.legend()
    plt.savefig(WORKING_DIR + 'white_light_' + simulation_noisy)

    # Plot only one wavelength
    l = 100  # Indice of wavelength to plot

    plt.figure()
    plt.plot(time_min, f_array_noisy_norm[:, l], '.', color='r', label=r'$\lambda$={:.3f}'.format(w[l]))
    plt.xlabel('Time [min]')
    plt.ylabel('Relative flux')
    plt.title(simulation_noisy)
    plt.legend()


    ### DISPERSION/PHOTON NOISE ###
    # Only the non-contaminated portion of the spectrum is used here
    new_w = w[1100:]
    new_f_array_noiseless = f_array_noiseless[:, 1100:]
    new_f_array_noisy = f_array_noisy[:, 1100:]

    # To save data:
    photon_noise = np.zeros(new_f_array_noiseless.shape[1], dtype='float')
    dispersion = np.zeros(new_f_array_noisy.shape[1], dtype='float')

    for n in range(new_f_array_noiseless.shape[1]):  # for each wavelength
        out_transit_noiseless = np.concatenate((new_f_array_noiseless[:t1, n],   # Noiseless
                                                 new_f_array_noiseless[t4:, n]))
        out_transit_noisy = np.concatenate((new_f_array_noisy[:t1, n], new_f_array_noisy[t4:, n]))  # Noisy
        out_transit_noiseless_elec = out_transit_noiseless * tint * gain  # Conversion in electrons
        photon_noise_elec = np.sqrt(np.mean(out_transit_noiseless_elec))
        # Photon noise (Poisson)
        photon_noise[n] = photon_noise_elec / gain / tint  # Reconversion in adu/s
        # Dispersion in data
        dispersion[n] = np.std(out_transit_noisy)

    ratio = dispersion / photon_noise
    ratio = ratio[:-5]

    # Plot ratio between dispersion and photon noise vs wavelength
    plt.figure()
    plt.plot(new_w[:-5], ratio, '.', color='b', label=simulation_noisy)
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel('Dispersion/Photon noise')
    plt.savefig(WORKING_DIR + 'disp_over_photnoise_' + simulation_noisy)

    print(simulation_noisy,"Mean dispersion-photon noise ratio =", np.mean(ratio))

    ### Relative difference ###
    relatDiff_white = box_kim.relative_difference(f_white_noisy, f_white_noiseless)

    plt.figure()
    plt.plot(time_min, relatDiff_white * 1e6, color='b')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Relative difference [ppm]')
    plt.title('Relative difference between {} and \n{}'.format(simulation_noisy, simulation_noiseless))
    plt.savefig(WORKING_DIR + 'relatDiff_white_' + simulation_noisy)
    plt.show()
