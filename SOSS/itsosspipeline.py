

# Read PATH config file

# Python imports

# Read Model parameters config file

# Read Planet model, star model, throughput file, trace parameters

# Resample planet and star on same grid

# Transit model setup

# Read PSF kernels

# Loop on orders to create convolved image

import numpy as np
from skimage.transform import downscale_local_mean, resize
import multiprocessing as mp
import os.path
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import scipy.constants as sc_cst
from scipy import interpolate #spline interpolation
import matplotlib.pyplot as plt
import time as clocktimer

from tqdm.notebook import tqdm as tqdm_notebook

import sys
#sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/')
import specgen.spgen as spgen
import trace.tracepol as tp
import csv


class paths():
    # PATHS
    path_home = '/genesis/jwst/'
    path_userland = '/genesis/jwst/jwst-user-soss/'
    path_tracemodel = path_home+'/jwst-ref-soss/trace_model/'
    path_starmodelatm = path_home+'/jwst-ref-soss/star_model_atm/'
    path_planetmodelatm = path_home+'/jwst-ref-soss/planet_model_atm/'
    path_spectralconvkernels = path_home+'/jwst-ref-soss/spectral_conv_kernels/'
    path_monochromaticpsfs = path_home+'/jwst-ref-soss/monochromatic_PSFs/'
    path_fortranlib = path_home+'/jwst-ref-soss/fortran_lib/'
    path_noisefiles = path_home+'/jwst-ref-soss/noise_files/'
    path_filtertransmission = path_home+'/github/jwst-mtl/SOSS/specgen/FilterSVO/'
    # Reference files
    simulationparamfile = path_userland+'simpars_wide.txt'
    tracefile = path_tracemodel+'/NIRISS_GR700_trace_extended.csv'
    throughputfile = path_tracemodel+'/NIRISS_Throughput_STScI.fits'


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
    pars.throughputfile = str(value[param =='THROUGHPUT_FILE'][0])
    pars.path_filtertransmission = str(value[param =='FILTERTRANSMISSION'][0])

def read_simu_cfg():
    with open('table_simulations.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel')
        for row in reader:
            print(row['MAGNITUDE'], row['STARMODEL'])
            # print(row)


def save_params(pathPars, simuPars, saved_inputs_file):

    # open file for writing, "w" is writing
    w = csv.writer(open(os.path.join(pathPars.path_userland, saved_inputs_file), 'w'))
    # loop over dictionary keys and values
    dict = pathPars.__dict__
    for key, val in dict.items():
        # write every key and value to file
        w.writerow([key, val])
    dict = simuPars.__dict__
    for key, val in dict.items():
        # write every key and value to file
        w.writerow([key, val])

    return

def planck(wave_micron, teff):
    """
    Black body spectrum
    :param wave_micron:
    :param teff:
    :return:
    """
    h = 6.626e-34
    c = 3.0e+8
    k = 1.38e-23

    # wave in microns, so:
    wave_meter = wave_micron * 1e-6
    a = 2.0*h*c**2
    b = h*c/(wave_meter*k*teff)
    # units of J/m2/s/m (Flambda)
    intensity = a / ( (wave_meter**5) * (np.exp(b) - 1.0) )

    return intensity


def readplanetmodel(planet_model_csvfile):
    # read the Benneke group planet models
    if False:
        # Caroline Piaulet and Benneke group planet model functionalities
        # Path for the model grid
        path_files = pathPars.path_planetmodelatm + "FwdRuns20210521_0.3_100.0_64_nLay60/"
        # planet_name = 'FwdRuns20210521_0.3_100.0_64_nLay60/HAT_P_1_b'
        planet_name = 'HAT_P_1_b'
        # Get a list of all parameters available
        planet_caselist = soss.get_atmosphere_cases(planet_name, path_files=path_files,
                                                    return_caselist=True, print_info=True)
        # select params that you want
        params_dict = soss.make_default_params_dict()
        params_dict["CtoO"] = 0.3
        # print(params_dict)
        # Get path to csv file that contains the model spectrum
        path_csv = soss.get_spec_csv_path(caselist=planet_caselist, params_dict=params_dict,
                                          planet_name=planet_name, path_files=path_files)
        planet_model_csvfile = path_csv

    t = ascii.read(planet_model_csvfile)
    print("\nSpectrum file:")
    print(t)
    # Wavelength in angstroms
    planetmodel_angstrom = np.array(t['wave']) * 1e+4
    # Rp/Rstar from depth in ppm
    planetmodel_rprs = np.sqrt(np.array(t['dppm']) / 1e+6)

    return planetmodel_angstrom, planetmodel_rprs


def starlimbdarkening(wave_angstrom, ld_type='flat'):
    """
    Generates an array of limb darkening coefficients when creating a star atmosphere model
    :return:
    """

    if ld_type == 'flat':
        # flat
        ld_coeff = np.zeros((np.size(wave_angstrom), 4))
    else:
        # no limb darkening - flat intensity
        ld_coeff = np.zeros((np.size(wave_angstrom), 4))

    return ld_coeff


def starmodel(simuPars, pathPars, tracePars, throughput, verbose=True):
    """
    From the simulation parameter file - determine what star atmosphere model
    to read or generate.
    :return:
    """
    if simuPars.modelfile == 'BLACKBODY' and simuPars.bbteff:
        if verbose: print('Star atmosphere model type: BLACKBODY')
        model_angstrom = np.linspace(4000, 55000, 2000001)
        model_flambda = planck(model_angstrom/10000, simuPars.bbteff)
        model_ldcoeff = starlimbdarkening(model_angstrom)
    elif simuPars.modelfile == 'CUSTOM' and simuPars.customstarmodel != 'null':
        if verbose: print('Star atmopshere model type: CUSTOM')
        print('CUSTOM star atmosphere model assumed angstrom and Flambda.')
        print('will be read this way:')
        print('a = fits.open(simuPars.customstarmodel)')
        print('model_angstrom = a[1].data["wavelength"]')
        print('model_flambda = a[1].data["flux"]')
        a = fits.open(simuPars.customstarmodel)
        model_angstrom = a[1].data['wavelength']
        model_flambda = a[1].data['flux']
        model_ldcoeff = starlimbdarkening(model_angstrom)
    elif simuPars.modelfile == 'CONSTANT_FLAMBDA':
        if verbose: print('Star atmosphere model type: CONSTANT_FLAMBDA')
        model_angstrom = np.linspace(4000, 55000, 2000001)
        model_flambda = np.ones(np.size(model_angstrom))
        model_ldcoeff = starlimbdarkening(model_angstrom)
    elif simuPars.modelfile == 'CONSTANT_FNU':
        if verbose: print('Star atmosphere model type: CONSTANT_FNU')
        model_angstrom = np.linspace(4000, 55000, 2000001)
        fnu = np.ones(np.size(model_angstrom))
        speedoflight = 3e+8
        model_flambda = speedoflight * fnu / (model_angstrom * 1e-10)**2
        model_ldcoeff = starlimbdarkening(model_angstrom)
    elif (simuPars.modelfile == 'ZODI_NOMINAL') | (simuPars.modelfile == 'ZODI'):
        if verbose: print('Star atmosphere model type: ZODI_NOMINAL')
        tab = ascii.read(pathPars.path_starmodelatm + 'zodi_nominal.txt')
        model_angstrom = np.array(tab['angstrom'])
        model_flambda = np.array(tab['W/m2/um/pixel'])
        model_ldcoeff = starlimbdarkening(model_angstrom)
    elif (simuPars.modelfile == 'CONSTANT_ADU'):
        if verbose: print('Star atmosphere model type: CONSTANT_ADU')
        os = 1
        spectral_order = 1
        model_angstrom = np.linspace(8000, 30000, 2000001)
        # Photon energy
        h = sc_cst.Planck
        c = sc_cst.speed_of_light
        joule_per_photon = h * c / (model_angstrom * 1e-10)
        # devise the dispersion
        micron_per_pixel = spectral_dispersion(tracePars, os=os,
                                               spectral_order=spectral_order,
                                               wavelength_angstrom=model_angstrom)
        # Throughput and Quantum yield resampling on same wavelength grid
        order_index = np.where(np.array(throughput.response_order) == spectral_order)[0][0]
        thruput = np.interp(model_angstrom, throughput.wv, throughput.response[order_index])
        qyield = np.interp(model_angstrom, throughput.wv, throughput.quantum_yield)
        # Constant adu per pixel definition
        model_aduperpixel = np.ones_like(model_angstrom)
        # Flambda is therefore
        #model_flambda = model_aduperpixel * joule_per_photon / micron_per_pixel / thruput / qyield
        model_flambda = model_aduperpixel * joule_per_photon / thruput / qyield
        # limb darkening coefficient
        model_ldcoeff = starlimbdarkening(model_angstrom)
        plt.figure()
        plt.plot(model_angstrom, thruput/np.max(thruput), label='Throughput')
        plt.plot(model_angstrom, qyield/np.max(qyield), label='QYIELD')
        plt.plot(model_angstrom, joule_per_photon/np.max(joule_per_photon), label='Photon/Joule')
        plt.plot(model_angstrom, micron_per_pixel/np.max(micron_per_pixel), label='micron/pixel')
        plt.plot(model_angstrom, model_flambda/np.max(model_flambda), label='Flambda')
        plt.plot(model_angstrom, model_aduperpixel/np.max(model_aduperpixel), label='ADU/pixel')
        plt.legend()
        plt.show()
    else:
        if verbose: print('Star atmosphere model assumed to be on disk.')
        # Read Stellar Atmosphere Model (wavelength in angstrom and flux in energy/sec/wavelength)
        model_angstrom, model_flambda, model_ldcoeff = spgen.readstarmodel(
                    pathPars.path_starmodelatm + simuPars.modelfile,
                    simuPars.nmodeltype, quiet=False)

    return model_angstrom, model_flambda, model_ldcoeff

def spectral_dispersion(tracePars, wavelength_angstrom=None, spectral_order=1, os=1):
    '''
    Devise the dispersion for a regular grid of x spectral pixels
    :param tracePars:
    :param spectral_order:
    :param os:
    :return:
    '''
    if not (wavelength_angstrom is None):
        # Wanted dispersion on a wavelength grid
        wave_micron = wavelength_angstrom*1e-4
        x, y, mask = tp.wavelength_to_pix(wave_micron, tracePars,
                                          m=spectral_order, frame='dms',
                                          subarray='SUBSTRIP256', oversample=os)
        pixel_per_micron = np.abs(np.gradient(x, wave_micron[1]-wave_micron[0]))
        micron_per_pixel = 1 / pixel_per_micron
        if False:
            plt.figure()
            plt.plot(wave_micron, micron_per_pixel)
            plt.show()
            sys.exit()
    else:
        # Wanted dispersion on the equally-spaced pixel grid
        #
        # The tp.pixel_to_wavelength() function is broken, so...
        # First get the trace x, y, wave at high sampling
        wave_micron_highR = np.linspace(0.5,5.5,100000)
        x_highR, y_highR, mask = tp.wavelength_to_pix(wave_micron_highR, tracePars,
                                          m=spectral_order, frame='dms',
                                          subarray='SUBSTRIP256', oversample=os)
        # Sort because interpolation requires it
        ind = np.argsort(x_highR)
        x_highR, y_highR, wave_micron_highR = x_highR[ind], y_highR[ind], wave_micron_highR[ind]
        # Interpolate for wavelength on an equally-spaced grid of oversampled pixels
        x = np.arange(2048*os)
        wave_micron = np.interp(x, x_highR, wave_micron_highR)
        # Take the derivative: dydx = np.gradient(y, dx)
        micron_per_pixel = np.abs(np.gradient(wave_micron, x[1]-x[0]))

    if False:
        # Write this on disk
        print('Writing dispersion on disk...')
        meta = {'description': 'Optics model dispersion (micron/pixel)'}
        formats = {'Wavelength': '{:.10f}', 'X': '{:.4f}', 'dispersion': '{:.10f}'}
        tab = Table([wave_micron, x, micron_per_pixel], names=formats, meta=meta)
        tab.write('/genesis/jwst/userland-soss/loic_review/dispersion_order1.ecsv', formats=formats)

        plt.figure()
        plt.plot(x, micron_per_pixel)
        plt.plot(x, pixel_per_micron)
        plt.plot(x_highR, wave_micron_highR)
        plt.plot(x, wave_micron)
        plt.show()

    return micron_per_pixel

def second2day(time_seconds):
    return(time_seconds / (3600.0*24.0))

def second2hour(time_seconds):
    return(time_seconds/3600.0)

def hour2day(time_hours):
    return(time_hours / 24.0)

def hour2second(time_hours):
    return(time_hours * 3600.0)


def barupdate(result):
    # Used to monitor progress for multiprocessing pools
    pbar.update()

def frames_to_exposure(frameseries, ng, nint, readpattern):
    # input is a cube of images representing the flux rate at intervals of frametime.

    nframes, dimy, dimx = np.shape(frameseries)

def generate_timesteps(simuPars, f277=False):

    ''' This function outputs the time steps for each simulated
    frame (read) or integration for the whole time-series.
    It returns the time at the center of the read or integration.
    It uses as inputs, the subarray, ngroup, tstart, tend of the
    simulation parameters.
    :param simuPars:
    :param f277: set to True to generate output for the F277W calibration
    :return:
    tintopen : time during which the detector is receiving photons
    in seconds during an integration.
    frametime : readout time of one frame in seconds.
    nint : number of integrations (integer)
    timesteps : The clock time of each integration (in seconds)
    '''

    # Determine what the frame time is from the subarray requested
    if simuPars.subarray == 'SUBSTRIP96':
        tpix, namps, colsover, rowsover = 1e-5, 1, 12, 2
        frametime = tpix * (96 / namps + colsover) * (2048 + rowsover)
        #frametime = 2.214   # seconds per read
    elif simuPars.subarray == 'SUBSTRIP256':
        tpix, namps, colsover, rowsover = 1e-5, 1, 12, 2
        frametime = tpix * (256 / namps + colsover) * (2048 + rowsover)
        #frametime = 5.494   # seconds per read
    elif simuPars.subarray == 'FULL':
        tpix, namps, colsover, rowsover = 1e-5, 4, 12, 1
        frametime = tpix * (2048 / namps + colsover) * (2048 + rowsover)
        #frametime = 10.73676  # seconds per read
    else:
        print('Need to specify the subarray requested for this simulation. Either SUBSTRIP96, SUBSTRIP256 or FF')
        sys.exit(1)
    # Update the frametime parameter in the simuPars structure
    simuPars.frametime = np.copy(frametime)


    # Compute the number of integrations to span at least the requested range of time
    # Use ngroup to define the integration time.
    intduration = frametime * (simuPars.ngroup + 1)
    # Determine the number of integrations
    #if f277:
    #    nint = simuPars.nintf277
    #else:
    nint = int(np.ceil(hour2second(simuPars.tend - simuPars.tstart)/intduration))
    # Update the frametime parameter in the simuPars structure
    simuPars.nint = np.copy(nint)

    # Generate an array of the time (in units of days) for all images to be simulated.
    # The choice is between two time precisions, or time granularities.
    # The time represents the time at mid-exposure (either mid frame or mid integration).
    # Two options:
    # 1) Generate an image at each frame read
    # 2) Generate an image valid for the whole integration (faster but not as realistic)
    if simuPars.granularity == 'FRAME':
        # arrays of time steps in units of days. Each step represents the time at the center of a frame (a read)
        timesteps = np.arange(0,nint*(simuPars.ngroup+1)) * frametime
        # add the time at the start and shift by half a frame time to center in mid-frame
        timesteps += hour2second(simuPars.tstart) + frametime/2
        # remove the reset frame
        indices = np.arange(0,nint*(simuPars.ngroup+1))
        isread = np.where(np.mod(indices,simuPars.ngroup+1) != 0)
        timesteps = timesteps[isread]
        # open shutter time, actually the time for which photons
        # reach the detector for simulation purposes
        tintopen = np.copy(frametime)
    elif simuPars.granularity == 'INTEGRATION':
        # TBC
        timesteps = np.arange(0,nint) * (simuPars.ngroup+1) * frametime
        # add the time at the start and shift by half an integration time to center in mid-integration
        # Notice that the center of an integration should exclude the reset frame
        # so the center is from read 1 (so add a frametime).
        timesteps += hour2second(simuPars.tstart) + frametime + simuPars.ngroup * frametime / 2
        # open shutter time, actually the time for which photons
        # reach the detector dor simulation purposes
        tintopen = frametime * simuPars.ngroup
    else:
        print('Time granularity of the simulation should either be FRAME or INTEGRATION.')
        sys.exit(1)

    if not f277:
        return tintopen, frametime, nint, timesteps
    else:
        # Generate the same for the F277W calibration
        nint_f277 = simuPars.nintf277
        f277_overhead = 600 # seconds
        if simuPars.granularity == 'FRAME':
            # arrays of time steps in units of days. Each step represents the time at the center of a frame (a read)
            timesteps_f277 = np.arange(0,nint_f277*(simuPars.ngroup+1)) * frametime
            # Determine the start of the F277W exposure - 5 minutes after the last science integration
            tstart_f277 = second2hour(np.max(timesteps) + f277_overhead)
            # add the time at the start and shift by half a frame time to center in mid-frame
            timesteps_f277 += hour2second(tstart_f277) + frametime/2
            # remove the reset frame
            indices = np.arange(0,nint_f277*(simuPars.ngroup+1))
            isread = np.where(np.mod(indices,simuPars.ngroup+1) != 0)
            timesteps_f277 = timesteps_f277[isread]
            # open shutter time, actually the time for which photons
            # reach the detector for simulation purposes
            tintopen = np.copy(frametime)
        elif simuPars.granularity == 'INTEGRATION':
            # TBC
            timesteps_f277 = np.arange(0, nint_f277) * (simuPars.ngroup+1) * frametime
            # Determine the start of the F277W exposure - 10 minutes after the last science integration
            tstart_f277 = second2hour(np.max(timesteps) + f277_overhead)
            # add the time at the start and shift by half an integration time to center in mid-integration
            # Notice that the center of an integration should exclude the reset frame
            # so the center is from read 1 (so add a frametime).
            timesteps_f277 += hour2second(tstart_f277) + frametime + simuPars.ngroup * frametime / 2
            # open shutter time, actually the time for which photons
            # reach the detector dor simulation purposes
            tintopen = frametime * simuPars.ngroup
        else:
            print('Time granularity of the simulation should either be FRAME or INTEGRATION.')
            sys.exit(1)

        return tintopen, frametime, nint_f277, timesteps_f277


def constantR_samples(wavelength_start, wavelength_end, resolving_power=100000):
    '''
    Generates an array of wavelengths at constant resolving power.
    :param wavelength_start:
    :param wavelength_end:
    :param resolving_power: R = lambda / delta_lambda
    :return:
    '''

    # One can show that wave_i+1 = wave_i * (2R+1)/(2R-1)
    term = (2 * resolving_power + 1) / (2 * resolving_power - 1)
    wavelength = []
    w = np.copy(wavelength_start)
    while w < wavelength_end:
        w = w * term
        wavelength.append(w)
    # remove the last sample as it is past wavelength_end
    wavelength = np.array(wavelength)
    wavelength = wavelength[0:-1]

    # The sample width is
    delta_wavelength = wavelength / resolving_power

    return wavelength, delta_wavelength

def resample_models(star_angstrom, star_flux, ld_coeff,
                    planet_angstrom, planet_rprs, simuPars, tracePars,
                    gridtype='planet', wavelength_start=None,
                    wavelength_end=None, resolving_power=None,
                    dispersion=None):

    '''

    :param star_angstrom:
    :param star_flux:
    :param ld_coeff:
    :param planet_angstrom:
    :param planet_rprs:
    :param simuPars:
    :param tracePars:
    :param gridtype:
    :param wavelength_start: angstrom
    :param wavelength_end: angstrom
    :param resolving_power: R = lambda/delta_lambda
    :param dispersion: angstrom per pixel
    :return:

    Note: An alternative is using bin_array instead of bin_array_conv . It is exact but much much slower.
    '''
    # Check that appropriate optional parameters are passed.
    if gridtype == 'constant_dispersion':
        if (dispersion is None) | (wavelength_start is None) | (wavelength_end is None):
            print('When using gridtype=constant_dispersion, need to specify dispersion,')
            print('wavelength_start and wavelength_end. One of these currently None.')
            sys.exit()
    if gridtype == 'constant_R':
        if (resolving_power is None) | (wavelength_start is None) | (wavelength_end is None):
            print('When using gridtype=constant_R, need to specify resolving_power,')
            print('wavelength_start and wavelength_end. One of these currently None.')
            sys.exit()

    # Define the grid over which to resample models
    if gridtype == 'planet':
        # Resample on the same grid as the planet model grid
        x_grid = np.copy(planet_angstrom)
        dx_grid = np.zeros_like(planet_angstrom)
        dx_grid[1:] = np.abs(x_grid[1:] - x_grid[0:-1])
        dx_grid[0] = dx_grid[1]
        # Resample star model and limb darkening matrix. But leave the planet model unchanged.
        bin_starmodel_flux = bin_array_conv(star_angstrom, star_flux, x_grid, dx_grid)
        bin_starmodel_wv = np.copy(planet_angstrom)
        bin_planetmodel_wv = np.copy(planet_angstrom)
        bin_planetmodel_rprs = np.copy(planet_rprs)
        bin_ld_coeff = bin_limb_darkening(star_angstrom, ld_coeff, x_grid, dx_grid)

    elif gridtype == 'constant_dispersion':
        # Resample on a constant dispersion grid
        #wavelength_start, wavelength_end = 5000, 55000
        #TODO: Check that wavelength_start/end are passed as input
        nsample = 1 + int((wavelength_end - wavelength_start) / dispersion)
        x_grid = np.arange(nsample) * dispersion + wavelength_start
        dx_grid = np.ones_like(x_grid) * dispersion
        # Resample the star model, the planet model as well as the LD coefficients.
        bin_starmodel_flux = bin_array_conv(star_angstrom, star_flux, x_grid, dx_grid)
        bin_starmodel_wv = np.copy(x_grid)
        bin_planetmodel_wv = np.copy(x_grid)
        bin_planetmodel_rprs = bin_array_conv(planet_angstrom, planet_rprs, x_grid, dx_grid)
        bin_ld_coeff = bin_limb_darkening(star_angstrom, ld_coeff, x_grid, dx_grid)

    elif gridtype == 'constant_R':
        # Resample on a constant resolving power grid
        #TODO: Check that resolving+power is passed as input
        x_grid, dx_grid = constantR_samples(wavelength_start, wavelength_end, resolving_power=resolving_power)
        # Resample the star model, the planet model as well as the LD coefficients.
        bin_starmodel_flux = bin_array_conv(star_angstrom, star_flux, x_grid, dx_grid)
        bin_starmodel_wv = np.copy(x_grid)
        bin_planetmodel_wv = np.copy(x_grid)
        bin_planetmodel_rprs = bin_array_conv(planet_angstrom, planet_rprs, x_grid, dx_grid)
        bin_ld_coeff = bin_limb_darkening(star_angstrom, ld_coeff, x_grid, dx_grid)
    else:
        print('Possible resample_models gridtype are: planet. constant_dispersion or constant_R.')
        sys.exit()

    return bin_starmodel_wv, bin_starmodel_flux, bin_ld_coeff, bin_planetmodel_wv, bin_planetmodel_rprs


def bin_limb_darkening(x, ld_coeff, x_grid, dx_grid):
    '''
    Bin down each parameter of the limb darkening coefficient
    on the new x_grid using the function bin_array()
    :param x:
    :param ld_coeff:
    :param x_grid:
    :param dx_grid:
    :return:
    '''

    # Shape of the ld_coeff input: (200001,4)
    # Shape of the output bin_ld_coeff: (n,4)
    bin_ld_coeff = np.zeros((len(x_grid), 4))
    for i in range(4):
        param_i = ld_coeff[:,i]
        bin_param_i = bin_array_conv(x, param_i, x_grid, dx_grid)
        bin_ld_coeff[:,i] = np.copy(bin_param_i)

    return bin_ld_coeff


def bin_array(x, fx, x_grid, dx_grid, debug=False):
    '''

    :param x:
    :param dx: width of the original array
    :param fx: represents the array values at each x sample
    :param x_grid: represents the sample center of the binned array
    :param dx_grid: represents the sample width of the bined array
    :return:
    '''

    # Arrays of the starting and ending x_grid values for each grid sample
    xgrid_left = x_grid - dx_grid/2
    xgrid_right = x_grid + dx_grid/2

    # Arrays of resampled fx at the position of the left and right x_grid
    fx_resampleft = np.interp(xgrid_left, x, fx)
    fx_resampright = np.interp(xgrid_right, x, fx)

    # Initialize the binned flux output array
    fx_grid = np.zeros_like(x_grid)

    # Sum up each original bin (fx * dx)
    grid_nsample = len(x_grid)
    for i in range(grid_nsample):
        if debug: print(i, grid_nsample)
        sum = 0.0
        # Original sample points fully included in new grid current bin
        ind = np.where((x >= xgrid_left[i]) & (x <= xgrid_right[i]))[0]
        # Integral for those full original samples
        for j in range(len(ind)-1):
            sum = sum + (fx[ind[j+1]] + fx[ind[j]])/2 * (x[ind[j+1]] - x[ind[j]])
        # Partial sample on the left
        sum = sum + (fx[ind[0]]+fx_resampleft[i])/2 * (x[ind[0]] - xgrid_left[i])
        # Partial sample on the right
        sum = sum + (fx[ind[-1]]+fx_resampright[i])/2 * (xgrid_right[i] - x[ind[-1]])
        #To get fx_grid, need to divide sum by the width of that binned grid
        fx_grid[i] = sum / dx_grid[i]

    return fx_grid

def bin_array_conv(x, fx, x_grid, dx_grid, debug=False):
    '''

    :param x:
    :param dx: width of the original array
    :param fx: represents the array values at each x sample
    :param x_grid: represents the sample center of the binned array
    :param dx_grid: represents the sample width of the bined array
    :return: fx on the new x_grid
    '''

    # Convert fx to constant resolving power. Find maximum R and resample at that R.
    R_max = np.max(x[:-1] / np.diff(x))
    x_Rcst, dx_Rcst = constantR_samples(np.min(x), np.max(x), resolving_power=R_max)
    fx_Rcst = np.interp(x_Rcst, x, fx)

    # Find what is the maximum resolving power of the new grid, x_grid
    R_max_newgrid = np.max(x_grid/dx_grid)

    # Sampling fwhm
    fwhm = R_max / R_max_newgrid
    # Convert FWHM to sigma for gaussian
    sigma = fwhm / np.sqrt(8*np.log(2))
    # Length of gaussian array
    length = int(5 * fwhm)
    # Make length an odd size
    if length%2 == 0: length = length + 1

    # x grid for gaussian
    x_gauss = np.mgrid[-length//2: length//2+1]
    # gaussian
    gauss = np.exp(-(x_gauss**2) / 2 / sigma**2)
    # normalization of gaussian
    gauss = gauss / np.sum(gauss)

    # Convolve the original flux by the gaussian
    fx_Rcst_conv = np.convolve(fx_Rcst, gauss, mode='same')

    # Interpolate at the desired x_grid position
    fx_grid = np.interp(x_grid, x_Rcst, fx_Rcst_conv)

    if debug is True:
        plt.figure()
        plt.scatter(x, fx, label='fx')
        plt.scatter(x_Rcst, fx_Rcst_conv, label='fx_Rcst_conv')
        plt.scatter(x_grid, fx_grid, label='fx_grid')
        plt.legend()
        plt.show()
        sys.exit()

    return fx_grid


def do_not_use_bin_star_model(mod_w, mod_f, mod_ldcoeff, w_grid, dw_grid):
    '''
    Bin high resolution stellar model down to a grid of wavelengths bins and its associated
    bin width. The limb darkening coefficients must be handled properly. We can't just average
    the LD parameters. We need to average the intensities then recover the new LD parameters.
    If, for example, we were averaging 3 high-resolution samples together, then:
    Ibin(mu) = (I(mu)_A + I(mu)_B + I(mu)_C)/3
               where I(mu)_A/Io_A = 1 - a1_A(1-mu^1/2) - a2_A() - a3_A() - a4_A()
    Then a1_bin = [ Io_A*a1_B + Io_B*a1_B + Io_C*a1_C ] / [Io_A + Io_B + Io_C]
    and so on for all 4 parameters.
    For partial samples then linear interpolation is used, in other words, a weight is assigned to each sample.
    :param mod_w:
    :param mod_f:
    :param mod_ldcoeff:
    :param w_grid: Assumes same units as input mod_w and sorted in increasing order
    :param dw_grid:
    :return:
    '''

    for i in range(len(w_grid)):
        w_start = w_grid[i] - dw_grid[i]/2
        w_end = w_grid[i] + dw_grid[i]/2

        #intensity_mu_bin = mod_ldcoeff[]

    # Arrays of the starting and ending wavelength values for each grid sample
    w_leftbound = w_grid - dw_grid/2
    w_rightbound = w_grid + dw_grid/2


    return

def input_trace_xytheta(xytheta_file, timesteps,
                        x_rms=0.0, y_rms=0.0, theta_rms=0.0,
                        x_slope=0.0, y_slope=0.0, theta_slope=0.0,
                        x_t0=0.0, y_t0=0.0, theta_t0=0.0):
    '''
    Reads from file or generate the seed trace position movement with time for each
    timestep in the time-series. dx, dy and dtheta are in native pixels and degrees.
    It is expected that the file contains as many lines as there are integrations.

    :param xytheta_file:
    :param timestep:
    :param x_noisy: if True, add some scatter (x_rms) to the seeded position
    :param x_rms:
    :param y_rms:
    :param theta_rms:
    :param x_slope:  native pixel per second
    :param y_slope:  native pixel per second
    :param theta_slope: degree per second
    :return:
    '''

    # Set time relative to first time step
    t = np.array(timesteps - np.min(timesteps))

    if os.path.isfile(xytheta_file) is True:
        # read it (3 columns, as many as timesteps)
        data = ascii.read(xytheta_file)
        dx = np.array(data[data.keys()[0]])
        dy = np.array(data[data.keys()[1]])
        dtheta = np.array(data[data.keys()[2]])

        if np.size(dx) != np.size(t):
            print('The input xytheta_file does not have the same number of lines as number of integrations.')
            sys.exit()
    else:
        # Assume a linear trend for all 3 parameters (0 by default, no noise)
        dx = x_t0 + t * x_slope + np.random.standard_normal(np.size(t)) * x_rms
        dy = y_t0 + t * y_slope + np.random.standard_normal(np.size(t)) * y_rms
        dtheta = theta_t0 + t * theta_slope + np.random.standard_normal(np.size(t)) * theta_rms

        # Save file on disk
        ascii.write({'dx': dx, 'dy': dy, 'dtheta': dtheta}, xytheta_file, delimiter=' ')

    return dx, dy, dtheta


def seed_trace_geometry(simuPars, tracePars, spectral_order, models_grid_wv,
                        specpix_offset=0, spatpix_offset=0):
    '''
    Sets up the seed trace image dimensions, position of the trace, left and
    right bounds of each spectral pixel, wavelength and width of each spectral
    pixel.

    :param simuPars:
    :param tracePars:
    :param spectral_order:
    :param models_grid_wv:
    :param specpix_offset: trace position offset to apply along the spectral direction
    :param spatpix_offset: trace position offset to apply along the spatial direction
    :return:
    x1, x2 : the left and right position bounds of a pixel in pixel coordinates
             if pixel is 0, then x1 = -0.5 and x2 = 0.5
    y1, y2 : the bottom and top boundaries of the spatial spread within a
             spectral pixel.
    w, dw : the wavelength in the center of the pixel and its width in wavelength
    paddimy, paddimx : dimensions of the padded and oversampled image seed
    padposx, padposy : the trace centroid positions in that padded image

    '''

    # Will need to pass these down at the input line
    # TODO: implement a non-zero dx,dy functionality and stick it here
    specmov, spatmov = np.copy(specpix_offset), np.copy(spatpix_offset)

    # Dimensions of the real (unpadded but oversampled) image
    dimx = simuPars.xout * simuPars.noversample
    dimy = simuPars.yout * simuPars.noversample

    # Dimensions of the 2D image on which we seed the trace
    paddimx = dimx + 2 * simuPars.xpadding * simuPars.noversample
    paddimy = dimy + 2 * simuPars.ypadding * simuPars.noversample

    # Convert star model wavelengths to trace positions.
    # Note that specpix is in range[0,2048]*noversample.
    specpix, spatpix, mask = tp.wavelength_to_pix(models_grid_wv / 10000, tracePars, m=spectral_order,
                                                  frame='dms', oversample=simuPars.noversample,
                                                  subarray=simuPars.subarray)

    # The trace position on the real (unpadded but oversampled) image
    posx = specpix + (specmov * simuPars.noversample)
    posy = spatpix + (spatmov * simuPars.noversample)

    # The trace positions in the padded image after adding padding
    padposx = posx + simuPars.xpadding * simuPars.noversample
    padposy = posy + simuPars.ypadding * simuPars.noversample

    # Compute the wavelength borders of each image pixel,
    # as well as the width (in wavelength) of each pixel
    x1 = np.arange(paddimx) - 0.5
    x2 = x1 + 1

    # independent array must be sorted for np.interp to work
    ind = np.argsort(padposx)
    w1 = np.interp(x1, padposx[ind], models_grid_wv[ind])
    w2 = np.interp(x2, padposx[ind], models_grid_wv[ind])
    # Wavelength and width of each pixel center.
    dw = np.abs(w2 - w1)
    w = (w1 + w2) / 2

    # Project along the spatial axis. y1 and y2 are the limits of the
    # spatial extent of the trace, in pixels.
    y1 = np.interp(x1, padposx[ind], padposy[ind])
    y2 = np.interp(x2, padposx[ind], padposy[ind])

    return x1, x2, y1, y2, w, dw, paddimy, paddimx, padposx, padposy



def loictrace(simuPars, response, bin_models_wv, bin_starmodel_flux, bin_ld_coeff,
              bin_planetmodel_rprs, time, itime, solin, spectral_order, tracePars,
              specpix_trace_offset=0, spatpix_trace_offset=0):
    '''
    :param simuPars:
    :param response:
    :param bin_models_wv:
    :param bin_starmodel_flux:
    :param bin_ld_coeff:
    :param bin_planetmodel_rprs:
    :param time:
    :param itime:
    :param solin:
    :param spectral_order:
    :param tracePars:
    :param specpix_trace_offset:
    :param spatpix_trace_offset:
    :return:
    '''
    # Get the pixel bounds, central wavelength and seed image dimensions
    x1, x2, y1, y2, w, dw, paddimy, paddimx, padposx, padposy = seed_trace_geometry(
                simuPars, tracePars, spectral_order, bin_models_wv,
                specpix_offset=specpix_trace_offset, spatpix_offset=spatpix_trace_offset)

    # Create functions to interpolate over response and quantum yield
    quantum_yield_spline_function = interpolate.splrep(response.wv, response.quantum_yield, s=0)
    # -- Index of the spectral order of interest in the response class.
    # -- (response_order is of type 'list')
    order_index = np.where(np.array(response.response_order) == spectral_order)[0][0]
    response_spline_function = interpolate.splrep(response.wv, response.response[order_index], s=0)
    # Interpolate response and quantum yield on the same wavelength grid as models
    bin_response = interpolate.splev(bin_models_wv, response_spline_function, der=0)
    bin_quantum_yield = interpolate.splev(bin_models_wv, quantum_yield_spline_function, der=0)

    # Generate Transit Model
    npt = len(bin_models_wv)  # number of wavelengths in model
    time_array = np.ones(npt) * time  # Transit model expects array
    itime_array = np.ones(npt) * itime  # Transit model expects array
    rdr_array = np.ones((1, npt)) * bin_planetmodel_rprs  # r/R* -- can be multi-planet
    tedarray = np.zeros((1, npt))  # secondary eclipse -- can be multi-planet
    planet_flux_ratio = spgen.transitmodel(solin, time_array, \
                                     bin_ld_coeff[:, 0], bin_ld_coeff[:, 1], bin_ld_coeff[:, 2], bin_ld_coeff[:, 3], \
                                     rdr_array, tedarray, itime=itime_array)

    bin_models_flux = planet_flux_ratio * bin_starmodel_flux * bin_response * bin_quantum_yield

    #TODO: return planet_flux_ratio * bin_starmodel_flux to calling program to enable
    # saving this "input spectrum".

    # Flux along x-axis pixels (in e-/column/s) for all columns
    pixelflux = bin_array(bin_models_wv, bin_models_flux, w, dw, debug=False)
    # Make sure that all flux are finite, or make Nans, zero
    pixelflux[~np.isfinite(pixelflux)] = 0.0

    if False:
        plt.figure()
        plt.scatter(bin_models_wv, bin_starmodel_flux, label="bin_starmodel_flux")
        plt.scatter(bin_models_wv, bin_models_flux, label="bin_models_flux")
        plt.scatter(np.arange(len(pixelflux)), pixelflux, label="pixelflux")
        plt.legend()
        plt.show()
        sys.exit()

    # Now need to distribute this flux along the y-axis using the trace centroid position
    # still want to seed a 1-pixel high trace. But in regions where the curvature is strong,
    # determine the center of mass of the flux (y * f) within a column as a better y position.
    seedtrace = np.zeros((paddimy, paddimx))
    # for now just assign the value to all rows
    make_trace_straight = False

    if make_trace_straight is False:
        # Assume that the flux is uniformly distributed.
        for i in range(paddimx):
            if True:
                # Grey pixels
                seedtrace[:, i] = np.transpose(spread_spatially(paddimy, y1[i], y2[i], pixelflux[i]))
            else:
                # Black or white pixels - was proven to produce large flux
                # oscillations - do not use
                ycenter = (y1[i]+y2[i])/2
                seedtrace[int(np.round(ycenter)), i] = pixelflux[i]
    else:
        # Generate horizontal straight traces without curvature.
        # This can be useful for testing and debugging.
        # Seed the 1, 2 3 traces so they are evenly spaced
        if spectral_order == 1:
            y = int(paddimy * 0.5)
        elif spectral_order == 2:
            y = int(paddimy * 0.25)
        elif spectral_order == 3:
            y = int(paddimy * 0.75)
        else:
            y = int(paddimy * 0.05)
        seedtrace[y, :] = pixelflux

    return seedtrace


def spread_spatially(dimy, y1, y2, flux):
    '''
    Spread the flux along a column, handling the partial pixels
    at both ends of the y1 to y2 range. Total flux should equal 1.
    :param dimy:
    :param y1:
    :param y2:
    :param flux:
    :return:
    '''

    # Make sure that y1 is the smallest
    if y1 > y2:
        tmp = y1
        y1 = np.copy(y2)
        y2 = np.copy(tmp)

    # Determine the flux density. y2-y1 should be >=1
    y_spread = (y2 - y1)
    # Make sure that spread is at least 1 pixel
    if y_spread < 1: y_spread = 1

    if y_spread > 1:
        # Case where the trace curvature is large (and/or os large)
        flux_density = flux / y_spread
        # A y pixel runs from -0.5 to +0.5 (center at 0)
        yindmin = int(np.floor(y1+0.5))

        # Dispatch the flux and properly handle ends
        column = np.zeros(dimy)
        yrange_bot = 1 - (y1+0.5)%1
        yrange_top = (y2+0.5)%1
        yrange_int = int(np.round(y_spread - yrange_bot - yrange_top))
        column[yindmin] =  yrange_bot * flux_density
        column[yindmin+1:yindmin+yrange_int+1] = flux_density
        column[yindmin+yrange_int+1] = yrange_top * flux_density

    else:
        # case for most pixels, seed a 1-pixel wide seed
        flux_density = flux / 1.0
        ycenter = (y1+y2)/2
        y1 = ycenter-0.5
        y2 = ycenter+0.5

        # A y pixel runs from -0.5 to +0.5
        yindmin = int(np.floor(y1 + 0.5))
        yindmax = yindmin + 1
        #print(yindmin, yindmax)
        #if yindmin < 0: yindmin = 0
        #if yindmax > dimy: yindmax = dimy

        # Dispatch the flux and properly handle ends
        column = np.zeros(dimy)
        if (yindmin >= 0) & (yindmin < dimy):
            column[yindmin] = (1 - (y1 + 0.5) % 1) * flux_density
        if (yindmax <= dimy-1) & (yindmax >= 0):
            column[yindmax] = ((y2 + 0.5) % 1) * flux_density

    return column


def add_wings(convolved_slice, noversample):
    '''
    Takes an image of the convolved trace and extends the wings across the
    image along the columns, assuming that the wing slop eis constant (in log(flux)).
    :param convolved_slice: a 2D image of the convolved trace, may be oversampled
    :param noversample: the oversampling of the image
    :return:
    '''

    # The wings are roughly a linear trend in log(flux). CV3 characterization
    # by Michael Radica. log(flux) goes from 1.5 to 1.0 over about 100 native
    # pixels.

    print('     Extending wings on both sides of the trace.')

    # Flux threshold above which a trace is considered seeded on a pixel
    fluxthreshold = 30.0
    slope = -0.032 # in log(flux) per native pixel

    # Obtain the image dimensions
    dimy, dimx = np.shape(convolved_slice)
    # Repeat for each column
    for col in range(dimx):
        data = np.copy(convolved_slice[:,col])
        indzero = np.where(data < fluxthreshold)[0]
        nzero = np.size(indzero)
        indtrace = np.where(data >= fluxthreshold)[0]
        ntrace = np.size(indtrace)
        # A branch of possibilities here
        if nzero < dimy:
            # Go further unless all pixels of this column are zero
            indmax = np.max(indtrace)
            indmin = np.min(indtrace)
            if indmax < dimy-1:
                # At least 1 pixel is zero at the top of the column
                # take value of pixel below and extend across column above
                logflux = np.log10(fluxthreshold) + slope * np.arange(dimy - indmax) / noversample
                # convolved_slice[indmax:,col] = np.exp(logflux)
                convolved_slice[indmax:,col] = np.power(10, logflux)
                if False:
                    if col%100 == 0:
                        plt.figure()
                        y = np.arange(dimy)
                        plt.scatter(y, np.log10(data), s=2)
                        plt.scatter(y[indmax], np.log10(data[indmax]))
                        y_wing = np.arange(dimy - indmax) + indmax
                        plt.scatter(y_wing, logflux)
            if indmin > 0:
                # At least 1 pixel is zero at the bottom of the column
                # take value of pixel above and extend across column below
                logflux = np.log10(fluxthreshold) + slope * np.arange(indmin) / noversample
                # convolved_slice[:indmin,col] = np.exp(np.flip(logflux))
                convolved_slice[:indmin, col] = np.power(10, np.flip(logflux))
                if False:
                    if col%100 == 0:
                        plt.scatter(y[indmax], np.log10(data[indmax]))
                        y_wing = np.arange(indmin)
                        plt.scatter(y_wing, np.flip(logflux))
                        plt.plot(y, np.log10(convolved_slice[:,col]))
                        plt.show()
    return convolved_slice


def readmonochromatickernels(psfdir, wls=0.5, wle=5.2, dwl=0.05, os=4, wfe=0):
    """
    Inputs
     psfdir    : psf FITS files directory
     wls       : wavelength start for Kernels, inclusive (um)
     wle       : wavelength end for Kernels, inclusive (um)
     dwl       : wavelength spacing for Kernels (uw)
     os        : amount of oversampling.  Much be an integer >=1.
     wfe       : WF map realization (integer between 0 and 9)
    """

    kernels = []
    kernels_wv = []

    # Handle the pixel oversampling
    if (os >=1) & (os<=10):
        os_string = str(int(os))
    else:
        print('Warning. Monochromatic PSF kernels oversampled by os outside 1-10 integer was requested. Assume os=4.')
        os_string = '4'
    prename = 'SOSS_os'+os_string+'_128x128_'

    # Select which Wavefront realization to use
    if (wfe >= 0) & (wfe < 10):
        # Make sure realization is a string between 0 and 9
        wfe_nbr = str(int(wfe))
        extname = '_wfe' + wfe_nbr + '.fits'
    else:
        print(
            'Warning. Wavefront Error Realization different from an integer between 0 and 9 was passed. Defaulting to 0.')
        extname = '_wfe0.fits'

    wl = np.copy(wls)
    while wl <= wle:
        wname = '{0:.6f}'.format(int(wl * 100 + 0.1) / 100)
        # fname=workdir+kerneldir+kdir+prename+wname+extname
        # (2020/09/02) New path in order to harmonize with rest of code
        fname = psfdir + prename + wname + extname
        # print(fname)
        hdulist = fits.open(fname)
        # Extract data and rotate (ds9 to sim coordinates). Deprecated.
        # kernel_1=hdulist[0].data.T
        # Extract data (of PSFs generated in the DMS coordinates)
        kernel_1 = hdulist[0].data
        # Normalize PSF flux to 1
        kernel_1 = kernel_1 / np.sum(kernel_1)
        # Stack up each wavelength's PSF
        kernels.append(np.copy(kernel_1))
        kernels_wv.append(np.float(wl))

        hdulist.close()

        wl += dwl

    return np.array(kernels), np.array(kernels_wv)


def generate_traces(savingprefix, pathPars, simuPars, tracePars, throughput,
                    star_angstrom, star_flux, ld_coeff,
                    planet_angstrom, planet_rprs,
                    timesteps, granularitytime,
                    specpix_trace_offset=0.0, spatpix_trace_offset=0.0):
    '''
    :param pathPars:
    :param simuPars:
    :param tracePars:
    :param throughput:
    :param star_angstrom:
    :param star_flux:
    :param ld_coeff:
    :param planet_angstrom:
    :param planet_rprs:
    :param timesteps: clock time of the whole frame or integration series (in seconds)
    :param granularitytime: time duration of each frame or integration (in seconds)
    :param trace_position_dxdy:
    :return:
    '''

    # output is a cube (1 slice per spectral order) at the requested
    # pixel oversampling.

    # Resample star and planet models to common wavelength grid.
    print('Resampling star and planet model')
    # Old Jason method which contains a bug where the specing is not actually uniform but
    # is rather binomial.
    #if False:
    #    # Get wavelength spacing to use for resampling
    #    dw, dwflag = spgen.get_dw(star_angstrom, planet_angstrom, simuPars, tracePars)
    #    # dw = dw/100
    #    print("Wavelength spacing (angstroms): ", dw, dwflag)
    #    star_angstrom_bin, star_flux_bin, ld_coeff_bin, planet_angstrom_bin, planet_rprs_bin = \
    #        spgen.resample_models(dw, star_angstrom, star_flux, ld_coeff, planet_angstrom, planet_rprs, simuPars,
    #        tracePars)
    # New rewritten function for resampling - dispersion in angstrom/pixel (0.1 was used for most sims)
    star_angstrom_bin, star_flux_bin, ld_coeff_bin, planet_angstrom_bin, planet_rprs_bin = resample_models(
        star_angstrom, star_flux, ld_coeff, planet_angstrom, planet_rprs, simuPars,
        tracePars, gridtype='constant_dispersion', dispersion = 0.1, wavelength_start=5000, wavelength_end=55000)
    # April 4 2022 - playing with the binning to see if it explains the 400 ppm offset in the extracted
    # transit spectrum extracted by Michael Radica.
    #star_angstrom_bin, star_flux_bin, ld_coeff_bin, planet_angstrom_bin, planet_rprs_bin = resample_models(
    #    star_angstrom, star_flux, ld_coeff, planet_angstrom, planet_rprs, simuPars,
    #    tracePars, gridtype='constant_dispersion', dispersion = 0.01, wavelength_start=5000, wavelength_end=55000)

    # Checked and this is absolutely flat, as expected for constant F_lambda
    # plt.figure()
    # plt.scatter(star_angstrom_bin, star_flux_bin)
    # plt.show()

    # Convert star_flux to photon flux (which is what's expected for addflux2pix in gen_unconv_image)
    print('Converting F_lambda to photon fluxes (e-/s/m2/ang)')
    h = sc_cst.Planck
    c = sc_cst.speed_of_light
    photon_energy = h * c / (star_angstrom_bin * 1e-10)
    star_flux_bin = star_flux_bin / photon_energy

    # plt.figure()
    # plt.scatter(star_angstrom_bin, star_flux_bin)
    # plt.show()
    # sys.exit()

    # Transit model
    print('Setting up Transit Model Parameters')
    # This will become a routine
    # Setup static Solution parameters (LD,r/R* and TED are wavelength dependent)
    # This is a single planet example
    solin = np.zeros(8 + 10 * simuPars.nplanet)  # TF5 style transit solution array
    time = 0.0  # time
    itime = 0.0001  # integration time (days)
    solin[0] = np.copy(simuPars.sol[0])  # Mean stellar density
    solin[8] = np.copy(simuPars.sol[1])  # EPO
    solin[9] = np.copy(simuPars.sol[2])  # Period
    solin[10] = np.copy(simuPars.sol[3])  # Impact parameter
    solin[12] = np.copy(simuPars.sol[4])  # ECW
    solin[13] = np.copy(simuPars.sol[5])  # ESW
    solin[14] = np.copy(simuPars.sol[6])  # KRV
    solin[16] = np.copy(simuPars.sol[7])  # ELL
    solin[17] = np.copy(simuPars.sol[8])  # ALB

    # Read in Kernels
    print('Reading in and resampling PSF Kernel')
    #kernels, kernels_wv = spgen.readkernels(pathPars.path_monochromaticpsfs)
    # New code to read the monochromatic PSF kernels
    kernels, kernels_wv = readmonochromatickernels(pathPars.path_monochromaticpsfs,
                                                   os=simuPars.noversample,
                                                   wfe=simuPars.wfrealization)
    kernel_resize = kernels
    if False:
        # resize Kernels
        # limit oversampling to be: 1<10
        kernel_resize = []
        for k in kernels:
            resized = resize(k, (128 * simuPars.noversample, 128 * simuPars.noversample))
            thesum = np.sum(resized)
            kernel_resize.append(resized / thesum)
        # Convert to numpy array (no flux scaling conservation from resize is required)
        kernel_resize = np.array(kernel_resize) #* (10 / simuPars.noversample)**2

    #The number of images (slices) that will be simulated is equal the number of orders
    # Don't worry, that 'cube' will be merged down later after flux normalization.
    #nframes = np.size(simuPars.orderlist)

    # Defines the dimensions of the arrays, depends on the oversampling
    #print('Bug traceing. xpadding={:d} ypadding={:d}'.format(simuPars.xpadding, simuPars.ypadding))
    xmax = (simuPars.xout + 2*simuPars.xpadding) * simuPars.noversample
    ymax = (simuPars.yout + 2*simuPars.ypadding) * simuPars.noversample
    #xmax =simuPars.xout*simuPars.noversample
    #ymax =simuPars.yout*simuPars.noversample

    # If a time-dependent trace position offset array was passed as input,
    # then use it.
    # First initialize the offset arrays
    specpix_offset_array = np.copy(specpix_trace_offset)
    spatpix_offset_array = np.copy(spatpix_trace_offset)
    # If a scalar value (a constant) was passed then make it an array matching the number of time steps
    if (np.size(specpix_trace_offset) == 1):
        specpix_offset_array = np.zeros_like(timesteps) + specpix_trace_offset
    if (np.size(spatpix_trace_offset) == 1):
        spatpix_offset_array = np.zeros_like(timesteps) + spatpix_trace_offset
    # Make sure that the offset arrays have the same size as timesteps array
    if np.size(specpix_offset_array) != np.size(spatpix_offset_array):
        print('Error in generate_traces(), trace position offsets must be a constant or have the same size as timesteps.')
        sys.exit()

    # Initialize the array that will contain all orders at a given time step
    nimage = len(simuPars.orderlist)
    convolved_image=np.zeros((nimage,ymax,xmax))
    trace_image=np.zeros((nimage,ymax,xmax))

    # list of temporary filenames
    filelist = []
    # Loop over all time steps for the entire Time-Series duration
    for t in range(len(timesteps)):
        # Loop over all spectral orders
        for m in range(len(simuPars.orderlist)):
            spectral_order = int(np.copy(simuPars.orderlist[m]))  # very important to pass an int here or tracepol fails
            currenttime = second2day(timesteps[t])
            exposetime = second2day(granularitytime)
            print('Time step {:} hours - Order {:}'.format(currenttime*24, spectral_order))
            if False:
                pixels=spgen.gen_unconv_image(simuPars, throughput, star_angstrom_bin, star_flux_bin,
                                      ld_coeff_bin, planet_rprs_bin,
                                      currenttime, exposetime, solin, spectral_order, tracePars)

                pixels_t=np.copy(pixels.T)
            else:
                print('     Seeding flux onto a narrow trace on a 2D image')
                pixels_t = loictrace(simuPars, throughput, star_angstrom_bin, star_flux_bin,
                                     ld_coeff_bin, planet_rprs_bin,
                                     currenttime, exposetime, solin, spectral_order, tracePars,
                                     specpix_trace_offset=specpix_offset_array[t],
                                     spatpix_trace_offset=spatpix_offset_array[t])



            if False:
                # Replace by a simple horizontal trace of intensity 10000
                a = np.zeros_like(pixels_t)
                a[500,:] = 10000
                pixels_t = a


            #Enable threads (not working?!?!)
            #pyfftw.config.NUM_THREADS = 1 #do not need multi-cpu for pools
            #with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft):
            # Turn on the cache for optimum performance
            #pyfftw.interfaces.cache.enable()

            #do the convolution
            print('     Convolving trace with monochromatic PSFs')
            # x=pixels_t*0+1.0e-10

            # Identify wavelengths whose wings fall on the detector
            wv_indices = spgen.convolve_1wv_wave_indices(kernel_resize, kernels_wv, simuPars, spectral_order, tracePars)
            kernels_wv_subgroup = kernels_wv[wv_indices]
            kernel_resize_subgroup = kernel_resize[wv_indices,:,:]

            nwv=len(kernels_wv_subgroup) #number of wavelengths to process
            #pbar = tqdm_notebook(total=nwv)  #Will make a progressbar to monitor processing.
            ncpu = 16
            pool = mp.Pool(processes=ncpu)  #Use lots of threads - because we can!

            #arguments = (pixels_t, kernel_resize, kernels_wv, wv_idx, simuPars, spectral_order, tracePars,)
            #results = [pool.apply_async(spgen.convolve_1wv, args = (pixels_t, kernel_resize, kernels_wv, wv_idx, simuPars, spectral_order, tracePars,), callback=barupdate) for wv_idx in range(nwv)]
            tic = clocktimer.perf_counter()


            results = [pool.apply_async(spgen.convolve_1wv, args = (pixels_t, kernel_resize_subgroup, kernels_wv_subgroup, wv_idx, simuPars, spectral_order, tracePars,)) for wv_idx in range(nwv)]


            pixels_c = [p.get() for p in results]

            pool.close()
            pool.join()

            # Save the intermediate convolutions
            #if spectral_order == 1:
            #    dimy, dimx = np.shape(pixels_t)
            #    conv_intermediate = np.zeros((nwv, dimy, dimx))

            #bring together the results
            x=pixels_t*0+1.0e-10
            n = 0
            for p in pixels_c:
                x+=p
                #if spectral_order == 1: conv_intermediate[n, :, :] = np.copy(p)
                n = n + 1

            toc = clocktimer.perf_counter()
            print('     Elapsed time = {:.4f}'.format(toc - tic))
            pixels_c=None #release Memory

            #if spectral_order == 1:
            #    h = fits.PrimaryHDU(conv_intermediate)
            #    h.writeto('/genesis/jwst/userland-soss/loic_review/convolved_intermediate.fits', overwrite=True)
            #    conv_intermediate=None

            trace_image[m,:,:] = np.copy(pixels_t)
            convolved_image[m,:,:] = np.copy(x) - 1e-10

            # Sum in the flux for that order
            y1 = simuPars.ypadding * simuPars.noversample
            y2 = y1 + simuPars.yout * simuPars.noversample
            x1 = simuPars.xpadding * simuPars.noversample
            x2 = x1 + simuPars.xout * simuPars.noversample
            actual_counts = np.sum(convolved_image[m,y1:y2,x1:x2])
            print('     Actual counts measured on the simulation = {:} e-/sec'.format(actual_counts))

            # Extend wings across the columns (neglect the flux associated with it)
            if simuPars.addwings == True:
                convolved_image[m,:,:] = add_wings(convolved_image[m,:,:], simuPars.noversample)

            print()

        tmp = write_intermediate_fits(trace_image, savingprefix+'_trace', t, simuPars)
        tmpfilename = write_intermediate_fits(convolved_image, savingprefix, t, simuPars)
        filelist.append(tmpfilename)

    return(filelist)





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



def rebin(image, noversampling, flux_method='mean'):
    """
    Takes an oversampled image and bins it down to native pixel size, taking
    the mean of the pixel values.
    :param arr:
    :param noversampling:
    :return:
    """
    ndim = image.ndim
    print('rebin : image dimensions = {:}'.format(ndim))
    if ndim == 2:
        dimy, dimx = np.shape(image)
        newdimy, newdimx = int(dimy/noversampling), int(dimx/noversampling)
        shape = (newdimy, image.shape[0] // newdimy,
                newdimx, image.shape[1] // newdimx)
        if flux_method == 'sum':
            return image.reshape(shape).sum(-1).sum(1)
        else:
            return image.reshape(shape).mean(-1).mean(1)
    elif ndim == 3:
        dimz, dimy, dimx = np.shape(image)
        newdimy, newdimx = int(dimy/noversampling), int(dimx/noversampling)
        cube = np.zeros((dimz, newdimy, newdimx))
        for i in range(dimz):
            image2D = image[i,:,:]
            shape = (newdimy, image2D.shape[0] // newdimy,
                     newdimx, image2D.shape[1] // newdimx)
            if flux_method == 'sum':
                cube[i,:,:] = image2D.reshape(shape).sum(-1).sum(1)
            else:
                cube[i,:,:] = image2D.reshape(shape).mean(-1).mean(1)
        return cube
    else:
        print('rebin accepts 2D or 3D arrays, nothing else!')
        return 1


def write_simu_fits(image, filename):

    # Writes the simulated data set as simulated, independently of the DMS constraints
    hdu = fits.PrimaryHDU(image)
    # Save on disk
    hdu.writeto(filename, overwrite=True)

    return

def write_intermediate_fits(image, savingprefix, timestep_index, simuPars):
    # Write image for a single time step. Differents spectral orders stored in third dimension
    # of the array. filename is the name of the final product. Intermediate filenames will
    # be forged based on that.

    directory_name = os.path.dirname(savingprefix)
    filename_current = savingprefix+'_{:06d}'.format(timestep_index)+'.fits'

    # Create a list of HDU with primary and extension HDUs
    hdu = fits.PrimaryHDU(image)
    # Add headers
    hdu.header['XPADDING'] = simuPars.xpadding
    hdu.header['YPADDING'] = simuPars.ypadding
    hdu.header['NOVRSAMP'] = simuPars.noversample

    # create a directory if it does not yet exists
    if os.path.exists(directory_name) is False:
        os.mkdir(directory_name)

    # Save on disk
    hdu.writeto(filename_current, overwrite=True)

    return(filename_current)

def write_dmsready_fits_init(imagelist, normalization_scale,
                             ngroup, nint, frametime, granularity,
                             verbose=None, os=1):
    if verbose:
        print('Entered write_dmsready_fits_init')
        print('imagelist =', imagelist)
        print('nint={:}, ngroup={:}, frametime={:} sec '.format(nint, ngroup, frametime))
        print('Granularity = {:}'.format(granularity))

    ntimesteps = len(imagelist)
    for t in range(ntimesteps):
        # Read the current image
        with fits.open(imagelist[t]) as hdu:
            image = np.array(hdu[0].data)
            image = rebin(image, os)
            # Create the cube of reads (if first iteration in loop)
            if t == 0:
                # First image, use dimensiosn and create a large cube
                norders, dimy, dimx = np.shape(image)
                #print('Bug tracing - norder={:d} dimy={:d} dimx={:d}'.format(norders, dimy, dimx))
                fluxratecube = np.zeros((ntimesteps, dimy, dimx))
            # Scale the flux for each order by the normalization factor passed as input
            for m in range(norders):
                image[m,:,:] = image[m,:,:] * normalization_scale[m]
            imflat = np.sum(image, axis=0)
            fluxratecube[t,:,:] = np.copy(imflat)

    # At this point, each image is a slope image (calibrated flux per second) for
    # a chunk of time that is either at the frame granularity or at the integration
    # granularity. It is time to divide in frame with the properly scaled flux as
    # happens during an integraiton.
    #ngroup = np.copy(simuPars.ngroup)
    #nint = np.copy(simuPars.nint)
    #frametime = np.copy(simuPars.frametime)
    print('nint={:}, ngroup={:}, frametime={:} sec '.format(nint,ngroup,frametime))

    # Initialize the exposure array containing up-the-ramp reads.
    exposure = np.zeros((nint, ngroup, dimy, dimx), dtype=float)
    if granularity == 'FRAME':
        # Then we already have a rate image for each individual read.
        for i in range(nint):
            cumulative = np.zeros((dimy, dimx))
            for g in range(ngroup):
                n = i*ngroup+g
                cumulative = cumulative + fluxratecube[n,:,:] * frametime
                print('i={:} g={:} n={:} flux={:} rate={:}'.format(i,g,n,np.sum(cumulative),np.sum(fluxratecube[n,:,:])))
                exposure[i, g, :, :] = np.copy(cumulative.reshape((1,1,dimy,dimx)))
    elif granularity == 'INTEGRATION':
        # We need to create ngroup reads per simulated rate image.
        for i in range(nint):
            for g in range(ngroup):
                n = np.copy(i)
                print('{:} {:}'.format(i,g))
                exposure[i, g, :, :] = fluxratecube[n, :, :].reshape((1, 1, dimy, dimx)) * frametime * (g+1)
    else:
        print('We are missing the granularity parameter in the simulation.')
        sys.exit()

    return(exposure)



def write_dmsready_fits(image, filename, os=1, xpadding=0, ypadding=0,
                        input_frame='sim', verbose=True, f277=False,
                        **kwargs):
    '''
    This script writes DMS ready fits files. It uses as input a simulated
    noiseless image to which it adds the required minimum set of FITS keywords
    required in order to minimally run the STScI pipeline. Based on
    the Fortran code specgen/utils/writefits_phdu.f90

    :param image: is a numpy array with 2,3 or 4 dimensions
             4 dim --> nint,ngroup,dimy,dimx are the dimensions expected
             3 dim --> ngroup,dimy,dimx are the dimensions expected
             2 dim --> dimy,dimx are the dimensions expected
             The input image is expected to be in the DMS orientation already
    :param filename: is the name of the fits file that will be written on disk
    :param os: is the oversampling integer 1 or more of the input image.
    :param input_frame: is either dms, native or sim. represents the coordinate frame of
            input image.
    :param verbose:
    :return:
    '''

    if verbose: print('Start of write_dmsready_fits')
    # Expects a numpy array with n dimensions =2 or 3, representing a single
    # integration. The size of each dimension is left open to accomodate
    # oversampled images or oversized images (to include order 0 for example).

    # First deal with possibly oversampled input images.
    # At the end of this exercice, a 4-dim array at native pixel size should exist.
    if (os >= 1) & (os % 1 == 0):
        size = np.shape(image)
        if len(size) == 4:
            # 4 dimensional. Assume nint,ngroup,dimy,dimx. Bin dimy,dimx dimensions.
            nint, ngroup, dimy, dimx = size
            if verbose: print('4 dimensional array')
            # Create a 4-dimensional array with native pixel size dimensions
            data = np.zeros((nint, ngroup, int(dimy / os), int(dimx / os)))
            # For each int and group, bin the image to native pixel size (handling flux properly)
            for i in range(nint):
                for j in range(ngroup):
                    #data[i, j, :, :] = downscale_local_mean(image[i, j, :, :], (os, os)) * os**2
                    data[i, j, :, :] = rebin(image[i, j, :, :], os)
        elif len(size) == 3:
            if verbose: print('3 dimensional array')
            nint = 1
            ngroup, dimy, dimx = size
            data = np.zeros((nint, ngroup, int(dimy / os), int(dimx / os)))
            for j in range(ngroup):
                #data[0, j, :, :] = downscale_local_mean(image[j, :, :], (os, os)) * os**2
                data[0, j, :, :] = rebin(image[j, :, :], os)
        elif len(size) == 2:
            if verbose: print('2 dimensional array')
            nint = 1
            ngroup = 1
            dimy, dimx = size
            data = np.zeros((nint, ngroup, int(dimy / os), int(dimx / os)))
            #data[0, 0, :, :] = downscale_local_mean(image, (os, os)) * os**2
            data[0, 0, :, :] = rebin(image, os)
        else:
            print('There is a problem with the image passed to write_dmsread_fits.')
            print('Needs to have 2 to 4 dimensions.')
            sys.exit()
        # Now that data is in native pixels, remove the padding
        if False:
            print('Bug tracing: size of data before removing padding')
            nint, ngroup, dimy, dimx = np.shape(data)
            print('nint={:}, ngroup={:}, dimy={:}, dimx={:}'.format(nint, ngroup, dimy, dimx))
            print('so?')
        data = data[:,:,ypadding:-ypadding,xpadding:-xpadding]

        # Reset the dimx, dimy parameters now that all is in native pixel size, unpadded
        nint, ngroup, dimy, dimx = np.shape(data)
    else:
        print('The oversampling of the input image should be 1 or higher integer. Stop')
        sys.exit()
    if verbose: print('nint={:}, ngroup={:}, dimy={:}, dimx={:}'.format(nint, ngroup, dimy, dimx))

    # Handle the case where the input_frame optional keyword is set
    if input_frame == 'dms':
        # Then the array is already in the desired coordinate frame, i.e: dms
        print('')  #
    elif input_frame == 'sim':
        # Need to flip along dimy axis
        data = np.flip(data, axis=2)
        nint, ngroup, dimy, dimx = np.shape(data)
    elif input_frame == 'native':
        # Need to mirror and rotate
        data = np.flip(np.flip(np.swapaxes(data, 2, 3), axis=3), axis=2)
        nint, ngroup, dimy, dimx = np.shape(data)
    else:
        # bad choice
        print('input_frame should be dms, sim or native.')
        sys.exit()

    # The format of a JWST file is:
    # A primary header, containing only a header with all important keywords.
    # An extension, EXTNAME=SCI, containing a minimal header and the data
    #
    # Here is an example thar runs successfully in the STScI pipeline:
    # SIMPLE  =                    T / file does conform to FITS standard
    # BITPIX  =                  -32 / number of bits per data pixel
    # NAXIS   =                    1 / number of data axes
    # NAXIS1  =                    0 / length of data axis 1
    # EXTEND  =                    T / FITS dataset may contain extensions
    # COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
    # COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H
    # NRSTSTRT=                    1 / / the number of resets at the start of the expo
    # NRESETS =                    1 / / the number of resets between integrations
    # DATE    = '2019-12-05T11:09:28.097' / / [yyyy-mm-ddThh:mm:ss.ss] UTC date file c
    # FILENAME= 'jw00001001001_0110100001_NISRAPID_cal_c.fits' / / Name of the file
    # DATAMODL= 'RampModel'          / / Type of data model
    # TELESCOP= 'JWST    '           / / Telescope used to acquire the data
    # DATE-OBS= '2020-02-05'         / / [yyyy-mm-dd] Date of observation
    # TIME-OBS= '11:08:45.000'       / / [hh:mm:ss.sss] UTC time at start of exposure
    # TARG_RA =            188.38685 / / Target RA at mid time of exposure
    # TARG_DEC=  -10.146173055555559 / / Target Dec at mid time of exposure
    # SRCTYPE = 'POINT   '           / / Advised source type (point/extended)
    # INSTRUME= 'NIRISS  '           / / Instrument used to acquire the data
    # DETECTOR= 'NIS     '           / / Name of detector used to acquire the data
    # FILTER  = 'CLEAR   '           / / Name of the filter element used
    # PUPIL   = 'GR700XD '           / / Name of the pupil element used
    # EXP_TYPE= 'NIS_SOSS'           / / Type of data in the exposure
    # READPATT= 'NISRAPID'           / / Readout pattern
    # NINTS   =                    1 / / Number of integrations in exposure
    # NGROUPS =                   10 / / Number of groups in integration
    # NFRAMES =                    1 / / Number of frames per group
    # GROUPGAP=                    0 / / Number of frames dropped between groups
    # TFRAME  =                5.494 / / [s] Time between frames
    # TGROUP  =                5.494 / / [s] Time between groups
    # DURATION=                  1.0 / / [s] Total duration of exposure
    # SUBARRAY= 'SUBSTRIP256'        / / Subarray used
    # SUBSTRT1=                    1 / / Starting pixel in axis 1 direction
    # SUBSTRT2=                 1793 / / Starting pixel in axis 2 direction
    # SUBSIZE1=                 2048 / / Number of pixels in axis 1 direction
    # SUBSIZE2=                  256 / / Number of pixels in axis 2 direction
    # FASTAXIS=                   -2 / / Fast readout axis direction
    # SLOWAXIS=                   -1 / / Slow readout axis direction
    # END
    #
    # XTENSION= 'IMAGE   '           / IMAGE extension
    # BITPIX  =                  -32 / number of bits per data pixel
    # NAXIS   =                    4 / number of data axes
    # NAXIS1  =                 2048 / length of data axis 1
    # NAXIS2  =                  256 / length of data axis 2
    # NAXIS3  =                   10 / length of data axis 3
    # NAXIS4  =                    1 / length of data axis 4
    # PCOUNT  =                    0 / required keyword; must = 0
    # GCOUNT  =                    1 / required keyword; must = 1
    # EXTNAME = 'SCI     '
    # END

    # Create primary HDU
    prim_hdu = fits.PrimaryHDU()
    phdr = prim_hdu.header
    phdr.set('NRSTSTRT', 1, 'the number of resets at the start of the expo')
    phdr.set('NRESETS', 1, 'the number of resets between integrations')
    phdr.set('DATE', '2019-12-05T11:09:28.097', '[yyyy-mm-ddThh:mm:ss.ss] UTC date file cre')
    phdr.set('FILENAME', filename, 'Name of the file')
    phdr.set('DATAMODL', 'RampModel', 'Type of data model')
    phdr.set('TELESCOP', 'JWST', 'Telescope used to acquire the data')
    phdr.set('DATE-OBS', '2020-02-05', '[yyyy-mm-dd] Date of observation')
    phdr.set('TIME-OBS', '11:08:45.000', '[hh:mm:ss.sss] UTC time at start of exposure')
    phdr.set('TARG_RA', 188.38685, 'Target RA at mid time of exposure')
    phdr.set('TARG_DEC', -10.14617305555556, 'Target Dec at mid time of exposure')
    phdr.set('SRCTYPE', 'POINT', 'Advised source type (point/extended)')
    phdr.set('INSTRUME', 'NIRISS', 'Instrument used to acquire the data')
    phdr.set('DETECTOR', 'NIS', 'Name of detector used to acquire the data')
    if f277:
        phdr.set('FILTER', 'F277W', 'Name of the filter element used')
    else:
        phdr.set('FILTER', 'CLEAR', 'Name of the filter element used')
    phdr.set('PUPIL', 'GR700XD', 'Name of the pupil element used')
    phdr.set('EXP_TYPE', 'NIS_SOSS', 'Type of data in the exposure')
    phdr.set('READPATT', 'NISRAPID', 'Readout pattern')
    # Check that the data is a valid subarray
    if (dimy != 2048) & (dimy != 256) & (dimy != 96):
        print('WARNING. The array Y-axis needs to be 96, 256 or 2048 pixels.')
    if dimx != 2048:
        print('WARNING. The array X-axis needs to be 2048 pixels.')
    phdr.set('NINTS', nint, 'Number of integrations in exposure')
    phdr.set('NGROUPS', ngroup, 'Number of groups in integration')
    phdr.set('NFRAMES', 1, 'Number of frames per group')
    phdr.set('GROUPGAP', 0, 'Number of frames dropped between groups')
    phdr.set('TFRAME', 5.494, '[s] Time between frames')
    phdr.set('TGROUP', 5.494, '[s] Time between groups')
    phdr.set('DURATION', 1.0, '[s] Total duration of exposure')

    if (dimy == 96) & (dimx == 2048):
        subarray = 'SUBSTRIP96'
        substrt1, subsize1 = 1, 2048
        substrt2, subsize2 = 1803, 96
        # Put reference pixels at zero
        data[:, :, :, 0:4] = 0.0
        data[:, :, :, -4:] = 0.0
        tpix, namps, colsover, rowsover = 1e-5, 1, 12, 2
        tframe = tpix * (dimy / namps + colsover) * (dimx + rowsover)
        phdr.set('TFRAME', tframe, '[s] Time between frames')
        phdr.set('TGROUP', tframe, '[s] Time between groups')

    elif (dimy == 256) & (dimx == 2048):
        subarray = 'SUBSTRIP256'
        substrt1, subsize1 = 1, 2048
        substrt2, subsize2 = 1793, 256
        # Put reference pixels at zero
        data[:, :, :, 0:4] = 0.0
        data[:, :, :, -4:] = 0.0
        data[:, :, -4:, :] = 0.0
        tpix, namps, colsover, rowsover = 1e-5, 1, 12, 2
        tframe = tpix * (dimy / namps + colsover) * (dimx + rowsover)
        phdr.set('TFRAME', tframe, '[s] Time between frames')
        phdr.set('TGROUP', tframe, '[s] Time between groups')
    elif (dimy == 2048) & (dimx == 2048):
        subarray = 'FULL'
        substrt1, subsize1 = 1, 2048
        substrt2, subsize2 = 1, 2048
        data[:, :, :, 0:4] = 0.0
        data[:, :, :, -4:] = 0.0
        data[:, :, -4:, :] = 0.0
        data[:, :, 0:4, :] = 0.0
        tpix, namps, colsover, rowsover = 1e-5, 4, 12, 1
        tframe = tpix * (dimy / namps + colsover) * (dimx + rowsover)
        phdr.set('TFRAME', tframe, '[s] Time between frames')
        phdr.set('TGROUP', tframe, '[s] Time between groups')
    else:
        print('WARNING. image size not correct.')
        subarray = 'CUSTOM'
        substrt1, subsize1 = 1, dimx
        substrt2, subsize2 = 1, dimy
    phdr.set('SUBARRAY', subarray, 'Subarray used')
    phdr.set('SUBSTRT1', substrt1, 'Starting pixel in axis 1 direction')
    phdr.set('SUBSTRT2', substrt2, 'Starting pixel in axis 2 direction')
    phdr.set('SUBSIZE1', subsize1, 'Number of pixels in axis 1 direction')
    phdr.set('SUBSIZE2', subsize2, 'Number of pixels in axis 2 direction')
    phdr.set('FASTAXIS', -2, 'Fast readout axis direction')
    phdr.set('SLOWAXIS', -1, 'Slow readout axis direction')
    phdr.set('SIMOVRSP', int(os), 'Oversampling used in the simulation')

    # keywords related to the noise sources and reference files
    for key, value in kwargs.items():
        # reference files
        if key == 'dark_ref': phdr.set('DARKREF', value, 'Dark reference file')
        if key == 'flat_ref': phdr.set('FLATREF', value, 'Flat reference file')
        if key == 'superbias_ref': phdr.set('SBIASREF', value, 'Superbias reference file')
        if key == 'nlcoeff_ref': phdr.set('NLCOEREF', value, 'Non-linearity coefficients reference file')
        if key == 'zodi_ref': phdr.set('ZODIREF', value, 'Zodiacal background reference file')
        if key == 'nonlin_ref': phdr.set('NLREF', value, 'Non-linearity reference file')
        # noise sources
        if key == 'readout': phdr.set('READNS', value, 'Readout Noise Source included?')
        if key == 'zodibackg': phdr.set('ZODINS', value, 'Zodiacal light background Noise Source included?')
        if key == 'photon': phdr.set('PHOTNS', value, 'Photon Noise Source included?')
        if key == 'superbias': phdr.set('SBIASNS', value, 'Superbias Noise Source included?')
        if key == 'flatfield': phdr.set('FLATNS', value, 'Flatfield Noise Source included?')
        if key == 'nonlinearity': phdr.set('NONLINNS', value, 'Non-linearity Noise Source included?')
        if key == 'oneoverf': phdr.set('ONEOFNS', value, 'One over f Noise Source included?')
        if key == 'darkcurrent': phdr.set('DARKNS', value, 'Readout Noise Source included?')
        if key == 'cosmicray': phdr.set('CRAYNS', value, 'Cosmic ray Noise Source included?')

    # Create extension HDU
    ext_hdu = fits.ImageHDU(data)
    # Add keywords to the extension header
    xhdr = ext_hdu.header
    xhdr.set('EXTNAME', 'SCI')
    ## In order to prevent wrapping of negative values to 65535, add some bias
    #xhdr.set('BZERO', 32768+10000)
    # Create a list of HDU with primary and extension HDUs
    hdulist = fits.HDUList([prim_hdu, ext_hdu])
    # Save on disk
    hdulist.writeto(filename, overwrite=True)
    hdulist.close()
    if verbose: print(
        '{:} was written with dimensions nint={:}, ngroup={:}, dimy={:}, dimx={:}, subarray={:}'.format(filename,
                                                                                                        nint, ngroup,
                                                                                                        dimy, dimx,
                                                                                                        subarray))
    if verbose: print('write_dmsready_fits completed successfuly.')


def aperture_extract(exposure, x_input, aperture_map, mask=None):
    '''
    Make a box extraction of exactly box_size.
    Author of original version: Antoine Darveau Bernier
    Parameters
    ----------
    data: 2d array of shape (n_row, n_columns)
        scidata
    lam_col: 1d array of shape (n_columns)
        wavelength associated with each columns. If not given,
        the column position is taken.
    cols: numpy valid index
        Which columns to extract
    box_weights: 2d array, same shape as data
        pre-computed weights for box extraction.
        If not given, compute with `get_aperture_weights`
        and `n_pix` and `aperture` will be needed.
    box_size: scalar (float or integer)
        length of the box in pixels
    mask: 2d array, boolean, same shape as data
        masked pixels
    Output
    ------
    (flux)
    '''



    # Check that requested x position indices are integers
    if all(i % 1 == 0 for i in x_input):
        # Convert x to integer indices
        x = x_input.astype(int)
    else:
        print('ERROR: x_input positions must be integer column indices, not floats')
        sys.exit()

    # Determine how many integrations and image size
    if exposure.ndim == 3:
        nintegration, dimy, dimx = np.shape(exposure)
    elif exposure.ndim == 2:
        nintegration = 1
        dimy, dimx = np.shape(exposure)
    else:
        print('exposure has to be of 2 or 3 dimensions. Check.')
        sys.exit()

    # Define mask if not given
    if mask is None:
        # False everywhere
        mask = np.zeros((dimy,dimx), dtype=bool)

    print('mask', np.shape(mask))
    print('x=',x)

    flux = np.zeros((nintegration, dimx)) * np.nan
    # Loop for all integrations
    for n in range(nintegration):
        # Make a copy of arrays with only needed columns
        # so it is not modified outside of the function
        if exposure.ndim == 3:
            #data = np.copy(exposure[n, :, x])
            data = exposure[n][:,x]
        else:
            #data = np.copy(exposure[:, x])
            data = exposure[:,x]
        if False:
            # Extract the aperture and mask for that same column
            box_weights = aperture_map[:, x].copy()
            mask = mask[:, x].copy()
            # Initialize the output with nans
            out = np.ones_like(x) * np.nan
            # Mask potential nans in data
            mask_nan = np.isnan(data)
            # Combine with user specified mask
            mask = (mask_nan | mask)
            # Apply to weights
            box_weights[mask_nan] = np.nan
            # Normalize considering the masked pixels
            norm = np.nansum(box_weights**2, axis=0)
            # Valid columns index
            idx = norm > 0
            # Normalize only valid columns
            out[idx] = np.nansum(box_weights*data, axis=0)[idx]
            out[idx] /= norm[idx]
        else:
            out = np.sum(data * aperture_map, axis=0)

        flux[n, :] = np.copy(out)

    if exposure.ndim == 2:
        flux = flux[0,:]

    return flux



def box_aperture(exposure, x_index, y_index, box_width=None, os=1):
    '''
    Creates a 2D image representing the box aperture. Normalized such that
    each column adds to 1.
    Author of original version: Antoine Darveau Bernier
    :param exposure:
    :param x_index:
    :param y_index:
    :param box_width:
    :return:
    '''

    #TODO: Check that the output aperture is centered on the trace center, not 1 pixel off.

    if box_width is None:
        box_width = 30*os

    # Check that requested x position indices are integers
    if all(i % 1 == 0 for i in x_index):
        # Convert x to integer indices
        x = x_index.astype(int)
    else:
        print('ERROR: box_aperture - x_input positions must be integer column indices, not floats')
        print(x_index)
        for i in range(x_index):
            print(i, x_index[i])
        sys.exit()

    # shape of the aperture map
    shape = np.shape(exposure)
    if np.size(shape) == 2:
        dimy, dimx = shape
    elif np.size(shape) == 3:
        nint, dimy, dimx = shape
    else:
        print('ERROR: exposure should have 2 or 3 dimensions.')
        sys.exit()
    shape = (dimy, dimx)

    # Box limits for all cols (+/- n_pix/2)
    row_lims = [y_index - box_width / 2, y_index + box_width / 2]
    row_lims = np.array(row_lims).T
    # Compute weights
    # For max lim:
    # w = center + n_pix/2 - (rows - 0.5)
    # For min lim:
    # w = rows + 0.5 - center + n_pix/2
    rows = np.indices(shape)[0]
    weights = rows[:, :, None] - row_lims[None, :, :]
    weights *= np.array([1, -1])
    weights += 0.5
    # Between 0 and 1
    weights = np.clip(weights, 0, 1)
    # Keep min weight
    weights = weights.min(axis=-1)
    # Normalize
    #weights /= weights.sum(axis=0)
    # Return with the same shape as aperture and
    # with zeros where the aperture is not define
    out = np.zeros(shape, dtype=float)
    out[:, x] = weights
    # Mirror about vertical axis
    out = np.fliplr(out)

    return out

def elecflux_to_flambda(flux, wavelength, area=25.0, gain=1.6):

    # From flux [adu/pixel/sec] to Flambda [J/sec/m2/um]
    h = 6.62606957e-34
    c = 3e+8
    #gain = 1.6 # e-/adu
    #Area = 25.0 # m2
    Eph = h * c / (wavelength*1e-6) # J/photon
    dispersion = np.zeros_like(wavelength) # um/pixel
    dispersion[1:] = np.abs(wavelength[1:]-wavelength[0:-1])
    dispersion[0] = dispersion[1]
    Flambda = flux * gain * Eph / area / dispersion

    return Flambda


def write_spectrum(wavelength, delta_wave, MJD, integtime, flux, flux_err, fitsname):
    '''
    Writes a MEF FITS file containing the measured spectrum for a time-series.
    :param wavelength: array of wavelengths (microns) for each of the flux measurements
    :param delta_wave: array
    :param MJD: Modified Julian date at the mid-time of each integration.
    :param integtime:
    :param flux: Measured flux at a given wavelength spanning a given delta_wave range. Units
                 are electrons integrated over the full integration during integtime. flux has
                 dimensions of n (wavelength) x m (BJD)
    :param flux_err: Measured flux uncertainty. Same units and dimensions as flux.
    :return:
        The FITS file that contains as many extensions as there are time steps. The time
        is captured in each extension header as MJD-OBS and the integration time in the header
        as INTTIME in seconds.
    '''
    # Create a primary header with the JDOFFSET keyword
    hdr = fits.Header()
    hdr['JDOFFSET'] = 2459518
    primary = fits.PrimaryHDU(header=hdr)
    hdu = fits.HDUList([primary])
    # Append the extensions (1 for each time stamp)
    for i in range(np.size(MJD)):
        h = fits.BinTableHDU(Table([wavelength, delta_wave, flux[i, :], flux_err[i, :]],
                                   names=('wavelength', 'wavelength_width', 'flux', 'flux_err'),
                                   #dtype=('F10.8', 'F10.8', 'F16.4', 'F16.4')))
                                   dtype=(np.float, np.float, np.float, np.float)))
        h.header['MJD-OBS'] = MJD[i]
        h.header['INTTIME'] = integtime[i]
        hdu.append(h)
    hdu.writeto(fitsname, overwrite=True)
    return

def read_spectrum(fitsname):
    '''
    Reads a FITS spectrum created by write_spectrum()
    :param fitsname:
    :return:
    '''

    return MJD, integtime, wavelength, wavelength_delta, flux, flux_err



"""
@author: caroline

Utility functions for reading in spectra from a grid
"""

# import modules
#import numpy as np
#import matplotlib.pyplot as plt
from astropy import io as aio
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel, Trapezoid1DKernel
from copy import deepcopy

# %%
# Utilities functions

class PlanetSpectrum:
    def __init__(self, caselist=None, params_dict={"ComposType": "ChemEqui", "Metallicity": 1.,
                                                   "CtoO": 0.54, "pCloud": 1e3, "cHaaze": 1e-10, "pQuench": 1e-99,
                                                   "Tint": 100., "HeatDistFactor": 0.25, "BondAlbedo": 0.1},
                 planet_name="HAT_P_1_b", path_files="./"):
        """
        Object that contains all info about a spectrum (wavelength, transit depth, planet name,
        atmosphere model params)
        """

        spec_csv_path = get_spec_csv_path(caselist=caselist, params_dict=params_dict,
                                          planet_name=planet_name, path_files=path_files)

        f = aio.ascii.read(spec_csv_path)

        self.spec_csv_path = spec_csv_path
        self.params = params_dict
        self.planet_name = planet_name

        self.wavelength_um = np.array(f["wave"])
        self.dppm = np.array(f["dppm"])
        self.rprs = np.sqrt(np.array(f["dppm"]) * 1e-6)

        # define res_power the way it is in scarlet
        ind = 3
        self.res_power = self.wavelength_um[ind] / (self.wavelength_um[ind + 1] - self.wavelength_um[ind])

    def plot(self, ax=None, res_power=None, lab=None,
             RpRs_or_dppm="dppm", wavelim_um=None, **kwargs):
        """
        Plot the spectrum at a given resolving power (res_power, if None keep original resolving power)
        ax: axis on which to plot (if None, making one)
        RpRs_or_dppm: plot Rp/Rs if "RpRs" otherwise plot transit depth in ppm if "dppm"
        lab: optional label
        wavelim_um: limits of the wavelengths shown on the plots in microns. of the form [min, max]
        kwargs: additional keyword arguments for plotting
        """
        if RpRs_or_dppm == "dppm":
            spec = self.dppm
            ylab = "Transit depth [ppm]"
        elif RpRs_or_dppm == "RpRs":
            spec = self.rprs
            ylab = r"R$_p$/R$_s$"

        if res_power is not None:
            if res_power < self.res_power:
                kernel = Gaussian1DKernel(self.res_power / res_power / 2.35)
                l = int(kernel.shape[0] / 2)
                wave = self.wavelength_um[l:-l]
                spec = convolve(spec, kernel)[l:-l]
            rp = str(res_power)
        else:
            wave = self.wavelength_um
            rp = str(int(self.res_power * 10.) / 10.)

        if lab is None:
            lab = "met=" + str(self.params["Metallicity"])
            lab = lab + "; C/O=" + str(self.params["CtoO"])
            lab = lab + "; pcloud=" + str(self.params["pCloud"] / 100.) + " mbar"

        ax.plot(wave, spec, label=lab + '; R=' + rp, **kwargs)
        ax.set_ylabel(ylab)
        ax.set_xlabel(r"Wavelength [$\mu$m]")
        if wavelim_um is not None:
            ax.set_xlim(wavelim_um)
        ax.legend()

        return ax


def get_atmosphere_cases(planet_name, path_files="./", return_caselist=True, print_info=True):
    """
    Get list of cases (sets of atmospheric properties) available for a given planet
    planet_name: str, name of the planet
    path_files: str, path to the grid of models for all planets
    return_caselist: if True, returns an astropy table containing the information on all models for this planet
    print_info: if True, print sets of parameters covered by the grid
    """

    summary_file_path = path_files + planet_name + "/" + "CaseList.csv"

    columns = ['ComposType', 'Metallicity', 'CtoO', 'pCloud', 'cHaaze', 'pQuench', 'Tint', 'HeatDistFactor',
               'BondAlbedo']
    summary = aio.ascii.read(summary_file_path)

    if print_info:
        print("\n** Cases for ", planet_name, ": **\n")
        print(summary[columns])
        print("\n** Grid span: **\n")
        for col in columns:
            print(col + ":", np.unique(np.array(summary[col])))

    if return_caselist:
        return summary


def make_default_params_dict():
    """
    Make a default parameters dictionary
    """
    params_dict = {"ComposType": "ChemEqui", "Metallicity": 1.,
                   "CtoO": 0.54, "pCloud": 1e3, "cHaaze": 1e-10, "pQuench": 1e-99,
                   "Tint": 100., "HeatDistFactor": 0.25, "BondAlbedo": 0.1}
    return params_dict


def get_spec_csv_path(caselist=None, params_dict={"ComposType": "ChemEqui", "Metallicity": 1.,
                                                  "CtoO": 0.54, "pCloud": 1e3, "cHaaze": 1e-10, "pQuench": 1e-99,
                                                  "Tint": 100., "HeatDistFactor": 0.25, "BondAlbedo": 0.1},
                      planet_name="HAT_P_1_b", path_files="./"):
    """
    Returns full path to the csv file that contains the planet's spectrum
    """
    if caselist is None:
        caselist = ut.get_atmosphere_cases(planet_name, path_files=path_files, return_caselist=True,
                                           print_info=False)
    case = deepcopy(caselist)
    for p in params_dict.keys():
        case = case[case[p] == params_dict[p]]

    spec_csv_path = path_files + planet_name + "/" + case["fileroot"].data[0] + "Spectrum_FullRes_dppm.csv"
    return spec_csv_path


def towardtrace(file_rateints, file_model):

    stack_integrations(file_rateints, '/genesis/jwst/userland-soss/loic_review/stack.fits')
    clean1, clean2 = decontaminate_exposure(file_rateints, file_model)

    print('towardtrace - medianing clean1...')
    stack_clean1 = np.nanmedian(clean1, axis=0)
    print('medianing clean2...')
    stack_clean2 = np.nanmedian(clean2, axis=0)

    print('writing clean stacks to disk')
    h = fits.PrimaryHDU(stack_clean1)
    h.writeto('/genesis/jwst/userland-soss/loic_review/stackclean1.fits', overwrite=True)
    h = fits.PrimaryHDU(stack_clean2)
    h.writeto('/genesis/jwst/userland-soss/loic_review/stackclean2.fits', overwrite=True)


def stack_integrations(filename, stackname, integ_start=0, integ_end=-1):

    h = fits.open(filename)
    allintegs = h[1].data
    nints = h[0].header['NINTS']

    print('Stacking rateints')
    stack = np.nanmedian(allintegs[integ_start:integ_end,:,:], axis=0)
    print('Writing on disk...')
    hout = fits.PrimaryHDU(stack)
    hout.writeto(stackname, overwrite=True)

def decontaminate_exposure(file_rateints, file_model):

    '''Decontaminate the rateint exposure using the ATOCA models'''

    print('Decontaminating integrations - reading rateints')
    h = fits.open(file_rateints)
    allintegs = h[1].data

    # The ATOCA models has 6 extensions, 1,2 are the models in orders 1 and 2
    # 3,4,5 are the apertures in orders 1 to 3. Ext 6 is the number of good pixels in
    # the aperture.
    print('Reading the models')
    m = fits.open(file_model)
    model1 = m[1].data
    model2 = m[2].data

    # Subtract model from rateints, for each integration
    clean1 = allintegs - model2
    clean2 = allintegs - model1

    print('Retrun the cleaned integrations...')
    return clean1, clean2

def extract_seeded_trace(tmp_path, extract1d_file):



    clear_trace_000175.fits


    from jwst import datamodels

    # Read the extracted spectra created by the extract1d step
    extract1d = datamodels.open(extract1d_file)

    # List all clear_trace_??????.fits files typically found in the tmp/ directory of every simulation run
    imagelist = glob.glob(tmp_path + 'clear_trace_??????.fits')


    # Initialize the output model and output references (model of the detector and box aperture weights).
    trace = datamodels.MultiSpecModel()
    trace.update(extract1d)  # Copy meta data from input to output.

    # spectra are stored at indice 1 (order 1), then 2 (order2) then 3 (order 3) then 4 (order 1, 2nd time step), ...
    nint = trace[0].header['NINTS']
    norder = 3

    wavelength = np.zeros((nint, norder, 2048))
    flux = np.zeros((nint, norder, 2048))
    order = np.zeros((nint, norder, 2048))
    integ = np.zeros((nint, norder, 2048))

    for ext in range(np.size(data) - 2):
        m = data[ext + 1].header['SPORDER']
        i = data[ext + 1].header['INT_NUM']
        wavelength[i - 1, m - 1, :] = data[ext + 1].data['WAVELENGTH']
        flux[i - 1, m - 1, :] = data[ext + 1].data['FLUX']



    # bla bla

    # Copy spectral data for each order into the output model.
    for order in wavelengths.keys():

        table_size = len(wavelengths[order])

        out_table = np.zeros(table_size, dtype=datamodels.SpecModel().spec_table.dtype)
        out_table['WAVELENGTH'] = wavelengths[order]
        out_table['FLUX'] = fluxes[order]
        out_table['FLUX_ERROR'] = fluxerrs[order]
        out_table['DQ'] = np.zeros(table_size)
        out_table['BACKGROUND'] = col_bkg
        out_table['NPIXELS'] = npixels[order]

        spec = datamodels.SpecModel(spec_table=out_table)

        # Add integration number and spectral order
        spec.spectral_order = order_str_2_int[order]
        spec.int_num = i + 1  # integration number starts at 1, not 0 like python

        output_model.spec.append(spec)

    result.write(spectrum_file)
