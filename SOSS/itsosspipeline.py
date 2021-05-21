

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
from tqdm.notebook import tqdm as tqdm_notebook

import sys
#sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/')
import specgen.spgen as spgen

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


def starlimbdarkening(wave_angstrom, ld_type='flat'):
    """
    Generates an array of limb darkening coefficients when creating a star atmosphere model
    :return:
    """

    if ld_type == 'flat':
        # flat
        a1234 = [0, 0, 0, 0]
        ld_coeff = np.zeros((np.size(wave_angstrom), 4))
        #ld_coeff[]
        #print(np.shape(ld_coeff))
    else:
        # no limb darkening - flat intensity
        ld_coeff = np.zeros((np.size(wave_angstrom), 4))

    return ld_coeff


def starmodel(simuPars, pathPars, verbose=True):
    """
    From the simulation parameter file - determine what star atmosphere model
    to read or generate.
    :return:
    """
    if simuPars.modelfile == 'BLACKBODY' and simuPars.bbteff:
        if verbose: print('Star atmosphere model type: BLACKBODY')
        model_angstrom = np.linspace(4000, 55000, 200001)
        model_flambda = planck(model_angstrom/10000, simuPars.bbteff)
        model_ldcoeff = starlimbdarkening(model_angstrom)
    elif simuPars.modelfile == 'CONSTANT_FLAMBDA':
        if verbose: print('Star atmosphere model type: CONSTANT_FLAMBDA')
        model_angstrom = np.linspace(4000, 55000, 200001)
        model_flambda = np.ones(np.size(model_angstrom))
        model_ldcoeff = starlimbdarkening(model_angstrom)
    elif simuPars.modelfile == 'CONSTANT_FNU':
        if verbose: print('Star atmosphere model type: CONSTANT_FNU')
        model_angstrom = np.linspace(4000, 55000, 200001)
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
    else:
        if verbose: print('Star atmosphere model assumed to be on disk.')
        # Read Stellar Atmosphere Model (wavelength in angstrom and flux in energy/sec/wavelength)
        model_angstrom, model_flambda, model_ldcoeff = spgen.readstarmodel(
                    pathPars.path_starmodelatm + simuPars.modelfile,
                    simuPars.nmodeltype, quiet=False)

    return model_angstrom, model_flambda, model_ldcoeff


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
        frametime = 2.214   # seconds per read
    elif simuPars.subarray == 'SUBSTRIP256':
        frametime = 5.494   # seconds per read
    elif simuPars.subarray == 'FF':
        frametime = 10.737  # seconds per read
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



def generate_traces(savingprefix, pathPars, simuPars, tracePars, throughput,
                    star_angstrom, star_flux, ld_coeff,
                    planet_angstrom, planet_rprs,
                    timesteps, granularitytime):
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
    :return:
    '''

    # output is a cube (1 slice per spectral order) at the requested
    # pixel oversampling.

    # Resample star and planet models to common uniform in wavelength grid.
    print('Resampling star and planet model')
    # Get wavelength spacing to use for resampling
    dw, dwflag = spgen.get_dw(star_angstrom, planet_angstrom, simuPars, tracePars)
    # dw = dw/100
    print("Wavelength spacing (angstroms): ", dw, dwflag)
    # Resample onto common grid.
    star_angstrom_bin, star_flux_bin, ld_coeff_bin, planet_angstrom_bin, planet_rprs_bin = \
        spgen.resample_models(dw, star_angstrom, star_flux, ld_coeff, planet_angstrom, planet_rprs, simuPars, tracePars)


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
    kernels, kernels_wv = spgen.readkernels(pathPars.path_monochromaticpsfs)
    # resize Kernels
    # limit oversampling to be: 1<10
    kernel_resize = []
    for k in kernels:
        kernel_resize.append(resize(k, (128 * simuPars.noversample, 128 * simuPars.noversample)))

    #The number of images (slices) that will be simulated is equal the number of orders
    # Don't worry, that 'cube' will be merged down later after flux normalization.
    #nframes = np.size(simuPars.orderlist)

    # Defines the dimensions of the arrays, depends on the oversanmpling
    xmax=simuPars.xout*simuPars.noversample
    ymax=simuPars.yout*simuPars.noversample

    # Initialize the array that will contain all orders at a given time step
    nimage = len(simuPars.orderlist)
    convolved_image=np.zeros((nimage,ymax,xmax))

    # list of temporary filenames
    filelist = []
    # Loop over all time steps for the entire Time-Series duration
    for t in range(len(timesteps)):
        # Loop over all spectral orders
        for m in range(len(simuPars.orderlist)):
            spectral_order = int(np.copy(simuPars.orderlist[m]))  # very important to pass an int here or tracepol fails
            currenttime = second2day(timesteps[t])
            exposetime = second2day(granularitytime)
            print('Time step {:} minutes - Order {:}'.format(currenttime/60, spectral_order))
            pixels=spgen.gen_unconv_image(simuPars, throughput, star_angstrom_bin, star_flux_bin,
                                      ld_coeff_bin, planet_rprs_bin,
                                      currenttime, exposetime, solin, spectral_order, tracePars)

            pixels_t=np.copy(pixels.T)

            #Enable threads (not working?!?!)
            #pyfftw.config.NUM_THREADS = 1 #do not need multi-cpu for pools
            #with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft):
            # Turn on the cache for optimum performance
            #pyfftw.interfaces.cache.enable()

            #do the convolution
            x=pixels_t*0+1.0e-10

            nwv=len(kernels_wv) #number of wavelengths to process
            #pbar = tqdm_notebook(total=nwv)  #Will make a progressbar to monitor processing.
            ncpu = 16
            pool = mp.Pool(processes=ncpu)  #Use lots of threads - because we can!

            #arguments = (pixels_t, kernel_resize, kernels_wv, wv_idx, simuPars, spectral_order, tracePars,)
            #results = [pool.apply_async(spgen.convolve_1wv, args = (pixels_t, kernel_resize, kernels_wv, wv_idx, simuPars, spectral_order, tracePars,), callback=barupdate) for wv_idx in range(nwv)]
            results = [pool.apply_async(spgen.convolve_1wv, args = (pixels_t, kernel_resize, kernels_wv, wv_idx, simuPars, spectral_order, tracePars,)) for wv_idx in range(nwv)]

            pixels_c = [p.get() for p in results]

            pool.close()
            pool.join()

            #bring together the results
            x=pixels_t*0+1.0e-10
            for p in pixels_c:
                x+=p

            pixels_c=None #release Memory

            convolved_image[m,:,:] = np.copy(x)

            # Sum in the flux for that order
            actual_counts = np.sum(convolved_image[m,:,:])
            print('Actual counts measured on the simulation = {:} e-/sec'.format(actual_counts))
            print()

        tmpfilename = write_intermediate_fits(convolved_image, savingprefix, t)
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



def rebin(image2D, noversampling):
    """
    Takes an oversampled image and bins it down to native pixel size, taking
    the mean of the pixel values.
    :param arr:
    :param noversampling:
    :return:
    """
    dimy, dimx = np.shape(image2D)
    newdimy, newdimx = int(dimy/noversampling), int(dimx/noversampling)
    shape = (newdimy, image2D.shape[0] // newdimy,
             newdimx, image2D.shape[1] // newdimx)

    return image2D.reshape(shape).mean(-1).mean(1)


def write_simu_fits(image, filename):

    # Writes the simulated data set as simulated, independently of the DMS constraints
    hdu = fits.PrimaryHDU(image)
    # Save on disk
    hdu.writeto(filename, overwrite=True)

    return

def write_intermediate_fits(image, savingprefix, timestep_index):
    # Write image for a single time step. Differents spectral orders stored in third dimension
    # of the array. filename is the name of the final product. Intermediate filenames will
    # be forged based on that.

    directory_name = os.path.dirname(savingprefix)
    filename_current = savingprefix+'_{:06d}'.format(timestep_index)+'.fits'

    # Create a list of HDU with primary and extension HDUs
    hdu = fits.PrimaryHDU(image)

    # create a directory if it does not yet exists
    if os.path.exists(directory_name) is False:
        os.mkdir(directory_name)

    # Save on disk
    hdu.writeto(filename_current, overwrite=True)

    return(filename_current)

def write_dmsready_fits_init(imagelist, normalization_scale,
                             ngroup, nint, frametime, granularity,
                             verbose=None):
    if verbose:
        print('Entered write_dmsready_fits_init')
        print('imagelist =', imagelist)
        print('nint={:}, ngroup={:}, frametime={:} sec '.format(nint, ngroup, frametime))
        print('Granularity = {:}'.format(granularity))

    ntimesteps = len(imagelist)
    for t in range(ntimesteps):
        # Read the current image
        hdu = fits.open(imagelist[t])
        image = hdu[0].data
        # Create the cube of reads (if first iteration in loop)
        if t == 0:
            # First image, use dimensiosn and create a large cube
            norders, dimy, dimx = np.shape(image)
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



def write_dmsready_fits(image, filename, os=1, input_frame='sim', verbose=True, f277=False):
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
    :param input_frame: is either dms, native or sim. represents the coorinate frame of
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
                    data[i, j, :, :] = downscale_local_mean(image[i, j, :, :], (os, os)) * os**2
        elif len(size) == 3:
            if verbose: print('3 dimensional array')
            nint = 1
            ngroup, dimy, dimx = size
            data = np.zeros((nint, ngroup, int(dimy / os), int(dimx / os)))
            for j in range(ngroup):
                data[0, j, :, :] = downscale_local_mean(image[j, :, :], (os, os)) * os**2
        elif len(size) == 2:
            if verbose: print('2 dimensional array')
            nint = 1
            ngroup = 1
            dimy, dimx = size
            data = np.zeros((nint, ngroup, int(dimy / os), int(dimx / os)))
            data[0, 0, :, :] = downscale_local_mean(image, (os, os)) * os**2
        else:
            print('There is a problem with the image passed to write_dmsread_fits.')
            print('Needs to have 2 to 4 dimensions.')
            sys.exit()
        # Reset the dimx, dimy parameters now that all is in native pixel size
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
    # TFRAME  =                5.491 / / [s] Time between frames
    # TGROUP  =                5.491 / / [s] Time between groups
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
    phdr.set('TFRAME', 5.491, '[s] Time between frames')
    phdr.set('TGROUP', 5.491, '[s] Time between groups')
    phdr.set('DURATION', 1.0, '[s] Total duration of exposure')

    if (dimy == 96) & (dimx == 2048):
        subarray = 'SUBSTRIP96'
        substrt1, subsize1 = 1, 2048
        substrt2, subsize2 = 1803, 96
    elif (dimy == 256) & (dimx == 2048):
        subarray = 'SUBSTRIP256'
        substrt1, subsize1 = 1, 2048
        substrt2, subsize2 = 1793, 256
    elif (dimy == 2048) & (dimx == 2048):
        subarray = 'FULL'
        substrt1, subsize1 = 1, 2048
        substrt2, subsize2 = 1, 2048
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

    # Create extension HDU
    ext_hdu = fits.ImageHDU(data)
    # Add keywords to the extension header
    xhdr = ext_hdu.header
    xhdr.set('EXTNAME', 'SCI')
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
    Make a box extraction of exactly box_size
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
            data = np.copy(exposure[n, :, x])
        else:
            data = np.copy(exposure[:, x])
        print('data',np.shape(data))
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

        flux[n, :] = np.copy(out)

    if exposure.ndim == 2:
        flux = flux[0,:]

    return flux



def box_aperture(exposure, x_index, y_index, box_width=None):

    if box_width is None:
        box_width = 30

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
    weights /= weights.sum(axis=0)
    # Return with the same shape as aperture and
    # with zeros where the aperture is not define
    out = np.zeros(shape, dtype=float)
    out[:, x] = weights
    return out