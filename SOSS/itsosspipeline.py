

# Read PATH config file

# Python imports

# Read Model parameters config file

# Read Planet model, star model, throughput file, trace parameters

# Resample planet and star on same grid

# Transit model setup

# Read PSF kernels

# Loop on orders to create convolved image

import sys

import numpy as np

import spgen as spgen

from skimage.transform import resize

from tqdm.notebook import tqdm as tqdm_notebook

import multiprocessing as mp

import os.path

from astropy.io import fits

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

def second2day(time_seconds):
    return(time_seconds / (3600.0*24.0))


def hour2day(time_hours):
    return(time_hours / 24.0)


def barupdate(result):
    # Used to monitor progress for multiprocessing pools
    pbar.update()

def frames_to_exposure(frameseries, ng, nint, readpattern):
    # input is a cube of images representing the flux rate at intervals of frametime.

    nframes, dimy, dimx = np.shape(frameseries)

def generate_timesteps(simuPars):

    # This function outputs the time steps for each simulated
    # frame (read) or integration for the whole time-series.
    # It returns the time at the center of the read or integration.
    # It uses as inputs, the subarray, ngroup, tstart, tend of the
    # simulation parameters.

    # Determine what the frame time is from the subarray requested
    if simuPars.subarray == 'SUBSTRIP96':
        frametime = second2day(2.214)   # seconds per read
    elif simuPars.subarray == 'SUBSTRIP256':
        frametime = second2day(5.494)   # seconds per read
    elif simuPars.subarray == 'FF':
        frametime = second2day(10.737)  # seconds per read
    else:
        print('Need to specify the subarray requested for this simulation. Either SUBSTRIP96, SUBSTRIP256 or FF')
        sys.exit(1)
    # Update the frametime parameter in the simuPars structure
    simuPars.frametime = np.copy(frametime)


    # Compute the number of integrations to span at least the requested range of time
    # Use ngroup to define the integration time.
    intduration = frametime * (simuPars.ngroup + 1)
    nint = int(np.ceil((hour2day(simuPars.tend - simuPars.tstart)/intduration)))
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
        timesteps += hour2day(simuPars.tstart) + frametime/2
        # remove the reset frame
        indices = np.arange(0,nint*(simuPars.ngroup+1))
        isread = np.where(np.mod(indices,simuPars.ngroup+1) != 0)
        timesteps = timesteps[isread]
        # open shutter time, actually the time for which photons
        # reach the detector dor simulation purposes
        tintopen = np.copy(frametime)
    elif simuPars.granularity == 'INTEGRATION':
        # TBC
        timesteps = np.arange(0,nint) * (simuPars.ngroup+1) * frametime
        # add the time at the start and shift by half an integration time to center in mid-integration
        # Notice that the center of an integration should exclude the reset frame
        # so the center is from read 1 (so add a frametime).
        timesteps += hour2day(simuPars.tstart) + frametime + simuPars.ngroup * frametime / 2
        # open shutter time, actually the time for which photons
        # reach the detector dor simulation purposes
        tintopen = frametime * simuPars.ngroup
    else:
        print('Time granularity of the simulation should either be FRAME or INTEGRATION.')
        sys.exit(1)

    return(tintopen, frametime, nint, timesteps)

def generate_traces(pathPars, simuPars, tracePars, throughput,
                    star_angstrom, star_flux, ld_coeff,
                    planet_angstrom, planet_rprs,
                    timesteps, granularitytime):

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
            currenttime = np.copy(timesteps[t])
            print('Time step {:} minutes - Order {:}'.format(currenttime*24*60, spectral_order))
            pixels=spgen.gen_unconv_image(simuPars, throughput, star_angstrom_bin, star_flux_bin,
                                      ld_coeff_bin, planet_rprs_bin,
                                      currenttime, granularitytime, solin, spectral_order, tracePars)

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

        tmpfilename = write_intermediate_fits(convolved_image, pathPars.path_userland+'tmp.fits',t)
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



def write_simu_fits(image, filename):

    # Writes the simulated data set as simulated, independently of the DMS constraints
    hdu = fits.PrimaryHDU(image)
    # Save on disk
    hdu.writeto(filename, overwrite=True)

    return

def write_intermediate_fits(image, filename, timestep_index):
    # Write image for a single time step. Differents spectral orders stored in third dimension
    # of the array. filename is the name of the final product. Intermediate filenames will
    # be forged based on that.

    name, suffix = os.path.splitext(filename)
    directory_name = name+'/'
    filename_current = directory_name+'tmp_{:06d}'.format(timestep_index)+suffix

    # Create a list of HDU with primary and extension HDUs
    hdu = fits.PrimaryHDU(image)

    # create a directory if it does not yet exists
    if os.path.exists(directory_name) is False:
        os.mkdir(directory_name)

    # Save on disk
    hdu.writeto(filename_current, overwrite=True)

    return(filename_current)

def write_dmsready_fits_init(imagelist, normalization_scale, simuPars):
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
    # granularity. It is time to divide in frame with the propely scaled flux as
    # happens during an integraiton.
    ngroup = np.copy(simuPars.ngroup)
    nint = np.copy(simuPars.nint)
    frametime = np.copy(simuPars.frametime)
    print(nint,ngroup,frametime)

    # Initialize the exposure array containing up-the-ramp reads.
    exposure = np.zeros((nint, ngroup, dimy, dimx), dtype=float)
    if simuPars.granularity == 'FRAME':
        # Then we already have a rate image for each individual read.
        for i in range(nint):
            cumulative = np.zeros((dimy, dimx))
            for g in range(ngroup):
                n = i*ngroup+g
                cumulative = cumulative + fluxratecube[n,:,:] * frametime
                print('i={:} g={:} n={:} flux={:} rate={:}'.format(i,g,n,np.sum(cumulative),np.sum(fluxratecube[n,:,:])))
                exposure[i, g, :, :] = np.copy(cumulative.reshape((1,1,dimy,dimx)))
    elif simuPars.granularity == 'INTEGRATION':
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



