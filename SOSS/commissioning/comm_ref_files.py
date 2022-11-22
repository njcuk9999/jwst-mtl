'''
Script used to generate the various iterations of reference files used by ATOCA during Commissioning.
It uses SOSS.dms.soss_ref_files.ipynb as a example.
The first iteration of reference files is after the wavelength calibration observation.
    - Arpita Roy ran the wavelength calibration script and sent me the wave vs x pixel for orders 1 and 3 (not 2).
    - I have ran the trace centroids on a depp stack.
    - Michael Radica ran his applesoss trace profile building script and sent me the results orders 1 and 2).
    - monochromatic tilt is set to zero.
    --> There is no speckernel update, no throughput update.
The second iteration is the same but for the observation 2 of the wavelength calibration program where the trace was
shifted by 1000 pixels down (to observe the background).

The third iteration will be based on the flux calibration or the transit observation, whichever comes first.
'''
import os

import numpy as np

from astropy.io import fits, ascii

from SOSS.trace import tracepol
from SOSS.dms.soss_ref_files import init_spec_trace, calc_2d_wave_map, init_wave_map, init_spec_profile, \
    init_spec_kernel, check_spec_trace, check_2dwave_map, check_profile_map

import matplotlib.pyplot as plt

import SOSS.dms.soss_oneoverf as soss_oneoverf

from jwst import datamodels

from jwst.pipeline import calwebb_detector1

from scipy.optimize import curve_fit

import sys


def run_iteration1(dataset='nis18obs02', wavecaldataset=None, subarray='SUBSTRIP256'):

    ######################## spectrace #########################

    # We will use the following sources for this reference file. These can be modified as needed.
    throughput_file = '../dms/files/NIRISS_Throughput_STScI.fits'
    tilt_file = '../dms/files/SOSS_wavelength_dependent_tilt.ecsv'

    # All sources will be modified to correspond to this wavelength grid.
    wavemin = 0.5
    wavemax = 5.5
    nwave = 5001
    wave_grid = np.linspace(wavemin, wavemax, nwave)

    # Read the SOSS total throughput as a function of wavelength.
    tab, hdr = fits.getdata(throughput_file, ext=1, header=True)
    #if dataset == 'commrevA':
    if True:
        # Read the measured throughputs (from Kevin Volk) - wave units of microns, value units of 0 to 1
        tab1 = ascii.read('files/throughput_o1_commrevA.txt')
        tab2 = ascii.read('files/throughput_o2_commrevA.txt')
        tab3 = ascii.read('files/throughput_o3_commrevA.txt')
        # Interpolate to the reference wavelength grid.
        throughput = np.zeros((nwave, 3))
        throughput[:, 0] = np.interp(wave_grid, tab1['col1'], tab1['col2'])
        throughput[:, 1] = np.interp(wave_grid, tab2['col1'], tab2['col2'])
        throughput[:, 2] = np.interp(wave_grid, tab3['col1'], tab3['col2'])
    else:
        # Interpolate to the reference wavelength grid.
        throughput = np.zeros((nwave, 3))
        throughput[:, 0] = np.interp(wave_grid, tab[0]['LAMBDA'] / 1e3, tab[0]['SOSS_ORDER1'])
        throughput[:, 1] = np.interp(wave_grid, tab[0]['LAMBDA'] / 1e3, tab[0]['SOSS_ORDER2'])
        throughput[:, 2] = np.interp(wave_grid, tab[0]['LAMBDA'] / 1e3, tab[0]['SOSS_ORDER3'])

    fig = plt.figure(figsize=(8,6))
    plt.plot(wave_grid, throughput[:,0], color='black', label='Order 1 Measured')
    plt.plot(wave_grid, throughput[:,1], color='blue', label='Order 2 Measured')
    plt.plot(wave_grid, throughput[:,2], color='red', label='Order 3 Measured')
    plt.legend()
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength')
    plt.show()


    # Fix small negative throughput values.
    throughput = np.where(throughput < 0, 0, throughput)

    # Read the tilt as a function of wavelength.
    tab = ascii.read(tilt_file)

    # Interpolate the tilt to the same wavelengths as the throughput.
    # Default bounds handling (constant boundary) is fine.
    tilt = np.zeros((nwave, 3))
    tilt[:, 0] = np.interp(wave_grid, tab['Wavelength'], tab['order 1'])
    tilt[:, 1] = np.interp(wave_grid, tab['Wavelength'], tab['order 2'])
    tilt[:, 2] = np.interp(wave_grid, tab['Wavelength'], tab['order 3'])
    # For now, set tilt to zero until we can measure it
    ##################
    tilt = tilt * 0.0
    ##################

    # Read the trace positions
    #if wavecaldataset == None:
    if True:
        o1 = ascii.read('files/centroids_o1_substrip256_'+dataset+'.txt')
        x_o1, y_o1 = np.array(o1['x']), np.array(o1['y'])
        o2 = ascii.read('files/centroids_o2_substrip256_'+dataset+'.txt')
        x_o2, y_o2 = np.array(o2['x']), np.array(o2['y'])
        o3 = ascii.read('files/centroids_o3_substrip256_'+dataset+'.txt')
        x_o3, y_o3 = np.array(o3['x']), np.array(o3['y'])
    if False:
        o1 = ascii.read('files/centroids_o1_substrip256_'+wavecaldataset+'.txt')
        x_o1, y_o1 = np.array(o1['x']), np.array(o1['y'])
        o2 = ascii.read('files/centroids_o2_substrip256_'+wavecaldataset+'.txt')
        x_o2, y_o2 = np.array(o2['x']), np.array(o2['y'])
        o3 = ascii.read('files/centroids_o3_substrip256_'+wavecaldataset+'.txt')
        x_o3, y_o3 = np.array(o3['x']), np.array(o3['y'])

    # CUSTOM wavecal order 1
    # Read the wavelength calibration files
    if wavecaldataset == None:
        wcal_o1 = ascii.read('files/wavecal_o1_'+dataset+'.txt')
        w_o1 = np.array(wcal_o1['wavelength'])
    else:
        wcal_o1 = ascii.read('files/wavecal_o1_'+wavecaldataset+'.txt')
        w_o1 = np.array(wcal_o1['wavelength'])
    # Resample to the desired wavelength sampling
    #ind = np.argsort(w_o1) # needs an increasing w_o1 grid or it fails
    #xtrace_order1 = np.interp(wave_grid, w_o1[ind], x_o1[ind])
    #ytrace_order1 = np.interp(wave_grid, w_o1[ind], y_o1[ind])
    xtrace_order1 = extrapolate_to_wavegrid(wave_grid, w_o1, x_o1)
    ytrace_order1 = extrapolate_to_wavegrid(wave_grid, w_o1, y_o1)

    # CUSTOM wavecal order 2
    # Read the wavelength calibration files
    if wavecaldataset == None:
        wcal_o2 = ascii.read('files/wavecal_o2_' + dataset + '.txt')
        w_o2 = np.array(wcal_o2['wavelength'])
    else:
        wcal_o2 = ascii.read('files/wavecal_o2_' + wavecaldataset + '.txt')
        w_o2 = np.array(wcal_o2['wavelength'])
    w_o2_tmp = np.array(wcal_o2['wavelength'])
    w_o2 = np.zeros(2048)*np.nan
    w_o2[:1783] = w_o2_tmp
    # Fill for column > 1783 with linear extrapolation
    m = w_o2[1782] - w_o2[1781]
    dx = np.arange(2048-1783)+1
    w_o2[1783:] = w_o2[1782] + m * dx
    xtrace_order2 = extrapolate_to_wavegrid(wave_grid, w_o2, x_o2)
    ytrace_order2 = extrapolate_to_wavegrid(wave_grid, w_o2, y_o2)


    # CUSTOM wavecal order 3
    # only 800 columns in wavecal_o3
    if wavecaldataset == None:
        wcal_o3 = ascii.read('files/wavecal_o3_'+dataset+'.txt')
    else:
        wcal_o3 = ascii.read('files/wavecal_o3_' + wavecaldataset + '.txt')
    w_o3_tmp = np.array(wcal_o3['wavelength'])
    w_o3 = np.zeros(2048)*np.nan
    w_o3[:800] = w_o3_tmp
    # Fill for column > 800 with linear extrapolation
    m = w_o3[799] - w_o3[798]
    dx = np.arange(2048-800)+1
    w_o3[800:] = w_o3[799] + m * dx
    xtrace_order3 = extrapolate_to_wavegrid(wave_grid, w_o3, x_o3)
    ytrace_order3 = extrapolate_to_wavegrid(wave_grid, w_o3, y_o3)

    # Temporarily fill orders2 and 3 with past wavelength calib
    ############################################################
    # Read the trace position parameters.
    #tracepars = tracepol.get_tracepars('../trace/NIRISS_GR700_trace_extended.csv')
    #xtrace_order2, ytrace_order2, mask = tracepol.wavelength_to_pix(wave_grid, tracepars, m=2, frame='dms',
    #                                                                subarray='SUBSTRIP256')
    #xtrace_order2 = np.where(mask, xtrace_order2, np.nan)
    # Use the measured y position on 2048 columns and interpolate it at the same sampling as xtrace_order2
    #ytrace_order2 = np.interp(xtrace_order2, np.arange(2048), y_o2)

    #xtrace_order3, ytrace_order3, mask = tracepol.wavelength_to_pix(wave_grid, tracepars, m=3, frame='dms',
    #                                                                subarray='SUBSTRIP256')
    #xtrace_order3 = np.where(mask, xtrace_order3, np.nan)
    # Use the measured y position on 2048 columns and interpolate it at the same sampling as xtrace_order2
    #ytrace_order3 = np.interp(xtrace_order3, np.arange(2048), y_o3)
    ############################################################

    fig = plt.figure(figsize=(6,4))
    plt.scatter(x_o1, w_o1, marker='.', color='black', label='Order 1 Measured')
    plt.plot(xtrace_order1, wave_grid, color='red', label='Order 1 Extrapolated and Resampled')
    plt.scatter(x_o2, w_o2, marker='.', color='blue', label='Order 2 Measured')
    plt.plot(xtrace_order2, wave_grid, color='red', label='Order 2 Extrapolated and Resampled')
    plt.scatter(x_o3, w_o3, marker='.', color='green', label='Order 3 Measured')
    plt.plot(xtrace_order3, wave_grid, color='red', label='Order 3 Extrapolated and Resampled')
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Wavelength')
    plt.show()



    xtrace = np.zeros((nwave, 3))
    xtrace[:, 0] = xtrace_order1
    xtrace[:, 1] = xtrace_order2
    xtrace[:, 2] = xtrace_order3

    ytrace = np.zeros((nwave, 3))
    ytrace[:, 0] = ytrace_order1
    ytrace[:, 1] = ytrace_order2
    ytrace[:, 2] = ytrace_order3

    # Massage inputs according to requested output subarray
    if subarray == 'SUBSTRIP96':
        ytrace[:, 0] = ytrace[:, 0] - 10
        ytrace[:, 1] = ytrace[:, 1] - 10
        ytrace[:, 2] = ytrace[:, 2] - 10
        #print('Actually do nothing. soss_ref_files.py handles it')
    elif subarray == 'FULL':
        ytrace[:, 0] = ytrace[:, 0] + (2048-256)
        ytrace[:, 1] = ytrace[:, 1] + (2048-256)
        ytrace[:, 2] = ytrace[:, 2] + (2048-256)


    # Call init_spec_trace with the cleaned input data. This will perform checks on the input and built the fits file structure.
    hdul = init_spec_trace(wave_grid, xtrace, ytrace, tilt, throughput, subarray) #'SUBSTRIP256')

    # If necessary manual changes and additions can be made here, before saving the file.
    #filename = hdul[0].header['FILENAME']
    if dataset == 'nis18obs02':
        trace_file = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis17':
        trace_file = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis34':
        trace_file = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == '02589_obs001':
        trace_file = '/Users/albert/NIRISS/Commissioning/analysis/T1/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == '02589_obs002':
        trace_file = '/Users/albert/NIRISS/Commissioning/analysis/T1_2/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == '01201_obs101':
        trace_file = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/' + hdul[0].header['FILENAME']
    else:
        print('Add entry for this target in com_ref_files.py...')
        sys.exit()
    hdul.writeto(trace_file, overwrite=True)
    #hdul.writeto(trace_file + '.gz', overwrite=True)


    ######################## 2d wavelength map #########################

    # We will use the following sources for this reference file. These can be modified as needed.
    #trace_file = 'SOSS_ref_trace_table_FULL.fits'

    padding = 20
    oversample = 1

    dimx = oversample * (2048 + 2 * padding)
    dimy = oversample * (2048 + 2 * padding)

    wave_map_2d = np.zeros((dimx, dimy, 3))

    # Read the 1D trace reference file.
    data = fits.getdata(trace_file, ext=1)

    # Compute the 2D wavelength map.
    wave_map_2d[:, :, 0] = calc_2d_wave_map(data['WAVELENGTH'], data['X'], data['Y'], data['TILT'],
                                            oversample=oversample, padding=padding)

    # Read the 1D trace reference file.
    data = fits.getdata(trace_file, ext=2)

    # Compute the 2D wavelength map.
    wave_map_2d[:, :, 1] = calc_2d_wave_map(data['WAVELENGTH'], data['X'], data['Y'], data['TILT'],
                                            oversample=oversample, padding=padding)

    # Read the 1D trace reference file.
    data = fits.getdata(trace_file, ext=3)

    # Compute the 2D wavelength map.
    wave_map_2d[:, :, 2] = calc_2d_wave_map(data['WAVELENGTH'], data['X'], data['Y'], data['TILT'],
                                            oversample=oversample, padding=padding)

    hdul = init_wave_map(wave_map_2d, oversample, padding, subarray)#'SUBSTRIP256')

    # If necessary manual changes and additions can be made here, before saving the file.
    if dataset == 'nis18obs02':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis17':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis34':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == '02589_obs001':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/T1/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == '02589_obs002':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/T1_2/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == '01201_obs101':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'+hdul[0].header['FILENAME']
    else:
        print('Add entry for this target in com_ref_files.py...')
        sys.exit()
    hdul.writeto(filename, overwrite=True)
    #hdul.writeto(filename + '.gz', overwrite=True)



    ######################## 2d trace profile #########################

    # We will use the following sources for this reference file. These can be modified as needed.
    if dataset == 'nis18obs01':
        profile_file = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/michael_trace/applesoss_traceprofile.fits'
    if dataset == 'nis18obs02':
        profile_file = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/michael_trace/applesoss_traceprofile.fits'
    if dataset == 'nis17':
        profile_file = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/michael_trace/applesoss_traceprofile.fits'
    if dataset == 'nis34':
        profile_file = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/michael_trace/applesoss_traceprofile.fits'
    if dataset == '02589_obs001':
        profile_file = '/Users/albert/NIRISS/Commissioning/analysis/T1/michael_trace/applesoss_traceprofile.fits'
    if dataset == '02589_obs002':
        profile_file = '/Users/albert/NIRISS/Commissioning/analysis/T1_2/michael_trace/applesoss_traceprofile.fits'
    if dataset == '01201_obs101':
        profile_file = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/michael_trace/applesoss_traceprofile.fits'
    else:
        print('Add entry for this target in com_ref_files.py...')
        sys.exit()
    # Read the profile file provided by Loïc.
    #profile_2d = fits.getdata(profile_file, ext=0)
    #profile_2d = np.moveaxis(profile_2d, 0, -1)
    # Read the profile file provided by Michael.
    aaa = fits.open(profile_file) # contains orders 1 and 2 only, fill order 3 with zeros
    # set to zero x> 1780 order 2 (bad negative stuff there)
    prof2 = aaa[2].data
    #prof2[:, 1760:] = 0.0
    if dataset == 'nis18obs01':
        profile_2d = np.array([aaa[1].data, prof2, aaa[2].data*0.0])
    else:
        profile_2d = np.array([aaa[1].data, prof2, aaa[3].data])
    profile_2d = np.moveaxis(profile_2d, 0, -1)

    # The padding and oversamspling used to generate the 2D profile.
    #padding = 20
    #oversample = 1

    # The provided file is for SUBSTRIP256, we pad this to the FULL subarray.
    nrows, ncols, _ = profile_2d.shape
    dimy = oversample * (2048 + 2 * padding)
    dimx = oversample * (2048 + 2 * padding)

    tmp = np.full((dimy, dimx, 3), fill_value=np.nan)
    tmp[-nrows:] = profile_2d
    profile_2d = tmp

    # Call init_spec_profile with the prepared input data.
    hdul = init_spec_profile(profile_2d, oversample, padding, subarray)

    # If necessary manual changes and additions can be made here, before saving the file.
    filename = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'+hdul[0].header['FILENAME']
    if dataset == 'nis18_obs02':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis17':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis34':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == '02589_obs001':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/T1/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == '02589_obs002':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/T1_2/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == '01201_obs101':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'+hdul[0].header['FILENAME']
    else:
        print('Add entry for this target in com_ref_files.py...')
        sys.exit()
    hdul.writeto(filename, overwrite=True)
    #hdul.writeto(filename + '.gz', overwrite=True)

    return


def extrapolate_to_wavegrid(w_grid, wavelength, quantity):
    '''
    Extrapolates quantities on the right and the left of a given array of quantity
    '''
    sorted = np.argsort(wavelength)
    q = quantity[sorted]
    w = wavelength[sorted]

    # Determine the slope on the right of the array
    slope_right = (q[-1] - q[-2])/(w[-1] - w[-2])
    # extrapolate at wavelengths larger than the max on the right
    indright = np.where(w_grid > w[-1])[0]
    q_right = q[-1] + (w_grid[indright] - w[-1]) * slope_right
    # Determine the slope on the left of the array
    slope_left = (q[1] - q[0])/(w[1] - w[0])
    # extrapolate at wavelengths smaller than the min on the left
    indleft = np.where(w_grid < w[0])[0]
    q_left = q[0] + (w_grid[indleft] - w[0]) * slope_left
    # Construct and extrapolated array of the quantity
    w = np.concatenate((w_grid[indleft], w, w_grid[indright]))
    q = np.concatenate((q_left, q, q_right))

    # resample at the w_grid everywhere
    q_grid = np.interp(w_grid, w, q)

    fig = plt.figure()
    plt.scatter(w, q, marker='.', color='black')
    plt.scatter(w_grid, q_grid, marker='.', color='red')
    plt.show()

    return q_grid


def linefitthruzero(x, slope):
    # Curve fitting function passing thru zero
    return slope * x


def build_dark_ref_file():

    # To replace the DMS ref files superbias and dark (to apply in a single step)
    # Doing this because of 1/f residuals in the darks and superbias.
    uncaldir = '/Users/albert/NIRISS/Commissioning/analysis/darks/'
    uncal_list = [
        'jw01081107001_02101_00001_nis_uncal.fits',
        'jw01081108001_02101_00001_nis_uncal.fits',
        'jw01081109001_02101_00001_nis_uncal.fits',
        'jw01081110001_02101_00001_nis_uncal.fits',
        'jw01081111001_02101_00001_nis_uncal.fits',
        'jw01081112001_02101_00001_nis_uncal.fits',
        'jw01081113001_02101_00001_nis_uncal.fits',
        'jw01081114001_02101_00001_nis_uncal.fits',
        'jw01081115001_02101_00001_nis_uncal.fits',
        'jw01081116001_02101_00001_nis_uncal.fits'
    ]
    red_list = [
        'jw01081107001_02101_00001_nis_saturationstep.fits',
        'jw01081108001_02101_00001_nis_saturationstep.fits',
        'jw01081109001_02101_00001_nis_saturationstep.fits',
        'jw01081110001_02101_00001_nis_saturationstep.fits',
        'jw01081111001_02101_00001_nis_saturationstep.fits',
        'jw01081112001_02101_00001_nis_saturationstep.fits',
        'jw01081113001_02101_00001_nis_saturationstep.fits',
        'jw01081114001_02101_00001_nis_saturationstep.fits',
        'jw01081115001_02101_00001_nis_saturationstep.fits',
        'jw01081116001_02101_00001_nis_saturationstep.fits'
    ]

    # Process the files thru first 3 steps of the pipeline
    if False:
        for i in range(np.size(uncal_list)):
        #for i in range(1):
            exposurename = uncaldir+uncal_list[i]
            result = calwebb_detector1.group_scale_step.GroupScaleStep.call(exposurename, output_dir=uncaldir, save_results=False)
            result = calwebb_detector1.dq_init_step.DQInitStep.call(result, output_dir=uncaldir, save_results=False)
            result = calwebb_detector1.saturation_step.SaturationStep.call(result, output_dir=uncaldir, save_results=True)
            #result = calwebb_detector1.superbias_step.SuperBiasStep.call(result, output_dir=uncaldir, save_results=True)
            #result = calwebb_detector1.refpix_step.RefPixStep.call(result, output_dir=uncaldir, save_results=True)
            #result = calwebb_detector1.linearity_step.LinearityStep.call(result, output_dir=uncaldir, save_results=True)
            #result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(result, output_dir=uncaldir, save_results=True)
            #result = calwebb_detector1.jump_step.JumpStep.call(result, output_dir=uncaldir, save_results=True)
            #result = calwebb_detector1.ramp_fit_step.RampFitStep.call(result, output_dir=uncaldir, save_results=True)

    if False:
        rawcube3x50 = np.zeros((15,50,256,2048))
        rawcube20x3 = np.zeros((100,3,256,2048))

        for i in range(np.size(uncal_list)):
            dark_i = datamodels.open(uncaldir+red_list[i])
            if i == 0:
                rawcube20x3 = dark_i.copy()
                rawcube20x3.meta.exposure.ngroups = 3 # unchanged
                rawcube20x3.meta.exposure.nints = 100
                rawcube20x3.meta.exposure.integration_start = 1
                rawcube20x3.meta.exposure.integration_end = 100
                rawcube20x3.meta.exposure.filename = 'darks.fits'
                rawcube20x3.meta.subarray.name = 'SUBSTRIP256'
                rawcube20x3.data = np.zeros((100, 3, 256, 2048))
                rawcube20x3.groupdq = np.zeros((100, 3, 256, 2048))
            if i == 5:
                rawcube3x50 = dark_i.copy()
                rawcube3x50.meta.exposure.ngroups = 50 # unchanged
                rawcube3x50.meta.exposure.nints = 15
                rawcube3x50.meta.exposure.integration_start = 1
                rawcube3x50.meta.exposure.integration_end = 15
                rawcube3x50.meta.exposure.filename = 'darks.fits'
                rawcube3x50.meta.subarray.name = 'SUBSTRIP256'
                rawcube3x50.data = np.zeros((15, 50, 256, 2048))
                rawcube3x50.groupdq = np.zeros((15, 50, 256, 2048))

            if i < 5:
                rawcube20x3.data[i*20:(i+1)*20,:,:,:] = np.copy(dark_i.data)
                rawcube20x3.groupdq[i*20:(i+1)*20,:,:,:] = np.copy(dark_i.groupdq)
            if i >=5:
                j = i-5
                rawcube3x50.data[j * 3:(j + 1) * 3, :, :, :] = np.copy(dark_i.data)
                rawcube3x50.groupdq[j * 3:(j + 1) * 3, :, :, :] = np.copy(dark_i.groupdq)

        # Save jwst models
        rawcube20x3.save(uncaldir+'rawcube100x3_uncal.fits')
        rawcube3x50.save(uncaldir+'rawcube15x50_uncal.fits')

    if True:
        iter=2
        if iter == 1:

            if False:
                # Correct the darks for the 1/f noise and save
                model20x3 = soss_oneoverf.applycorrection(rawcube20x3, output_dir = uncaldir, save_results = True)
                model3x50 = soss_oneoverf.applycorrection(rawcube3x50, output_dir = uncaldir, save_results = True)

                hdu = fits.PrimaryHDU(model20x3.data)
                hdu.writeto(uncaldir+'dark100x3_oofcorr_pass1.fits', overwrite=True)
                hdu = fits.PrimaryHDU(model3x50.data)
                hdu.writeto(uncaldir+'dark15x50_oofcorr_pass1.fits', overwrite=True)

                # Correct the darks for the 1/f noise and save - second pass
                model20x3 = soss_oneoverf.applycorrection(model20x3, output_dir = uncaldir, save_results = True)
                model3x50 = soss_oneoverf.applycorrection(model3x50, output_dir = uncaldir, save_results = False)

                hdu = fits.PrimaryHDU(model20x3.data)
                hdu.writeto(uncaldir+'dark100x3_oofcorr_pass2.fits', overwrite=True)
                hdu = fits.PrimaryHDU(model3x50.data)
                hdu.writeto(uncaldir+'dark15x50_oofcorr_pass2.fits', overwrite=True)



            if False:
                # Make a plot of the median dark signal versus frame number
                dark15x50 = fits.getdata(uncaldir + 'dark15x50_oofcorr_pass2.fits')
                stack15x50 = np.median(dark15x50, axis=0)
                cds = stack15x50 - stack15x50[0, :, :]
                lvl = np.nanmedian(cds, axis=(1, 2))
                print(lvl)
                print(np.shape(lvl))

                par = np.polyfit(np.arange(50), lvl, 1)
                dark_slope = np.copy(par[0])
                print('dark slope = ', dark_slope)
                yfit = np.polyval(par, np.arange(50))

                # Curve fitting
                xfit = np.arange(50)
                params = curve_fit(linefitthruzero, xfit, lvl)
                [slope] = params[0]
                yfit = slope * xfit
                print('slope = ', slope)

                plt.scatter(np.arange(50), lvl, marker='.', color='black')
                plt.plot(xfit, yfit, color='red', label='Slope = {:5f} ADU/Frame'.format(slope))
                plt.xlabel('CDS (Read i - Read 1)')
                plt.ylabel('Median Count [ADU]')
                plt.legend()
                plt.savefig(uncaldir + 'Dark_counts_with_frame.png')

            if False:
                # Average the 300 integrations (applying a blind offset between different reads of 0.92 ADU)
                dark100x3 = fits.getdata(uncaldir + 'dark100x3_oofcorr_pass2.fits')
                deepdark = np.zeros((300, 256, 2048))
                deepdark[0:100, :, :] = np.copy(dark100x3[:, 0, :, :])
                deepdark[100:200, :, :] = np.copy(dark100x3[:, 1, :, :]) - 0.92
                deepdark[200:300, :, :] = np.copy(dark100x3[:, 2, :, :]) - (2 * 0.92)
                deepstack = np.nanmedian(deepdark, axis=0)
                hdu = fits.PrimaryHDU(deepstack)
                hdu.writeto(uncaldir + 'deepdark300.fits', overwrite=True)


        if iter == 2:
            # Stack to generate read1 stack
            #    read1_stack = median(100 read#1 + 100 read#2 - 1*slope + 100 read#3 - 2*slope)
            # Create the diff image between current read and deep read 1 stack
            #    diff_i = read2_i - read1_stack - 1*slope
            # Correct for 1/f
            #    diff_i_oofcorr = oof(diff_i)
            # Recover the read2_i
            #    read2_i_clean = diff_oofcorr + 1*slope + read1_stack

            read1_stack = fits.getdata(uncaldir+'deepdark300.fits')
            raw100x3 = datamodels.open(uncaldir+'rawcube100x3_uncal.fits')
            raw15x50 = datamodels.open(uncaldir+'rawcube15x50_uncal.fits')

            tmp_3reads = raw100x3.copy()
            tmp_50reads = raw15x50.copy()
            dark_slope = 0.91149684
            # Remove dark signal
            for i in range(3):
                tmp_3reads.data[:,i,:,:] = tmp_3reads.data[:,i,:,:] - dark_slope * i
            for i in range(50):
                tmp_50reads.data[:,i,:,:] = tmp_50reads.data[:,i,:,:] - dark_slope * i

            # oof correction, forcing the read 1 stack as the deepstack
            tmp_3reads, _, _, outliers_3reads = soss_oneoverf.applycorrection(tmp_3reads, output_dir=uncaldir, save_results=False,
                                          deepstack_custom=read1_stack, return_intermediates=True)
            tmp_50reads, _, _, outliers_50reads = soss_oneoverf.applycorrection(tmp_50reads, output_dir=uncaldir, save_results=False,
                                          deepstack_custom=read1_stack, return_intermediates=True)
            # Add back dark signal
            for i in range(3):
                tmp_3reads.data[:,i,:,:] = tmp_3reads.data[:,i,:,:] + dark_slope*i
            for i in range(50):
                tmp_50reads.data[:,i,:,:] = tmp_50reads.data[:,i,:,:] + dark_slope*i

            #hdu = fits.PrimaryHDU(tmp_50reads.data)
            #hdu.writeto(uncaldir+'test.fits', overwrite=True)

            # Average all integrations for each read starting with the 50-read cube
            dark_upto50 = np.nanmedian(tmp_50reads.data * outliers_50reads, axis=0)
            # For the first 3 reads, replace by the higher SNR 3-read cube
            dark_upto50[:3,:,:] = np.nanmedian(tmp_3reads.data * outliers_3reads, axis=0)

            ref = datamodels.open('/Users/albert/NIRISS/CRDS_CACHE/references/jwst/niriss/jwst_niriss_dark_0171.fits')
            ref.data = np.copy(dark_upto50)
            ref.save(uncaldir+'jwst_niriss_dark_loiccustom.fits')

            #hdu = fits.PrimaryHDU(dark_upto50)
            #hdu.writeto(uncaldir+'dark50reads_final.fits', overwrite=True)
            cds = dark_upto50 - dark_upto50[0]
            hdu = fits.PrimaryHDU(cds)
            hdu.writeto(uncaldir+'dark50reads_final_cds.fits', overwrite=True)





    if False:
        # Look at the 1/f rms values, as a function of deepstack nbr of integrations
        #dark100x3 = fits.getdata(uncaldir + 'dark100x3_oofcorr_pass2.fits')
        dark100x3 = fits.getdata(uncaldir + 'rawcube20x3.fits')
        plt.figure()
        for i in range(10):
            # Analyze read 1 only
            stack = np.median(dark100x3[:(i+1)*10, 0, :, :], axis=0)
            diff = dark100x3[:, 0, :, :] - stack
            #print(np.shape(diff))
            oof = np.nanmedian(diff, axis=1)
            #print(np.shape(oof))
            print(np.nanstd(oof, axis=1))
            plt.scatter(np.ones(100)*(i+1)*10, np.nanstd(oof, axis=1), marker='.', color='black')
        plt.xlabel('Number of Integrations Used in Deep Stack of Reads')
        plt.ylabel('1/f rms across CDS for 100 integrations')
        plt.show()


    if False:
        # Make a second pass thru 1/f correction with the corrected cube as input
        correctedcube = datamodels.open('/Users/albert/NIRISS/Commissioning/analysis/darks/jw01081112001_02101_00001_nis_custom1overf.fits')
        model = soss_oneoverf.applycorrection(correctedcube, output_dir = uncaldir, save_results = True)

        # stack the data
        outliers = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/darks/supplemental_jw01081112001_02101_00001_nis/outliers.fits')
        dark = np.nanmedian(model.data * outliers, axis=0)
        hdu = fits.PrimaryHDU(dark)
        hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/darks/dark_loic_pass2.fits', overwrite=True)

    if False:
        # Check that all went as expected (take CDS)
        dark = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/darks/dark_loic_pass2.fits')
        cds = dark[1:50,:,:] - dark[0:49,:,:]
        hdu = fits.PrimaryHDU(cds)
        hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/darks/dark_loic_cds_pass2.fits', overwrite=True)

        # For reference, produce the dark CDS diagnostic with original uncorrected data
        rawdata = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/darks/rawcube.fits')
        outliers = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/darks/supplemental_jw01081112001_02101_00001_nis/outliers.fits')
        dark = np.nanmedian(rawdata * outliers, axis=0)
        cds = dark[1:50, :, :] - dark[0:49, :, :]
        hdu = fits.PrimaryHDU(cds)
        hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/darks/dark_loic_cds_pass0.fits', overwrite=True)

    return

def alternate_saturation_map():
    '''
    Creates an alternative jwst_niriss_saturation_XXXX.fits map to cut at a lower threshold
    '''

    GAIN = 1.62
    sat_limit_electron = [35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000]
    for satlim_electron in sat_limit_electron:
        SAT_LIMIT = satlim_electron/GAIN

        crds_dir = '/Users/albert/NIRISS/CRDS_CACHE/references/jwst/niriss/'
        satmap = datamodels.open(crds_dir+'jwst_niriss_saturation_0014.fits')
        superbias = datamodels.open(crds_dir+'jwst_niriss_superbias_0181.fits')

        #Subtract the superbias from satmap then multiply difference by a scale

        newsatmap = satmap.copy()
        tmp256 = satmap.data[-256:,:] - superbias.data
        ind = tmp256 >= SAT_LIMIT
        tmp256[ind] = SAT_LIMIT
        #newsatmap.data[:256,:] = newsatmap.data[:256,:] - superbias.data
        #newsatmap.data[:256,:] = np.where(newsatmap.data[:256,:] >= SAT_LIMIT, SAT_LIMIT, )
        newsatmap.data[-256:,:] = tmp256 + superbias.data

        newsatmap.meta.filename = 'custom_saturation_'+str(satlim_electron)+'e.fits'
        newsatmap.write(crds_dir+newsatmap.meta.filename)

    return



if __name__ == "__main__":
    #refdir = '/Users/albert/NIRISS/CRDS_CACHE/references/jwst/niriss/'
    #check_spec_trace(refdir+'jwst_niriss_spectrace_0018.fits')
    #check_2dwave_map(refdir+'jwst_niriss_wavemap_0014.fits')
    #check_profile_map(refdir+'jwst_niriss_specprofile_0017.fits')

    if True:
        a = alternate_saturation_map()

    if False:
        a = build_dark_ref_file()

    if False:
        dataset, dataset_dirname = '02589_obs001', 'T1'
        dataset, dataset_dirname = '02589_obs002', 'T1_2'
        dataset, dataset_dirname = '01201_obs101', 'T1_3'
        subarray_list = ['FULL', 'SUBSTRIP256', 'SUBSTRIP96']
        for subarray in subarray_list:
        #if True:
            #subarray = 'FULL'
            if False:
                refdir = '/Users/albert/NIRISS/Commissioning/analysis/'+dataset_dirname+'/ref_files/'
                run_iteration1(dataset='nis17', wavecaldataset='commrevA', subarray=subarray)
            if False:
                refdir = '/Users/albert/NIRISS/Commissioning/analysis/'+dataset_dirname+'/ref_files/'
                run_iteration1(dataset='nis18obs02', wavecaldataset='commrevA', subarray=subarray)
            if True:
                refdir = '/Users/albert/NIRISS/Commissioning/analysis/'+dataset_dirname+'/ref_files/'
                run_iteration1(dataset=dataset, wavecaldataset='commrevA', subarray=subarray)
            check_spec_trace(refdir+'SOSS_ref_trace_table_'+subarray+'.fits')
            check_2dwave_map(refdir+'SOSS_ref_2D_wave_'+subarray+'.fits')
            check_profile_map(refdir+'SOSS_ref_2D_profile_'+subarray+'.fits')
