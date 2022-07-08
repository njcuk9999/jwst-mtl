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

import numpy as np

from astropy.io import fits, ascii

from SOSS.trace import tracepol
from SOSS.dms.soss_ref_files import init_spec_trace, calc_2d_wave_map, init_wave_map, init_spec_profile, \
    init_spec_kernel, check_spec_trace, check_2dwave_map, check_profile_map

import matplotlib.pyplot as plt

import sys

def run_iteration1(dataset='nis18obs02', wavecaldataset=None):

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

    # Call init_spec_trace with the cleaned input data. This will perform checks on the input and built the fits file structure.
    hdul = init_spec_trace(wave_grid, xtrace, ytrace, tilt, throughput, 'SUBSTRIP256')#,
                           #filename='/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/')

    # If necessary manual changes and additions can be made here, before saving the file.
    #filename = hdul[0].header['FILENAME']
    if dataset == 'nis18obs02':
        trace_file = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis17':
        trace_file = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis34':
        trace_file = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/ref_files/'+hdul[0].header['FILENAME']

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

    hdul = init_wave_map(wave_map_2d, oversample, padding, 'SUBSTRIP256')

    # If necessary manual changes and additions can be made here, before saving the file.
    if dataset == 'nis18obs02':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis17':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis34':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/ref_files/'+hdul[0].header['FILENAME']

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
    hdul = init_spec_profile(profile_2d, oversample, padding, 'SUBSTRIP256')

    # If necessary manual changes and additions can be made here, before saving the file.
    filename = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'+hdul[0].header['FILENAME']
    if dataset == 'nis18_obs02':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis17':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'+hdul[0].header['FILENAME']
    elif dataset == 'nis34':
        filename = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/ref_files/'+hdul[0].header['FILENAME']

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



if __name__ == "__main__":
    #refdir = '/Users/albert/NIRISS/CRDS_CACHE/references/jwst/niriss/'
    #check_spec_trace(refdir+'jwst_niriss_spectrace_0018.fits')
    #check_2dwave_map(refdir+'jwst_niriss_wavemap_0014.fits')
    #check_profile_map(refdir+'jwst_niriss_specprofile_0017.fits')

    if True:
        #refdir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'
        #refdir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'
        refdir = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/ref_files/'
        #run_iteration1(dataset='nis17', wavecaldataset='commrevA')
        #run_iteration1(dataset='nis18obs02', wavecaldataset='commrevA')
        run_iteration1(dataset='nis34', wavecaldataset='commrevA')
        check_spec_trace(refdir+'SOSS_ref_trace_table_SUBSTRIP256.fits')
        check_2dwave_map(refdir+'SOSS_ref_2D_wave_SUBSTRIP256.fits')
        check_profile_map(refdir+'SOSS_ref_2D_profile_SUBSTRIP256.fits')