import numpy as np

from jwst import datamodels

import matplotlib.pyplot as plt

def mediandev(x, axis=None):
    med = np.nanmedian(x, axis=axis)

    return np.nanmedian(np.abs(x - med), axis=axis) / 0.67449

def ktc():
    '''
    Looks at the raw (uncal) time series of a few pixels to understand
    how kTC noise operates.
    '''

    p1 = [1193,31] #[1159,53]
    p2 = [1160,53]

    datadir = '/Users/albert/NIRISS/Commissioning/analysis/T1/'
    uncal_list = ['jw02589001001_04101_00001-seg001_nis_uncal.fits',
                  'jw02589001001_04101_00001-seg002_nis_uncal.fits']

    # Read in the whole time series for 2 pixels
    pixel1 = []
    pixel2 = []
    nint = 0
    for i in range(np.size(uncal_list)):
        # Read current file
        data_model = datamodels.open(datadir+uncal_list[i])
        # Extract temp pixel 1
        pix = data_model.data[:, :, p1[1]-1, p1[0]-1]
        nint_i, ngroup = np.shape(pix)
        # Append current pixel 1 to previously read pixel 1
        pixel1 = np.append(pixel1, pix).reshape((nint+nint_i, ngroup))
        # Proceed to temp pixel 2
        pix = data_model.data[:, :, p2[1]-1, p2[0]-1]
        nint_i, ngroup = np.shape(pix)
        pixel2 = np.append(pixel2, pix).reshape((nint+nint_i, ngroup))
        # Update the full number of integration
        nint = np.shape(pixel1)[0]


    nint, ngroup = np.shape(pixel1)
    print('nint, ngroup =', nint, ngroup)
    plt.scatter(np.arange(nint*ngroup), pixel1, marker='.', color='black', label='Pixel (x,y)=({:},{:})'.format(p1[0], p1[1]))
    plt.scatter(np.arange(nint*ngroup), pixel2, marker='.', color='green', label='Pixel (x,y)=({:},{:})'.format(p2[0], p2[1]))
    plt.xlim((0,nint*ngroup))
    #plt.ylim((12200,13700))
    plt.legend()
    plt.grid()
    plt.title('Raw pixels')
    plt.xlabel('Frame Number')
    plt.ylabel('Raw ADU Values (uncal.fits)')
    plt.show()


    # Subtract the best estimate of the KTC level
    pixel1_reads_median = np.median(pixel1, axis=0)
    pixel1_sub = np.copy(pixel1)
    pixel1_offset = np.zeros(nint)
    for i in range(nint):
        pixel1_offset[i] = np.mean(pixel1[i,:] - pixel1_reads_median)
        pixel1_sub[i,:] = pixel1_sub[i,:] - pixel1_offset[i]
        plt.plot([0,nint*ngroup], [pixel1_reads_median, pixel1_reads_median], color='grey', ls='dotted')

    pixel2_reads_median = np.median(pixel2, axis=0)
    pixel2_sub = np.copy(pixel2)
    pixel2_offset = np.zeros(nint)
    for i in range(nint):
        pixel2_offset[i] = np.mean(pixel2[i,:] - pixel2_reads_median)
        pixel2_sub[i,:] = pixel2_sub[i,:] - pixel2_offset[i]

    for i in range(nint):
        plt.plot(np.arange(ngroup)+(i*ngroup), pixel1_sub[i,:], color='red')
    plt.scatter(np.arange(nint*ngroup), pixel1_sub, marker='.', color='red', label='KTC shifted Pixel ({:},{:})'.format(p1[0], p1[1]))
    plt.scatter(np.arange(nint*ngroup), pixel1, marker='.', color='black', label='Raw Pixel (x,y)=({:},{:})'.format(p1[0], p1[1]))
    plt.xlim((0,nint*ngroup))
    #plt.ylim((12200,13300))
    plt.legend()
    #plt.grid()
    plt.title('Raw pixel vs KTC correction (+ arbitrary 500 adu offset)')
    plt.xlabel('Frame Number')
    plt.ylabel('ADU Values')
    plt.show()

    # Plot the offsets measured to see deviants
    pixel1_ktcmed = np.median(pixel1_offset)
    pixel1_ktcdev = mediandev(pixel1_offset)
    pixel2_ktcmed = np.median(pixel2_offset)
    pixel2_ktcdev = mediandev(pixel2_offset)
    print('Pixel 1 KTC med, dev = {:},{:}'.format(pixel1_ktcmed, pixel1_ktcdev))
    print('Pixel 2 KTC med, dev = {:},{:}'.format(pixel2_ktcmed, pixel2_ktcdev))

    # Outliers (1 = bad data, np.nan = good data)
    outliers1 = np.where(np.abs(pixel1_offset-pixel1_ktcmed)/pixel1_ktcdev > 4, 1, np.nan)
    outliers2 = np.where(np.abs(pixel2_offset-pixel2_ktcmed)/pixel2_ktcdev > 4, 1, np.nan)

    plt.scatter(np.arange(nint), pixel1_offset, marker='.', color='black', label='Pixel 1 KTC Offsets')
    plt.scatter(np.arange(nint), pixel2_offset, marker='.', color='green', label='Pixel 2 KTC Offsets')
    plt.scatter(np.arange(nint), pixel1_offset*outliers1, marker='o', color='black', label='Outliers')
    plt.scatter(np.arange(nint), pixel2_offset*outliers2, marker='o', color='black', label='Outliers')

    plt.plot([0,nint],[pixel1_ktcmed,pixel1_ktcmed], color='grey', ls='dotted')
    plt.plot([0,nint],[pixel1_ktcmed,pixel1_ktcmed]+pixel1_ktcdev, color='black', ls='dotted')
    plt.plot([0,nint],[pixel1_ktcmed,pixel1_ktcmed]-pixel1_ktcdev, color='black', ls='dotted')
    plt.plot([0,nint],[pixel2_ktcmed,pixel2_ktcmed], color='grey', ls='dotted')
    plt.plot([0,nint],[pixel2_ktcmed,pixel2_ktcmed]+pixel2_ktcdev, color='green', ls='dotted')
    plt.plot([0,nint],[pixel2_ktcmed,pixel2_ktcmed]-pixel2_ktcdev, color='green', ls='dotted')
    plt.xlabel('Integration Number')
    plt.ylabel('KTC Offset applied [ADU]')
    plt.legend()
    plt.title('KTC Offset applied at each integration (to spot outliers)')
    plt.show()


    # Try spotting frames that strongly depart from their expected value
    # Method 1 - One constraint only
    reads_expected = np.median(pixel1_sub, axis=0)
    reads_scatter = mediandev(pixel1_sub, axis=0)
    print('median is =', reads_expected)
    print('scatter is =', reads_scatter)

    sigdeviation = np.copy(pixel1) * 0
    for i in range(nint):
        # How many sigma away from expected is the actual value
        sigdeviation[i, :] = np.abs(pixel1_sub[i, :] - reads_expected) / reads_scatter
    # Make a mask (bad data = 1, good data = np.nan)
    badmask_iter1 = np.where(sigdeviation > 4, 1, np.nan)

    # Method 2 - Other constraint
    badmask_iter2 = np.zeros((nint,ngroup))
    for i in range(nint):
        badmask_iter2[i,:] = np.copy(outliers1[i])

    # All constraints together
    badmask = np.where(np.isfinite(badmask_iter1) | np.isfinite(badmask_iter2), 1, np.nan)
    goodmask = np.where(np.isfinite(badmask), np.nan, 1)

    frames = np.arange(nint*ngroup).reshape((nint,ngroup))
    plt.scatter(frames, pixel1_sub, marker='.', color='green', label='KTC shifted Pixel (x,y)=(110,198) ')
    plt.scatter(frames, pixel1_sub*badmask_iter1, marker='o', color='grey', label='Masking - 1 constraint')
    plt.scatter(frames, pixel1_sub*badmask_iter2, marker='.', color='black', label='Masking - other constraint')
    plt.scatter(frames, pixel1_sub*badmask, marker='x', color='red', label='Masking - all constraints')
    plt.title('Time-Series and masked data')
    plt.xlabel('Frame Number')
    plt.ylabel('KTC-Corrected Values [adu]')
    plt.legend()
    plt.show()



    # How do the signals correlate? - it gets tighter but otherwise, not clear what info this gives...
    plt.scatter(pixel1, pixel2, marker='.', color='black', label='Raw [ADU]')
    for i in range(ngroup):
        plt.scatter(pixel1_sub[:,i]*goodmask[:,i], pixel2_sub[:,i]*goodmask[:,i], marker='.', label='KTC Corrected [ADU]')
    #plt.xlim((12200,12900))
    #plt.ylim((13000,13700))
    plt.title('2-Pixel Correlation with/without KTC Correction')
    plt.grid()
    #plt.legend()
    plt.show()







    return

if __name__ == "__main__":

    ktc()