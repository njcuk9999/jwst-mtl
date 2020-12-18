def make_trace_mask(stack, tracex, tracey, halfwidth=15, masktop=None,
                    maskbottom=None, header=None):
    ''' Builds a mask of the same size as the input stack image. That mask masks a band
    of pixels around the trace position (tracex, tracey) of width = 2*halfwidth*yos pixels.
    Option masktop additionally masks all pixels above the trace. Option maskbottom addtionally
    masks all pixels below the trace.
    Parameters
    ----------
    stack
    '''

    # Value assigned to a pixel that should be masked out
    mask_value = np.nan

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subaaray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header)

    # Intitialize the mask array. All 1 except potential NaNs.
    mask = np.copy(stack)*0+1
    print(np.shape(mask))

    # Column by column, mask out pixels beyond the halfwidth of the trace center
    y = np.arange(dimy)
    for i in range(dimx):
        print(i,tracex[i],tracey[i])
        d = np.abs(y - tracey[i])
        ind = d > halfwidth * yos
        mask[ind, i] = mask_value
        # If masktop is set then mask pixels above the y position
        if masktop:
            ind = (y - tracey[i]) > 0
            mask[ind, i] = mask_value
        # If masktop is set then mask pixels below the y position
        if maskbottom:
            ind = (y - tracey[i]) < 0
            mask[ind, i] = mask_value

    return mask

def determine_stack_dimensions(stack, header=None):
    ''' Determine the size of the stack array. Will be called by get_uncontam_centroids
    and make_trace_mask.
    Parameters
    ----------
    stack : array of floats (2D)
        Data frame. Assumes DMS orientation.
        This array could be a native pixel size SOSS subarray or FF.
        It could also be a 2D trace reference file in which case padding exists
        around the edges, and the pixels may be oversampled by some integer factor.
    header : fits header
        Header associated to the stack array.
        If the header is None then some assumptions will be made regarding the stack array.
        If the header is passed, then specific keywords will be read in it to assess what
        the stack array is. This ensures that a 2D Trace Reference file will be digested
        properly.
    Returns
    -------
    dimx, dimy : The dimensions of the stack array.
    xos, yos : The oversampling factor (integer) of the stack array.
    xnative, ynative : The dimensions of the stack image, expressed in native pixels units.
    padding : the size of padding all around the image, in units of native pixels.
    working_pixel_bool : a 2D array of the same size as stack with boolean values of
        False where pixels are not light sensitive (the reference pixels). True elsewhere.
    '''

    # Dimensions of the subarray.
    dimy, dimx = np.shape(stack)

    # Determine what is the input stack based either on its dimensions or on
    # the header if passed. Construct a mask of working pixels in case the
    # stack contains reference pixels.
    if header is None:
        # No header passed - Assume that the stack is a valid SOSS subarray or FF, i.e.
        # 2048x256 or 2040x252 (working pixels) or multiple of if oversampled
        # 2048x96 or 2040x96 (working pixels) or multiple of
        # 2048x2048 or 2040x2040 (working pixels) or multiple of
        if (dimx % 2048) == 0:
            # stack is a multiple of native pixels.
            xnative = 2048
            # The x-axis oversampling is:
            xos = int(dimx / 2048)
        elif (dimx % 2040) == 0:
            # stack is a multiple of native *** working *** pixels.
            xnative = 2040
            # The y-axis oversampling is:
            xos = int(dimx / 2040)
        else:
            # stack x dimension has unrecognized size.
            print('Stack X dimension has unrecognized size of {:}. Accepts 2048, 2040 or multiple of.'.format(dimx))
            sys.exit()
        # Check if the Y axis is consistent with the X axis.
        acceptable_ydim = [96,256,252,2040,2048]
        yaxis_consistent = False
        for accdim in acceptable_ydim:
            if dimy / (accdim*xos) == 1:
                # Found the acceptable dimension
                yos = np.copy(xos)
                ynative = np.copy(accdim)
                yaxis_consistent = True
        if yaxis_consistent == False:
            # stack y dimension is inconsistent with the x dimension.
            print('Stack Y dimension ({:}) is inconsistent with X dimension ({:}) for acceptable SOSS arrays'.format(dimy,dimx))
            sys.exit()
        # Construct a boolean mask (true or false) of working pixels
        working_pixel_bool = np.full((dimy, dimx), True)

        # For dimensions where reference pixels would have been included in the stack,
        # mask those reference pixels out.
        # Sizes 96, 252 and 2040 should not contain any reference pixel.
        if xnative == 2048:
            # Mask out the left and right columns of reference pixels
            working_pixel_bool[:, 0:xos * 4] = False
            working_pixel_bool[:, -xos * 4:] = False
        if ynative == 2048:
            # Mask out the top and bottom rows of reference pixels
            working_pixel_bool[0:yos * 4, :] = False
            working_pixel_bool[-yos * 4:, :] = False
        if ynative == 256:
            # Mask the top rows of reference pixels
            working_pixel_bool[-yos * 4:,:] = False

        # Initialize padding to zero in this case because it is not a 2D Trace ref file
        padding = int(0)

    else:
        # header was passed
        # Read in the relevant keywords
        xos, yos = int(header['OVERSAMP']), int(header['OVERSAMP'])
        padding = int(header['PADDING'])
        # The 2D Trace profile is for FULL FRAME so 2048x2048
        xnative, ynative = int(2048), int(2048)
        # Check that the stack respects its intended format
        if dimx != ((xnative+2*padding)*xos):
            # Problem
            print('The header passed is inconsistent with the X dimension of the stack.')
            sys.exit()
        if dimy != ((ynative+2*padding)*yos):
            # Problem
            print('The header passed is inconsistent with the Y dimension of the stack.')
            sys.exit()
        # Construct a mask of working pixels. The 2D Trace REFERENCE file does
        # not contain any reference pixel. So all are True.
        working_pixel_bool = np.full((dimy, dimx), True)

    # For debugging purposes...
    if True:
        print('dimx={:}, dimy={:}, xos={:}, yos={:}, xnative={:}, ynative={:}'.format(dimx, dimy, xos, yos, xnative, ynative))

    return(dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool)


def get_uncontam_centroids(stack, header=None, badpix=None, tracemask=None):
    '''Determine the x, y positions of the trace centroids from an
    exposure using a center-of-mass analysis. Works for either order if there
    is no contamination, or for order 1 on a detector where the two orders
    are overlapping.
    This is an adaptation of Loïc's get_order1_centroids which can better
    deal with a bright second order.
    Parameters
    ----------
    stack : array of floats (2D)
        Data frame. Assumes DMS orientation.
        This array could be a native pixel size SOSS subarray or FF.
        It could also be a 2D trace reference file in which case padding exists
        around the edges, and the pixels may be oversampled by some integer factor.
    header : fits header
        Header associated to the stack array.
        If the header is None then some assumptions will be made regarding the stack array.
        If the header is passed, then specific keywords will be read in it to assess what
        the stack array is. This ensures that a 2D Trace Reference file will be digested
        properly.
    badpix : array of floats (2D) with anything different than zero meaning a bad pixel
        Optional input bad pixel mask to apply to the stack. Should be of
        the same dimensions as the stack.
    tracemask : array of floats (2D) with anything different than zero meaning a
        masked out pixel. The spirit is to have zeros along one spectral order with
        a certain width.
    Returns
    -------
    tracexbest : np.array
        Best estimate data x centroid.
    traceybest : np.array
        Best estimate data y centroids.
    '''

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subaaray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header)

    # Make a numpy mask array of the working pixels
    working_pixel_mask = np.ma.array(np.ones((dimy, dimx)), mask = np.invert(working_pixel_bool))
    # Fill the working pixel mask with NaN
    working_pixel_mask = np.ma.filled(working_pixel_mask, np.nan)

    # Check for the optional input badpix and create a bad pixel numpy mask
    if badpix != None:
        # 1) Check the dimension is the same as stack
        # TODO:
        # 2) Create the numpy.ma array with it
        # The bad pixel mask has values of 'one' for valid pixels.
        badpix_mask = np.ma.array(np.ones((dimy, dimx)), mask = (badpix != 0) )
    else:
        # Create a mask with all valid pixels (all ones)
        badpix_mask = np.ma.array(np.ones((dimy, dimx)))
    # Fill the bad pixels with NaN
    badpix_mask = np.ma.filled(badpix_mask, np.nan)

    # Check for the optional input tracemask and create a trace numpy mask
    if tracemask != None:
        # 1) Check the dimension is the same as stack
        # TODO:
        # 2) Create the numpy.ma array with it
        # The trace mask has values of 'one' for valid pixels.
        trace_mask = np.ma.array(np.ones((dimy, dimx)), mask = (tracemask != 0) )
    else:
        # Create a mask with all pixels in the trace (all ones)
        trace_mask = np.ma.array(np.ones((dimy, dimx)))
    # Fill the trace mask with NaN
    trace_mask = np.ma.filled(trace_mask, np.nan)

    # Multiply working pixel mask, bad pixel mask and trace mask
    # The stack image with embedded numpy mask is stackm
    stackm = stack * badpix_mask * working_pixel_mask * trace_mask



    # Identify the floor level of all 2040 working cols to subtract it first.
    floorlevel = np.nanpercentile(stackm, 10, axis=0)
    backsub = stackm - floorlevel
    # Find centroid - first pass, use all pixels in the column.
    # Normalize each column
    norm = backsub / np.nanmax(backsub, axis=0)
    # Create 2D Array of pixel positions
    rows = (np.ones((dimx, dimy)) * np.arange(dimy)).T
    # CoM analysis to find centroid
    com = (np.nansum(norm * rows, axis=0) / np.nansum(norm, axis=0)).data
    # Adopt these trace values as best
    tracex_best = np.arange(dimx)
    tracey_best = np.copy(com)
    # Second pass, find centroid on a subset of pixels
    # from an area around the centroid determined earlier.
    tracex = np.arange(dimx)
    tracey = np.zeros(dimx)*np.nan
    row = np.arange(dimy)
    w = 30 * yos
    for i in range(dimx):
        miny = np.int(np.nanmax([np.around(tracey_best[i] - w), 0]))
        maxy = np.int(np.nanmax([np.around(tracey_best[i] + w), dimy - 1]))
        val = backsub[miny:maxy, i] / np.nanmax(backsub[:, i])
        ind = np.where(np.isfinite(val))
        thisrow = (row[miny:maxy])[ind]
        thisval = val[ind]
        com = np.sum(thisrow * thisval) / np.sum(thisval)
        # Ensure that the centroid position is not getting too close to an edge
        # such that it is biased.
        if (not np.isfinite(com)) or (com <= 5*yos) or (com >= (ynative-6)*yos):
            continue
        # For a bright second order, it is likely that the centroid at this
        # point will be somewhere in between the first and second order.
        # If this is the case (i.e. the pixel value of the centroid is very low
        # compared to the column average), restrict the range of pixels
        # considered to be above the current centroid.
        if backsub[int(com)][i] < np.nanmean(backsub[(int(com) - w):(int(com) + w), i]):
            miny = np.int(np.nanmax([np.around(com), 0]))
            maxy = np.int(np.nanmin([np.around(com + 2*w), dimy - 1]))
            val = backsub[miny:maxy, i] / np.nanmax(backsub[:, i])
            ind = np.where(np.isfinite(val))
            thisrow = (row[miny:maxy])[ind]
            thisval = val[ind]
            com = np.sum(thisrow * thisval) / np.sum(thisval)
        tracey[i] = com
    # Adopt these trace values as best.
    tracex_best = np.copy(tracex)
    tracey_best = np.copy(tracey)

    # Third pass - fine tuning.
    tracex = np.arange(dimx)
    tracey = np.zeros(dimx) * np.nan
    row = np.arange(dimy)
    w = 16 * yos
    for i in range(len(tracex_best)):
        miny = np.int(np.nanmax([np.around(tracey_best[i] - w), 0]))
        maxy = np.int(np.nanmax([np.around(tracey_best[i] + w), dimy - 1]))
        val = backsub[miny:maxy, i] / np.nanmax(backsub[:, i])
        ind = np.where(np.isfinite(val))
        thisrow = (row[miny:maxy])[ind]
        thisval = val[ind]
        com = np.sum(thisrow * thisval) / np.sum(thisval)
        tracex[i] = np.copy(tracex_best[i])
        tracey[i] = np.copy(com)
    # Update with the best estimates
    tracex_best = np.copy(tracex)
    tracey_best = np.copy(tracey)

    if True:
        plt.figure()
        plt.plot(tracex_best, tracey_best)

    # Final pass : Fitting a polynomial to the measured (noisy) positions
    if padding == 0:
        # Only use the non NaN pixels.
        induse = np.isfinite(tracex_best) & np.isfinite(tracey_best)
    else:
        # Important steps in the case of the 2D Trace reference file.
        # Mask out the padded pixels from the fit so it is rigorously the
        # same as for regular science images.
        induse = np.isfinite(tracex_best) & np.isfinite(tracey_best) & \
                 (tracex_best >= xos*padding) & (tracex_best < (dimx-xos*padding))
    # Use a *** fixed *** polynomial order of 11 to keep results consistent
    # from data set to data set. Any systematics would remain fixed.
    param = np.polyfit(tracex_best[induse], tracey_best[induse], 11)
    tracey_best = np.polyval(param, tracex_best)

    if True:
        plt.plot(tracex_best, tracey_best, color='r')
        plt.show()

    return tracex_best, tracey_best



# test the script

import numpy as np
import sys
from astropy.io import fits
import matplotlib.pylab as plt

im = np.zeros((256*2,2040*2))
a = fits.open('/genesis/jwst/userland-soss/loic_review/stack_256_ng3_DMS.fits')
im = a[0].data

x, y = get_uncontam_centroids(im)
mask_o1 = make_trace_mask(im, x, y, halfwidth=40, masktop=True)

plt.imshow(mask_o1, origin='bottom')