def get_uncontam_centroids(stack, header=None, badpix=None, specpix=None):
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
    badpix : array of floats (2D) with anything different than zero meaning a good pixel
        Optional input bad pixel mask to apply to the stack. Should be of
        the same dimensions as the stack.
    specpix : list of floats
        Pixel x values at which to extract the trace centroids.
        It is TBD whether we mean native pixels or oversampled pixels...
    Returns
    -------
    tracexbest : np.array
        Best estimate data x centroid.
    traceybest : np.array
        Best estimate data y centroids.
    '''


    # Dimensions of the subarray.
    #dimx = len(specpix)
    #dimy = np.shape(stack)[0]
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
        # Construct a mask of working pixels
        working_pixel_mask = np.full((dimy, dimx), True)
        # For dimensions where reference pixels would have been included in the stack,
        # mask those reference pixels out.
        # Sizes 96, 252 and 2040 should not contain any reference pixel.
        if xnative == 2048:
            # Mask out the left and right columns of reference pixels
            working_pixel_mask[:, 0:xos * 4] = False
            working_pixel_mask[:, -xos * 4:] = False
        if ynative == 2048:
            # Mask out the top and bottom rows of reference pixels
            working_pixel_mask[0:yos * 4, :] = False
            working_pixel_mask[-yos * 4:, :] = False
        if ynative == 256:
            # Mask the top rows of reference pixels
            working_pixel_mask[-yos * 4:,:] = False
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
        working_pixel_mask = np.full((dimy, dimx), True)

    # For debugging purposes...
    if True:
        print('dimx={:}, dimy={:}, xos={:}, yos={:}, xnative={:}, ynative={:}'.format(dimx, dimy, xos, yos, xnative, ynative))

    if badpix != None:
        # 1) Check the dimension is the same as stack
        # TODO:
        # 2) Create the numpy.ma array with it
        badpix_mask = np.ma.array(np.ones((dimy, dimx)), mask = (badpix != 0) )

    # Multiply working pixel mask and bad pixel mask
    # TODO:

    sys.exit()



    # Identify the floor level of all 2040 working cols to subtract it first.
    floorlevel = np.nanpercentile(stack, 10, axis=0)
    backsub = stack - floorlevel
    # Find centroid - first pass, use all pixels in the column.
    # Normalize each column
    norm = backsub[:, 4:2044] / np.nanmax(backsub[:, 4:2044], axis=0)
    # Create 2D Array of pixel positions
    rows = (np.ones((2040, 256)) * np.arange(256)).T
    # Mask any nan values
    norm_mask = np.ma.masked_invalid(norm)
    # CoM analysis to find centroid
    cx = (np.nansum(norm_mask * rows, axis=0) / np.nansum(norm, axis=0)).data
    # Adopt these trace values as best
    tracex_best = np.arange(2040)+4
    tracey_best = cx
    # Second pass, find centroid on a subset of pixels
    # from an area around the centroid determined earlier.
    tracex = []
    tracey = []
    row = np.arange(dimy)
    w = 30
    for i in range(dimx - 8):
        miny = np.int(np.nanmax([np.around(tracey_best[i] - w), 0]))
        maxy = np.int(np.nanmax([np.around(tracey_best[i] + w), dimy - 1]))
        val = backsub[miny:maxy, i + 4] / np.nanmax(backsub[:, i + 4])
        ind = np.where(np.isfinite(val))
        thisrow = (row[miny:maxy])[ind]
        thisval = val[ind]
        cx = np.sum(thisrow * thisval) / np.sum(thisval)
        # Ensure that the centroid position is not getting too close to an edge
        # such that it is biased.
        if not np.isfinite(cx) or cx <= 5 or cx >= 250:
            continue
        # For a bright second order, it is likely that the centroid at this
        # point will be somewhere in between the first and second order.
        # If this is the case (i.e. the pixel value of the centroid is very low
        # compared to the column average), restrict the range of pixels
        # considered to be above the current centroid.
        if backsub[int(cx)][i+4] < np.nanmean(backsub[(int(cx) - w):(int(cx)+w), i+4]):
            miny = np.int(np.nanmax([np.around(cx), 0]))
            maxy = np.int(np.nanmin([np.around(cx + 2*w), dimy - 1]))
            val = backsub[miny:maxy, i + 4] / np.nanmax(backsub[:, i + 4])
            ind = np.where(np.isfinite(val))
            thisrow = (row[miny:maxy])[ind]
            thisval = val[ind]
            cx = np.sum(thisrow * thisval) / np.sum(thisval)
        tracex.append(i + 4)
        tracey.append(cx)
    # Adopt these trace values as best.
    tracex_best = np.array(tracex) * 1
    tracey_best = np.array(tracey) * 1
    # Third pass - fine tuning.
    tracex = []
    tracey = []
    row = np.arange(dimy)
    w = 16
    for i in range(len(tracex_best)):
        miny = np.int(np.nanmax([np.around(tracey_best[i] - w), 0]))
        maxy = np.int(np.nanmax([np.around(tracey_best[i] + w), dimy - 1]))
        val = backsub[miny:maxy, i + 4] / np.nanmax(backsub[:, i + 4])
        ind = np.where(np.isfinite(val))
        thisrow = (row[miny:maxy])[ind]
        thisval = val[ind]
        cx = np.sum(thisrow * thisval) / np.sum(thisval)
        tracex.append(tracex_best[i])
        tracey.append(cx)
    tracex_best = np.array(tracex)
    tracey_best = np.array(tracey)
    return tracex_best, tracey_best



# test the script

import numpy as np

import sys

im = np.zeros((256*2,2040*2))
get_uncontam_centroids(im)