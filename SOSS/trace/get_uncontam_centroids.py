def determine_stack_dimensions(stack, header=None, verbose=False):
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
    if verbose == True:
        print('dimx={:}, dimy={:}, xos={:}, yos={:}, xnative={:}, ynative={:}'.format(dimx, dimy, xos, yos, xnative, ynative))

    return(dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool)



def make_trace_mask(stack, tracex, tracey, halfwidth=30, extend_redward=None,
                    extend_blueward=None, header=None, verbose=False):
    ''' Builds a mask of the same size as the input stack image. That mask masks a band
    of pixels around the trace position (tracex, tracey) of width = 2*halfwidth*yos pixels.
    Option masktop additionally masks all pixels above the trace. Option extend_redward
    additionally masks all pixels below the trace.
    Parameters
    ----------
    stack
    '''

    # Value assigned to a pixel that should be masked out
    masked_value = np.nan
    notmasked_value = 1.0

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subaaray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header)

    # Intitialize the mask array to unmasked value.
    mask = np.ones((dimy, dimx)) * notmasked_value
    if verbose == True: print(np.shape(mask))

    # Column by column, mask out pixels beyond the halfwidth of the trace center
    y = np.arange(dimy)
    for i in range(dimx):
        if verbose == True: print(i,tracex[i],tracey[i])
        # Mask the pixels in the trace
        d = np.abs(y - tracey[i])
        ind = d < halfwidth * yos
        mask[ind, i] = masked_value
        # If extend_redward is set then mask pixels redward (in the spatial axis)
        # of the trace (so not only mask the trace but all pixels redward of that
        # along the spatial axis).
        if extend_redward:
            ind = (y - tracey[i]) < 0
            mask[ind, i] = masked_value
        # If extend_blueward is set then mask pixels blueward along the spatial axis
        if extend_blueward:
            ind = (y - tracey[i]) > 0
            mask[ind, i] = masked_value

    return mask



def make_mask_vertical(stack, header=None, masked_side = 'blue', native_x = 1700):
    ''' Builds a mask where there are two sides: left and right, one being masked, the
    other not. In other words, this masks a region along the spectral dispersion axis.
    native_x : the boundary position in native x pixels (0 to 2047)
    masked_side : either 'blue' or 'red'
    '''

    # Value assigned to a pixel that should be masked out
    masked_value = np.nan
    notmasked_value = 1.0

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subarray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header)

    if xnative == 2040: xcut = (native_x-4+padding)*xos
    if xnative == 2048: xcut = (native_x+padding)*xos

    mask = np.ones((dimy,dimx)) * notmasked_value

    if masked_side == 'blue':
        mask[:,xcut:] = masked_value
    if masked_side == 'red':
        mask[:,0:xcut] = masked_value

    return mask



def make_mask_sloped(stack, header=None, masked_side='blue', pt1 = [0,0], pt2 = [2048,0]):
    ''' Draw a sloped line and mask on one side of it (the side is defined with respect to
    the spectral dispersion axis. Requires the x,y position of two points that define the
    line. The x,y must be given in native size pixels. along the x axis: 0-2047, along the
    y-axis, it depends on the array size. For SUBSTRIP256, y=0-255, for FF, y=0-2047'''

    # Value assigned to a pixel that should be masked out
    masked_value = np.nan
    notmasked_value = 1.0

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subarray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header)

    # Simplify one's life and simply fit the two points
    thex = (np.array([pt1[0],pt2[0]])+padding) * xos
    they = (np.array([pt1[1],pt2[1]])+padding) * yos
    param = np.polyfit(thex, they, 1)

    print(param)

    # Initialize a mask
    mask = np.ones((dimy,dimx)) * notmasked_value

    # Compute the position of the line at every x position
    fitx = np.arange(dimx)
    fity = np.polyval(param, fitx) # round it
    # Make sure negative values in fity get floored to zero, to be able
    # to index in array (below) without wrapping.
    fity[fity < 0] = 0

    # Branch depending on side that needs masking and sign of the slope
    if masked_side == 'blue':
        if param[0] < 0:
            for i in range(dimx): mask[int(fity[i]):,i] = masked_value
        else:
            for i in range(dimx): mask[0:int(fity[i]), i] = masked_value
    if masked_side == 'red':
        if param[0] < 0:
            for i in range(dimx): mask[0:int(fity[i]),i] = masked_value
        else:
            for i in range(dimx): mask[int(fity[i]):, i] = masked_value

    return mask



def make_mask_butorder3(stack, header=None):
    ''' Builds the mask that will be handy to get 3rd order centroids
    '''

    # Value assigned to a pixel that should be masked out
    masked_value = np.nan
    notmasked_value = 1.0

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subarray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header)

    # Line to mask redward of order 3: (x,y)_1=(0,132) to (x,y)_2=(1000,163)
    slope = (163 - 132) / 1000.
    yintercept = 132.0
    # vertical line redward of which no order 3 is detected
    maxcolumn = 700

    mask = np.ones((dimy,dimx)) * notmasked_value
    if ynative == 96:
        # Nothing to be done because order 3 can not be present.
        return(mask)
    if (ynative == 252) or (ynative == 256):
        m = slope * yos / xos
        if (xnative == 2040):
            b = (yintercept + 4*m + padding) * yos
            xmax = (maxcolumn - 4 + padding) * xos
        if (xnative == 2048):
            b = (yintercept + padding) * yos
            xmax = (maxcolumn + padding) * xos
    if (ynative == 2040):
        m = slope * yos/xos
        if (xnative == 2040):
            b = (1788 + yintercept + 4*m + padding) * yos
            xmax = (maxcolumn - 4 + padding) * xos
        if (xnative == 2048):
            b = (1788 + yintercept + padding) * yos
            xmax = (maxcolumn + padding) * xos
    if (ynative == 2048):
        m = slope * yos/xos
        if (xnative == 2040):
            b = (1792 + yintercept + 4*m + padding) * yos
            xmax = (maxcolumn - 4 + padding) * xos
        if (xnative == 2048):
            b = (1792 + yintercept + padding) * yos
            xmax = (maxcolumn + padding) * xos

    # Mask redward (spatially) of the sloped line
    for i in range(dimx):
        y = np.int(np.round(b + m * i))
        mask[0:y,i] = masked_value
    # Mask redward (spectrally) of the vertical line
    mask[:,xmax:] = masked_value

    return(mask)






def get_edge_centroids(stack, header=None, badpix=None, mask=None, verbose=False,
                       return_what='edgemean_param', polynomial_order=2):
    ''' Determine the x, y positions of the trace centroids from an exposure 
    using the two edges and the width of the traces. This should be performed on a very high SNR
    stack.

    return_what : What to return. Either x,y positions or polynomial parameters,
    either for one of the edges or for the mean of both edges (i.e. trace center)
    'edgemean_param' : polynomial fit parameters to the mean of both edges, i.e. trace center
    'edgeman_xy' : x, y values for the mean of both edges
    'rededge_param' : polynomial fit parameters to the red edge (spatially)
    'rededge_xy' : x, y values for the red edge
    'blueedge_param' : polynomial fit parameters to the blue edge (spatially)
    'blueedge_xy' : x, y values for the blue edge
    'tracewidth' : scalar representing the median of (red edge - blue edge) y values
    '''

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subarray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header)

    edge1, edge2 = [], []
    for i in range(dimx):
        #if i >= 1000: verbose = True
        y1, y2 = edge_trigger(stack[:,i]*mask[:,i], triggerscale=5, verbose=verbose)
        #print(i, y1, y2)
        edge1.append(y1)
        edge2.append(y2)
    edge1 = np.array(edge1)
    edge2 = np.array(edge2)

    # Fit the red edge
    x_red = np.arange(dimx)
    ind = np.where(np.isfinite(edge1))
    param_red = polyfit_sigmaclip(x_red[ind], edge1[ind], polynomial_order)
    y_red = np.polyval(param_red, x_red)
    # Fit the blue edge
    x_blue = np.arange(dimx)
    ind = np.where(np.isfinite(edge2))
    param_blue = polyfit_sigmaclip(x_blue[ind], edge2[ind], polynomial_order)
    y_blue = np.polyval(param_blue, x_blue)
    # Fit the mean of both edges
    x_both = np.arange(dimx)
    both = (edge1+edge2)/2.
    ind = np.where(np.isfinite(both))
    param_both = polyfit_sigmaclip(x_both[ind], both[ind], polynomial_order)
    y_both = np.polyval(param_both, x_both)

    # Plot the edge position as a function of x
    if True:
        plt.plot(edge1)
        plt.plot(edge2)
        plt.plot(both)
        plt.plot(x_red, y_red)
        plt.plot(x_blue, y_blue)
        plt.plot(x_both, y_both)
        plt.show()

    if return_what == 'edgemean_param' : return param_both
    if return_what == 'rededge_param' : return param_red
    if return_what == 'blueedge_param' : return param_blue
    if return_what == 'edgemean_xy' : return x_both, y_both
    if return_what == 'rededge_xy' : return x_red, y_red
    if return_what == 'blueedge_xy' : return x_blue, y_blue
    if return_what == 'tracewidth' : return np.nanmedian(np.abs(y_blue - y_red))


def polyfit_sigmaclip(x, y, order, sigma=4):

    polyorder = np.copy(order)
    ind = np.where(y == y)
    for n in range(5):
        if n == 4: polyorder = order + 1
        param = np.polyfit(x[ind], y[ind], polyorder)
        yfit = np.polyval(param, x)
        dev = np.abs(yfit - y)
        ind = np.where(dev <= sigma)

    return param




def edge_trigger(column, triggerscale=5, verbose=False):
    # triggerscale must be odd integer

    # dimension of the column array
    dim, = np.shape(column)
    halftrig = int((triggerscale-1)/2)
    # positions along that column where the full triggerscale is accessible
    ic = halftrig + np.arange(dim-triggerscale)
    # slope of the flux at position datax
    slope = []
    datax = np.arange(triggerscale)
    # For each position, grab current n pixels, exclude NaN, fit a slope
    for i in ic:
        data = column[i-halftrig:i+halftrig+1]
        ind = np.where(np.isfinite(data))
        if np.size(ind) >=3:
            param = np.polyfit(datax[ind],data[ind],1)
            slope.append(param[0])
        else:
            slope.append(np.nan)
    slope = np.array(slope)

    # Determine which x sees the maximum slope
    indmax = np.argwhere(slope == np.nanmax(slope))
    edgemax = np.nan # default value because ref pixels produce no slope
    if indmax.size > 0: edgemax = ic[indmax[0][0]]

    # Determine which x sees the minimum slope
    indmin = np.argwhere(slope == np.nanmin(slope))
    edgemin = np.nan
    if indmin.size > 0: edgemin = ic[indmin[0][0]]

    # Make a plot if verbose is True
    if verbose == True:
        plt.plot(ic,slope)
        plt.show()

    return edgemax, edgemin



def get_uncontam_centroids(stack, header=None, badpix=None, tracemask=None, verbose=False,
                           specpix_bounds=None):
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
    specpix_bounds : native spectral pixel bounds to consider in fitting the trace. Most
        likely used for the 2nd and 3rd orders, not for the 1st order.
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
    if badpix is not None:
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
    if tracemask is not None:
        # 1) Check the dimension is the same as stack
        # TODO:
        # 2) Create the numpy.ma array with it
        # The trace mask has values of 'one' for valid pixels.
        trace_mask = np.ma.array(np.ones((dimy, dimx)), mask = (tracemask == 0) )
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

    if verbose == True:
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
    # Find indices of pixels to include in the polynomial fit if specpix_bounds
    if specpix_bounds != None:
        indfit = (tracex_best >= np.min(specpix_bounds)) & (tracex_best <= np.max(specpix_bounds))
    else:
        indfit = np.copy(induse)
    # Use a *** fixed *** polynomial order of 11 to keep results consistent
    # from data set to data set. Any systematics would remain fixed.
    if specpix_bounds != None:
        polyorder = 2
    else:
        polyorder = 11
    param = np.polyfit(tracex_best[induse & indfit], tracey_best[induse & indfit], polyorder)
    tracey_best = np.polyval(param, tracex_best)

    if verbose == True:
        plt.plot(tracex_best, tracey_best, color='r')
        plt.show()

    return tracex_best, tracey_best




def trace_centroids(stack, header=None, badpix=None, verbose=False):
    ''' Sequential scheme to get the trace centroids, depending on the image dimensions

    '''

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subarray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header)


    ######################################################################
    ############################ FIRST ORDER #############################
    ######################################################################
    # First, call the get_uncontam_centroids for the first order.
    # All sub-array sizes should have a first order. SUBSTRIP256 and SUBSTRIP96 will
    # have it very close to the bottom edge, i.e. We assume that the brightest
    # trace is the one of interest. For FULLFRAME, things may be a bit different
    # as a field contaminant may be present. So for FF, we apply a mask to mask
    # out pixels outside the 256x2048 region.
    if (ynative == 96) or (ynative == 256) or (ynative == 252):
        # Call the centroiding algorithm
        x_o1, y_o1 = get_uncontam_centroids(stack, header=header, badpix=badpix,
                                            verbose=verbose)
    if (ynative == 2048):
        # Build a mask to mask out what is below the 1st order trace
        fieldmask = np.ones(dimy, dimx)
        # Adjust the row where we cut according to whether the reference pixels are included or not
        rowcut = 1792  # works for 2048
        if (ynative == 2040): rowcut = 1788
        # Set to 0 that outside region
        fieldmask[(0 + padding) * yos:(1792 + padding) * yos, :] = 0
        # Call the centroiding algorithm on that masked image (use tracemask optional keyword)
        x_o1, y_o1 = get_uncontam_centroids(stack, header=header, badpix=badpix,
                                            tracemask=fieldmask, verbose=verbose)

    if (ynative == 96):
        return x_o1, y_o1
    else:
        ######################################################################
        ############################ THIRD ORDER #############################
        ######################################################################
        # Now, we are ready to get the THIRD order trace positions. This applies only to SUBSTRIP256
        # and to FF images. The SECOND order trace will ba handled last.

        # Make a mask to measure the 3rd order trace
        mask_allbut3 = make_mask_butorder3(stack, header=header)
        # Get the centroid position by locking on the traces edges and returning their mean
        x_o3, y_o3 = get_edge_centroids(stack, header=header, badpix=badpix, mask=mask_allbut3,
                                        return_what='edgemean_xy', polynomial_order=2, verbose=verbose)

        ######################################################################
        ########################### SECOND ORDER #############################
        ######################################################################
        # Now, we are ready to get the SECOND order trace position. We will make use of our
        # knowledge of order 1 and 3 to create a masks and draw lines for further masks.

        ######################################################################
        # Determine the trace width in the uncontaminated region of order 2
        ######################################################################
        # First, the order 1 trace needs to be masked out. Construct a mask that not
        # only covers the order 1 trace but everything redward along the spatial axis.
        mask_o1 = make_trace_mask(stack, x_o1, y_o1, halfwidth=34, extend_redward=True,
                                  verbose=verbose, header=header)
        # Do the same to mask out order 3 - this one is fainter so make a narrower mask.
        # Also mask all pixels blueward (spatially)
        mask_o3 = make_trace_mask(stack, x_o3, y_o3, halfwidth=15, extend_blueward=True,
                                  verbose=verbose, header=header)
        # Mask what is on the left side where orders 1 and 2 are well blended
        mask_red = make_mask_vertical(stack, header=header, masked_side='red', native_x=1100)
        # Mask everything to the right of where the 2nd order goes out of the image
        mask_blue = make_mask_vertical(stack, header=header, masked_side='blue', native_x=1500)
        # Combine masks
        fullmask = mask_o1 * mask_o3 * mask_red * mask_blue
        # Get the trace width by locking on both trace edges in a region without contamination
        tracewidth = get_edge_centroids(stack, header=header, badpix=badpix, mask=fullmask,
                                        return_what='tracewdith', polynomial_order=3, verbose=verbose)

        print('tracewidth =', tracewidth)
        print(np.shape(tracewidth))
        ######################################################################
        # Find the inner trace edge positions
        ######################################################################
        # Mask what is on the left side where orders 1 and 2 are well blended
        mask_red = make_mask_vertical(stack, header=header, masked_side='red', native_x=500)
        # Mask along a sloped line closely following the bottom of order 2 (rejecting toward 1st order)
        mask_slope = make_mask_sloped(stack, header=header, masked_side='blue',
                                      pt1 = [1450, ynative-161], pt2 = [1775, ynative-0])
        # Mask everything to the right of where the 2nd order goes out of the image
        mask_blue = make_mask_vertical(stack, header=header, masked_side='blue', native_x=1775)
        # Combine all masks
        fullmask = mask_o1 * mask_o3 * mask_red * mask_slope * mask_blue
        # Get the centroid position by locking on the trace blue edge only (provides best tracing
        # capabilities)
        x_o2_edge, y_o2_edge = get_edge_centroids(stack, header=header, badpix=badpix, mask=fullmask,
                                        return_what='blueedge_xy', polynomial_order=3, verbose=verbose)

        ######################################################################
        # Combine the inner edge position with the trace width to get solution
        ######################################################################
        x_o2 = np.copy(x_o2_edge)
        y_o2 = y_o2_edge - tracewidth/2.

        return (x_o1, y_o1, x_o2, y_o2, x_o3, y_o3)





# test the script

import numpy as np
import sys
from astropy.io import fits
import matplotlib.pylab as plt

# Get an image of the traces
a = fits.open('/genesis/jwst/userland-soss/loic_review/stack_256_ng3_DMS.fits')
im = a[0].data
#im = np.zeros((256*2,2040*2))

x_o1, y_o1, x_o2, y_o2, x_o3, y_o3 = trace_centroids(im, verbose=False)

plt.imshow(im, origin='bottom')
plt.plot(x_o1, y_o1)
plt.plot(x_o2, y_o2)
plt.plot(x_o3, y_o3)
plt.ylim((0,2048))
plt.show()

sys.exit()

mask = make_mask_butorder3(im)

# Save the mask on disk
hdu = fits.PrimaryHDU()
hdu.data = mask
hdu.writeto('/genesis/jwst/userland-soss/loic_review/mask.fits', overwrite=True)

sys.exit()

if True:
    edge1, edge2 = [], []
    for i in range(2048):
        y1, y2 = edge_trigger(im[:,i]*mask_o1[:,i], triggerscale=5, verbose=False)
        print(i, y1, y2)
        edge1.append(y1)
        edge2.append(y2)
    plt.plot(edge1)
    plt.plot(edge2)
    plt.show()

sys.exit()


#a = plt.figure(figsize=(8,4))
print('vas-y!')
#plt.imshow(mask_o2*im, origin='bottom')
plt.imshow(im*mask_o1, origin='bottom')
plt.plot(x_o1, y_o1)
plt.plot(x_o2, y_o2)
plt.ylim((0,2048))
plt.show()



# Save the mask on disk
hdu = fits.PrimaryHDU()
hdu.data = mask_o1
hdu.writeto('/genesis/jwst/userland-soss/loic_review/mask.fits', overwrite=True)

# Save the masked CV3 stack on disk
hdu = fits.PrimaryHDU()
hdu.data = mask_o1 * im
hdu.writeto('/genesis/jwst/userland-soss/loic_review/maskedcv3.fits', overwrite=True)

