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




def get_edge_centroids(stack, header=None, badpix=None, mask=None, verbose=False,
                       return_what='edgemean_param', polynomial_order=2,
                       triggerscale=5):
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
        y1, y2 = edge_trigger(stack[:,i]*mask[:,i], triggerscale=triggerscale, verbose=verbose)
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
        plt.plot(np.abs(edge1-edge2))
        plt.legend()
        plt.show()

    # Trace width
    tracewidth = np.nanmedian(np.abs(edge1 - edge2))


    if return_what == 'edgemean_param' : return param_both
    if return_what == 'rededge_param' : return param_red
    if return_what == 'blueedge_param' : return param_blue
    if return_what == 'edgemean_xy' : return x_both, y_both
    if return_what == 'rededge_xy' : return x_red, y_red
    if return_what == 'blueedge_xy' : return x_blue, y_blue
    if return_what == 'tracewidth' : return tracewidth



def polyfit_sigmaclip(x, y, order, sigma=4):

    polyorder = np.copy(order)
    ind = np.where(y == y)
    for n in range(5):
        #if n == 4: polyorder = order + 1
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



# Here is how to call the code
import numpy as np
import sys
from astropy.io import fits
import matplotlib.pylab as plt

# Read in the 2D trace reference file (each extension has an isolated
# trace). When it exists, make sure to pass the deader as well in the call
# to the get_edge_centroids function. For now, we are missing that file so
# use the CV3 stack instead.
#
# im = read.the.2Dtrace.ref.file
# hdr = is.its.header
a = fits.open('/genesis/jwst/userland-soss/loic_review/stack_256_ng3_DMS.fits')
im = a[0].data

# Triggers on the rising and declining edges of the trace. Make a polynomial
# fit to those and return the x,y fit. Alternatively, the parameters of that
# fit coudl be returned by using return_what='edgemean_param'.
x_o1, y_o1 = get_edge_centroids(im, header=hdr, return_what='edgemean_xy',
                                polynomial_order=10, verbose=False)
