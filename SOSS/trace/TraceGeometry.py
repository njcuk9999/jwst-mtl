import numpy as np

import sys

#from astropy.io import fits

#import matplotlib.pylab as plt


class Orders:
    def __init__(self):
        self.micron = None
        self.x = None
        self.y = None
        # A mask that defines the trace with some width
        self.aperture = None
        # A mask that masks out all but this order pixels to ease centroiding
        self.mask = None

class TraceGeometry:
    def __init__(self, image, header=None, verbose=False, badpix=None, order1_apex_y=None):
        '''
        image : array of floats (2D)
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
        '''
        self.image = image
        self.header = header
        self.verbose = verbose
        self.badpix = badpix
        # order1_apex_y is the y position roughly going thru the apex of the order 1 trace.
        # Usefull guess when dealing with trace at unconventional position.
        self.yapex_order1 = order1_apex_y
        self.dimx = None
        self.dimy = None
        self.order1 = Orders()
        self.order2 = Orders()
        self.order3 = Orders()

    def __str__(self):
        return 'TraceGeometry class'

    def __repr__(self):
        return self.__str__


    # Initialize the image dimensions and build a boolean mask defining working pixels
    def get_dimensions(self, verbose=False):
        ''' Determine the size of the stack array. Will be called by get_centerofmass_centroids
        and make_trace_mask.
        Parameters
        ----------
        image : array of floats (2D)
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
        self.dimy, self.dimx = np.shape(self.image)

        # Determine what is the input stack based either on its dimensions or on
        # the header if passed. Construct a mask of working pixels in case the
        # stack contains reference pixels.
        if self.header is None:
            # No header passed - Assume that the stack is a valid SOSS subarray or FF, i.e.
            # 2048x256 or 2040x252 (working pixels) or multiple of if oversampled
            # 2048x96 or 2040x96 (working pixels) or multiple of
            # 2048x2048 or 2040x2040 (working pixels) or multiple of
            if (self.dimx % 2048) == 0:
                # stack is a multiple of native pixels.
                self.xnative = 2048
                # The x-axis oversampling is:
                self.xos = int(self.dimx / 2048)
            elif (self.dimx % 2040) == 0:
                # stack is a multiple of native *** working *** pixels.
                self.xnative = 2040
                # The y-axis oversampling is:
                self.xos = int(self.dimx / 2040)
            else:
                # stack x dimension has unrecognized size.
                print('Stack X dimension has unrecognized size of {:}. Accepts 2048, 2040 or multiple of.'.format(self.dimx))
                sys.exit()
            # Check if the Y axis is consistent with the X axis.
            acceptable_ydim = [96, 256, 252, 2040, 2048]
            yaxis_consistent = False
            for accdim in acceptable_ydim:
                if self.dimy / (accdim * self.xos) == 1:
                    # Found the acceptable dimension
                    self.yos = np.copy(self.xos)
                    self.ynative = np.copy(accdim)
                    yaxis_consistent = True
            if yaxis_consistent == False:
                # stack y dimension is inconsistent with the x dimension.
                print(
                    'Stack Y dimension ({:}) is inconsistent with X dimension ({:}) for acceptable SOSS arrays'.format(
                        self.dimy, self.dimx))
                sys.exit()
            # Construct a boolean mask (true or false) of working pixels
            self.working_pixel_bool = np.full((self.dimy, self.dimx), True)

            # For dimensions where reference pixels would have been included in the stack,
            # mask those reference pixels out.
            # Sizes 96, 252 and 2040 should not contain any reference pixel.
            if self.xnative == 2048:
                # Mask out the left and right columns of reference pixels
                self.working_pixel_bool[:, 0:self.xos * 4] = False
                self.working_pixel_bool[:, -self.xos * 4:] = False
            if self.ynative == 2048:
                # Mask out the top and bottom rows of reference pixels
                self.working_pixel_bool[0:self.yos * 4, :] = False
                self.working_pixel_bool[-self.yos * 4:, :] = False
            if self.ynative == 256:
                # Mask the top rows of reference pixels
                self.working_pixel_bool[-self.yos * 4:, :] = False

            # Initialize padding to zero in this case because it is not a 2D Trace ref file
            self.padding = int(0)

        else:
            # header was passed
            # Read in the relevant keywords
            self.xos, self.yos = int(self.header['OVERSAMP']), int(self.header['OVERSAMP'])
            self.padding = int(self.header['PADDING'])
            # The 2D Trace profile is for FULL FRAME so 2048x2048
            self.xnative, self.ynative = int(2048), int(2048)
            # Check that the stack respects its intended format
            if self.dimx != ((self.xnative + 2 * self.padding) * self.xos):
                # Problem
                print('The header passed is inconsistent with the X dimension of the stack.')
                sys.exit()
            if self.dimy != ((self.ynative + 2 * self.padding) * self.yos):
                # Problem
                print('The header passed is inconsistent with the Y dimension of the stack.')
                sys.exit()
            # Construct a mask of working pixels. The 2D Trace REFERENCE file does
            # not contain any reference pixel. So all are True.
            self.working_pixel_bool = np.full((self.dimy, self.dimx), True)

        # For debugging purposes...
        if self.verbose == True or verbose == True:
            print('dimx={:}, dimy={:}, xos={:}, yos={:}, xnative={:}, ynative={:}'.format(
                self.dimx, self.dimy, self.xos, self.yos, self.xnative, self.ynative))

