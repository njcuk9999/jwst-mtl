#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from astropy.io import fits


class RefTraceTable:

    def __init__(self, filename):

        self.filename = filename

        return

    def __call__(self, column, wavelengths=None, subarray='SUBSTRIP256', order=1):

        trace_table, header, = fits.getdata(self.filename, header=True, ext=order)

        if subarray == 'FULL':
            pass
        elif subarray == 'SUBSTRIP256':
            trace_table['Y'] = trace_table['Y'] + header['DYSUB96']
        elif subarray == 'SUBSTRIP96':
            trace_table['Y'] = trace_table['Y'] + header['DYSUB256']
        else:
            raise ValueError('Unknown subarray: {}'.format(subarray))

        if wavelengths is None:
            wavelengths = trace_table['WAVELENGTH']
            values = trace_table[column]
        else:
            values = np.interp(wavelengths, trace_table['WAVELENGTH'], trace_table[column], left=np.nan, right=np.nan)

        return wavelengths, values


class Ref2dProfile:

    def __init__(self, filename):

        self.filename = filename

        return

    @staticmethod
    def _binning(array, shape, os):

        # The 2D profile is normalized so that columns sum to 1.
        # To preserve this columns must be averaged and rows summed.
        binned_array = array.reshape(shape[0], os, shape[1], os)
        binned_array = binned_array.mean(-1).sum(1)

        return binned_array

    def __call__(self, order=1, subarray='SUBSTRIP256', offset=None,
                 native=True, only_prof=True):

        if offset is None:
            offset = [0, 0]  # x, y

        # TODO hardcoded?
        if subarray == 'FULL':
            shape = [2048, 2048]  # row, col
            origin = [0, 0]  # row, col
        elif subarray == 'SUBSTRIP256':
            shape = [256, 2048]  # row, col
            origin = [1792, 0]  # row, col
        elif subarray == 'SUBSTRIP96':
            shape = [96, 2048]  # row, col
            origin = [1802, 0]  # row, col
        else:
            raise ValueError('Unknown subarray: {}'.format(subarray))

        # Read the reference file.
        ref_2d_profile, header, = fits.getdata(self.filename, header=True, ext=order)

        # Read necessary header information.
        os = header['OVERSAMP']
        pad = header['PADDING']

        # Select the relevant area.
        if native:
            minrow = os*(pad + origin[0]) + int(os*offset[1])
            maxrow = minrow + os*shape[0]
            mincol = os*(pad + origin[1]) + int(os*offset[0])
            maxcol = mincol + os*shape[1]
        else:
            minrow = os*origin[0] + int(os*offset[1])
            maxrow = minrow + os*(shape[0] + 2*pad)
            mincol = os*origin[1] + int(os*offset[0])
            maxcol = mincol + os*(shape[1] + 2*pad)

        ref_2d_profile = ref_2d_profile[minrow:maxrow, mincol:maxcol]

        # Bin down to native resolution.
        if native:
            ref_2d_profile = self._binning(ref_2d_profile, shape, os)

        # Return amount of oversampling and padding if requested.
        if only_prof is True:
            return ref_2d_profile
        else:
            return ref_2d_profile, os, pad


class Ref2dWave(Ref2dProfile):

    def __init__(self, filename):

        Ref2dProfile.__init__(self, filename)

        return

    @staticmethod
    def _binning(array, shape, os):

        binned_array = array.reshape(shape[0], os, shape[1], os)
        binned_array = binned_array.mean(-1).mean(1)

        return binned_array


class RefKernels:

    def __init__(self, filename):

        self.filename = filename

        return

    def __call__(self):

        kernels, header = fits.getdata(self.filename, header=True)
        wavelengths = np.linspace(header['WAVE0'], header['WAVEN'], header['NWAVE'])

        return wavelengths, kernels


def main():

    return


if __name__ == '__main__':
    main()
