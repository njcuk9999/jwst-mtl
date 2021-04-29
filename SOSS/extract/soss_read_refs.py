#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np

from astropy.io import fits


# Where the ref files are to be found depends who runs it.
PATH = '/Users/michaelradica/Documents/School/Ph.D./Research/SOSS/Extraction/Input_Files/SOSS_Ref_Files/'
PATH = '/genesis/jwst/jwst-ref-soss/dms/'


class RefTraceTable:

    def __init__(self, filenames=None):

        if filenames is None:
            filenames = {'FULL': 'SOSS_ref_trace_table_FULL.fits.gz',
                         'SUBSTRIP96': 'SOSS_ref_trace_table_SUBSTRIP96.fits.gz',
                         'SUBSTRIP256': 'SOSS_ref_trace_table_SUBSTRIP256.fits.gz'}

        self.filenames = filenames

        return

    def __call__(self, column, wavelengths=None, subarray='SUBSTRIP256', order=1):

        if subarray not in ['FULL', 'SUBSTRIP96', 'SUBSTRIP256']:
            raise ValueError('Unknown subarray: {}'.format(subarray))

        filename = os.path.join(PATH, self.filenames[subarray])
        trace_table, header, = fits.getdata(filename, header=True, extname='ORDER', extver=order)

        if wavelengths is None:
            wavelengths = trace_table['WAVELENGTH']
            values = trace_table[column]
        else:
            values = np.interp(wavelengths, trace_table['WAVELENGTH'], trace_table[column], left=np.nan, right=np.nan)

        return wavelengths, values


class Ref2dProfile:

    def __init__(self, filenames=None):

        if filenames is None:
            filenames = {'FULL': 'SOSS_ref_2D_profile_FULL.fits.gz',
                         'SUBSTRIP96': 'SOSS_ref_2D_profile_SUBSTRIP96.fits.gz',
                         'SUBSTRIP256': 'SOSS_ref_2D_profile_SUBSTRIP256.fits.gz'}

        self.filenames = filenames

        return

    @staticmethod
    def _binning(array, shape, ovs):

        # The 2D profile is normalized so that columns sum to 1.
        # To preserve this columns must be averaged and rows summed.
        binned_array = array.reshape(shape[0], ovs, shape[1], ovs)
        binned_array = binned_array.mean(-1).sum(1)

        return binned_array

    def __call__(self, order=1, subarray='SUBSTRIP256', offset=None,
                 native=True, only_prof=True):

        if offset is None:
            offset = [0, 0]  # x, y

        # TODO hardcoded?
        if subarray == 'FULL':
            shape = [2048, 2048]  # row, col
        elif subarray == 'SUBSTRIP256':
            shape = [256, 2048]  # row, col
        elif subarray == 'SUBSTRIP96':
            shape = [96, 2048]  # row, col
        else:
            raise ValueError('Unknown subarray: {}'.format(subarray))

        # Read the reference file.
        filename = os.path.join(PATH, self.filenames[subarray])
        ref_2d_profile, header, = fits.getdata(filename, header=True, extname='ORDER', extver=order)

        # Read necessary header information.
        ovs = header['OVERSAMP']
        pad = header['PADDING']

        if native:

            # Select the relevant area.
            minrow = ovs*pad + int(ovs*offset[1])
            maxrow = minrow + ovs*shape[0]
            mincol = ovs*pad + int(ovs*offset[0])
            maxcol = mincol + ovs*shape[1]
            ref_2d_profile = ref_2d_profile[minrow:maxrow, mincol:maxcol]

            # Bin down to native resolution.
            ref_2d_profile = self._binning(ref_2d_profile, shape, ovs)

        # Return amount of oversampling and padding if requested.
        if only_prof is True:
            return ref_2d_profile
        else:
            return ref_2d_profile, ovs, pad


class Ref2dWave(Ref2dProfile):

    def __init__(self, filenames=None):

        if filenames is None:
            filenames = {'FULL': 'SOSS_ref_2D_wave_FULL.fits.gz',
                         'SUBSTRIP96': 'SOSS_ref_2D_wave_SUBSTRIP96.fits.gz',
                         'SUBSTRIP256': 'SOSS_ref_2D_wave_SUBSTRIP256.fits.gz'}

        Ref2dProfile.__init__(self, filenames=filenames)

        return

    @staticmethod
    def _binning(array, shape, ovs):

        binned_array = array.reshape(shape[0], ovs, shape[1], ovs)
        binned_array = binned_array.mean(-1).mean(1)

        return binned_array


class RefKernels:

    def __init__(self, filename=None):

        if filename is None:
            filename = 'SOSS_ref_spectral_kernel.fits.gz'

        self.filename = filename

        return

    def __call__(self):

        filename = os.path.join(PATH, self.filename)
        wavelengths, header = fits.getdata(filename, header=True, extname='WAVELENGTHS')
        kernels, header = fits.getdata(filename, header=True, extname='KERNELS')

        return wavelengths, kernels


def main():

    return


if __name__ == '__main__':
    main()
