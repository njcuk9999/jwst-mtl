#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:35:32 2020

@author: albert
"""

from astropy.io import fits

# Bit that reads our current best estimate of the SOSS total throughput.
a, hdr = fits.getdata('/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_Throughput_STScI.fits',ext=1,header=True)

wave_micron = a[0]['LAMBDA'] / 1000.
th_order1 = a[0]['SOSS_ORDER1']
th_order2 = a[0]['SOSS_ORDER2']
th_order3 = a[0]['SOSS_ORDER3']

print(wave_micron)
print(th_order1)


# Bit that reads the monochromatic tilt measured at CV3

# Initialize arrays read from reference file.
w = [] # wavelength in the table
o1, o2, o3 = [], [], [] # tilts in the table for the 3 orders
# Read in the reference tilt file
f = open('/genesis/jwst/github/jwst-mtl/SOSS/extract/Generate_ref_files/SOSS_wavelength_dependent_tilt.txt', 'r')
for line in f:
    # Ignore comments (lines starting with #
    if line[0] != '#':
        columns = line.split()
        w.append(float(columns[0]))
        o1.append(float(columns[1]))
        o2.append(float(columns[2]))
        o3.append(float(columns[3]))
# Make sure to convert from lists to numpy arrays
w = np.array(w)
o1 = np.array(o1)
o2 = np.array(o2)
o3 = np.array(o3)

# Interpolate at the same wavelength as throughput (every nm)
tilt_degrees_o1 = np.interp(wave_micron, w, o1)
tilt_degrees_o2 = np.interp(wave_micron, w, o2)
tilt_degrees_o3 = np.interp(wave_micron, w, o3)
# TODO: Should handle the values higher than 3 microns more explicitely here.


# Bit about the trace position from tracepol
import tracepol as tp

# Get the trace parameters, function found in tracepol imported above
trace_file = '/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv'
tracepars = tp.get_tracepars(trace_file)
# Have to do it for 3 subarrays (e.g. SUBSTRIP96 is shifted by 11 pixels).
# Note that the x axis for all these are the same. Only the y axis change with
# subarray.
# SUBSTRIP256
y_256_1, x_256_1, mask = tp.wavelength_to_pix(wave_micron, tracepars, m = 1,
                                  frame = 'dms', subarray = 'SUBSTRIP256')
y_256_2, x_256_2, mask = tp.wavelength_to_pix(wave_micron, tracepars, m = 2,
                                  frame = 'dms', subarray = 'SUBSTRIP256')
y_256_3, x_256_3, mask = tp.wavelength_to_pix(wave_micron, tracepars, m = 3,
                                  frame = 'dms', subarray = 'SUBSTRIP256')
# SUBSTRIP96
y_96_1, x_96_1, mask = tp.wavelength_to_pix(wave_micron, tracepars, m = 1,
                                  frame = 'dms', subarray = 'SUBSTRIP96')
y_96_2, x_96_2, mask = tp.wavelength_to_pix(wave_micron, tracepars, m = 2,
                                  frame = 'dms', subarray = 'SUBSTRIP96')
y_96_3, x_96_3, mask = tp.wavelength_to_pix(wave_micron, tracepars, m = 3,
                                  frame = 'dms', subarray = 'SUBSTRIP96')
# Full Frame
y_FF_1, x_FF_1, mask = tp.wavelength_to_pix(wave_micron, tracepars, m = 1,
                                  frame = 'dms', subarray = 'FF')
y_FF_2, x_FF_2, mask = tp.wavelength_to_pix(wave_micron, tracepars, m = 2,
                                  frame = 'dms', subarray = 'FF')
y_FF_3, x_FF_3, mask = tp.wavelength_to_pix(wave_micron, tracepars, m = 3,
                                  frame = 'dms', subarray = 'FF')



# TODO: Finally, you need to save that in a single file. The format is TBD.

# wave_micron
# th_order1
# th_order2
# th_order3
# tilt_degrees_o1
# tilt_degrees_o2
# tilt_degrees_o3
# x_256_1
# y_96_1
# y_256_1
# y_FF_1
# x_256_2
# y_96_2
# y_256_2
# y_FF_2
# x_256_3
# y_96_3
# y_256_3
# y_FF_3


