#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:09:15 2020

@author: albert
"""

# This is the piece of code that should be embedded in the Specgen Prototype.ipynb
# The steps are the following:
# 1) Use the bin_starmodel_wv and bin_starmodel_flux to calibrate the flux
#    based on a a filter magnitude
# 2) Integrate the calibrated flux within the SOSS Order 1 wavelength span
#    Use Geert Jan's p2w to determine the wavelength coverage.

# import numpy. When you embed in specgen, remove that line
import numpy as np
import tracepol as tr

# Define those arrays. When you embed in specgen, remove those 2 lines
bin_starmodel_wv = np.linspace(0.5,6.0,1000)*10000.0
bin_starmodel_flux = np.ones(1000)

# My renormalization package
import synthesizeMagnitude as smag

filtername = 'J'
mag = 8.5
# Choice of filter names can be found here: 
# synthesizeMagnitude.py --> readFilter
# When you embed in specgen, replace paths by relative path in the github hierarchy.
pathvega = '/Users/albert/NIRISS/SOSSpipeline/sandbox/'
pathfilter = '/Users/albert/NIRISS/SOSSpipeline/sandbox/filterSVO/'

# Do the synthetic magnitude call on the model spectrum
# waves should be in microns
# Flux should be Flambda, not Fnu (Make sure that is the case)
syntmag = smag.syntMag(bin_starmodel_wv/10000,bin_starmodel_flux,filtername,
                    path_filter_transmission=pathfilter,
                    path_vega_spectrum=pathvega)
print('Synthetic magnitude of input spectrum: {:6.2f}'.format(syntmag))

# The nromalization scale is the difference between synthetic mag and 
# requested mag
bin_starmodel_flux_normalized = bin_starmodel_flux * 10**(-0.4*(mag-syntmag))

# Check that thsi worked by re synthesizing the magnitude
syntmag_check = smag.syntMag(bin_starmodel_wv/10000,bin_starmodel_flux_normalized,
                       filtername, path_filter_transmission=pathfilter,
                       path_vega_spectrum=pathvega)
print('Synthetic magnitude of normalized spectrum: {:6.2f}'.format(syntmag_check))

# Now that our spectrum is normalized in flux (W/m2/um), 
# compute what the flux count (in electrons) should be for the order 1.

# Get the trace parameters, function found in tracepol imported above
tracepars = tr.get_tracepars()
# Get wavelength (in um) of first and last pixel of the Order 1 trace
pixel1lba = tr.x2wavelength(1, tracepars, m=1)
pixel2048lba = tr.x2wavelength(2048, tracepars, m=1)
# The number of photons per second integrated over that range is
# (Flambda: J/sec/m2/um requires dividing by photon energy to get counts)
# Ephoton = h*c/lambda
#
# Model spectrum wavelength step size (in microns)
lambdastep = np.abs(bin_starmodel_wv[1] - bin_starmodel_wv[0])/10000
#
# Spectrum converted to photonumber of photons
hc = 6.62607015e-34 * 299792458.0 # J * m
photon_per_sec_per_um = bin_starmodel_flux_normalized * bin_starmodel_wv * 1e-10 / hc
order1range = (bin_starmodel_wv/10000 >= pixel1lba) & (bin_starmodel_wv/10000 <= pixel2048lba)
counts_per_sec = np.sum(photon_per_sec_per_um[order1range]*lambdastep)

print('Electrons per second in SOSS Order 1 = {:}'.format(counts_per_sec))

