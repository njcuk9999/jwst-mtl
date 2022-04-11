#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-11

@author: cook
"""
from soss_tfit.core import base_classes


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.parameters.py'
# Get the parameter dictionary class
ParamDict = base_classes.ParamDict
# Get the Fit Parameter class
FitParam = base_classes.FitParam
# set up parameter dictionary
params = base_classes.ParamDict()


# =============================================================================
# Define default global parameters
# =============================================================================
# Number of planets (use for N planets) [FLOAT]
params.set(key='NPLANETS', value=1, source=__NAME__,
           desc='Number of planets to use',
           dtype=int, path='global_params.nplanets')

# input extracted spctrum - aboslute path (must be set)
params.set(key='INSPECTRUM', value=None, source=__NAME__,
           desc='input extracted spctrum - aboslute path (must be set)',
           dtype=str, not_none=True, path='global_params.inspectrum')

# output directory for results (created if doesn't exist)
params.set(key='OUTDIR', value='./output', source=__NAME__,
           desc='output directory for results (created if doesnt exist)',
           dtype=str, path='global_params.outdir')

# model file path - absolute path (set None to not plot model)
params.set(key='MODELPATH', value=None, source=__NAME__,
           desc='model file path - absolute path (set None to not plot model)',
           dtype=str, path='global_params.modelpath')

# =============================================================================
# Define default fitting parameters
# =============================================================================
# Scale to apply to photometric errors [PARAM]
params.set(key='ERROR_SCALE', value=None, source=__NAME__,
           desc='Scale to apply to photometric errors [PARAM]',
           dtype=FitParam)

# GP Kernel Amplitude (default is Matern 3/2) [PARAM]

# GP length scale (default is Matern 3/2) [PARAM]


# =============================================================================
# Define misc parameters
# =============================================================================
# Parameters for TTVs ntt = number of transit times

# Observed centre of transit times

# O-C values for each transit


# =============================================================================
# Define star parameters
# =============================================================================
# Mean stellar density [units?] [PARAM]

# Limb-darkening param 1. Set ld1=ld2=0 and ld3=q1 and ld4=q2 [PARAM]

# Limb-darkening param 2. Set ld1=ld2=0 and ld3=q1 and ld4=q2 [PARAM]

# Limb-darkening param 3. Set ld1=ld2=0 and ld3=q1 and ld4=q2 [PARAM]

# Limb-darkening param 4. Set ld1=ld2=0 and ld3=q1 and ld4=q2 [PARAM]

# Stellar dilution 0=none, 0.99 means 99% of light from other source [PARAM]

# Out of transit baseline (set in code) [PARAM]

# =============================================================================
# Define planet parameters
# =============================================================================
# Center of transit time [PARAM]

# Orbital period [PARAM]

# Impact parameter [PARAM]

# Scale planet radius [PARAM]

# sqrt(e)cos(w) [PARAM]

# sqrt(e)cos(w) [PARAM]

# Secondard eclipse depth (ppm) [PARAM]

# Amplitude of ellipsoidal variations (ppm) [PARAM]

# Amplitude of reflected/emission phase curve (ppm) - Lambertian [PARAM]


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
