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
#       Note FitParam value = None dtype=FitParam
# =============================================================================
# Scale to apply to photometric errors [PARAM]
params.set(key='ERROR_SCALE', value=None, source=__NAME__,
           desc='Scale to apply to photometric errors',
           dtype=FitParam, path='global_params.error_scale',
           label='DSC')

# GP Kernel Amplitude (default is Matern 3/2) [PARAM]
params.set(key='AMPLITUDE_SCALE', value=None, source=__NAME__,
           desc='GP Kernel Amplitude (default is Matern 3/2)',
           dtype=FitParam, path='global_params.amplitude_scale',
           label='ASC')

# GP length scale (default is Matern 3/2) [PARAM]
params.set(key='LENGTH_SCALE', value=None, source=__NAME__,
           desc='GP Kernel Amplitude (default is Matern 3/2)',
           dtype=FitParam, path='global_params.amplitude_scale',
           label='LSC')

# =============================================================================
# Define misc parameters
#       Note FitParam value = None dtype=FitParam
# =============================================================================
# Parameters for TTVs ntt = number of transit times
params.set(key='NTT', value=0, source=__NAME__,
           desc='Parameters for TTVs ntt = number of transit times',
           dtype=int, path='global_params.ntt')

# Observed centre of transit times
params.set(key='T_OBS', value=0.0, source=__NAME__,
           desc='Observed centre of transit times',
           dtype=float, path='global_params.tobs')

# O-C values for each transit
params.set(key='OMC', value=0.0, source=__NAME__,
           desc='O-C values for each transit',
           dtype=float, path='global_params.omc')

# =============================================================================
# Define star parameters
#       Note FitParam value = None dtype=FitParam
# =============================================================================
# Mean stellar density [units?] [PARAM]
params.set(key='RHO_STAR', value=None, source=__NAME__,
           desc='Mean stellar density',
           dtype=FitParam, path='star.rhostar', label='p')

# Limb-darkening param 1. Set ld1=ld2=0 and ld3=q1 and ld4=q2 [PARAM]
params.set(key='LD1', value=None, source=__NAME__,
           desc='Limb-darkening param 1 (c1)',
           dtype=FitParam, path='star.ld1', label='c1')

# Limb-darkening param 2. Set ld1=ld2=0 and ld3=q1 and ld4=q2 [PARAM]
params.set(key='LD2', value=None, source=__NAME__,
           desc='Limb-darkening param 2 (c2)',
           dtype=FitParam, path='star.ld2', label='c2')

# Limb-darkening param 3. Set ld1=ld2=0 and ld3=q1 and ld4=q2 [PARAM]
params.set(key='LD3', value=None, source=__NAME__,
           desc='Limb-darkening param 3 (q1)',
           dtype=FitParam, path='star.ld3', label='q1')

# Limb-darkening param 4. Set ld1=ld2=0 and ld3=q1 and ld4=q2 [PARAM]
params.set(key='LD4', value=None, source=__NAME__,
           desc='Limb-darkening param 4 (q2)',
           dtype=FitParam, path='star.ld4', label='q2')

# Stellar dilution 0=none, 0.99 means 99% of light from other source [PARAM]
params.set(key='DILUTION', value=None, source=__NAME__,
           desc='Stellar dilution 0=none, 0.99 means 99% of light from '
                'other source',
           dtype=FitParam, path='star.dilution', label='DIL')

# Out of transit baseline (set in code) [PARAM]
params.set(key='ZEROPOINT', value=None, source=__NAME__,
           desc='Out of transit baseline (set in code)',
           dtype=FitParam, path='star.dilution', label='ZPT')

# =============================================================================
# Define planet parameters
#       Note FitParam value = None dtype=FitParam
#       Note planet path must have an {N} in - one for each planet
# =============================================================================
# Center of transit time [PARAM]
# TODO: comment units
params.set(key='T0', value=None, source=__NAME__,
           desc='Center of transit time',
           dtype=FitParam, path='planet{N}.t0', label='EP')

# Orbital period [PARAM]
# TODO: comment units
params.set(key='PERIOD', value=None, source=__NAME__,
           desc='Orbital period',
           dtype=FitParam, path='planet{N}.period', label='PE')

# Impact parameter [PARAM]
params.set(key='B', value=None, source=__NAME__,
           desc='Impact parameter',
           dtype=FitParam, path='planet{N}.b', label='BB')

# Scale planet radius [PARAM]
params.set(key='RPRS', value=None, source=__NAME__,
           desc='Scale planet radius',
           dtype=FitParam, path='planet{N}.rprs', label='RD')

# sqrt(e)cos(w) [PARAM]
params.set(key='SQRT_E_COSW', value=None, source=__NAME__,
           desc='sqrt(e)cos(w)',
           dtype=FitParam, path='planet{N}.sqrt_e_cosw', label='EC')

# sqrt(e)sin(w) [PARAM]
params.set(key='SQRT_E_SINW', value=None, source=__NAME__,
           desc='sqrt(e)sin(w)',
           dtype=FitParam, path='planet{N}.sqrt_e_sinw', label='ES')

# Secondary eclipse depth (ppm) [PARAM]
params.set(key='ECLIPSE_DEPTH', value=None, source=__NAME__,
           desc='Secondary eclipse depth (ppm)',
           dtype=FitParam, path='planet{N}.eclipse_depth', label='TED')

# Amplitude of ellipsoidal variations (ppm) [PARAM]
params.set(key='ELLIPSOIDAL', value=None, source=__NAME__,
           desc='Amplitude of ellipsoidal variations (ppm)',
           dtype=FitParam, path='planet{N}.ellipsoidal', label='ELL')

# Amplitude of reflected/emission phase curve (ppm) - Lambertian [PARAM]
params.set(key='PHASECURVE', value=None, source=__NAME__,
           desc='Amplitude of reflected/emission phase curve (ppm)',
           dtype=FitParam, path='planet{N}.phasecurve', label='ALB')


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
