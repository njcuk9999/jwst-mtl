#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-06

@author: cook
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import transitfit5 as tf

from tqdm import tqdm
from jwst import datamodels

# =============================================================================
# Define variables
# =============================================================================
data_dir = '/data/jwst-soss/data/jwst-mtl-user/wasp52b/'
extracted_tso_file = 'substrip256_extract1d_extract1dstep.fits'
model_file = ('WASP_52_b_HR_Metallicity100_CtoO0.54_pQuench1e-99_TpNonGray'
              'Tint75.0f0.25A0.1_pCloud100000.0mbar_Spectrum_FullRes.csv')

# =============================================================================
# Define classes
# =============================================================================
class PhotClass:
    def __init__(self):
        # Initialize arrays

        # Wavelength of observation (should be a single number)
        self.wavelength = np.array([])
        # Time-stamps array -- mid-exposure time (days)
        self.time = np.array([])
        # Observed flux array
        self.flux = np.array([])
        # Error in flux array
        self.ferr = np.array([])
        # Integration time (seconds)
        self.itime = np.array([])


class SptransitModelClass(object):
    def __init__(self):
        # Wavelength of each observation
        self.wavelength = np.array([])
        # Mean stellar density
        self.rhostar = []
        # Limb-darkening. Set ld1=ld2=0 and ld3=q1 and ld4=q2
        self.ld1 = []
        self.ld2 = []
        self.ld3 = []
        self.ld4 = []
        # Stellar dilution 0=none, 0.99 means 99% of light from other source
        self.dilution = []
        # Out of transit baseline
        self.zeropoint = []
        # Number of planets
        self.nplanet = 1
        # Center of transit time
        self.t0 = []
        # Orbital period
        self.period = []
        # Impact parameter
        self.b = []
        # Scale planet radius
        self.rprs = []
        # sqrt(e)cos(w)
        self.sqrt_e_cosw = []
        # sqrt(e)cos(w)
        self.sqrt_e_sinw = []
        # Secondard eclipse depth (ppm)
        self.eclipse_depth = []
        # Amplitude of ellipsoidal variations (ppm)
        self.ellipsoidal = []
        # Amplitude of reflected/emission phase curve (ppm) - Lambertian
        self.phasecurve = []

        self.error_scale = []  # Scale to apply to photometric errors
        self.amplitude_scale = []  # GP Kernel Amplitude (default is Matern 3/2)
        self.length_scale = []  # GP length scale (default is Matern 3/2)

        self.ntt = []  # Parameters for TTVs ntt = number of transit times
        self.tobs = []  # Observed centre of transit times
        self.omc = []  # O-C values for eachtransit


class SptransitModelParameters(SptransitModelClass):
    def __init__(self, num):
        SptransitModelClass.__init__(self)
        nwav = len(num[0])

        # Wavelength for each dataset
        zpt = []
        for p in num[0]:
            self.wavelength = np.append(self.wavelength, p.wavelength)
            zpt.append(np.median(p.flux))
        zpt = np.array(zpt)

        # Star Parameters
        self.rhostar = [np.ones(1), 'bolometric', 'fit',
                        np.array([1.0e-4, 1000])]
        self.ld1 = [np.zeros(nwav), 'chromatic', 'fixed',
                    np.array([-1, 1])]
        self.ld2 = [np.zeros(nwav), 'chromatic', 'fixed',
                    np.array([-1, 1])]
        self.ld3 = [np.ones(nwav) * 0.5, 'chromatic', 'fit',
                    np.array([0, 1])]
        self.ld4 = [np.ones(nwav) * 0.5, 'chromatic', 'fit',
                    np.array([0, 1])]
        self.dilution = [np.zeros(nwav), 'chromatic', 'fixed',
                         np.array([0, 1])]
        self.zeropoint = [zpt, 'chromatic', 'fit',
                          np.array([0, 1.0e9])]

        # Planet Parameters
        nplanet = num[1]
        self.nplanet = num[1] * 1
        for i in range(nplanet):
            self.t0.append([np.ones(1), 'bolometric', 'fit',
                            np.array([0, 2])])
            self.period.append([np.ones(1), 'bolometric', 'fit',
                                np.array([0, 2])])
            self.b.append([np.ones(1) * 0.5, 'bolometric', 'fit',
                           np.array([0, 2])])
            self.rprs.append([np.ones(nwav) * 0.01, 'chromatic', 'fit',
                              np.array([0, 1])])
            self.sqrt_e_cosw.append([np.zeros(1), 'bolometric', 'fixed',
                                     np.array([-1, 1])])
            self.sqrt_e_sinw.append([np.zeros(1), 'bolometric', 'fixed',
                                     np.array([-1, 1])])
            self.eclipse_depth.append([np.zeros(nwav), 'chromatic', 'fixed',
                                       np.array([0, 1.0e4])])
            self.ellipsoidal.append([np.zeros(nwav), 'chromatic', 'fixed',
                                     np.array([0, 1.0e4])])
            self.phasecurve.append([np.zeros(nwav), 'chromatic', 'fixed',
                                    np.array([0, 1.0e4])])

        # Error model
        self.error_scale = [np.ones(1) * 1.0, 'bolometric', 'fit',
                            np.array([0, 2])]
        self.amplitude_scale = [np.ones(nwav) * 1.0, 'chromatic', 'fixed',
                                np.array([0, 2000])]
        self.length_scale = [np.ones(nwav) * 1.0, 'chromatic', 'fixed',
                             np.array([0, 1])]

        self.ntt = 0
        self.tobs = 0
        self.omc = 0


# =============================================================================
# Define functions
# =============================================================================
def get_fitted_parameters(tpars):
    x = []

    if tpars.rhostar[2] == 'fit':
        for p in tpars.rhostar[0]:
            x.append(p)

    if tpars.ld1[2] == 'fit':
        for p in tpars.ld1[0]:
            x.append(p)

    if tpars.ld2[2] == 'fit':
        for p in tpars.ld2[0]:
            x.append(p)

    if tpars.ld3[2] == 'fit':
        for p in tpars.ld3[0]:
            x.append(p)

    if tpars.ld4[2] == 'fit':
        for p in tpars.ld4[0]:
            x.append(p)

    if tpars.dilution[2] == 'fit':
        for p in tpars.dilution[0]:
            x.append(p)

    if tpars.zeropoint[2] == 'fit':
        for p in tpars.zeropoint[0]:
            x.append(p)

    for i in range(tpars.nplanet):

        if tpars.t0[i][2] == 'fit':
            for p in tpars.t0[i][0]:
                x.append(p)

        if tpars.period[i][2] == 'fit':
            for p in tpars.period[i][0]:
                x.append(p)

        if tpars.b[i][2] == 'fit':
            for p in tpars.b[i][0]:
                x.append(p)

        if tpars.rprs[i][2] == 'fit':
            for p in tpars.rprs[i][0]:
                x.append(p)

        if tpars.sqrt_e_cosw[i][2] == 'fit':
            for p in tpars.sqrt_e_cosw[i][0]:
                x.append(p)

        if tpars.sqrt_e_sinw[i][2] == 'fit':
            for p in tpars.sqrt_e_sinw[i][0]:
                x.append(p)

        if tpars.eclipse_depth[i][2] == 'fit':
            for p in tpars.eclipse_depth[i][0]:
                x.append(p)

        if tpars.ellipsoidal[i][2] == 'fit':
            for p in tpars.ellipsoidal[i][0]:
                x.append(p)

        if tpars.phasecurve[i][2] == 'fit':
            for p in tpars.phasecurve[i][0]:
                x.append(p)

    if tpars.error_scale[2] == 'fit':
        for p in tpars.error_scale[0]:
            x.append(p)

    if tpars.amplitude_scale[2] == 'fit':
        for p in tpars.amplitude_scale[0]:
            x.append(p)

    if tpars.length_scale[2] == 'fit':
        for p in tpars.length_scale[0]:
            x.append(p)

    x = np.array(x)

    return x


def get_all_parameters(tpars, photospectra):
    nhp = 3  # Potential number of hyper-parameters

    npars = 8 + 10 * tpars.nplanet + nhp
    nwav = len(photospectra)

    sol = np.zeros([npars, nwav])

    if tpars.rhostar[1] == 'bolometric':
        sol[0][:] = np.ones(nwav) * tpars.rhostar[0][0]
    else:
        sol[0][:] = tpars.rhostar[0]

    if tpars.ld1[1] == 'bolometric':
        sol[1][:] = np.ones(nwav) * tpars.ld1[0][0]
    else:
        sol[1][:] = tpars.ld1[0]

    if tpars.ld2[1] == 'bolometric':
        sol[2][:] = np.ones(nwav) * tpars.ld2[0][0]
    else:
        sol[2][:] = tpars.ld2[0]

    if tpars.ld3[1] == 'bolometric':
        sol[3][:] = np.ones(nwav) * tpars.ld3[0][0]
    else:
        sol[3][:] = tpars.ld3[0]

    if tpars.ld4[1] == 'bolometric':
        sol[4][:] = np.ones(nwav) * tpars.ld4[0][0]
    else:
        sol[4][:] = tpars.ld4[0]

    if tpars.dilution[1] == 'bolometric':
        sol[5][:] = np.ones(nwav) * tpars.dilution[0][0]
    else:
        sol[5][:] = tpars.dilution[0]

    if tpars.zeropoint[1] == 'bolometric':
        sol[7][:] = np.ones(nwav) * tpars.zeropoint[0][0]
    else:
        sol[7][:] = tpars.zeropoint[0]

    for i in range(tpars.nplanet):
        nc = 10 * i

        if tpars.t0[i][1] == 'bolometric':
            sol[8 + nc][:] = np.ones(nwav) * tpars.t0[i][0][0]
        else:
            sol[8 + nc][:] = tpars.t0[i][0]

        if tpars.period[i][1] == 'bolometric':
            sol[9 + nc][:] = np.ones(nwav) * tpars.period[i][0][0]
        else:
            sol[9 + nc][:] = tpars.period[i][0]

        if tpars.b[i][1] == 'bolometric':
            sol[10 + nc][:] = np.ones(nwav) * tpars.b[i][0][0]
        else:
            sol[10 + nc][:] = tpars.b[i][0]

        if tpars.rprs[i][1] == 'bolometric':
            sol[11 + nc][:] = np.ones(nwav) * tpars.rprs[i][0][0]
        else:
            sol[11 + nc][:] = tpars.rprs[i][0]

        if tpars.sqrt_e_cosw[i][1] == 'bolometric':
            sol[12 + nc][:] = np.ones(nwav) * tpars.sqrt_e_cosw[i][0][0]
        else:
            sol[12 + nc][:] = tpars.sqrt_e_cosw[i][0]

        if tpars.sqrt_e_sinw[i][1] == 'bolometric':
            sol[13 + nc][:] = np.ones(nwav) * tpars.sqrt_e_sinw[i][0][0]
        else:
            sol[13 + nc][:] = tpars.sqrt_e_sinw[i][0]

        if tpars.eclipse_depth[i][1] == 'bolometric':
            sol[15 + nc][:] = np.ones(nwav) * tpars.eclipse_depth[i][0][0]
        else:
            sol[15 + nc][:] = tpars.eclipse_depth[i][0]

        if tpars.ellipsoidal[i][1] == 'bolometric':
            sol[16 + nc][:] = np.ones(nwav) * tpars.ellipsoidal[i][0][0]
        else:
            sol[16 + nc][:] = tpars.ellipsoidal[i][0]

        if tpars.phasecurve[i][1] == 'bolometric':
            sol[17 + nc][:] = np.ones(nwav) * tpars.phasecurve[i][0][0]
        else:
            sol[17 + nc][:] = tpars.phasecurve[i][0]

    if tpars.error_scale[2] == 'fit':
        if tpars.error_scale[1] == 'bolometric':
            sol[npars - 3][:] = np.ones(nwav) * tpars.error_scale[0][0]
        else:
            sol[npars - 3][:] = tpars.error_scale[0]

    if tpars.amplitude_scale[2] == 'fit':
        if tpars.amplitude_scale[1] == 'bolometric':
            sol[npars - 2][:] = np.ones(nwav) * tpars.amplitude_scale[0][0]
        else:
            sol[npars - 2][:] = tpars.amplitude_scale[0]

    if tpars.length_scale[2] == 'fit':
        if tpars.length_scale[1] == 'bolometric':
            sol[npars - 2][:] = np.ones(nwav) * tpars.length_scale[0][0]
        else:
            sol[npars - 2][:] = tpars.length_scale[0]

    return sol


def update_sol(tpars, x, sol):
    """
    Uses tpars and x to make an parameter set that will work with our
    transit model.
    """

    solnew = np.copy(sol)  # make a copy of the input sol array.
    nwav = sol.shape[1]  # number of bandpasses
    npars = sol.shape[0]  # number of model parameters

    xc = 0  # counts position as we work through the x array.

    if tpars.rhostar[2] == 'fit':
        if tpars.rhostar[1] == 'bolometric':
            solnew[0][:] = np.ones(nwav) * x[xc]
            xc += 1
        else:
            solnew[0][:] = x[xc:xc + nwav]
            xc += nwav

    if tpars.ld1[2] == 'fit':
        if tpars.ld1[1] == 'bolometric':
            solnew[1][:] = np.ones(nwav) * x[xc]
            xc += 1
        else:
            solnew[1][:] = x[xc:xc + nwav]
            xc += nwav

    if tpars.ld2[2] == 'fit':
        if tpars.ld2[1] == 'bolometric':
            solnew[2][:] = np.ones(nwav) * x[xc]
            xc += 1
        else:
            solnew[2][:] = x[xc:xc + nwav]
            xc += nwav

    if tpars.ld3[2] == 'fit':
        if tpars.ld3[1] == 'bolometric':
            solnew[3][:] = np.ones(nwav) * x[xc]
            xc += 1
        else:
            solnew[3][:] = x[xc:xc + nwav]
            xc += nwav

    if tpars.ld4[2] == 'fit':
        if tpars.ld4[1] == 'bolometric':
            solnew[4][:] = np.ones(nwav) * x[xc]
            xc += 1
        else:
            solnew[4][:] = x[xc:xc + nwav]
            xc += nwav

    if tpars.dilution[2] == 'fit':
        if tpars.dilution[1] == 'bolometric':
            solnew[5][:] = np.ones(nwav) * x[xc]
            xc += 1
        else:
            solnew[5][:] = x[xc:xc + nwav]
            xc += nwav

    if tpars.zeropoint[2] == 'fit':
        if tpars.zeropoint[1] == 'bolometric':
            solnew[7][:] = np.ones(nwav) * x[xc]
            xc += 1
        else:
            solnew[7][:] = x[xc:xc + nwav]
            xc += nwav

    for i in range(tpars.nplanet):
        nc = 10 * i

        if tpars.t0[i][2] == 'fit':
            if tpars.t0[i][1] == 'bolometric':
                solnew[8 + nc][:] = np.ones(nwav) * x[xc]
                xc += 1
            else:
                solnew[8 + nc][:] = x[xc:xc + nwav]
                xc += nwav

        if tpars.period[i][2] == 'fit':
            if tpars.period[i][1] == 'bolometric':
                solnew[9 + nc][:] = np.ones(nwav) * x[xc]
                xc += 1
            else:
                solnew[9 + nc][:] = x[xc:xc + nwav]
                xc += nwav

        if tpars.b[i][2] == 'fit':
            if tpars.b[i][1] == 'bolometric':
                solnew[10 + nc][:] = np.ones(nwav) * x[xc]
                xc += 1
            else:
                solnew[10 + nc][:] = x[xc:xc + nwav]
                xc += nwav

        if tpars.rprs[i][2] == 'fit':
            if tpars.rprs[i][1] == 'bolometric':
                solnew[11 + nc][:] = np.ones(nwav) * x[xc]
                xc += 1
            else:
                solnew[11 + nc][:] = x[xc:xc + nwav]
                xc += nwav

        if tpars.sqrt_e_cosw[i][2] == 'fit':
            if tpars.sqrt_e_cosw[i][1] == 'bolometric':
                solnew[12 + nc][:] = np.ones(nwav) * x[xc]
                xc += 1
            else:
                solnew[12 + nc][:] = x[xc:xc + nwav]
                xc += nwav

        if tpars.sqrt_e_sinw[i][2] == 'fit':
            if tpars.sqrt_e_sinw[i][1] == 'bolometric':
                solnew[13 + nc][:] = np.ones(nwav) * x[xc]
                xc += 1
            else:
                solnew[13 + nc][:] = x[xc:xc + nwav]
                xc += nwav

        if tpars.eclipse_depth[i][2] == 'fit':
            if tpars.eclipse_depth[i][1] == 'bolometric':
                solnew[15 + nc][:] = np.ones(nwav) * x[xc]
                xc += 1
            else:
                solnew[15 + nc][:] = x[xc:xc + nwav]
                xc += nwav

        if tpars.amplitude_scale[i][2] == 'fit':
            if tpars.amplitude_scale[i][1] == 'bolometric':
                solnew[16 + nc][:] = np.ones(nwav) * x[xc]
                xc += 1
            else:
                solnew[16 + nc][:] = x[xc:xc + nwav]
                xc += nwav

        if tpars.length_scale[i][2] == 'fit':
            if tpars.length_scale[i][1] == 'bolometric':
                solnew[17 + nc][:] = np.ones(nwav) * x[xc]
                xc += 1
            else:
                solnew[17 + nc][:] = x[xc:xc + nwav]
                xc += nwav

    if tpars.error_scale[2] == 'fit':
        if tpars.error_scale[1] == 'bolometric':
            solnew[npars - 3][:] = np.ones(nwav) * x[xc]
            xc += 1
        else:
            solnew[npars - 3][:] = x[xc:xc + nwav]
            xc += nwav

    if tpars.amplitude_scale[2] == 'fit':
        if tpars.amplitude_scale[1] == 'bolometric':
            solnew[npars - 2][:] = np.ones(nwav) * x[xc]
            xc += 1
        else:
            solnew[npars - 2][:] = x[xc:xc + nwav]
            xc += nwav

    if tpars.length_scale[2] == 'fit':
        if tpars.length_scale[1] == 'bolometric':
            solnew[npars - 2][:] = np.ones(nwav) * x[xc]
            xc += 1
        else:
            solnew[npars - 2][:] = x[xc:xc + nwav]
            xc += nwav

    return solnew


def get_names(sol, clabels, tpars):
    """
    Assign labels to all the parameters
    """

    # solnew = np.copy(sol)  # make a copy of the input sol array.
    nwav = sol.shape[1]  # number of bandpasses
    # npars = sol.shape[0]  # number of model parameters

    tran_par_names = []

    if tpars.rhostar[2] == 'fit':
        if tpars.rhostar[1] == 'bolometric':
            tran_par_names.append(clabels[0])
        else:
            for j in range(nwav):
                tran_par_names.append(clabels[0])

    if tpars.ld1[2] == 'fit':
        if tpars.ld1[1] == 'bolometric':
            tran_par_names.append(clabels[1])
        else:
            for j in range(nwav):
                tran_par_names.append(clabels[1])

    if tpars.ld2[2] == 'fit':
        if tpars.ld2[1] == 'bolometric':
            tran_par_names.append(clabels[2])
        else:
            for j in range(nwav):
                tran_par_names.append(clabels[2])

    if tpars.ld3[2] == 'fit':
        if tpars.ld3[1] == 'bolometric':
            tran_par_names.append(clabels[3])
        else:
            for j in range(nwav):
                tran_par_names.append(clabels[3])

    if tpars.ld4[2] == 'fit':
        if tpars.ld4[1] == 'bolometric':
            tran_par_names.append(clabels[4])
        else:
            for j in range(nwav):
                tran_par_names.append(clabels[4])

    if tpars.dilution[2] == 'fit':
        if tpars.dilution[1] == 'bolometric':
            tran_par_names.append(clabels[5])
        else:
            for j in range(nwav):
                tran_par_names.append(clabels[5])

    if tpars.zeropoint[2] == 'fit':
        if tpars.zeropoint[1] == 'bolometric':
            tran_par_names.append(clabels[7])
        else:
            for j in range(nwav):
                tran_par_names.append(clabels[7])

    for i in range(tpars.nplanet):
        # nc = 10 * i

        if tpars.t0[i][2] == 'fit':
            if tpars.t0[i][1] == 'bolometric':
                tran_par_names.append(clabels[8])
            else:
                for j in range(nwav):
                    tran_par_names.append(clabels[8])

        if tpars.period[i][2] == 'fit':
            if tpars.period[i][1] == 'bolometric':
                tran_par_names.append(clabels[9])
            else:
                for j in range(nwav):
                    tran_par_names.append(clabels[9])

        if tpars.b[i][2] == 'fit':
            if tpars.b[i][1] == 'bolometric':
                tran_par_names.append(clabels[10])
            else:
                for j in range(nwav):
                    tran_par_names.append(clabels[10])

        if tpars.rprs[i][2] == 'fit':
            if tpars.rprs[i][1] == 'bolometric':
                tran_par_names.append(clabels[11])
            else:
                for j in range(nwav):
                    tran_par_names.append(clabels[11])

        if tpars.sqrt_e_cosw[i][2] == 'fit':
            if tpars.sqrt_e_cosw[i][1] == 'bolometric':
                tran_par_names.append(clabels[12])
            else:
                for j in range(nwav):
                    tran_par_names.append(clabels[12])

        if tpars.sqrt_e_sinw[i][2] == 'fit':
            if tpars.sqrt_e_sinw[i][1] == 'bolometric':
                tran_par_names.append(clabels[13])
            else:
                for j in range(nwav):
                    tran_par_names.append(clabels[13])

        if tpars.eclipse_depth[i][2] == 'fit':
            if tpars.eclipse_depth[i][1] == 'bolometric':
                tran_par_names.append(clabels[15])
            else:
                for j in range(nwav):
                    tran_par_names.append(clabels[15])

        # Question: set twice?
        if tpars.amplitude_scale[i][2] == 'fit':
            if tpars.amplitude_scale[i][1] == 'bolometric':
                tran_par_names.append(clabels[16])
            else:
                for j in range(nwav):
                    tran_par_names.append(clabels[16])

        # Question: set twice?
        if tpars.length_scale[i][2] == 'fit':
            if tpars.length_scale[i][1] == 'bolometric':
                tran_par_names.append(clabels[17])
            else:
                for j in range(nwav):
                    tran_par_names.append(clabels[17])

    if tpars.error_scale[2] == 'fit':
        if tpars.error_scale[1] == 'bolometric':
            tran_par_names.append(clabels[18])
        else:
            for j in range(nwav):
                tran_par_names.append(clabels[18])

    if tpars.amplitude_scale[2] == 'fit':
        if tpars.amplitude_scale[1] == 'bolometric':
            tran_par_names.append(clabels[19])
        else:
            for j in range(nwav):
                tran_par_names.append(clabels[19])

    if tpars.length_scale[2] == 'fit':
        if tpars.length_scale[1] == 'bolometric':
            tran_par_names.append(clabels[20])
        else:
            for j in range(nwav):
                tran_par_names.append(clabels[20])

    return tran_par_names


def checksolution(tpars, solnew, badlpr):
    """
    Make sure model parameters are valid.
    """

    logl = 1.0e0

    nwav = sol.shape[1]  # number of bandpasses
    npars = sol.shape[0]  # number of model parameters

    for j in range(nwav):
        sol1 = np.array([s[j] for s in solnew])

        if tpars.rhostar[2] == 'fit':
            lo = tpars.rhostar[3][0]
            hi = tpars.rhostar[3][1]
            if (sol1[0] > hi) | (sol1[0] < lo):
                logl = badlpr

        if tpars.ld1[2] == 'fit':
            lo = tpars.ld1[3][0]
            hi = tpars.ld1[3][1]
            if (sol1[1] > hi) | (sol1[1] < lo):
                logl = badlpr

        if tpars.ld2[2] == 'fit':
            lo = tpars.ld2[3][0]
            hi = tpars.ld2[3][1]
            if (sol1[2] > hi) | (sol1[2] < lo):
                logl = badlpr

        if tpars.ld3[2] == 'fit':
            lo = tpars.ld3[3][0]
            hi = tpars.ld3[3][1]
            if (sol1[3] > hi) | (sol1[3] < lo):
                logl = badlpr

        if tpars.ld4[2] == 'fit':
            lo = tpars.ld4[3][0]
            hi = tpars.ld4[3][1]
            if (sol1[4] > hi) | (sol1[4] < lo):
                logl = badlpr

        if tpars.dilution[2] == 'fit':
            lo = tpars.dilution[3][0]
            hi = tpars.dilution[3][1]
            if (sol1[5] > hi) | (sol1[5] < lo):
                logl = badlpr

        if tpars.zeropoint[2] == 'fit':
            lo = tpars.zeropoint[3][0]
            hi = tpars.zeropoint[3][1]
            if (sol1[7] > hi) | (sol1[7] < lo):
                logl = badlpr

        for i in range(tpars.nplanet):
            nc = 10 * i

            if tpars.t0[i][2] == 'fit':
                lo = tpars.t0[i][3][0]
                hi = tpars.t0[i][3][1]
                if (sol1[8 + nc] > hi) | (sol1[8 + nc] < lo):
                    logl = badlpr

            if tpars.period[i][2] == 'fit':
                lo = tpars.period[i][3][0]
                hi = tpars.period[i][3][1]
                if (sol1[9 + nc] > hi) | (sol1[9 + nc] < lo):
                    logl = badlpr

            if tpars.b[i][2] == 'fit':
                lo = tpars.b[i][3][0]
                hi = tpars.b[i][3][1]
                if (sol1[10 + nc] > hi) | (sol1[10 + nc] < lo):
                    logl = badlpr

            if tpars.rprs[i][2] == 'fit':
                lo = tpars.rprs[i][3][0]
                hi = tpars.rprs[i][3][1]
                if (sol1[11 + nc] > hi) | (sol1[11 + nc] < lo):
                    logl = badlpr

            if tpars.sqrt_e_cosw[i][2] == 'fit':
                lo = tpars.sqrt_e_cosw[i][3][0]
                hi = tpars.sqrt_e_cosw[i][3][1]
                if (sol1[12 + nc] > hi) | (sol1[12 + nc] < lo):
                    logl = badlpr

            if tpars.sqrt_e_sinw[i][2] == 'fit':
                lo = tpars.sqrt_e_sinw[i][3][0]
                hi = tpars.sqrt_e_sinw[i][3][1]
                if (sol1[13 + nc] > hi) | (sol1[13 + nc] < lo):
                    logl = badlpr

            if tpars.eclipse_depth[i][2] == 'fit':
                lo = tpars.eclipse_depth[i][3][0]
                hi = tpars.eclipse_depth[i][3][1]
                if (sol1[15 + nc] > hi) | (sol1[15 + nc] < lo):
                    logl = badlpr

            if tpars.amplitude_scale[i][2] == 'fit':
                lo = tpars.amplitude_scale[i][3][0]
                hi = tpars.amplitude_scale[i][3][1]
                if (sol1[15 + nc] > hi) | (sol1[15 + nc] < lo):
                    logl = badlpr

            if tpars.length_scale[i][2] == 'fit':
                lo = tpars.length_scale[i][3][0]
                hi = tpars.length_scale[i][3][1]
                if (sol1[15 + nc] > hi) | (sol1[15 + nc] < lo):
                    logl = badlpr

        if tpars.error_scale[2] == 'fit':
            lo = tpars.error_scale[3][0]
            hi = tpars.error_scale[3][1]
            if (sol1[npars - 3] > hi) | (sol1[npars - 3] < lo):
                logl = badlpr

        if tpars.amplitude_scale[2] == 'fit':
            lo = tpars.amplitude_scale[3][0]
            hi = tpars.amplitude_scale[3][1]
            if (sol1[npars - 2] > hi) | (sol1[npars - 2] < lo):
                logl = badlpr

        if tpars.length_scale[2] == 'fit':
            lo = tpars.length_scale[3][0]
            hi = tpars.length_scale[3][1]
            if (sol1[npars - 1] > hi) | (sol1[npars - 1] < lo):
                logl = badlpr

    return logl


def gelmanrubin(*chain, burnin, npt):
    """
    # See pdf doc BrooksGelman for info
    "Estimating PSRF"
    """
    # Number of chains
    m = len(chain)
    # Assuming all chains have the same size.
    n = chain[0].shape[0] - burnin
    # Number of parameters
    npars = chain[0].shape[1]
    # Allocate array to hold mean calculations
    pmean = np.zeros(shape=(m, npars))
    # Allocate array to hold variance calculations
    pvar = np.zeros(shape=(m, npars))

    for i in range(0, m):
        currentchain = chain[i]
        for j in range(0, npars):
            # Generate means for each parameter in each chain
            pmean[i, j] = np.mean(currentchain[burnin:, j])
            # Generate variance for each parameter in each chain
            pvar[i, j] = np.var(currentchain[burnin:, j])
    # Allocate array for posterior means
    posteriormean = np.zeros(npars)
    for j in range(0, npars):
        # Calculate posterior mean for each parameter
        posteriormean[j] = np.mean(pmean[:, j])

    # Calculate between chains variance
    b = np.zeros(npars)
    for j in range(0, npars):
        for i in range(0, m):
            b[j] += np.power((pmean[i, j] - posteriormean[j]), 2)
    b = b * n / (m - 1.0)

    # Calculate within chain variance
    w = np.zeros(npars)
    for j in range(0, npars):
        for i in range(0, m):
            w[j] += pvar[i, j]
    w = w / m

    # Calculate the pooled variance
    v = (n - 1) * w / n + (m + 1) * b / (m * n)
    # Degrees of freedom
    dof = npt - 1
    # PSRF from Brooks and Gelman (1997)
    rc = np.sqrt((dof + 3.0) / (dof + 1.0) * v / w)

    # Calculate Ru
    # qa=0.95
    # ru=np.sqrt((dof+3.0)/(dof+1.0)*((N-1.0)/N*W+(m+1.0)/m*qa))

    return rc


def demhmcmc(x, llx, loglikelihood, beta, buffer, corbeta):
    """
    A Metropolis-Hastings MCMC with Gibbs sampler
    """

    nbuffer = len(buffer[:, 0])
    # draw a random number to decide which sampler to use
    rsamp = np.random.rand()
    # if rsamp is less than 0.5 use a Gibbs sampler
    if rsamp < 0.5:
        # make a copy of our current state to the trail state
        xt = np.copy(x)
        # number of parameters
        npars = len(x)
        # random select a parameter to vary.
        n = int(np.random.rand() * npars)
        # Step 2: Generate trial state with Gibbs sampler
        xt[n] = xt[n] + np.random.normal(0.0, beta[n])

    else:  # use our deMCMC sampler

        n = -1  # tell the accept array that we used the deMCMC sampler
        i1 = int(np.random.rand() * nbuffer)
        i2 = int(np.random.rand() * nbuffer)
        vectorjump = buffer[i1, :] - buffer[i2, :]
        xt = x + vectorjump * corbeta

    # Step 3 Compute log(p(x'|d))=log(p(x'))+log(p(d|x'))
    llxt = loglikelihood(xt)

    # Step 4 Compute the acceptance probability
    alpha = min(np.exp(llxt - llx), 1.0)

    u = np.random.rand()  # Step 5 generate a random number

    if u <= alpha:  # Step 6, compare u and alpha
        xp1 = np.copy(xt)  # accept new trial
        llxp1 = np.copy(llxt)
        ac = [0, n]  # Set ac to mark acceptance
    else:
        xp1 = np.copy(x)  # reject new trial
        llxp1 = np.copy(llx)
        ac = [1, n]  # Set ac to mark rejectance

    xp1 = np.array(xp1)
    return xp1, llxp1, ac


def betarescale(x, beta, niter, burnin, mcmcfunc, loglikelihood, imax=20,
                x_name=None):
    """
    Calculate rescaling of beta to improve acceptance rates
    """
    # alow, ahigh define the acceptance rate range we want
    alow = 0.22
    ahigh = 0.28
    # parameter controling how fast corscale changes - from Gregory 2011.
    delta = 0.01
    # Number of parameters
    npars = len(x)
    acorsub = np.zeros(npars)
    # total number of accepted proposals
    nacor = np.zeros(npars)
    # total number of accepted proposals immediately prior to rescaling
    nacorsub = np.zeros(npars)
    # total number of proposals
    npropp = np.zeros(npars)
    # total number of proposals immediately prior to rescaling
    nproppsub = np.zeros(npars)
    # acrate = np.zeros(npars)  # current acrate
    corscale = np.ones(npars)

    # inital run
    chain, accept = genchain(x, niter, beta, mcmcfunc, loglikelihood)  # Get a MC
    nchain = len(chain[:, 0])

    # calcalate initial values of npropp and nacor
    for i in range(burnin, nchain):
        # get accept flag value
        j = accept[i, 1]
        # update total number of proposals
        npropp[j] += 1
        # update total number of accepted proposals
        nacor[j] += 1 - accept[i, 0]

    # update x
    # we can continue to make chains by feeding the current state
    #    back into genchain
    xin = chain[niter, :]
    # inital acceptance rate
    acrate = nacor / npropp
    # afix is an integer flag to indicate which beta entries need to be updated
    afix = np.ones(npars)
    for i in range(0, npars):
        # we strive for an acceptance rate between alow,ahigh
        if (acrate[i] < ahigh) & (acrate[i] > alow):
            # afix=1 : update beta, afix=0 : do not update beta
            afix[i] = 0

    # We will iterate a maximum of imax times - avoid infinite loops
    icount = 0  # counter to track iterations
    while np.sum(afix) > 0:
        icount += 1  # track number of iterations

        if icount > 1:
            npropp = np.copy(nproppsub)
            nacor = np.copy(nacorsub)
        nacorsub = np.zeros(npars)  # reset nacorsub counts for each loop
        nproppsub = np.zeros(npars)  # reset nproppsub counts for each loop

        # Make another chain starting with xin
        # New beta for Gibbs sampling
        betain = beta * corscale
        # Get a MC
        chain, accept = genchain(xin, niter, betain, mcmcfunc, loglikelihood)
        xin = chain[niter, :]  # Store current parameter state

        # scan through Markov-Chains and count number of states and acceptances
        for i in range(burnin, nchain):
            j = accept[i, 1]
            # if acrate[j]>ahigh or acrate[j]<alow:
            # update total number of proposals
            npropp[j] += 1
            # update total number of accepted proposals
            nacor[j] += 1 - accept[i, 0]
            # Update current number of proposals
            nproppsub[j] += 1
            # Update current number of accepted proposals
            nacorsub[j] += 1 - accept[i, 0]

        # calculate acceptance rates for each parameter that is to updated
        for i in range(0, npars):
            # calculate current acrates
            acrate[i] = nacorsub[i] / nproppsub[i]

            # calculate acorsub
            acorsub[i] = (nacor[i] - nacorsub[i]) / (npropp[i] - nproppsub[i])

            if afix[i] > 0:
                # calculate corscale
                part1 = (acorsub[i] + delta) * 0.75 / (0.25 * (1.0 - acorsub[i] + delta))
                corscale[i] = np.abs(corscale[i] * np.power(part1, 0.25))

        if x_name is None:
            # report acceptance rates
            print('Iter', icount, 'Current Acceptance: ', acrate)
        else:
            # report acceptance rates
            print('Iter', icount, 'Current Acceptance: ')
            for i in range(npars):
                print(x_name[i], acrate[i])

        # check which parameters have achieved required acceptance rate
        for i in range(0, npars):
            if ahigh > acrate[i] > alow:
                afix[i] = 0

        if icount > imax:  # if too many iterations, then we give up and exit
            afix = np.zeros(npars)
            print("Too many iterations: icount > imax")

    if x_name is None:
        print('Final Acceptance: ', acrate)  # report acceptance rates
    else:
        print('Final Acceptance: ')  # report acceptance rates
        for i in range(npars):
            print(x_name[i], acrate[i])

    return corscale


def calcacrate(accept, burnin):  # ,label):
    """
    Calculate Acceptance Rates
    """
    nchain = len(accept[:, 0])

    pargs = [(nchain - burnin - np.sum(accept[burnin:, 0])) / (nchain - burnin)]
    print('Global Acceptance Rate: {0:.3f}'.format(*pargs))

    denprop = 0  # this is for deMCMC
    deacrate = 0  # this is for deMCMC

    for j in range(max(accept[burnin:, 1]) + 1):
        denprop = 0  # this is for deMCMC
        deacrate = 0  # this is for deMCMC

        nprop = 0  # number of proposals
        acrate = 0  # acceptance rate

        for i in range(burnin, nchain):  # scan through the chain.
            if accept[i, 1] == j:
                nprop = nprop + 1
                acrate = acrate + accept[i, 0]
            if accept[i, 1] == -1:
                denprop = denprop + 1
                deacrate = deacrate + accept[i, 0]

        # print('%s Acceptance Rate %.3f' % (label[j],(nprop-acrate)/nprop))
        pargs = [str(j), (nprop - acrate) / (nprop + 1)]
        print('{0} Acceptance Rate {1:.3f}'.format(*pargs))

    # if we have deMCMC results, report the acceptance rate.
    if denprop > 0:
        pargs = ['deMCMC', (denprop - deacrate) / denprop]
        print('{0} Acceptance Rate {1:.3f}'.format(*pargs))

    return


def genchain(x, niter, beta, mcmcfunc, loglikelihood, buffer=None, corbeta=1.0,
             progress=False):
    """Generate Markov Chain
    x - starting model parameters

    All variables needed by mcmcfunc are passed

    returns: chain, accept
        chain - Markov-Chain dimensions [npars,iter]
        accept - tracking acceptance
    """
    if buffer is None:
        buffer = []

    chain = []  # Initialize list to hold chain values
    accept = []  # Track our acceptance rate
    chain.append(x)  # Step 1: start the chain
    accept.append((0, 0))  # track acceptance rates for each parameter
    llx = loglikelihood(x)  # pre-compute the log-likelihood for Step 3

    if progress:
        for _ in tqdm(range(0, niter)):
            x, llx, ac = mcmcfunc(x, llx, loglikelihood, beta, buffer, corbeta)
            chain.append(x)
            accept.append(ac)
    else:
        for _ in range(0, niter):
            x, llx, ac = mcmcfunc(x, llx, loglikelihood, beta, buffer, corbeta)
            chain.append(x)
            accept.append(ac)

    chain = np.array(chain)  # Convert list to array
    accept = np.array(accept)

    return chain, accept


def mhgmcmc(x, llx, loglikelihood, beta, buffer=None, corbeta=1):
    """A Metropolis-Hastings MCMC with Gibbs sampler
    x - np.array : independent variable
    llx - real : previous value of logL
    loglikeihood : returns log-likelihood
    beta - Gibb's factor : characteristic step size

    buffer - used when we discuss deMCMC

    returns: xp1,llxp1,ac
      xpl - next state (new parameters)
      llxp1 - log-likelihood of new state
      ac - if trial state was accepted or rejected.
    """
    # not used?
    _ = corbeta

    if buffer is None:
        buffer = []
    # not used?
    _ = buffer

    xt = np.copy(x)  # make a copy of our current state to the trail state
    npars = len(x)  # number of parameters
    n = int(np.random.rand() * npars)  # random select a parameter to vary.

    xt[n] += np.random.normal(0.0, beta[n])  # Step 2: Generate trial state with Gibbs sampler

    llxt = loglikelihood(xt)  # Step 3 Compute log(p(x'|d))=log(p(x'))+log(p(d|x'))

    alpha = min(np.exp(llxt - llx), 1.0)  # Step 4 Compute the acceptance probability

    u = np.random.rand()  # Step 5 generate a random number

    if u <= alpha:  # Step 6, compare u and alpha
        xp1 = np.copy(xt)  # accept new trial
        llxp1 = np.copy(llxt)
        ac = [0, n]  # Set ac to mark acceptance
    else:
        xp1 = np.copy(x)  # reject new trial
        llxp1 = np.copy(llx)
        ac = [1, n]  # Set ac to mark rejectance

    # return new state and log(p(x|d))
    return xp1, llxp1, ac


def plotchains(chain, burnin, label):
    npars = chain.shape[1]
    fig, ax = plt.subplots(nrows=npars, figsize=(12, 1.5 * npars))
    for i in range(npars):
        # fig[i].subplot(npars, 1, i+1)
        ax[i].plot(chain[burnin:, i])  # ,c=colour[i])
        ax[i].tick_params(direction='in', length=10, width=2)
        ax[i].set_ylabel(label[i])
        if i + 1 < npars:
            ax[i].set_xticklabels([])

    plt.show()


def stack_multi_spec(multi_spec,
                     quantities=('WAVELENGTH', 'FLUX', 'FLUX_ERROR')):
    all_spec = {sp_ord: {quantity: [] for quantity in quantities}
                for sp_ord in [1, 2, 3]}
    for spec in multi_spec.spec:
        sp_ord = spec.spectral_order
        for quantity in quantities:
            all_spec[sp_ord][quantity].append(spec.spec_table[quantity])

    for sp_ord in all_spec:
        for key in all_spec[sp_ord]:
            all_spec[sp_ord][key] = np.array(all_spec[sp_ord][key])

    return all_spec


def lnprob(x):
    """
    ln prob model
    Nested sampling and many MCMC routines use global variables.  Thus, only the fitted parameters are passed.
    Fitted parameters are contained in the input 'x' array.
    """

    # logl = 1.0e0  # initialize log-likelihood to some value.
    # check validity of array
    badlpr = -np.inf  # if outside bounds, then mark poor likelihood.

    nwav = sol.shape[1]  # number of bandpasses
    npars = sol.shape[0]  # number of model parameters

    solnew = update_sol(tpars, x, sol)  # contains sol but updated with values from 'x'

    # check validity of array
    logl = checksolution(tpars, solnew, badlpr)

    if logl > badlpr:

        for i in range(nwav):
            dscale = solnew[npars - 3][i]  # photometric scale
            ascale = solnew[npars - 2][i]  # photometric scale
            lscale = solnew[npars - 1][i]  # length scale for GP

            # check dscale, ascale and lscale hyper parameters (they must be positive)
            if (dscale <= 0.0) & (tpars.error_scale[2] == 'fit'):
                logl = badlpr
            if (ascale <= 0.0) & (tpars.amplitude_scale[2] == 'fit'):
                logl = badlpr
            if (lscale <= 0.0) & (tpars.length_scale[2] == 'fit'):
                logl = badlpr

            if (tpars.amplitude_scale[2] == 'fit') | (tpars.length_scale[2] == 'fit'):
                modeltype = 1  # GP model
            else:
                modeltype = 0  # uncorrelated noise model

            # npt = len(photospectra[i].time)  # number of data points

            sol1 = np.array([s[i] for s in solnew])

            # zpt=np.copy(sol1[7])
            # sol1[7] -= 1.0 ###DL: this is screwing things up!

            if logl > badlpr:  # check that we have a valid model to use
                # Retrieve transit model using sol3 array
                ans = tf.transitmodel(sol1, photospectra[i].time,
                                      itime=photospectra[i].itime,
                                      ntt=tpars.ntt, tobs=tpars.tobs,
                                      omc=tpars.omc)
                # ans = ans*zpt #put in zero-point

                # check for NaNs -- we don't want these.
                if not np.isnan(np.sum(ans)):

                    if modeltype == 0:  # non-correlated noise-model

                        part1 = np.sum(np.log(photospectra[i].ferr ** 2 *
                                              dscale ** 2))
                        part2 = np.sum((photospectra[i].flux - ans) ** 2 /
                                       (photospectra[i].ferr ** 2 * dscale ** 2))

                        logl += -0.5 * (part1 + part2)
                else:
                    logl = badlpr

            # plt.plot(photospectra[i].time,photospectra[i].flux)
            # plt.plot(photospectra[i].time,ans)
            # plt.show()
            # Add Priors here...

    return logl


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Load model and data
    # -------------------------------------------------------------------------

    # load data
    data = datamodels.open(data_dir + extracted_tso_file)
    # rearrange data
    all_spec = stack_multi_spec(data)

    # create array of time
    n_group = data.meta.exposure.ngroups  # Number of groups
    t_group = data.meta.exposure.group_time  # Reading time [s]
    n_int = data.meta.exposure.nints
    time_int = (n_group + 1) * t_group

    del data  # no longer needed

    # -------------------------------------------------------------------------

    # create the time arrays to go along with the spectra, and add it to all_spec
    for i in [1, 2]:
        n_pix = all_spec[i]['FLUX'].shape[1]
        n_wl = all_spec[i]['WAVELENGTH'].shape[1]

        pix0 = 0  # if n_pix not 2048 must check and update this value, will it always be 2048?
        time_pix = (pix0 + np.arange(n_pix) - 1024) / 2048 * time_int

        # the 2d array of time, for all integrations and all pixels (wavelengths)
        # check order here, which end of spectrum is read first?
        time_obs = np.zeros(n_wl) + (np.arange(n_int) * time_int)[:, None]
        time_obs += time_pix[None, :]
        t0 = time_int / 2  # could choose appropriate time here
        time_obs += t0
        time_obs /= 86400  # convert to days

        # add this to the dict
        all_spec[i]['TIME'] = time_obs

    # -------------------------------------------------------------------------

    # Delete wavelengths corresponding to nul flux
    for i in [1, 2]:
        bad_idx, = ((all_spec[i]['FLUX'] == 0) | np.isnan(all_spec[i]['FLUX'])).all(axis=0).nonzero()
        all_spec[i]['WAVELENGTH'] = np.delete(all_spec[i]['WAVELENGTH'], bad_idx, axis=1)
        all_spec[i]['TIME'] = np.delete(all_spec[i]['TIME'], bad_idx, axis=1)
        all_spec[i]['FLUX'] = np.delete(all_spec[i]['FLUX'], bad_idx, axis=1)
        all_spec[i]['FLUX_ERROR'] = np.delete(all_spec[i]['FLUX_ERROR'], bad_idx, axis=1)

    # -------------------------------------------------------------------------

    # TODO: Must make sure binning is consistent between orders
    # apply spectral binning
    all_spec_binned = {1: {}, 2: {}}
    # num_bins = [30, 15]  # for order 1 and order 2
    num_bins = [10, 5]  # for order 1 and order 2

    for order, n_bins in zip([1, 2], num_bins):
        n_wl = all_spec[order]['FLUX'].shape[1]
        bin_size = n_wl // n_bins

        all_spec_binned[order]['WAVELENGTH'] = np.zeros((n_int, n_bins))
        all_spec_binned[order]['TIME'] = np.zeros((n_int, n_bins))
        all_spec_binned[order]['FLUX'] = np.zeros((n_int, n_bins))
        all_spec_binned[order]['FLUX_ERROR'] = np.zeros((n_int, n_bins))
        all_spec_binned[order]['BIN_LIMITS'] = np.zeros((2, n_bins))

        for i in range(n_bins):

            start, end = i * bin_size, (i + 1) * bin_size

            bin_wave = np.mean(all_spec[order]['WAVELENGTH'][:, start:end], axis=1)
            all_spec_binned[order]['WAVELENGTH'][:, i] = bin_wave

            bin_time = np.mean(all_spec[order]['TIME'][:, start:end], axis=1)
            all_spec_binned[order]['TIME'][:, i] = bin_time

            bin_flux = np.mean(all_spec[order]['FLUX'][:, start:end], axis=1)
            all_spec_binned[order]['FLUX'][:, i] = bin_flux

            bin_flux_err = np.sqrt(np.mean(all_spec[order]['FLUX_ERROR'][:, start:end] ** 2, axis=1))
            all_spec_binned[order]['FLUX_ERROR'][:, i] = bin_flux_err

            wave_start = all_spec[order]['WAVELENGTH'][0, start]
            wave_end = all_spec[order]['WAVELENGTH'][0, end - 1]

            all_spec_binned[order]['BIN_LIMITS'][:, i] = (wave_start, wave_end)

    # -------------------------------------------------------------------------
    # normalization of flux
    tnorm0 = 0.05  # time value before transit
    tnorm1 = 0.135  # time value after transit

    for order in [1, 2]:
        t = all_spec_binned[order]['TIME']
        i_norm, = ((t < tnorm0) | (t > tnorm1)).all(axis=1).nonzero()  # integration before/after transit
        f_norm = all_spec_binned[order]['FLUX'][i_norm, :].mean(axis=0)  # mean out-of-transit flux at each wavelength

        all_spec_binned[order]['FLUX'] /= f_norm[None, :]
        all_spec_binned[order]['FLUX_ERROR'] /= f_norm[None, :]

    # -------------------------------------------------------------------------

    # check that all looks good
    order = 2
    i = 0
    t = all_spec_binned[order]['TIME'][:, i]
    f = all_spec_binned[order]['FLUX'][:, i]
    e = all_spec_binned[order]['FLUX_ERROR'][:, i]
    print(all_spec_binned[order]['WAVELENGTH'][0, i])
    plt.errorbar(t, f, yerr=e)

    # -------------------------------------------------------------------------

    # here I want to remove a few lambdas from order 1, S/N too low
    order = 2
    imin = 4
    all_spec_binned[order]['WAVELENGTH'] = all_spec_binned[order]['WAVELENGTH'][:, imin:]
    all_spec_binned[order]['TIME'] = all_spec_binned[order]['TIME'][:, imin:]
    all_spec_binned[order]['FLUX'] = all_spec_binned[order]['FLUX'][:, imin:]
    all_spec_binned[order]['FLUX_ERROR'] = all_spec_binned[order]['FLUX_ERROR'][:, imin:]
    all_spec_binned[order]['BIN_LIMITS'] = all_spec_binned[order]['BIN_LIMITS'][:, imin:]

    # -------------------------------------------------------------------------

    # create and fill the photospectra list
    photospectra = []

    itime = np.ones(n_int) * (n_group - 1) * t_group / 60. / 60. / 24.  # Integration time [days]
    for order in [1, 2]:
        w = all_spec_binned[order]['WAVELENGTH']
        t = all_spec_binned[order]['TIME']
        f = all_spec_binned[order]['FLUX']
        fe = all_spec_binned[order]['FLUX_ERROR']
        for i in range(w.shape[1]):
            phot = PhotClass()  # Each wavelength has its class (arrays)
            phot.wavelength = np.copy(w[:, i])
            phot.time = np.copy(t[:, i])
            phot.flux = np.copy(f[:, i])
            phot.ferr = np.copy(fe[:, i])
            phot.itime = np.copy(itime)
            photospectra.append(phot)  # Stores phot. class

    # -------------------------------------------------------------------------

    # Show a plot of the data. Each colour is a different wavelength.
    matplotlib.rcParams.update({'font.size': 30})  # Adjust font
    matplotlib.rcParams['axes.linewidth'] = 2.0
    fig = plt.figure(figsize=(12, 8))  # Adjust size of figure
    ax = plt.axes()
    ax.tick_params(direction='in', which='major', bottom=True, top=True,
                   left=True, right=True, length=10, width=2)
    ax.tick_params(direction='in', which='minor', bottom=True, top=True,
                   left=True, right=True, length=4, width=2)

    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Flux')

    for p in photospectra:
        ax.plot(p.time, p.flux)
    # fig.savefig(WORKING_DIR + "transit_curves.png")
    fig.show()

    # -------------------------------------------------------------------------
    # Let's fit the multi-spectrum model
    # -------------------------------------------------------------------------

    # Number of planets to include
    nplanet = 1

    # Set up default parameters
    tpars = SptransitModelParameters([photospectra, nplanet])

    # Fill in a few necessary parameters
    # (Overwrites default parameters that were given in def)

    # [g/cm³]
    tpars.rhostar[0] = np.array([2.48])

    # boundaries for valid models, if needed.
    tpars.rhostar[3] = np.array([2.0, 3.0])

    # limb dark. coeff1
    tpars.ld3[0][:] = 0.2
    # limb dark. coeff2
    tpars.ld4[0][:] = 0.15

    # is added to transit curve
    tpars.zeropoint[0] = np.full(len(photospectra), 0.)
    tpars.zeropoint[3] = np.array([-0.005, 0.005])

    # planet
    tpars.period[0][0] = np.array([1.7497798])  # [days]
    tpars.period[0][2] = 'fixed'

    tpars.t0[0][0] = np.array([0.093])  # t0: center of transit time
    tpars.t0[0][3] = np.array([0.093 - 0.03, 0.093 + 0.03])

    tpars.b[0][0] = np.array([0.6])  # impact parameter
    tpars.b[0][3] = np.array([0.4, 0.8])

    tpars.rprs[0][0] = np.ones(len(photospectra)) * 0.1645  # Rp/Rs
    tpars.rprs[0][3] = np.array([0.1, 0.2])

    # Simple labels to identify parameters.
    # NOTE: VOF is not used, TED not used
    clabels = ['p', 'c1', 'c2', 'q1', 'q2', 'DIL', 'VOF', 'ZPT', 'EP', 'PE',
               'BB', 'RD', 'EC', 'ES', 'KRV', 'TED', 'ELL', 'ALB', 'DSC',
               'ASC', 'LSC']

    # -------------------------------------------------------------------------

    nwalkers = 3  # Number of walkers for MCMC
    nsteps1 = 10000  # Total length of chain will be nwalkers*nsteps
    nsteps2 = 200000  # nstep1 is to check that MCMC is okay, nstep2 is the real work.
    nsteps_inc = 100000
    burninf = 0.5  # burn-in for evalulating convergence
    niter_cor = 5000
    burnin_cor = 1000
    nthin = 101
    nloopmax = 3
    converge_crit = 1.02  # Convergence criteria
    buf_converge_crit = 1.2  # Convergence criteria for buffer
    itermax = 5  # Maximum iterations allowed

    # Creates internal array used to create transit model.
    sol = get_all_parameters(tpars, photospectra)
    x = get_fitted_parameters(tpars)
    tran_par_names = get_names(sol, clabels, tpars)
    beta = np.random.rand(len(x)) * 1.0e-5
    # beta = x*0.01

    # -------------------------------------------------------------------------

    # added by DL, trying to adjust the beta factors get better acceptance
    # rates right away
    tran_par_names = np.array(tran_par_names)
    beta[tran_par_names == 'p'] = 0.03  # rho star
    beta[tran_par_names == 'q1'] = 0.2
    beta[tran_par_names == 'q2'] = 0.2
    beta[tran_par_names == 'ZPT'] = 1.e-3
    beta[tran_par_names == 'EP'] = 2.e-4  # T0 (EPoch?)
    beta[tran_par_names == 'BB'] = 2.e-3  # impact parameter
    beta[tran_par_names == 'RD'] = 3.e-3
    beta[tran_par_names == 'DSC'] = 0.002  # error scale

    # -------------------------------------------------------------------------

    # check that all is good
    i = 3
    ans = tf.transitmodel(sol, photospectra[i].time,
                          itime=photospectra[i].itime,
                          ntt=tpars.ntt, tobs=tpars.tobs, omc=tpars.omc)
    # sol[:,5]
    plt.figure()
    plt.plot(photospectra[i].time, ans)
    plt.plot(photospectra[i].time, photospectra[i].flux)
    plt.show()

    # -------------------------------------------------------------------------

    # corscale=1
    corscale = betarescale(x, beta, niter_cor, burnin_cor, mhgmcmc,
                           loglikelihood=lnprob, imax=2, x_name=tran_par_names)

    # -------------------------------------------------------------------------
    # check of the final beta values that will be used,
    #    i.e. beta*corscale, for information
    for i in range(x.size):
        print(i, x[i], tran_par_names[i], beta[i] * corscale[i])
    # -------------------------------------------------------------------------

    nloop = 0
    nsteps = np.copy(nsteps1)
    mcmcloop = True
    progress = True

    chain1 = []
    chain2 = []
    chain3 = []
    accept1 = []
    accept2 = []
    accept3 = []
    burnin = 0

    while mcmcloop:

        nloop += 1  # Count number of loops

        hchain1, haccept1 = genchain(x, nsteps, beta * corscale,
                                     mhgmcmc, lnprob, progress=progress)
        hchain2, haccept2 = genchain(x, nsteps, beta * corscale,
                                     mhgmcmc, lnprob, progress=progress)
        hchain3, haccept3 = genchain(x, nsteps, beta * corscale,
                                     mhgmcmc, lnprob, progress=progress)

        if nloop == 1:
            chain1 = np.copy(hchain1)
            chain2 = np.copy(hchain2)
            chain3 = np.copy(hchain3)
            accept1 = np.copy(haccept1)
            accept2 = np.copy(haccept2)
            accept3 = np.copy(haccept3)
        else:
            chain1 = np.concatenate((chain1, hchain1))
            chain2 = np.concatenate((chain2, hchain2))
            chain3 = np.concatenate((chain3, hchain3))
            accept1 = np.concatenate((accept1, haccept1))
            accept2 = np.concatenate((accept2, haccept2))
            accept3 = np.concatenate((accept3, haccept3))

        burnin = int(chain1.shape[0] * burninf)
        calcacrate(accept1, burnin)

        grtest = gelmanrubin(chain1, chain2, chain3, burnin=burnin,
                             npt=len(photospectra[-1].time))

        print('Gelman-Rubin Convergence:')
        print('parameter  Rc')
        for i in range(0, len(chain1[1, :])):
            print('%8s %3s %.4f' % (str(i), tran_par_names[i], grtest[i]))
        if int(np.sum(grtest[grtest < buf_converge_crit] / grtest[grtest < buf_converge_crit])) == len(grtest):
            mcmcloop = False
        else:
            mcmcloop = True
            nsteps += nsteps1

        # runtest=np.array(tf.checkperT0(chain1,burninf,TPnthin,sol,serr))
        # print('runtest:',runtest)
        # if int(np.sum(runtest[runtest<1.0]/runtest[runtest<1.0]))!=4.0:
        #    mcmcloop=False #run-away

        if nloop >= nloopmax:  # Break if too many loops
            mcmcloop = False

        # print("---- %s seconds ----" % (time.time() - start_time))

    # -------------------------------------------------------------------------
    chain = np.concatenate((chain1[burnin:, ], chain2[burnin:, ], chain3[burnin:, ]))
    accept = np.concatenate((accept1[burnin:, ], accept2[burnin:, ], accept3[burnin:, ]))

    for i in range(x.size):
        print(tran_par_names[i], np.median(chain[:, i]))

    # -------------------------------------------------------------------------

    mcmcloop = True
    nloop = 0
    nsteps = np.copy(nsteps2)
    nloopmax = 3

    # -------------------------------------------------------------------------
    # the main work, bigger chains
    while mcmcloop:

        nloop += 1  # Count number of loops
        print('loop', nloop)

        burnin = int(chain1.shape[0] * burninf)
        # Create buffer for deMCMC
        buffer = np.concatenate((chain1[burnin:], chain2[burnin:],
                                 chain3[burnin:]))
        x1 = np.copy(chain1[chain1.shape[0] - 1, :])
        x2 = np.copy(chain1[chain2.shape[0] - 1, :])
        x3 = np.copy(chain1[chain2.shape[0] - 1, :])
        corbeta = 0.3
        # burnin = int(chain1.shape[0] * burninf)
        chain1, accept1 = genchain(x1, nsteps, beta * corscale, demhmcmc,
                                   lnprob, buffer=buffer,
                                   corbeta=corbeta, progress=True)
        chain2, accept2 = genchain(x2, nsteps, beta * corscale, demhmcmc,
                                   lnprob, buffer=buffer,
                                   corbeta=corbeta, progress=True)
        chain3, accept3 = genchain(x3, nsteps, beta * corscale, demhmcmc,
                                   lnprob, buffer=buffer,
                                   corbeta=corbeta, progress=True)

        burnin = int(chain1.shape[0] * burninf)
        grtest = gelmanrubin(chain1, chain2, chain3, burnin=burnin,
                             npt=len(photospectra[-1].time))
        print('Gelman-Rubin Convergence:')
        print('parameter  Rc')
        for i in range(0, len(chain1[1, :])):
            print('%8s %3s %.4f' % (str(i), tran_par_names[i], grtest[i]))

        if int(np.sum(grtest[grtest < converge_crit] / grtest[grtest < converge_crit])) == len(grtest):
            mcmcloop = False
        else:
            mcmcloop = True

        burnin = int(chain1.shape[0] * burninf)
        chain = np.concatenate((chain1[burnin:, ], chain2[burnin:, ],
                                chain3[burnin:, ]))
        accept = np.concatenate((accept1[burnin:, ], accept2[burnin:, ],
                                 accept3[burnin:, ]))
        burnin = int(chain.shape[0] * burninf)
        calcacrate(accept, burnin)

        nsteps += nsteps_inc  # Make longer chain to help with convergence

        # check for run-away Chain.
        # runtest=np.array(tf.checkperT0(chain1,burninf,nthin,sol,serr))
        # print('runtest:',runtest)
        # if int(np.sum(runtest[runtest<1.0]/runtest[runtest<1.0]))!=4.0:
        #    mcmcloop=False #run-away

        if nloop >= nloopmax:  # Break if too many loops
            mcmcloop = False

        # break #DL, to do loops one by one manually

    # -------------------------------------------------------------------------

    # quick check of median value of posterior
    for i in range(x.size):
        print(i, tran_par_names[i], np.median(chain[:, i]))

    # -------------------------------------------------------------------------

    # quick check of chain
    plt.plot(chain[:, 140])

    # -------------------------------------------------------------------------

    # check agreement of fit
    solnew = update_sol(tpars, chain1[-1, :], sol)

    i = 5
    ans = tf.transitmodel(solnew[:, i], photospectra[i].time,
                          itime=photospectra[i].itime,
                          ntt=tpars.ntt, tobs=tpars.tobs, omc=tpars.omc)
    plt.figure()
    plt.plot(photospectra[i].time, ans)
    plt.plot(photospectra[i].time, photospectra[5].flux)
    plt.show()

    # -------------------------------------------------------------------------

    # quick check of posterior
    _ = plt.hist(chain[::10, 142])

    # -------------------------------------------------------------------------

    matplotlib.rcParams.update({'font.size': 12})  # adjust font
    plotchains(chain, 0, tran_par_names)

    # -------------------------------------------------------------------------

    # print the value and error of all parameters
    npars = len(chain[1, :])
    mm = np.zeros(npars)
    for i in range(0, npars):
        mode, x_eval, kde1 = tf.modekdestimate(chain[::10, i], 0)
        mm[i] = mode
        perc1 = tf.intperc(mode, x_eval, kde1)
        # ,perc=0.997)
        pargs = [tran_par_names[i], mode, np.abs(perc1[1] - mode),
                 np.abs(mode - perc1[0])]
        print('{0} = {1:.8f} + {2:.8f} - {3:.8f} (1 Sigma)'.format(*pargs))

    # -------------------------------------------------------------------------

    npars = len(chain[1, :])
    mm = np.zeros(npars)

    nthin = 10
    chain_thin = chain[::nthin, :]

    rprs_model = []
    rprs_model_ep = []
    rprs_model_em = []
    for i in range(npars):
        if tran_par_names[i] != 'RD':
            continue
        mode, x_eval, kde1 = tf.modekdestimate(chain_thin[:, i], 0)
        mm[i] = mode
        perc1 = tf.intperc(mode, x_eval, kde1)  # , perc=0.997)
        rprs_model.append(mode)
        rprs_model_ep.append(np.abs(perc1[1] - mode))
        rprs_model_em.append(-np.abs(mode - perc1[0]))

        pargs = [tran_par_names[i], mode, np.abs(perc1[1] - mode),
                 np.abs(mode - perc1[0])]
        print('{0} = {1:.8f} +{2:.8f} -{3:.8f} (1 Sigma)'.format(*pargs))

    rprs_model = np.array(rprs_model)
    rprs_model_ep = np.array(rprs_model_ep)
    rprs_model_em = np.array(rprs_model_em)

    # -------------------------------------------------------------------------
    # TODO: Needs updating wl_test does not exist any more
    #       we have all_spec_binned[order]['WAVELENGTH']  but need to figure out
    #       what to do with each order
    # save rp/rs result of MCMC
    # with open("trans_spec_result_ord12.txt", "w") as f:
    #     print("wave, rp/rs, err_low, err_hi", file=f)
    #     for i in range(n_wl):
    #         pargs = [wl_test[0, i], rprs_model[i], rprs_model_em[i], rprs_model_ep[i]]
    #         print("{:.6f} {:.8f} {:.8f} {:.8f}".format(),
    #             file=f)

    # -------------------------------------------------------------------------

    # load the input model for comparison
    input_model_filename = data_dir + model_file
    input_wl, input_dppm, _ = np.loadtxt(input_model_filename, delimiter=',',
                                         comments=['#', 'wave'], dtype=float,
                                         unpack=True)

    input_rprs = np.sqrt(input_dppm / 1.e6)

    # -------------------------------------------------------------------------
    # bin model to match retrieved spectrum
    n_wl = len(photospectra)
    input_rprs_b = np.zeros(n_wl)
    input_wl_b = np.zeros(n_wl)
    order_b = np.zeros(n_wl)

    for i in range(n_wl):
        for order in [1, 2]:
            # noinspection PyUnresolvedReferences
            j, = (all_spec_binned[order]['WAVELENGTH'][0, :] == photospectra[i].wavelength[0]).nonzero()
            if j.size > 0:
                break
        lam2, lam1 = all_spec_binned[order]['BIN_LIMITS'][:, j[0]]

        order_b[i] = order

        print(i, j, order)

        i1 = np.searchsorted(input_wl, lam1)
        i2 = np.searchsorted(input_wl, lam2)

        input_rprs_b[i] = input_rprs[i1:i2].mean()
        input_wl_b[i] = input_wl[i1:i2].mean()

    # -------------------------------------------------------------------------

    fig = plt.figure(figsize=(12, 8))
    plt.plot(input_wl, input_rprs, lw=1, color='lightgrey')

    order = 2
    i, = (order_b == order).nonzero()
    plt.plot(input_wl_b[i], input_rprs_b[i], color='b')
    w = [photospectra[ii].wavelength[0] for ii in i]
    plt.errorbar(w, rprs_model[i], yerr=np.array([-rprs_model_em[i], rprs_model_ep[i]]), fmt='o', lw=3,
                 zorder=2, mec='k', c='b')

    order = 1
    i, = (order_b == order).nonzero()
    plt.plot(input_wl_b[i], input_rprs_b[i], color='r')
    w = [photospectra[ii].wavelength[0] for ii in i]
    plt.errorbar(w, rprs_model[i], yerr=np.array([-rprs_model_em[i], rprs_model_ep[i]]), fmt='o', lw=3,
                 zorder=2, mec='k', c='r')

    plt.xlim(0.6, 3)
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel(r'$R_{p}/R_{\star}$')

# =============================================================================
# End of code
# =============================================================================
