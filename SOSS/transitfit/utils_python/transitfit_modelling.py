import numpy as np

class phot_class:
    def __init__(self):
        # initialize arrays
        self.wavelength = []  # wavelength of observation (should be a single number)
        self.time = []  # time-stamps array -- mid-exposure time (days)
        self.flux = []  # observed flux array
        self.ferr = []  # error in flux array
        self.itime = []  # integration time (seconds)







class psg_class:
    def __init__(self):
        # initialize arrays
        self.wavelength = []  # wavelength of observation array
        self.flux = []  # observed transit-depth due to atmosphere (ppm)
        self.ferr = []  # predicted noise (ppm)


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1  # array[idx-1]
    else:
        return idx  # array[idx]


def read_model(filename):
    '''Reads in simple space delimited textfile.
    '''

    data = []
    f = open(filename)
    icount = -1
    for line in f:
        line = line.strip()
        columns = line.split()  # break into columns
        icount += 1
        if (icount > 0) & (columns[0] != '#'):  # skip header info
            data.append([float(i) for i in columns])
            # data.append(columns)
    f.close()

    return data


class sptransit_model_class(object):
    def __init__(self):
        self.wavelength = []  # Wavelength of each observation

        self.rhostar = []  # mean stellar density
        self.ld1 = []  # limb-darkening. set ld1=ld2=0 and ld3=q1 and ld4=q2
        self.ld2 = []
        self.ld3 = []
        self.ld4 = []
        self.dilution = []  # stellar dilution 0=none, 0.99 means 99% of light from other source
        self.zeropoint = []  # out of transit baseline

        self.nplanet = 1  # number of planets
        self.t0 = []  # center of transit time
        self.period = []  # orbital period
        self.b = []  # impact parameter
        self.rprs = []  # scale planet radius
        self.sqrt_e_cosw = []  # sqrt(e)cos(w)
        self.sqrt_e_sinw = []  # sqrt(e)cos(w)
        self.eclipse_depth = []  # Secondard eclipse depth (ppm)
        self.ellipsoidal = []  # amplitude of ellipsoidal variations (ppm)
        self.phasecurve = []  # amplitude of reflected/emission phase curve (ppm) - Lambertian

        self.error_scale = []  # scale to apply to photometric errors
        self.amplitude_scale = []  # GP Kernel Amplitude (default is Matern 3/2)
        self.length_scale = []  # GP length scale (default is Matern 3/2)

        self.ntt = []  # parameters for TTVs ntt=number of transit times
        self.tobs = []  # observed centre of transit times
        self.omc = []  # O-C values for eachtransit


class sptransit_model_parameters(sptransit_model_class):
    def __init__(self, num):
        sptransit_model_class.__init__(self)
        nwav = len(num[0])

        # Wavelength for each dataset
        zpt = []
        for p in num[0]:
            self.wavelength.append(p.wavelength)
            zpt.append(np.median(p.flux))
        zpt = np.array(zpt)

        # Star Parameters
        self.rhostar = [np.ones(1), 'bolometric', 'fit', np.array([1.0e-4, 1000])]
        self.ld1 = [np.zeros(nwav), 'chromatic', 'fixed', np.array([-1, 1])]
        self.ld2 = [np.zeros(nwav), 'chromatic', 'fixed', np.array([-1, 1])]
        self.ld3 = [np.ones(nwav) * 0.5, 'chromatic', 'fit', np.array([0, 1])]
        self.ld4 = [np.ones(nwav) * 0.5, 'chromatic', 'fit', np.array([0, 1])]
        self.dilution = [np.zeros(nwav), 'chromatic', 'fixed', np.array([0, 1])]
        self.zeropoint = [zpt, 'chromatic', 'fit', np.array([0, 1.0e9])]

        # Planet Parameters
        nplanet = num[1]
        self.nplanet = num[1] * 1
        for i in range(nplanet):
            self.t0.append([np.ones(1), 'bolometric', 'fit', np.array([0, 2])])
            self.period.append([np.ones(1), 'bolometric', 'fit', np.array([0, 2])])
            self.b.append([np.ones(1) * 0.5, 'bolometric', 'fit', np.array([0, 2])])
            self.rprs.append([np.ones(nwav) * 0.01, 'chromatic', 'fit', np.array([0, 1])])
            self.sqrt_e_cosw.append([np.zeros(1), 'bolometric', 'fixed', np.array([-1, 1])])
            self.sqrt_e_sinw.append([np.zeros(1), 'bolometric', 'fixed', np.array([-1, 1])])
            self.eclipse_depth.append([np.zeros(nwav), 'chromatic', 'fixed', np.array([0, 1.0e4])])
            self.ellipsoidal.append([np.zeros(nwav), 'chromatic', 'fixed', np.array([0, 1.0e4])])
            self.phasecurve.append([np.zeros(nwav), 'chromatic', 'fixed', np.array([0, 1.0e4])])

        # Error model
        self.error_scale = [np.ones(1) * 1.0, 'bolometric', 'fit', np.array([0, 2])]
        self.amplitude_scale = [np.ones(nwav) * 1.0, 'chromatic', 'fixed', np.array([0, 2000])]
        self.length_scale = [np.ones(nwav) * 1.0, 'chromatic', 'fixed', np.array([0, 1])]

        self.ntt = 0
        self.tobs = 0
        self.omc = 0


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
    nhp = 3  # potential number of hyper-parameters

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
    '''Uses tpars and x to make an parameter set that will work with our transit model.
    '''

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


def tran_ll_transform(cube):
    params = cube.copy()  # cube contains random draws that need to be mapped to params
    ncube = len(cube)  # number of parameters that need to be mapped.

    nwav = sol.shape[1]  # number of bandpasses
    npars = sol.shape[0]  # number of model parameters

    xc = 0  # counter to keep track of how many parmaters we have.

    if tpars.rhostar[2] == 'fit':
        lo = tpars.rhostar[3][0]
        hi = tpars.rhostar[3][1]
        if tpars.rhostar[1] == 'bolometric':
            params[xc] = cube[xc] * (hi - lo) + lo
            xc += 1
        else:
            params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
            xc += nwav

    if tpars.ld1[2] == 'fit':
        lo = tpars.ld1[3][0]
        hi = tpars.ld1[3][1]
        if tpars.ld1[1] == 'bolometric':
            params[xc] = cube[xc] * (hi - lo) + lo
            xc += 1
        else:
            params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
            xc += nwav

    if tpars.ld2[2] == 'fit':
        lo = tpars.ld2[3][0]
        hi = tpars.ld2[3][1]
        if tpars.ld2[1] == 'bolometric':
            params[xc] = cube[xc] * (hi - lo) + lo
            xc += 1
        else:
            params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
            xc += nwav

    if tpars.ld3[2] == 'fit':
        lo = tpars.ld3[3][0]
        hi = tpars.ld3[3][1]
        if tpars.ld3[1] == 'bolometric':
            params[xc] = cube[xc] * (hi - lo) + lo
            xc += 1
        else:
            params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
            xc += nwav

    if tpars.ld4[2] == 'fit':
        lo = tpars.ld4[3][0]
        hi = tpars.ld4[3][1]
        if tpars.ld4[1] == 'bolometric':
            params[xc] = cube[xc] * (hi - lo) + lo
            xc += 1
        else:
            params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
            xc += nwav

    if tpars.dilution[2] == 'fit':
        lo = tpars.dilution[3][0]
        hi = tpars.dilution[3][1]
        if tpars.dilution[1] == 'bolometric':
            params[xc] = cube[xc] * (hi - lo) + lo
            xc += 1
        else:
            params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
            xc += nwav

    if tpars.zeropoint[2] == 'fit':
        lo = tpars.zeropoint[3][0]
        hi = tpars.zeropoint[3][1]
        if tpars.zeropoint[1] == 'bolometric':
            params[xc] = cube[xc] * (hi - lo) + lo
            xc += 1
        else:
            params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
            xc += nwav

    for i in range(tpars.nplanet):
        nc = 10 * i

        if tpars.t0[i][2] == 'fit':
            lo = tpars.t0[i][3][0]
            hi = tpars.t0[i][3][1]
            if tpars.t0[i][1] == 'bolometric':
                params[xc] = cube[xc] * (hi - lo) + lo
                xc += 1
            else:
                params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
                xc += nwav

        if tpars.period[i][2] == 'fit':
            lo = tpars.period[i][3][0]
            hi = tpars.period[i][3][1]
            if tpars.period[i][1] == 'bolometric':
                params[xc] = cube[xc] * (hi - lo) + lo
                xc += 1
            else:
                params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
                xc += nwav

        if tpars.b[i][2] == 'fit':
            lo = tpars.b[i][3][0]
            hi = tpars.b[i][3][1]
            if tpars.b[i][1] == 'bolometric':
                params[xc] = cube[xc] * (hi - lo) + lo
                xc += 1
            else:
                params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
                xc += nwav

        if tpars.rprs[i][2] == 'fit':
            lo = tpars.rprs[i][3][0]
            hi = tpars.rprs[i][3][1]
            if tpars.rprs[i][1] == 'bolometric':
                params[xc] = cube[xc] * (hi - lo) + lo
                xc += 1
            else:
                params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
                xc += nwav

        if tpars.sqrt_e_cosw[i][2] == 'fit':
            lo = tpars.sqrt_e_cosw[i][3][0]
            hi = tpars.sqrt_e_cosw[i][3][1]
            if tpars.sqrt_e_cosw[i][1] == 'bolometric':
                params[xc] = cube[xc] * (hi - lo) + lo
                xc += 1
            else:
                params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
                xc += nwav

        if tpars.sqrt_e_sinw[i][2] == 'fit':
            lo = tpars.sqrt_e_sinw[i][3][0]
            hi = tpars.sqrt_e_sinw[i][3][1]
            if tpars.sqrt_e_sinw[i][1] == 'bolometric':
                params[xc] = cube[xc] * (hi - lo) + lo
                xc += 1
            else:
                params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
                xc += nwav

        if tpars.eclipse_depth[i][2] == 'fit':
            lo = tpars.eclipse_depth[i][3][0]
            hi = tpars.eclipse_depth[i][3][1]
            if tpars.eclipse_depth[i][1] == 'bolometric':
                params[xc] = cube[xc] * (hi - lo) + lo
                xc += 1
            else:
                params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
                xc += nwav

        if tpars.amplitude_scale[i][2] == 'fit':
            lo = tpars.amplitude_scale[i][3][0]
            hi = tpars.amplitude_scale[i][3][1]
            if tpars.amplitude_scale[i][1] == 'bolometric':
                params[xc] = cube[xc] * (hi - lo) + lo
                xc += 1
            else:
                params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
                xc += nwav

        if tpars.length_scale[i][2] == 'fit':
            lo = tpars.length_scale[i][3][0]
            hi = tpars.length_scale[i][3][1]
            if tpars.length_scale[i][1] == 'bolometric':
                params[xc] = cube[xc] * (hi - lo) + lo
                xc += 1
            else:
                params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
                xc += nwav

    if tpars.error_scale[2] == 'fit':
        lo = tpars.error_scale[3][0]
        hi = tpars.error_scale[3][1]
        if tpars.error_scale[1] == 'bolometric':
            params[xc] = cube[xc] * (hi - lo) + lo
            xc += 1
        else:
            params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
            xc += nwav

    if tpars.amplitude_scale[2] == 'fit':
        lo = tpars.amplitude_scale[3][0]
        hi = tpars.amplitude_scale[3][1]
        if tpars.amplitude_scale[1] == 'bolometric':
            params[xc] = cube[xc] * (hi - lo) + lo
            xc += 1
        else:
            params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
            xc += nwav

    if tpars.length_scale[2] == 'fit':
        lo = tpars.length_scale[3][0]
        hi = tpars.length_scale[3][1]
        if tpars.length_scale[1] == 'bolometric':
            params[xc] = cube[xc] * (hi - lo) + lo
            xc += 1
        else:
            params[xc:xc + nwav] = cube[xc:xc + nwav] * (hi - lo) + lo
            xc += nwav

    return params


def get_names(clabels, tpars, sol):
    '''Assign labels to all the parameters
    '''

    solnew = np.copy(sol)  # make a copy of the input sol array.
    nwav = sol.shape[1]  # number of bandpasses
    npars = sol.shape[0]  # number of model parameters

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
        nc = 10 * i

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

        if tpars.amplitude_scale[i][2] == 'fit':
            if tpars.amplitude_scale[i][1] == 'bolometric':
                tran_par_names.append(clabels[16])
            else:
                for j in range(nwav):
                    tran_par_names.append(clabels[16])

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


#def checksolution(tpars, solnew, badlpr):
def checksolution(tpars, solnew, badlpr, sol):
    '''Make sure model parameters are valid.
    '''

    logl = 1.0e0

    nwav = sol.shape[1]  # number of bandpasses
    npars = sol.shape[0]  # number of model parameters

    for i in range(nwav):
        sol1 = np.array([s[i] for s in solnew])

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
