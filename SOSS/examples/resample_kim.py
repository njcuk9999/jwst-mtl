import matplotlib.pyplot as plt
import numpy as np


def resample_models(dw,starmodel_wv, starmodel_flux, ld_coeff, planetmodel_wv, planetmodel_rprs, pars, tracePars):
    """Resamples star and planet model onto common grid.

    Usage:
    bin_starmodel_wv,bin_starmodel_flux,bin_ld_coeff,bin_planetmodel_wv,bin_planetmodel_rprs\
      =resample_models(dw,starmodel_wv,starmodel_flux,ld_coeff,\
      planetmodel_wv,planetmodel_rprs,pars,tracePars)

      Inputs:
        dw - wavelength spacing.  This should be calculated using get_dw
        starmodel_wv - stellar model wavelength array
        starmodel_flux - stellar model flux array
        ld_coeff - non-linear limb-darkening coefficients array
        planetmodel_wv - planet model wavelength array
        planetmodel_rprs - planet model Rp/R* array
        pars - model parameters
        tracePars - trace solution

      Output:
        bin_starmodel_wv - binned star wavelength array
        bin_starmodel_flux - binned star model array
        bin_ld_coeff - binned limb-darkening array
        bin_planetmodel_wv - binned planet wavelength array (should be same size as bin_starmodel_wv)
        bin_planetmodel_rprs - binned Rp/R* array
    """
    bin_starmodel_wv = starmodel_wv
    bin_starmodel_flux = starmodel_flux
    plt.figure()
    plt.plot(bin_starmodel_wv[800000:804000], bin_starmodel_flux[800000:804000])
    plt.show()
    d_wv = np.empty(shape=(len(starmodel_wv)-1))
    for i in range(len(starmodel_wv)-1):
        d_wv[i] = starmodel_wv[i+1] - starmodel_wv[i]
    print(d_wv)
    return bin_starmodel_wv, bin_starmodel_flux, bin_ld_coeff,bin_planetmodel_wv, bin_planetmodel_rprs