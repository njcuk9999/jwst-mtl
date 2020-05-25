import numpy as np
from custom_numpy import is_sorted


def gaussian(x, x0=0, sig=1, amp=None):
    
    # Amplitude term
    if amp is None:
        amp = 1/np.sqrt(2 * np.pi * sig**2)
    
    return amp * np.exp(-0.5*((x - x0) / sig)**2)


def gauss_ker(x=None, mean=None, sigma=None, FWHM=None, nFWHM=7, oversample=None):
    
    if mean is None:
        mean = np.array([0])
    
    if sigma is None:
        sigma = np.array([1])
        
    if oversample is None:
        oversample = 1
    
    if x is None:
        if FWHM is None:
            FWHM = sigma2fwhm(sigma)
        # Length of the kernel is 2 x nFWHM times FWHM
        x = np.linspace(0, nFWHM*FWHM, int(nFWHM*FWHM*oversample + 1))
        x = np.concatenate([mean - x, mean + x])
        x = np.unique(x)
       
    if FWHM is not None:
        # Convert to sigma
        sigma = fwhm2sigma(FWHM)  # FWHM / (2 * np.sqrt(2 * np.log(2)))
    
    #
    # Compute gaussian ----------
    #
    # Exponential term
    G = np.exp(-0.5 * ((x - mean[:,None]) / sigma[:,None])**2)
    # Amplitude term
    G /= np.sqrt(2 * np.pi * sigma[:,None]**2)

    # Normalization
    G /= G.sum(axis=-1)[:,None]  

    return G.squeeze()


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def get_d_lam_poly(wv, sampling=2, order=5):
    """
    Returns the function of the width of the resolution
    kernel (delta lambda) in the wavelength space.
    The function is a polynomial fit.
    """
    
    # Use only valid columns
    wv = wv[wv > 0]
    
    # Get d_lam of the convolution kernel assuming sampling
    d_lam = np.abs(sampling*np.diff(wv))
    
    # Fit polynomial
    coeffs = np.polyfit(wv[1:], d_lam, order)
    
    # Return as a callable
    return np.poly1d(coeffs)
