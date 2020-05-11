import numpy as np

def _get_lam_p_or_m(lam):
    '''
    Compute lambda_plus and lambda_minus
    '''
    lam = lam.T  # Simpler to use transpose
    
    # Init
    lam_r = np.zeros_like(lam)
    lam_l = np.zeros_like(lam)
    
    # Def delta lambda
    d_lam = np.diff(lam, axis=0)
    
    # Define lambda left and lambda right of each pixels
    lam_r[:-1] = lam[:-1] + d_lam/2
    lam_r[-1] = lam[-1] + d_lam[-1]/2
    lam_l[1:] = lam[:-1] + d_lam/2  # Same values as lam_r
    lam_l[0] = lam[0] - d_lam[0]/2
    
    if (lam_r >= lam_l).all():
        return lam_r.T, lam_l.T
    elif (lam_r <= lam_l).all():
        return lam_l.T, lam_r.T
    else:
        raise ValueError('Bad pixel values for wavelength')