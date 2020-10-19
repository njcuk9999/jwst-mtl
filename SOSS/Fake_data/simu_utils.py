# Not intended to be official code.

from astropy.io import fits

def load_simu(filename, order=None, noisy=True):
    """ Load a homemade simulation """
    hdu = fits.open(filename)
    out = {'grid': hdu['FLUX'].data['lam_grid'],
           'f_k': hdu['FLUX'].data['f_lam'],
           'grid_c1': hdu['FLUX_C1'].data['lam_grid'],
           'f_c1': hdu['FLUX_C1'].data['f_lam'],
           'grid_c2': hdu['FLUX_C2'].data['lam_grid'],
           'f_c2': hdu['FLUX_C2'].data['f_lam']}
    
    if order is None:
        key = "FULL"
    else:
        key = f"ORD {order}"
        
    if noisy:
        key += " NOISY"
        
    out['data'] = hdu[key].data
    
    return out