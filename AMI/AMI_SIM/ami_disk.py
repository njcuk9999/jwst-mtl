import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def mk_disk(params, doplot = True):
    wpix = int(params['fov'] * params['oversampling'])  # width of simulation image in pixels

    x, y = np.indices([wpix, wpix]) - wpix / 2.0 + .5
    x*=(params['niriss_scale']/params['oversampling'])
    y*=(params['niriss_scale']/params['oversampling'])


    x1 = np.cos(params['roll'] * np.pi / 180) * x + np.sin(params['roll'] * np.pi / 180) * y
    y1 = -np.sin(params['roll'] * np.pi / 180) * x + np.cos(params['roll'] * np.pi / 180) * y

    x2 = x1 / np.cos(params['inclination'] * np.pi / 180)
    y2 = np.array(y1)

    rr = np.sqrt(x2 ** 2 + y2 ** 2)
    flux = np.exp(-0.5 * np.abs((rr - params['rad']) /  (params['width'] )) ** params['exponent'])

    if doplot:
        plt.imshow(flux)
        plt.show()

    return flux

def mk_bar(params, doplot = True):
    wpix = int(params['fov'] * params['oversampling'])  # width of simulation image in pixels

    x, y = np.indices([wpix, wpix]) - wpix / 2.0 + .5
    x*=(params['niriss_scale']/params['oversampling'])
    y*=(params['niriss_scale']/params['oversampling'])

    x1 = np.cos(params['roll'] * np.pi / 180) * x + np.sin(params['roll'] * np.pi / 180) * y
    y1 = -np.sin(params['roll'] * np.pi / 180) * x + np.cos(params['roll'] * np.pi / 180) * y


    y1b = (np.abs(y1)-params['rad'])/params['width']
    y1b[y1b<0] = 0
    flux = np.exp(-0.5*np.abs(x1/params['width'])**params['exponent'])*np.exp( -.5*y1b**params['exponent']  )

    if doplot:
        plt.imshow(flux)
        plt.show()
    return flux

params = dict()

params['fov'] = 30 # in NIRISS pixels
params['oversampling'] = 11  # oversampling of simulation pixels
params['niriss_scale'] = 0.065 # oversampling of simulation pixels

# For a disk
# long axis radius in arcsec for disk
# long axis of the bar
params['rad'] = 0.2
# e-width of annulus in arcsec for disk
# thickness of bar
params['width'] = 0.02
params['exponent'] = 2
params['inclination'] = 60 # in degrees -> tilt toward the line of sight
params['roll'] = 15 # in degrees -> rotation on the sky plane

#image = mk_disk(params,doplot = False)
#params['rad'] = 0.4
#params['exponent'] = 1.2
#image = image+mk_disk(params,doplot = False)
#plt.imshow(image)
#plt.show()

image = mk_bar(params,doplot = True)


