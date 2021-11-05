import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt

if False:

    im = fits.getdata('/home/kmorel/ongenesis/jwst-user-soss/PHY3030/WASP_52/IDTSOSS_clear_noisy--photon--darkcurrent--superbias--readout--oneoverf_rateints.fits')

    nim, dimy, dimx = np.shape(im)

    cut = im[:,0:100,1300:]

    for n in range(nim):
        for y in range(dimy):
            ref = np.array([im[n,y,0:4], im[n, y, -4:]])
            dc = np.mean(ref)
            dc = np.nanpercentile(im[n, y, :],2)
            im[n, y, :] = im[n, y, :] - dc

    fits.writeto('/genesis/jwst/userland-soss/loic_review/test.fits', im, overwrite=True)


#[]
im = fits.getdata('/genesis/jwst/userland-soss/loic_review/test.fits')
cut = im[:,0:100,1300:2000]
nim, dimy, dimx = np.shape(cut)

col = np.arange(dimx)
tim = np.arange(nim)

flux = np.zeros((nim, dimx))

for n in range(nim):
    for x in range(dimx):
        flux[n,x] = np.nansum(cut[n,:,x-50:x+50])

plt.figure()
plt.scatter(tim, flux[:,300])
plt.show()

