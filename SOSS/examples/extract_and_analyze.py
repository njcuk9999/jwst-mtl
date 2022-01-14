import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt

from jwst.extract_1d import Extract1dStep

import os

rateints_file = '/genesis/jwst/userland-soss/loic_review/GTO/wasp52b/IDTSOSS_clear_noisy_rateints.fits'
spectrum_file = '/genesis/jwst/userland-soss/loic_review/GTO/wasp52b/extracted_result.fits'
model_file = '/genesis/jwst/userland-soss/loic_review/test_modeloutput.fits'
flux2d_file = '/genesis/jwst/userland-soss/loic_review/test_flux.fits'

#rateints_file = '/genesis/jwst/userland-soss/loic_review/GTO/wasp52b/IDTSOSS_f277_noisy_rateints.fits'



if os.path.isfile(spectrum_file) is False:
    result = Extract1dStep.call(rateints_file, soss_transform=[0,0,0], soss_atoca=True, soss_modelname=model_file)
    result.write(spectrum_file)

data = fits.open(spectrum_file)
# spectra are stored at indice 1 (order 1), then 2 (order2) then 3 (order 3) then 4 (order 1, 2nd time step), ...
nint = data[0].header['NINTS']
norder = 3

wavelength = np.zeros((nint, norder, 2048))
flux = np.zeros((nint, norder, 2048))
order = np.zeros((nint, norder, 2048))
integ = np.zeros((nint, norder, 2048))

for ext in range(np.size(data)-2):
    m = data[ext+1].header['SPORDER']
    i = data[ext+1].header['INT_NUM']
    wavelength[i-1, m-1, :] = data[ext+1].data['WAVELENGTH']
    flux[i-1, m-1, :] = data[ext+1].data['FLUX']
    #print(m, i, np.size(w))

# Normalize each wavelength
flux = flux / np.nanmedian(flux, axis=0)

hdu = fits.PrimaryHDU(flux[:,0,:])
hdu.writeto(flux2d_file, overwrite=True)

plt.figure()
for i in range(nint):
    plt.plot(wavelength[i,0], flux[i,0]+100*i)
plt.show()

