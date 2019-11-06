import webbpsf

# Create a PSF with WebbPSF for the driver_scene.py input
def psf(outputname='myPSF.fits'):
	nis = webbpsf.NIRISS()
	nis.filter='F430M'
	nis.pupil_mask='MASK_NRM'
	nis.calc_psf(outputname, fov_pixels=77, oversample=11)


if __name__ == '__main__':
	psf()
