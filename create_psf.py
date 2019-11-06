# Create a PSF with WebbPSF for the driver_scene.py input

def psf(outputname='myPSF.fits'):
	import webbpsf
	nis = webbpsf.NIRISS()
	nis.filter='F430M'
	nis.calc_psf(outputname, oversample=11)

psf()    
