#!/usr/bin/env python
# -*- coding: utf-8 -*-
import webbpsf


# Create a PSF with WebbPSF for the driver_scene.py input
def psf(outputprefix='myPSF_',filter='F430M'):
    outputname=outputprefix+filter+'.fits'
    nis = webbpsf.NIRISS()
    nis.filter=filter
    nis.pupil_mask='MASK_NRM'
    nis.calc_psf(outputname, fov_pixels=77, oversample=11)

filters = 'F380M','F430M','F480M'

for filter in filters:
    psf(filter=filter)

