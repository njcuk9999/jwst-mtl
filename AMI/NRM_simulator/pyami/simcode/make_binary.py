#!/usr/bin/env python
#Code to simulate PSF of binary star pair with NIRISS NRM.  
#Jan 19 2016 Thatte mac os x
#Feb 10 2016 version emailed from Anand to Johannes


"""
This code is an updated version of make_binarypair_sim_jan14.py(January 14, 2014 version)
Updates - 4 dithers instead of 9 with updated pointings 
          TFRAME = 0.0745 sec instead of 2.688 sec, fov = 80 instead of 128
          Using 'nint' instead of exposures, using 'ngroups' instead of non_destructive_reads
          x and y are still arbitray similar to the earlier version
          Uses astropy.io.fits instead of pyfits
		  JS+AZG+AS: magA fluxratio sep pa filt ngroups nint Feb 10


		DT AS 21 September 2016  UNITS RATIONALIZATION
		Total number of photons in a Calibrator data cube
		(Target image has more photons because the companion is added to the point source)

		doc:  ngrp 3 nint 4....  5000 e- per frame  => ngrp*e/fr = 3 * 5000 = 15000
		4 integs = 15000 * 4 = 60000 e-
		"frame" is a saved readout of the  detector, so 
			nint 1 ngrp = 3 <arrayreset_arrayread_notsaved>//array_read//array_read//array_read  3 frames
			nint 2 ngrp = 3 <arrayreset_arrayread_notsaved>//array_read//array_read//array_read  3 frames
			nint 3 ngrp = 3 <arrayreset_arrayread_notsaved>//array_read//array_read//array_read  3 frames
			nint 4 ngrp = 3 <arrayreset_arrayread_notsaved>//array_read//array_read//array_read  3 frames
			12 frames total (in hardware) saved

		Fit a slope to each int 15000 - 5000 /(3-1) = 5000 e-/frame   -- this is the cube slice.

		nframes * e-/frame = 3 * 4 * 5000  =  60000 e-
		
		cube has nint slices, 4 slices, each  slice has 5000 e- per frame
		However 
			cube[0].sum() = 5000
			cube.sum() = 5000 * 4 = 20,000 accumulated e- in 4 frames
		Data that went into a slice of  the cube is ngroup readouts....
		so each slice used ngroup * slice value = 3 * 5000 = 15000 e-

		Convert cube.sum() to detected e- by multiplying by ngroup: 20000 * 3 = 60000 e- collected

	CONVERT cubes to e-, so after the initial slope fit, write out the cubes after multiplying by
	ngroup immediately before writing out the file

"""

import sys
import os
import numpy as np
from astropy.io import fits 
from astropy.io.fits import getheader
import webbpsf as wp
import pysynphot
#rom poppy import specFromSpectralType
from webbpsf import specFromSpectralType
import time

import pyami.simcode.utils as U   
 
####  osample = U.oversampling - now passed into the following routine...
# CREATE NOISELESS POINT SOURCE PSF USING WEBBPSF
def generate_starPSF(FILTER=None, fov=None, osample=None, spectraltype="A0V"):         
    niriss = wp.NIRISS()
    niriss.filter = FILTER
    niriss.pupil_mask = 'MASK_NRM'

    # set the WFE file to use...
    #iriss.pupilopd = ("OPD_RevV_niriss_162.fits", 3) old webbpsf
    path_to_webbpsf_data = wp.utils.get_webbpsf_data_path()
    opdfile = "OPD_RevW_ote_for_NIRISS_requirements.fits.gz"
    opd = os.path.join(path_to_webbpsf_data,"NIRISS", "OPD", opdfile) 
    opdslice = 3 # anywhere between 0 and 9:  10 realizations...
    niriss.pupilopd = (opd, opdslice)

    fov_pixels = fov # handoff no refactor
    oversample = osample # handoff no refactor

    niriss.pixelscale = U.pixscl # handoff no refactor
    src = specFromSpectralType(spectraltype)

    
    #Create an oversized array for star PSF. 
    #sf_fits = niriss.calcPSF(fov_pixels=fov + 4,oversample=osample,source=src,rebin=False,clobber=True) old call webbpsf
    psf_fits = niriss.calc_psf(oversample=oversample, source=src, fov_pixels=fov_pixels+4) # +4 because of jittering?
    psf_array = psf_fits[0].data
    psf_header = psf_fits[0].header
    print(psf_array.sum(), 'sum of star psf')  
    return psf_array, psf_header


def simulate_skydata(_trials, _binarystar_array, _cubename, _dithers,  _x_dith, _y_dith, 
                     ngroups, nint, filt='F430M', outDir = '', tmpDir = '', **kwargs):
    
    offset_x = kwargs['offset_x']
    offset_y = kwargs['offset_y']
    fluxratio = kwargs['fluxratio']
    fov = kwargs['fov']
    starmag = kwargs['starmag']
    magzeropoint = kwargs['magzeropoint']
    sptype = kwargs['sptype']
    osample = kwargs['osample']  #  oversampling set by the call to this function
    dim = kwargs['dim']
    utr = kwargs['utr']


    
    flux = 10**(-(starmag-magzeropoint)/2.5) # detected e-/sec
    if False:
        flux = 1.0e6/(U.tframe*0.14)
        print("hardwired debug flux e = %.2e detected e- / sec" % flux)
        ngroups, nint = 5, 3
        print("ngroups, nint", ngroups, nint)

    cube = np.zeros((nint, int(fov), int(fov)), np.float64)
    ipsarray_big = np.ones((U.ips_size*osample, U.ips_size*osample))

    #Create target data
    for p in range(_trials):
        #print 'Starting trial', p
        #CALCULATE LOCATIONS OF 4 DITHERS WITH 15 MAS ERROR ON 256 X 11 ARRAY
        mean_d, sigma_d = 0, U.dither_stddev_as * osample/U.pixscl
        x_dith_error = np.random.normal(mean_d,sigma_d, _dithers)
        x_dith_error_r = [int(round(n, 0)) for n in x_dith_error]
        #Accumulate dither errors
        x_dith_error_accum = np.cumsum(x_dith_error_r)
        dither_xcenter = [a + b for a, b in zip(_x_dith, x_dith_error_accum)] 
   
        y_dith_error = np.random.normal(mean_d,sigma_d, _dithers)
        y_dith_error_r = [int(round(n, 0)) for n in y_dith_error]
        #Accumulate dither errors
        y_dith_error_accum = np.cumsum(y_dith_error_r)
        dither_ycenter = [a + b for a, b in zip(_y_dith, y_dith_error_accum)]
    
        #print "\tPrinting commanded dither, accumulated error, final dither location"
        #print "  "
        #print "\tcommanded X dither", _x_dith
        #print "\tAccumulated X dither error", x_dith_error_accum
        #print "\tdither_xcenter", dither_xcenter      
        #print "  "
        #print "\tcommanded Y dither", _y_dith
        #print "\tAccumulated Y dither error", y_dith_error_accum
        #print "\tdither_ycenter", dither_ycenter
        #print "  "
    
        #POSITIONAL ERROR = DITHER + JITTER
    
        xjitter = list(range( _dithers))   #each of the 4 elements is an array of NINT jitters
        yjitter = list(range( _dithers))   #one set per dither location
        for i in range( _dithers):
            xjitter[i], yjitter[i] = U.jitter(nint, osample)
            #print '\t\tx jitter', xjitter[i]
            #print '\t\ty jitter', yjitter[i]
        xjitter_array = np.array(xjitter)
        x = list(range( _dithers))
        yjitter_array = np.array(yjitter)      
        y = list(range( _dithers))
        total_pos_error_x = list(range( _dithers))
        total_pos_error_y = list(range( _dithers))
        for i in range( _dithers):
            x[i] = dither_xcenter[i] + xjitter_array[i]
            y[i] = dither_ycenter[i] + yjitter_array[i] 
            total_pos_error_x[i] = x[i] - _x_dith[i]
            total_pos_error_y[i] = y[i] - _y_dith[i]
            #print " "
            #print '\ttotal positional error in X', total_pos_error_x[i]
            #print '\treal X pointing with dither and jitter', x[i]
            #print " "
            #print '\ttotal positional error in Y', total_pos_error_y[i]
            #print '\treal Y pointing with dither and jitter', y[i] 
            #print " "
            #
            for i_int,(ii,jj) in enumerate(zip(x[i],y[i])):
   
                #ips section for even FOV
                #print "ipsarray_big.shape", ipsarray_big.shape
                ips_section = ipsarray_big[ii-dim*osample:ii+dim*osample,jj-dim*osample:jj+dim*osample]
   
                binarystar_ips_array = ips_section * _binarystar_array
                #print '\tjitter error for the integration - x,y', ii-dither_xcenter[i],jj-dither_ycenter[i]
                binarystar_ips_array_sh = U.apply_padding_image(binarystar_ips_array,
                                            jj-dither_ycenter[i], ii-dither_xcenter[i], fov, osample)
                #print "\tinfo",dither_xcenter[i]-(dither_xcenter[i]//osample)*float(osample),osample-(dither_xcenter[i]
                # -(dither_xcenter[i]//osample)*osample)
                #print "\tinfo",dither_ycenter[i]-(dither_ycenter[i]//osample)*float(osample),osample-(dither_ycenter[i]
                # -(dither_ycenter[i]//osample)*osample) 
                im = np.pad(binarystar_ips_array_sh, 
                           [(int(dither_xcenter[i]-(dither_xcenter[i]//osample)*float(osample)),
                             int(osample-(dither_xcenter[i]-(dither_xcenter[i]//osample)*osample))),
                            (int(dither_ycenter[i]-(dither_ycenter[i]//osample)*float(osample)),
                             int(osample-(dither_ycenter[i]-(dither_ycenter[i]//osample)*osample)))],
                             mode='constant')
                #print '\tim.shape is', im.shape
                rebinned_array_81 = U.rebin(im, (osample,osample))
                rebinned_array = rebinned_array_81[0:80,0:80]
                
                ips_section_sh = U.apply_padding_ips(ips_section,jj-dither_ycenter[i],ii-dither_xcenter[i], fov, osample)
                #fits.writeto(tmpDir+'ips_section_sh'+str(i_int)+'.fits',ips_section_sh, clobber = True)
               
                im_ips=np.pad(ips_section_sh,
                             [(int(dither_xcenter[i]-(dither_xcenter[i]//osample)*float(osample)),
                               int(osample-(dither_xcenter[i]-(dither_xcenter[i]//osample)*osample))),
                              (int(dither_ycenter[i]-(dither_ycenter[i]//osample)*float(osample)),
                               int(osample-(dither_ycenter[i]-(dither_ycenter[i]//osample)*osample)))],
                               mode='constant', constant_values=(1,1))             
                #fits.writeto(tmpDir+'im_ips'+str(i_int)+'.fits',im_ips, clobber = True)
                rebinned_ips_flat_81 = U.rebin(im_ips, (osample,osample))/osample**2
                rebinned_ips_flat = rebinned_ips_flat_81[0:80,0:80]
                #print '\trebinned_ips_flat sum',rebinned_ips_flat.sum()
                rebinned_array /= rebinned_ips_flat
                counts_array_persec = flux * rebinned_array

                # up-the-ramp slices simulated, each readout or first&last
                ramp = U.create_ramp(counts_array_persec, fov, ngroups, utr)

                #fits.writeto(tmpDir+'ramp.fits',ramp, clobber = True)

                pflat = U.get_flatfield((fov,fov),outDir)

                # Do a double correlation subtraction or equivalent...
                integration = (U.create_integration(ramp) - U.darkcurrent - U.background) * pflat
                cube[i_int,:,:] = integration
                print('Integration[%d],'%i_int,  "%.1e e-" % cube[i_int,:,:].sum()) 
                #fits.writeto(tmpDir+'exp1.fits', integration, clobber = True)

   
            outfile = _cubename+str(p)+str(i)+".fits"
            print('\tcreated', _cubename+str(p)+str(i)+'.fits')
            (year, month, day, hour, minute, second, weekday, DOY, DST) =  time.gmtime()
            fitsobj = fits.HDUList()
            hdu = fits.PrimaryHDU(  )
            printhdr = hdu.header
      
            printhdr['INSTRUME'] =  'NIRISS'
            printhdr['pixscl'] = U.pixscl, 'Pixel scale (arcsec/pixel)'
            printhdr['NRMNAME'] =  'G7S6', 'Tuthill Anand Beaulieu Lightsey'
            printhdr['starmag'] = starmag,'Star magnitude'
            printhdr['sptype'] = sptype
            printhdr['NRM_GEOM'] =  'Hardcoded', 'Mathilde Beaulieu, PGT, AS'
            printhdr['NRM_X_A1'] =  0.00000, 'X coordinate (m) of NRM sub-aperture 0'          
            printhdr['NRM_Y_A1'] = -2.64000, 'Y coordinate (m) of NRM sub-aperture 0'         
            printhdr['NRM_X_A2'] = -2.28631, 'X coordinate (m) of NRM sub-aperture 1'          
            printhdr['NRM_Y_A2'] =  0.00000, 'Y coordinate (m) of NRM sub-aperture 1'          
            printhdr['NRM_X_A3'] =  2.28631, 'X coordinate (m) of NRM sub-aperture 2'          
            printhdr['NRM_Y_A3'] = -1.32000, 'Y coordinate (m) of NRM sub-aperture 2'          
            printhdr['NRM_X_A4'] = -2.28631, 'X coordinate (m) of NRM sub-aperture 3'          
            printhdr['NRM_Y_A4'] =  1.32000, 'Y coordinate (m) of NRM sub-aperture 3'          
            printhdr['NRM_X_A5'] = -1.14315, 'X coordinate (m) of NRM sub-aperture 4'          
            printhdr['NRM_Y_A5'] =  1.98000, 'Y coordinate (m) of NRM sub-aperture 4'          
            printhdr['NRM_X_A6'] =  2.28631, 'X coordinate (m) of NRM sub-aperture 5'          
            printhdr['NRM_Y_A6'] =  1.32000, 'Y coordinate (m) of NRM sub-aperture 5'          
            printhdr['NRM_X_A7'] =  1.14315, 'X coordinate (m) of NRM sub-aperture 6'          
            printhdr['NRM_Y_A7'] =  1.98000, 'Y coordinate (m) of NRM sub-aperture 6'   
            #rinthdr['nframe'] = 1,'Readout number of frames'  
            printhdr['ngroup'] = ngroups, 'Readout number of groups'  
            printhdr['nint'] = nint, 'number of integrations'  
            printhdr['framtime'] = U.tframe, "frame time for up the ramp"
            printhdr['UTR'] = utr, "1: up the ramp  0: last-first"
            printhdr['units'] = 'photoelectrons in the integration(s)'
            printhdr['PEAK_E'] = cube.max(), 'Brightest pixel in cube'
            printhdr['TOTAL_E'] = cube.sum(), 'Total number of e- in cube'
            printhdr['COMP_DX'] = offset_x*U.pixscl/float(osample), 'Companion separation in X, arcsec'
            printhdr['COMP_DY'] = offset_y*U.pixscl/float(osample), 'Companion separation in Y, arcsec'      
            printhdr['COMP_FR'] = fluxratio, 'Companion flux ratio' 
            printhdr['ffe_err'] =  U.flat_sigma*100, '% Flat field error stddev in %'
            printhdr['jitter'] =   U.jitter_stddev_as*1000, '1-axis jitter stddev mas'
            printhdr['dith_err'] = U.dither_stddev_as*1000, '1-axis dither placement stddev mas'
            printhdr['dithx%d'%i] = _x_dith[i]/osample, 'Commanded X dither (detpix in ipsarray)'
            printhdr['dithy%d'%i] = _y_dith[i]/osample, 'Commanded Y dither (detpix in ipsarray)'
            printhdr['dithx_r%d'%i] = dither_xcenter[i]/float(osample), 'Real X dither (detpix in ipsarray)'
            printhdr['dithy_r%d'%i] = dither_ycenter[i]/float(osample), 'Real Y dither (detpix in ipsarray)'
            printhdr['CODESRC'] = 'make_binary.py', '[thatte anand jsahlmann]@stsci.edu'

            # Append the header from psf_star.fits, likely created by WebbPSF
            wp_header = getheader(outDir +'star_array_fov80_%s.fits'%filt.lower())
            printhdr.extend(wp_header, update=True)

            #Delete and over-write keyword values written by WebbPSF
            del printhdr['PLANE1']
            del printhdr['DET_SAMP']
            printhdr['oversamp']= osample, 'Oversampling factor for MFT'
            printhdr['AUTHOR'] = '%s@%s' % (os.getenv('USER'), os.getenv('HOST')), 'username@host for calculation'
            printhdr['DATE'] = '%4d-%02d-%02dT%02d:%02d:%02d' %  (year, month, day, hour, minute, second), 'Date of calculation'
            hdu.data = cube
            fitsobj.append( hdu )
            fitsobj.writeto(outDir+outfile, overwrite = True)
            fitsobj.close()


            print("\nPeak pixel and total e- in each slice")
            for i in range(cube.shape[0]):
                print("%2d"%i, " %.2e"%(cube[i,:,:].max()), " %.2e"%(cube[i,:,:].sum()))
            print("")

            print("up-the-ramp %d"%utr, end=' ') 
            print("\nTotal e- in cube:", "%.2e   "%cube.sum())
