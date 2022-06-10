#!/usr/bin/env python
import sys
import astropy.io.fits as pyfits
import numpy as np
import crsim

def run(filein, ptype, outDir):
    #get the Exposure cube

    print('Running: addCRs2Exposure', filein, ptype, outDir)
    hdr = pyfits.getheader(filein)
    datain = pyfits.getdata(filein)

    #frame integration time (in real case will be read from fits header
    tframe = hdr['TFRAME']

    #Fixing the saturation problem for FULL images
    subarray = hdr['SUBARRAY']
    
    if subarray == 'FULL': 

        #Negative pixels
        ineg = np.where(datain < 0)
        datain[ineg] = np.abs(datain[ineg]) 
    
        #Normalize
        datain = datain/(np.max(datain))
    
         #Gain 
        datain = datain * 50000

    #NIRISS Detector 
    #noise files are in units of ADU so need gain to convert to electrons
    gain=1.62
    dataout, mask = crsim.addCRs(datain, tframe, ptype=ptype, f_ADC=gain) 

    #building output filename:
    n = filein.rfind('.')
    s = filein.rfind('/')
    print('Root of ouput filename is ', filein[s+1:n])
    fileout = outDir+filein[s+1:n]+'_CRs.fits'
    #writing fits with original header 
    hdr.add_history('Simulated CRs added by program addCRs2Exposure.py')
    pyfits.writeto(fileout,dataout,hdr,clobber=True)
    #writing CR mask
    maskfname = outDir+filein[s+1:n]+'_mask.fits'
    pyfits.writeto(maskfname,mask,None,clobber=True)


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('You must specify name of input exposure, whether simulation is for  SUNMIN or SUNMAX or FLARE, and the output directory')
        print('e.g. ./addCRs2Exposure.py cube.fits SUNMIN tmpdir')
        sys.exit()

        filein = sys.argv[1]
        ptype = sys.argv[2]
        outDir = sys.argv[3]

        run(filein, ptype, outDir)

