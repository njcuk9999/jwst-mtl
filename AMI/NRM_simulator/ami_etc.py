#!/usr/bin/env python
# Code to run the 'private' NIRISS team AMI ETC code:
# 1. run ETC to get desired electron count
# 
# 2016-03-04 Anand Sivaramakrishnan anand@stsci.edu
# based on ETC from Deepashri Thatte and Anand Sivaramakrishnan and structure and mods from Johannes Salhlmann


import sys, os, argparse
import numpy as np
from astropy.io import fits
import pyami.simcode.utils as U
import pyami.etc.NIRISSami_apt_calc_v3 as etc

def main(argv):

    parser = argparse.ArgumentParser(description="This script provides access to a 'private' NIRISS-AMI ETC code.")
    parser.add_argument('-st','--spectralType',  type=str, default='A0V', choices=["A0V","M5V"])
    #help="""SPECTRALTYPE must be one of:
    #O3V O5V O7V O9V :: B0V B1V B3V B5V B8V :: A0V A1V A3V A5V :: F0V F2V F5V F8V :: G0V G2V G5V G8V :: K0V K2V K5V K7V :: M0V M2V M5V
    #B0III B5III :: G0III G5III :: K0III K5III :: M0III
    #O6I O8I :: B0I B5I :: A0I A5I :: F0I F5I :: G0I G5I :: K0I K5I :: M0I M2I -  (A0V is the default)""" 
    #TBD - need to package supporting psf files for other sptypes - anand@stsci.edu
    parser.add_argument('filt', type=str, help='filter to be used, must be one of F277W,F380M,F430M,F480M', \
                       choices=["F277W", "F380M", "F430M", "F480M", "f277w", "f380m", "f430m", "f480m"])
    parser.add_argument('targetMagnitude', type=float, help='Target apparent magnitude in selected filter')
    parser.add_argument('totalElectrons', type=float, help='Requested total number of collected electrons')
    
    args = parser.parse_args(sys.argv[1:])

    
    pathname = os.path.dirname(sys.argv[0])
    fullPath = os.path.abspath(pathname)
    pyamiDataDir = fullPath + '/pyami/etc/NIRISSami_apt_calcPSF/';

    # ETC inputs
    MAG_T  = args.targetMagnitude; #11.0
    TOT_E = args.totalElectrons; # 1e6
    
    filt = args.filt.upper()
    sptype = args.spectralType


    #  run ETC prototype
    mag_t = MAG_T
    tot_e = TOT_E
    sat_e = U.SAT_E
    report, params = etc.generatePSF(filt=filt, fov=31, osample=3, cr=etc.cr_from_mag(mag_t, U.ZP[filt]), \
                                      tot_e=tot_e, sat_e=sat_e, SRC = sptype, return_params=1,\
                                      DATADIR=pyamiDataDir)
    ngroups, nint, nint_ceil = params # not needed here...
    #print(params)
    print(report)
    print("\tTarget magnitude is %.2f, total  number of detected photons is %.0e\n" % (MAG_T, TOT_E))
        
    #file = open('save_etc.txt')
    np.savetxt('save_etc.txt', [ngroups, int(nint)+1, TOT_E, etc.cr_from_mag(mag_t, U.ZP[filt])])
    #file.close()
    #return ngroups, nint
    
if __name__ == "__main__":

    main(sys.argv[1:])    
    
    sys.exit(0)
