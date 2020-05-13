#!/usr/bin/env python
# Code to run observation simulation for binary:
# 1. run ETC to get desired electron count
# 2. simulate AMI data of a binary with desired separation and flux ratio
# 
# 2016-02-15 Johannes Sahlmann
# based on ETC and binary simulation codes from Deepashri Thatte and Anand Sivaramakrishnan
#
# 2016-09-27 anand@stsci.edu 
#     Create calibrator observation with same readout as target,  same total flux
#     Places calibrator at location of first maximum of target (in np.where(tgt==tgt.max()) list)
#     rework driver_scene to create cal and tgt data cubes
#     rework make_scene to match reworked make_binary 
#         (utils create_ramp & create_integration changed,
#         new up-the-ramp handling to match binary simulation)
#     name cal and tgt data cubes c_* and t_*


import sys, os, argparse
import numpy as np
from astropy.io import fits 

import pyami.simcode.make_scene as scenesim 
import importlib
importlib.reload(scenesim)       
import pyami.simcode.utils as U   
importlib.reload(U)     


def main(argv):
    
    parser = argparse.ArgumentParser(description='''
        This script simulates the observation of a sky scene with a (Webb)PSF supplied by the user.
        
        WARNING - WE ASSUME A SINGLE DOMINANT POINT SOURCE IN THE SKY DATA:
        We match the countrate to that of the dominant point source in skydata.
        Thus if the skydata point source accounts for 0.99 of of target flux,
        then we place a unit delta function into caldata, and request a count
        rate of 0.99 * the user-requested count rate for creating calibrator
        data.                  
        
        To generate a calibration star observation, use a 'delta function' single positive
        pixel in an otherwise zero-filled array as your sky fits file.
        
        ''', formatter_class=argparse.RawTextHelpFormatter)
        
    parser.add_argument('-t','--target_dir',  type=str, default='niriss-ami_out/', help='output directory path (relative to home directory)')
    parser.add_argument('--output_absolute_path',  type=str, default=None, help='absolute output directory path, if specified it overrides --target_dir')
    
    parser.add_argument('-o','--overwrite',  type=int, default=0, help='overwrite yes/no, default 0 (no)', choices=[0,1])
    parser.add_argument('-utr','--uptheramp',  type=int, default=0, help='generate up-the-ramp fits file? yes/no, default 0 (no)', choices=[0,1])
    parser.add_argument('-f', '--filter', type=str, help='filter name (upper/lower case)', choices=["F277W", "F380M", "F430M", "F480M"])
    parser.add_argument('-p','--psf', type=str, help='absolute path to oversampled PSF fits file. Spectral type set in this')
    parser.add_argument('-s','--sky', type=str, help='absolute path to oversampled sky scene fits file, normalized to sum to unity')
    parser.add_argument('-os','--oversample', type=int, help='sky scene oversampling (must be odd integer number)', choices=[1,3,5,7,9,11,21])
    parser.add_argument('-I','--nint', type=int, default=1, help='number of integrations (IR community calls these exposures sometimes)')
    parser.add_argument('-G','--ngroups', type=int, default=1, help='number of up-the-ramp readouts')
    parser.add_argument('-c','--create_calibrator', type=int, default=1, help='create calibrator observation yes/no default 1 (yes)', choices=[0,1])
    parser.add_argument('-cr','--countrate', type=float, help='Photon count rate on 25m^2 per sec in the bandpass (CRclearp in ami_etc output)',)
    parser.add_argument('-tag','--tag', type=str, default='', help='Tag to include in the names of the produced files')
    parser.add_argument('--uniform_flatfield', type=int, default='0',help='Generate random-noise flatfield (default) or uniform noiseless flatfield (if set to 1) ', choices=[0,1])
    parser.add_argument('--random_seed' ,type=int, default=None, help='Random seed for all noise generations (seed is altered for every integration), allows for well-controlled simulations')
    parser.add_argument('--flatfield_dir' ,type=str, default=None, help='Directory for simulated flatfield. Defaults to targetDir.')
    parser.add_argument('--overwrite_flatfield',  type=int, default=0, help='Overwrite simulated flatfield. Defaults to No.', choices=[0,1])
    parser.add_argument('-v','--verbose',  type=int, default=0, help='Verbose output to screen. Default is off', choices=[0,1])
    parser.add_argument('--apply_dither',  type=int, default=1, help='Dither the observations. Default is on', choices=[0,1])
    parser.add_argument('--apply_jitter',  type=int, default=1, help='Include pointing errors in the observations. Default is on', choices=[0,1])
    parser.add_argument('--include_detection_noise',  type=int, default=1, help='Include photon noise, read noise, background noise, and dark current. Default is on', choices=[-1,0,1])
    
    
    args = parser.parse_args(argv)

    print('*** JWST NIRISS scene simulation of NRM observation ***')

    target_dir = args.target_dir 
    out_dir_0 = os.path.join(os.getenv('HOME') , target_dir);

    output_absolute_path = args.output_absolute_path
    if output_absolute_path is not None:
        #       override target_dir
        out_dir_0 = output_absolute_path
        


    overwrite = args.overwrite
    uptheramp = args.uptheramp
    calibrator = args.create_calibrator
    file_tag = args.tag
    uniform_flatfield = args.uniform_flatfield
    random_seed = args.random_seed
    flatfield_dir = args.flatfield_dir
    overwrite_flatfield = args.overwrite_flatfield
    countrate = args.countrate
    filt = args.filter
    psffile = args.psf
    skyfile = args.sky
    osample = args.oversample
    verbose = args.verbose  
    apply_dither = args.apply_dither
    apply_jitter = args.apply_jitter
    include_detection_noise = args.include_detection_noise

    nint = args.nint        # TBD: calculate internally to save the user prep time doing ETC work
    ngroups = args.ngroups  # TBD: calculate internally to save the user prep time doing ETC work


    if flatfield_dir is None:
        flatfield_dir = out_dir_0

    if verbose:
        print(argv)
        print("countrate input as %.2e photons/sec on 25m^2 primary in filter bandpass" % args.countrate)
    # rebin sky_conv_psf image to detector scale, use max of detector array to calculate nint, ngroups, data-collect-time

        
    # generate images  
    if verbose:
        print("oversampling set in top level driver to %d" % osample)
        
    trials = 1
    
    out_dir = os.path.join(out_dir_0 , '%s/' % (filt));
#         tmpDir = os.path.join(outDir0 , 'tmp/')
    if verbose:
        print(('Output directory set to %s' % out_dir))
    # NB outDir must exist to contain input files - clean up organization later?
    for dd in [out_dir]:#,tmpDir]:
        if not os.path.exists(dd):
            os.makedirs(dd)

    

    # FEEDER FOR SIMULATION - read in pre-made psf made by WebbPSF (or any other way)
    # File sizes: 
#     psfdata, psfhdr = fits.getdata(os.path.join(out_dir_0,psffile), header=True)
#     skydata, skyhdr = fits.getdata(os.path.join(out_dir_0,skyfile), header=True)
    psfdata, psfhdr = fits.getdata(psffile, header=True)
    skydata, skyhdr = fits.getdata(skyfile, header=True)
    skydata = skydata / skydata.sum()  # normalize sky data total to unity!
    skydata = skydata * countrate
    if verbose:
        print("psfdata", psfdata.shape, "totals %.2e (NRM throughput / full aperture throughput)"%psfdata.sum())
        print("skydata", skydata.shape, "totals %.2e (photons / s on 25^m in band)"%skydata.sum())

    caldata = np.zeros(skydata.shape, np.float64)
    maxloc = np.where(skydata==skydata.max())
    ptsrcfraction = skydata[maxloc]/skydata.sum()
    caldata[maxloc[0][0], maxloc[1][0]] = countrate#ptsrcfraction * countrate
    
    # DEFINE DITHER POINTING in det pixels
    ipsoffset = U.ips_size//2 - (skydata.shape[0]//osample)//2
    x_dith, y_dith = [(skydata.shape[0]//2)/osample + ipsoffset,], \
                     [(skydata.shape[0]//2)/osample + ipsoffset,]
    dithers = len(x_dith)
    if verbose:
        print("x_dith, y_dith", x_dith, y_dith)

        
    # now convert to oversampled pixels for the calculation:
    x_dith[:] = [(x*osample - osample//2+1) for x in x_dith]
    y_dith[:] = [(y*osample - osample//2+1) for y in y_dith]
    """ python 2to3 conversion"""
    print(type(x_dith))
    x_dith = np.array(x_dith).astype(np.int)
    y_dith = np.array(y_dith).astype(np.int)
    print(type(x_dith))

    if apply_dither == 0:
        # -osample/2 offset introduced so that produced target image is aligned with input PSF
        x_dith = [ (skydata.shape[0] )//2 - np.floor(osample/2)] * dithers
        y_dith = [ (skydata.shape[0] )//2 - np.floor(osample/2)] * dithers

    basename_sky = os.path.basename(skyfile)
    basename_psf = os.path.basename(psffile)
    file_name_seed = basename_sky.replace(".fits","__") + basename_psf.replace('.fits','_') + file_tag + '_'
    cubename = "t_" + file_name_seed
    if (not os.path.isfile(os.path.join(out_dir,cubename+'00.fits'))) | (overwrite): 
        scenesim.simulate_scenedata(trials, 
                                    skydata, psfdata, psfhdr, cubename, osample,
                                    dithers, x_dith, y_dith, apply_dither, apply_jitter,
                                    ngroups, nint, U.tframe, filt,include_detection_noise,
                                    out_dir, flatfield_dir, verbose, uptheramp,uniform_flatfield=uniform_flatfield,overwrite=overwrite,random_seed=random_seed,overwrite_flatfield=overwrite_flatfield)
    if calibrator:
        cubename = "c_" + file_name_seed
        
        # flatfield has been simulated in target simulation, it should therefore not be overwritten
        overwrite_flatfield = 0

        # noises should be independent between target and calibrator integrations
        random_seed_calibrator = random_seed + 1
        
        if (not os.path.isfile(os.path.join(out_dir,cubename+'00.fits'))) | (overwrite): 
            scenesim.simulate_scenedata(trials, 
                                        caldata, psfdata, psfhdr, cubename, osample,
                                        dithers, x_dith, y_dith, apply_dither, apply_jitter,
                                        ngroups, nint, U.tframe, filt,include_detection_noise,
                                        out_dir, flatfield_dir, verbose, uptheramp,uniform_flatfield=uniform_flatfield,overwrite=overwrite,random_seed=random_seed_calibrator,overwrite_flatfield=overwrite_flatfield)

    print('Scene simulation done!')
    
if __name__ == "__main__":
#     print sys.argv
    main(sys.argv[1:])        
    sys.exit(0)

