import numpy as np

import matplotlib.pyplot as plt

from scipy import ndimage

from time import time

from sys import path as sys_path
from os import path as os_path
from os import listdir, mkdir

from astropy.io import fits

import webbpsf

# These contain the functions used to determine the tilt
# of a PSF using cross-correlations (from Loic Albert)
sys_path.append("../../github/jwst-mtl/SOSS/PSFs/")
import soss_get_tilt_Frost as sgt




#--------------------------------------------------------------------------------#

def get_webbpsf( wavelist=None , view='dtect' , save=False , savepath=None
               , return_fits_HDUL=True , doPrint=False , doPlot=False
               , **wpsf_args):
    '''Utility function which calls the WebbPSF package to create monochromatic
PSFs. Currently, this code can only produce PSFs for NIRISS-SOSS obserations.


Parameters
----------

wavelist : list (or) float
    Float (or list) of wavelength(s) in meters for which to generate PSF(s)

view : str
    Represents the desired orientation of the PSFs.
    'dtect' : outputs the native detector orientation for NIRISS-SOSS
                  (horizontal axis = spatial)
                  (vertical axis = spectral (low to high wavelength))
    'dms'  :  outputs the DMS orientation of the NIRISS-SOSS detector
                  (horizontal axis = spectral (high to low wavelength))
                  (vertical axis = spatial (traces are curved upwards))

save, savepath : bool & string
    Whether to save PSFs to disk. If save=True, savepath is the output directory
doPrint, doPlot : boolean
    Equivalent to enabling 'verbose' mode, which decides whether to print/plot info

**wpsf_args : optional args
              Determine the instrument characteristics 

    - instrument : string
        Specify the desired JWST instrument 
    - filter : string
    - pupil_mask : string
    - pupil_opd : string
    - wfe_real : int
        Index of wavefront realization/error to use for the PSF
        (if non-default WFE realization is desired).
    - psf_dim : int
        Horizontal and vertical dimensions of the PSF in native pixels
    - oversamp : int
        Detector pixel oversampling factor
    - jitter, jitter_sigma : str and float
        Determines the way jitter is modeled onto the PSF images
        (example: jitter='gaussian' and jitter_sigma=0.007)


Returns
-------
None : NoneType
    If PSFs are written to disk.
psf : np.ndarray or FITS HDUL object
    If only one PSF is produced (one wavelength was supplied).
psf-list : list
    If many PSFs are produced (more than one wavelength supplied)
    List with elements of type 'psf' (either np.ndarray or FITS HDUL object).
'''

    
    if wavelist is None:
        # List of wavelengths to generate PSFs for
        wavelist = np.linspace(0.5, 5.2, 95) * 1e-6

######## Initiate wpsf_args elements if not already present #################
    if 'instrument' not in wpsf_args:     wpsf_args['instrument'] = 'NIRISS'
    if 'filter' not in wpsf_args:         wpsf_args['filter'] = 'CLEAR'
    if 'pupil_mask' not in wpsf_args:     wpsf_args['pupil_mask'] = 'GR700XD'
    if 'pupil_opd_file' not in wpsf_args: wpsf_args['pupil_opd_file'] = 'OPD_RevW_ote_for_NIRISS_predicted.fits.gz'
    if 'wfe_real' not in wpsf_args:       wpsf_args['wfe_real'] = 0
    if 'psf_dim' not in wpsf_args:        wpsf_args['psf_dim'] = 128
    if 'oversamp' not in wpsf_args:       wpsf_args['oversamp'] = 10
    if 'jitter' not in wpsf_args:         wpsf_args['jitter'] = None
    if 'jitter_sigma' not in wpsf_args:   wpsf_args['jitter_sigma'] = 0.0
    
######## Initialize save directory if necessary ###################
    if save:
        if savepath != None:
            PSF_DIR = savepath
        else:
            #PSF_DIR = '/genesis/jwst/jwst-ref-soss/monochromatic_PSFs/'
            PSF_DIR = 'monochromatic_PSFs/'
        if not os_path.isdir(PSF_DIR):
            mkdir(PSF_DIR)
                

################### Select instrument characteristics #############################
    if wpsf_args['instrument'] == "NIRISS":
        instrument = webbpsf.NIRISS()
        # Override the default minimum wavelength of 0.6 microns
        instrument.SHORT_WAVELENGTH_MIN = 0.5e-6
        # Set correct filter and pupil wheel components
        instrument.filter = wpsf_args['filter']
        instrument.pupil_mask = wpsf_args['pupil_mask']
        # Change the WFE realization if you so desire (accepted values 0 to 9)
        instrument.pupilopd = (wpsf_args['pupil_opd_file'], wpsf_args['wfe_real'])
        # Set telescope jitter options
        instrument.options['jitter'] = wpsf_args['jitter']
        instrument.options['jitter_sigma'] = wpsf_args['jitter_sigma']
        
    else:
        raise ValueError("Instrument "+wpsf_args['instrument']+" is not included in 'get_webbpsf' options")


############ Loop through all wavelengths to generate PSFs ###################
    if save is False:
        psf_list = []
    # If wavelist is not a list or np.array (meaning only 1 wave), then create list
    was_list = True
    if not isinstance(wavelist, list) and not isinstance(wavelist,np.ndarray):
        wavelist = [wavelist];  was_list = False
    
    
    for wave in wavelist:
        
        # Get instrument PSF for current wavelength
        if doPrint: print('Calculating PSF at wavelength ', wave/1e-6, ' microns')
        psf = instrument.calc_psf(monochromatic=wave, fov_pixels=wpsf_args['psf_dim'],
                              oversample=wpsf_args['oversamp'], display=doPlot)
        
        # Select which orientation the detector is viewed from
        if view == 'dtect':
            pass # (meaning instrument.calc_psf() returns 'dtect' view by default)
        elif view == 'dms':
            # Visually, 'dms' view is a 90° counter-clockwise rotation about the
            # center of 'dtect' that has been vertically flipped (along spatial axis)
            psf[0].data = np.flipud(np.fliplr(np.transpose(psf[0].data)))

        # Save psf realization to disk if desired
        if save is True:
            
            filename = '{:}SOSS_os{:d}_{:d}x{:d}_{:5f}_wfe{:d}.fits'.format(
                PSF_DIR, wpsf_args['oversamp'], wpsf_args['psf_dim'], wpsf_args['psf_dim']
                , wave*1e+6, wpsf_args['wfe_real']   )
            psf.writeto(filename, overwrite=True)
        else:
            if return_fits_HDUL:
                psf_list.append(psf)
            else:
                psf_list.append(psf[0].data)
        
        
############# Return ################
    if save is False:
        if was_list is False:
            return psf_list[0]
        else:
            return psf_list
    else:
        return None



    
    


#--------------------------------------------------------------------------------#

def get_webbpsf_tilt( PSF , oversamp=10 , view='dms'
                    , mode='ccf' , fitrad=20 , spatbox_size=23 , specbox_size=20
                    , doPlot=False):
    '''Function used to calculate/estimate the tilt
(with respect to the spatial axis) in NIRISS-SOSS PSFs


Parameters
----------

PSF : string (or) np.ndarray (or) FITS HDUL
    The PSF
    If string, represents the path+name of a '.fits' file to open
    If np.ndarray, represents a 2D array of the PSF
    If FITS HUDL, well the designation pretty much says what it is

oversamp : int
    Detector pixel oversampling factor

view : string
    Represents the desired orientation of the PSFs.
    'dtect' : outputs the native detector orientation for NIRISS-SOSS
                  (horizontal axis = spatial)
                  (vertical axis = spectral (low to high wavelength))
    'dms'  :  outputs the DMS orientation of the NIRISS-SOSS detector
                  (horizontal axis = spectral (high to low wavelength))
                  (vertical axis = spatial (traces are curved upwards))

mode : string
    Represents what technique is used to measure tilt.
    Currently, only the cross-correlation function method
    developped by Loic Albert is implemented
   
spatbox_size, specbox_size : int
    Represents the box dimensions of the region where tilt is calculated
    (centered about the center of the PSF). With mode='ccf', spatbox_size
    is not used.

doPlot : boolean
    Equivalent to 'verbose' mode, which decides whether to plot helpful stuff


Returns
-------
tilt : float
    The tilt with respect to the spatial axis of the PSF trace
'''
    
######## Open/Initialize the psf image as a numpy array ############
    if isinstance(PSF,np.ndarray) or isinstance(PSF,list):
        psf = np.array(PSF)
        PSF = None #clear up memory space
    elif isinstance(PSF,str):
        with fits.open(PSF) as hdul:
            psf = hdul[0].data
    else:# assume it is a .FITS HDUL object
        psf = PSF[0].data
    
    # Force even dimensions for the PSFs. Because I can.
    if psf.shape[0]%2 != 0 or psf.shape[1]%2 != 0:
        raise ValueError("PSF shape of "+str(psf.shape)+"."
                         +" For now, 'get_webbpsf_tilt' only works on PSFs with even dimensions."
                         +" This constraint was not respected here.")
    
######## Define psf center and extraction box limits ###############
    center = int(psf.shape[0]/2)
    spec_left = int(center - (oversamp*specbox_size/2))
    spec_right = int(center + (oversamp*(1+specbox_size/2)))
    # These last 2 mesures are not used if mode=='ccf'
    spat_left = int(center - (oversamp*spatbox_size/2))
    spat_right = int(center + (oversamp*(1+spatbox_size/2)))

###### Find tilt ########################################
    if mode=='ccf':
        lspat_left = center - int(12*oversamp);  lspat_right = center - int(4*oversamp)
        rspat_left = center + int(5*oversamp);   rspat_right = center + int(13*oversamp)
        if view == 'dtect':
            axis=1
            # Define two arrays representing vertical cuts through both peaks of the PSF
            leftslice = np.sum(psf[spec_left:spec_right, lspat_left:lspat_right], axis=axis)
            rightslice = np.sum(psf[spec_left:spec_right, rspat_left:rspat_right], axis=axis)
        elif view == 'dms':
            axis=0
            # Define two arrays representing vertical cuts through both peaks of the PSF
            leftslice = np.sum(psf[ lspat_left:lspat_right , spec_left:spec_right ], axis=axis)
            rightslice = np.sum(psf[ rspat_left:rspat_right , spec_left:spec_right ], axis=axis)
        
        if doPlot: # plot the boxes used to calculate leftslice and rightslice
            plt.figure(figsize=(7,7));  plt.title(view + " and its boxes")
            plt.imshow(psf , origin='lower' , zorder=1)
            if view == 'dtect':
                plt.gca().add_patch( plt.Rectangle( (lspat_left , spec_left)
                                                , lspat_right - lspat_left
                                                , spec_right - spec_left
                                                , fc='none',ec='green' , zorder=10 , lw=2
                                                )
                                   )
                plt.gca().add_patch( plt.Rectangle( (rspat_left , spec_left)
                                                , rspat_right - rspat_left
                                                , spec_right - spec_left
                                                , fc='none',ec='red' , zorder=10 , lw=2
                                                )
                                   )
            elif view == 'dms':
                plt.gca().add_patch( plt.Rectangle( (spec_left , lspat_left)
                                                , spec_right - spec_left
                                                , lspat_right - lspat_left
                                                , fc='none',ec='green' , zorder=10 , lw=2
                                                )
                                   )
                plt.gca().add_patch( plt.Rectangle( (spec_left , rspat_left)
                                                , spec_right - spec_left
                                                , rspat_right - rspat_left
                                                , fc='none',ec='red' , zorder=10 , lw=2
                                                )
                                   )
        
############ Calculate tilt ##############################
        # The number of columns between both slices
        dx = np.mean([rspat_left, rspat_right]) - np.mean([lspat_left, lspat_right]) 
        # Perform a cross-correlation correlation and fit its peak using a gaussian
        ccf2 = sgt.fitCCF(leftslice, rightslice, fitfunc='gauss', fitradius=fitrad, makeplot=doPlot)
        # The monochromatic tilt is then:
        tilt = np.rad2deg(np.arctan(ccf2 / dx))
    
    return tilt    
    
    


    
    
    
#--------------------------------------------------------------------------------#

def rotate_and_center_crop( image,rotation,crop_dims , reshape=False
                            , doPlot=False):
    '''Function used to rotate an image using a given rotation and direction,
as well as crop that image about it's center.


Parameters
----------

image : np.ndarray
    2D image to rotate and crop

rotation : int
    angle (in degrees) by which to rotate the image.
    Positive rotation value goes clockwise.
    Negative rotation goes counter-clockwise.
    
crop_dims : [int,int]
    List with 2 elements that specify the dimensions of the cropped image
    
reshape : boolean
    argument supplied to scipy.ndimage.rotate
    Recommended that it remains False to guarantee an error-free runtime.
    Refer to the scipy docs for info. 

doPlot : boolean
    Equivalent to 'verbose' mode, which decides whether to plot helpful stuff


Returns
-------
image_rot_crop : np.ndarray
    The rotated and cropped 2D np.array 
'''
    # Rotate image
    image_rot = ndimage.rotate(image, rotation, reshape=reshape)
    if doPlot:
        plt.figure(); plt.title("Raw Rotated Image")
        plt.subplot(1,2,1); plt.imshow(image_rot,origin='lower')
        plt.subplot(1,2,2); plt.imshow(image_rot-image,origin='lower')
    image_crop = np.empty((crop_dims[0],crop_dims[1]))
    
    # Define center of image
    center_indices = [ image_rot.shape[0]/2.0 , image_rot.shape[1]/2.0 ]
    
    # Get x and y slices based on crop_dims
    slices = [None] * 2
    dim_str = ['X','Y']
    for i in range(len(image_rot.shape)):
        if (image_rot.shape[i] % 2 == 0 and crop_dims[i] % 2 == 0) \
        or (image_rot.shape[i] % 2 != 0 and crop_dims[i] % 2 != 0):
                low = int(center_indices[i])-int(crop_dims[i]/2)
                high = int(center_indices[i])+int(crop_dims[i]/2)
                slices[i] = slice( low , high )
        else:
            raise Exception(dim_str[i]+"-dimensions of rotated and cropped images are " +str(image_rot.shape[i])
                            + " and "+str(image_crop.shape[i])+". Dimensions must be either both even or odd"
                            + " in the current implementation of 'rotate_and_center_crop'")
    
    # Crop image about it's center
    image_rot_crop = image_rot[slices[0],slices[1]]
    return image_rot_crop







#--------------------------------------------------------------------------------#

def generate_and_rotate_webbpsf( PSF_arg , wanted_tilt
                  , known_webbpsf_tilt = None
                  , save=False , savepath=None , **kwargs ):
    '''Function that takes a PSF (or an argument to create one)
and rotates it according to a desired tilt.


Parameters
----------

PSF_arg : string (or) np.ndarray (or) float
    If string, represents the path+name of a '.fits' file to open
    If np.ndarray, represents the PSF itself
    If float, represents a wavelength value from which a PSF
        is generated using 'get_webbpsf'.
        This is the preferred argument type, since it allows
        for the creation of a .FITS file with proper headers
        using functions in the 'webbpsf' package
        
wanted_tilt : float
    Represents the desired angle (in degrees) of the PSF.
    Remember that the angle is w.r.t the spatial axis
    
save, savepath : bool & string
    Whether to save PSFs to disk. If save=True, savepath is the output directory

**kwargs : optional arguments
    - psf_dim : int
        the size of both dimensions of the desired PSF
    - psf_dim_pad : int
        the size of both dimensions of the padded PSF
        (to be cropped to the dimensions of the desired PSF)
    - oversamp : int
        Detector pixel oversampling factor
    - view : string
        Represents the desired orientation of the PSFs.
        'dtect' : outputs the native detector orientation for NIRISS-SOSS
                      (horizontal axis = spatial)
                      (vertical axis = spectral (low to high wavelength))
        'dms'  :  outputs the DMS orientation of the NIRISS-SOSS detector
                      (horizontal axis = spectral (high to low wavelength))
                      (vertical axis = spatial (traces curved upwards))
    - doPlot, doPrint : boolean
        Equivalent to 'verbose' mode. Decides whether to plot/print helpful stuff
        
**wpsf_args : optional args used in 'get_webbpsf()' that are contained in **kwargs
    (see 'get_webbpsf()' docstring for description of these args)
    (instrument, filter, pupil_mask, pupil_opd_file, wfe_real, jitter, jitter_sigma)

**kwargs (continued) : optional args used in 'get_webbpsf_tilt()'
    (see 'get_webbpsf_tilt()' docstring for description of these args)
    (mode, spatbox_size, specbox_size)
        


Returns
-------
image_rot_crop : np.ndarray
    The rotated and cropped 2D np.array 
'''

################## Initiate kwargs if not already present #############################
    if 'psf_dim' not in kwargs:     kwargs['psf_dim'] = 128
    if 'psf_pad_dim' not in kwargs: kwargs['psf_pad_dim'] = 160
    if 'oversamp' not in kwargs:    kwargs['oversamp'] = 10
    if 'view' not in kwargs:        kwargs['view'] = 'dms'
    if 'doPlot' not in kwargs:      kwargs['doPlot'] = False
    if 'doPrint' not in kwargs:     kwargs['doPrint'] = False
    
    # Initiate args for 'get_webbpsf' if not already present in kwargs
    # These args are used if PSF_arg represents a wavelength
    if 'instrument' not in kwargs:     kwargs['instrument'] = 'NIRISS'
    if 'filter' not in kwargs:         kwargs['filter'] = 'CLEAR'
    if 'pupil_mask' not in kwargs:     kwargs['pupil_mask'] = 'GR700XD'
    if 'pupil_opd_file' not in kwargs: kwargs['pupil_opd_file'] = 'OPD_RevW_ote_for_NIRISS_predicted.fits.gz'
    if 'wfe_real' not in kwargs:       kwargs['wfe_real'] = 0
    if 'jitter' not in kwargs:         kwargs['jitter'] = 'gaussian'
    if 'jitter_sigma' not in kwargs:   kwargs['jitter_sigma'] = 0.007
    wpsf_keys = ['oversamp','instrument','filter','pupil_mask','pupil_opd_file','wfe_real','psf_dim'
                ,'jitter','jitter_sigma']
    wpsf_args = {key:kwargs[key] for key in wpsf_keys if key in kwargs}
    if kwargs['doPrint']:
        print("keyword args supplied to 'get_webbpsf':")
        print(wpsf_args); print()
    wpsf_args['psf_dim'] = kwargs['psf_pad_dim']

    # Initiate args for 'get webbpsf_tilt' if not already present in kwargs
    if 'mode' not in kwargs:         kwargs['tilt_measurement_mode'] = 'ccf'
    if 'spatbox_size' not in kwargs: kwargs['spatbox_size'] = 23
    if 'specbox_size' not in kwargs: kwargs['specbox_size'] = 20
    
        

############ Open/Initialize the psf image as a numpy array ###################
    # If the 'PSF_arg' argument is either a file or an array/list,
    # then the code assumes that the supplied PSF is already padded,
    # such that it ready to be rotated and cropped.
    is_hdu = False
    if isinstance(PSF_arg,str):
        with fits.open(PSF_arg) as a:
            psf = a[0].data
    elif isinstance(PSF_arg,np.ndarray) or isinstance(PSF_arg,list):
        psf = np.array(PSF_arg)
    # Else, the 'PSF_arg' argument is treated as a wavelength value, and 
    # the 'get_webbpsf' function is called to create that wave's padded PSF
    else:
        wave = PSF_arg
        psf_hdul = get_webbpsf( wave , view=kwargs['view']
                         , doPrint=kwargs['doPrint'] , doPlot=False
                         , **wpsf_args )
        psf = psf_hdul[0].data
        is_hdu = True
    

################ Get tilt of the padded PSF ##########################
    if known_webbpsf_tilt != None:
        current_tilt = known_webbpsf_tilt
    else:
        current_tilt = get_webbpsf_tilt( psf , oversamp=kwargs['oversamp'] , mode=kwargs['tilt_measurement_mode']
                                   , spatbox_size=kwargs['spatbox_size'] , specbox_size=kwargs['specbox_size']
                                   , view=kwargs['view'] , doPlot=kwargs['doPlot']
                                   , fitrad=(kwargs['oversamp'] if kwargs['oversamp'] >= 6 else 6)
                                   )
    
    rotation = wanted_tilt - current_tilt
    if kwargs['view'] == 'dtect' :
        # Based on DMS tilt convention by Loic Albert and William Frost
        # Since going from 'dtect' to 'dms' involves a symetrical flip
        # along the spatial axis, what would be measured as a positive
        # rotation in 'dms' view in instead a negative rotation in 'dtect'.
        # A correction is therefore necessary to maintain the DMS convention
        # in 'dtect' view.
        rotation = -rotation
    
    if kwargs['doPrint']:
        print("Current tilt (°) is",round(current_tilt,4),". Rotate",round(rotation,4)
             ,"to get to target of ",wanted_tilt  )
        print()

############## Rotate PSF and crop it to the expected size #################
    cropodims = [ kwargs['oversamp']*kwargs['psf_dim'] ] * 2
    new_psf = rotate_and_center_crop( psf , rotation , crop_dims=cropodims
                                    , reshape=False,  doPlot=False )
    
    # Optional plot of before and after rotation
    if kwargs['doPlot']:
        if kwargs['view'] == 'dtect':
            xlab = 'spatial'; ylab = 'spectral'
        elif kwargs['view'] == 'dms':
            xlab = 'spectral'; ylab = 'spatial'
        plt.figure()
        plt.subplot(1,2,1);  plt.title("Old"); plt.xlabel(xlab);plt.ylabel(ylab)
        plt.imshow( psf , origin='lower' )
        plt.subplot(1,2,2);  plt.title("New"); plt.xlabel(xlab);plt.ylabel(ylab)
        plt.imshow( new_psf , origin='lower' )
        plt.tight_layout()
    
    # Save procedure
    if save and is_hdu:
        if savepath == None: savepath = ''
        filename = '{:}SOSS_os{:d}_{:d}x{:d}_{:5f}_wfe{:d}.fits'.format( 
            savepath, kwargs['oversamp'], kwargs['psf_dim'], kwargs['psf_dim'] 
            , wave*1e+6, wpsf_args['wfe_real']                         )
        #print(filename)
        psf_hdul[0].data = new_psf
        psf_hdul[0].header.set('PSF_TILT' , wanted_tilt)
        psf_hdul.writeto(filename, overwrite=True)
        return
    else:
        return
    
    
    
    

    
    
#--------------------------------------------------------------------------------#
    
def correct_tilt_in_webbpsf_files( fpath , files_endwith='.fits' , savepath=None
                                 , webbpsf_tilts_file = None
                                 , wanted_tilts_file="SOSS_wavelength_dependent_tilt_extrapolated.txt"
                                 , wanted_os = []
                                 , **kwargs):
    ''' Function that takes a path to a directory containing PSF .fits files
and corrects them based on the desired tilt obtained from a reference file.


Parameters
---------

fpath : string
    path towards the PSFs in .fits file format
files_endwith : string
    file extension of the PSFs
savepath : string
    which path to save the PSFs
wanted_tilts_file : string
    path+name of file containing reference info
    on the tilt vs wavelength desired behaviour
**kwargs : optional args
    (see 'generate_and_rotate_webbpsf()' docstring for info on **kwargs)
'''
    
############### Initiate kwargs if not already present #############################
    if 'psf_pad_dim' not in kwargs: kwargs['psf_pad_dim'] = 160
    if 'view' not in kwargs:        kwargs['view'] = 'dms'
    if 'doPlot' not in kwargs:      kwargs['doPlot'] = False
    if 'doPrint' not in kwargs:     kwargs['doPrint'] = False
    
    # Initiate args for 'get_webbpsf' if not already present in kwargs
    if 'instrument' not in kwargs:     kwargs['instrument'] = 'NIRISS'
    if 'filter' not in kwargs:         kwargs['filter'] = 'CLEAR'
    if 'pupil_mask' not in kwargs:     kwargs['pupil_mask'] = 'GR700XD'
    if 'pupil_opd_file' not in kwargs: kwargs['pupil_opd_file'] = 'OPD_RevW_ote_for_NIRISS_predicted.fits.gz'
    if 'wfe_real' not in kwargs:       kwargs['wfe_real'] = 0
    if 'jitter' not in kwargs:         kwargs['jitter'] = 'gaussian'
    if 'jitter_sigma' not in kwargs:   kwargs['jitter_sigma'] = 0.007

    # Initiate args for 'get webbpsf_tilt' if not already present in kwargs
    if 'mode' not in kwargs:         kwargs['tilt_measurement_mode'] = 'ccf'
    if 'spatbox_size' not in kwargs: kwargs['spatbox_size'] = 23
    if 'specbox_size' not in kwargs: kwargs['specbox_size'] = 20
        
############ list all webbPSF files in fpath ##############
    flist = []
    for file in listdir(fpath):
        if file.endswith(files_endwith):
            flist.append(file)
    print(len(flist))
    
########## unpack data found in the wanted_tilts_file ##########
    data_wtf = np.loadtxt(wanted_tilts_file)
    ref_wavs = data_wtf[:,0]
    wanted_tilts = data_wtf[:,1]
    
    if savepath == None:
            savepath = fpath
    
    if webbpsf_tilts_file != None:
        wpsf_tilts = np.loadtxt(webbpsf_tilts_file)
    
########## Process each file through 'generate_and_rotate_webbpsf' ##########
    t_start = time()
    if len(wanted_os) == 0: os_str = ''
    else: os_str = "For each wavelength, "+str(len(wanted_os))+" files are created with different oversampling\n" 
    print("\n\nStarting the #process\n"
          + "Savepath is: " + savepath +"\n"
          + os_str
          + "To optimize viewer experience, we recommend you\n"
          + "sit back and enjoy your favorite tune while this runs\n\n")
    
    for i,f in enumerate(flist):
        print("Processing:  " + f)
        t_lap = time()
        
        # Get pars from file name
        os, psf_dim, wav = get_webbpsf_pars_from_filename(f)
        # interpolate to find desired tilt from reference file
        wanted_tilt = np.interp( wav , ref_wavs , wanted_tilts )
        
        # If you already know the PSF's tilt in advance, say it
        if webbpsf_tilts_file != None:
            where = np.where( np.logical_and( wav-0.00001 <= wpsf_tilts[:,0]
                                            , wpsf_tilts[:,0] <= wav+0.00001
                                            )
                            )
            if len(where[0]) != 0:
                known_webbpsf_tilt = wpsf_tilts[where[0][0],1]
            else:
                known_webbpsf_tilt = None
        else:
            known_webbpsf_tilt = None
            
        
        # Decides on if you desire multiple oversamp for each PSF
        # and generates them
        kwargs['psf_dim'] = psf_dim
        if len(wanted_os) == 0:
            kwargs['oversamp'] = os
            generate_and_rotate_webbpsf( wav*1e-6 , wanted_tilt , known_webbpsf_tilt=known_webbpsf_tilt
                                       , save=True , savepath=savepath , **kwargs
                                       )
            print("Runtime = " + get_time_str(time()-t_lap) + "\n")
        else:
            for oh_ess in wanted_os:
                t_lap2 = time()
                kwargs['oversamp'] = oh_ess
                generate_and_rotate_webbpsf( wav*1e-6 , wanted_tilt , known_webbpsf_tilt=known_webbpsf_tilt
                                           , save=True , savepath=savepath , **kwargs
                                           )
                print("Runtime (os"+str(oh_ess)+") = " + get_time_str(time()-t_lap2) )
            print("Runtime for this file = " + get_time_str(time()-t_lap) + '\n')
        
        if (i+1)%5 == 0:
            if len(wanted_os) == 0: os_str = ''
            else: os_str = "("+str(len(wanted_os))+" times each file for all wanted oversamps)"
            print('\n-------------------------------')
            print("UPDATE: "+str(i+1)+"/"+str(len(flist))+" files processed "+os_str)
            if i+1<len(flist):
                print("Elapsed time = " + get_time_str(time()-t_start) )
            print('-------------------------------\n\n')
        
    print("\nTOTAL RUNTIME = " +get_time_str(time()-t_start)+ " secs\n\n\n")
    




    
    
    
    
#--------------------------------------------------------------------------------#    
# Some minor helper functions
    
def get_time_str(duration):
    
    days = int(   duration//(24*3600)   )
    hours = int( (duration-(days*24*3600))//3600  )
    mins = int(  (duration-(days*24*3600)-(hours*3600))//60  )
    secs = int(   duration-(days*24*3600)-(hours*3600)-(mins*60)  )
    
    dur_str = ''
    if days > 0: dur_str += str(days)+'days:'
    if hours > 0: dur_str += str(hours)+'h:'
    if mins > 0: dur_str += str(mins)+'m:'
    if days<=0 and hours<=0 and mins<=0 :
        secs += round( duration - days-hours-mins-secs , 3 )
    dur_str += str(secs)+'s'
    
    return dur_str


def make_webbpsf_filename(os,pix_dim,wav):
    return "SOSS_os{:d}_{:d}x{:d}_{:5f}.fits".format(os, pix_dim, pix_dim, wav)


def get_webbpsf_pars_from_filename(fname):
    
    underscore_count = 0
    for c in fname:
        if c =='_': underscore_count+=1
    if underscore_count != 3:
        raise ValueError("Invalid amount of pars found in webbpsf file name."
                         + " Expected 3 pars (os, pix, wav) but got "+str(underscore_count)
                         + ". Make sure an underscore ('_') is in front of every param."
                         + " Example: 'SOSS_os10_128x128_1.234000.fits'")
    
    os=None; pix_dim=None; wav=None
    for i in range(len(fname)):
        
        if fname[i:i+2] == "os" and os == None:
            count=0
            while fname[i+2+count] != '_':
                count+=1
            try:
                os = int( fname[i+2:i+2+count] )
            except ValueError:
                raise Exception("File name has no valid attribute 'os'")
        
        if fname[i] == "x" and pix_dim == None:
            count1=0; count2=0
            while fname[i-1-count1] != '_':
                count1+=1
            while fname[i+1+count2] != '_':
                count2+=1
            try:
                pix_dim1 = int( fname[i-count1:i] )
            except ValueError:
                pix_dim1=None
            try:
                pix_dim2 = int( fname[i+1:i+1+count2] )
            except ValueError:
                pix_dim2=None
            if pix_dim1 != pix_dim2 or (pix_dim1==None and pix_dim2==None):
                raise Exception("File name has no valid attribute for psf pixel dimension")
            else:
                pix_dim = pix_dim1
            
        if fname[i:i+5] == ".fits" and wav==None:
            count=0
            while fname[i-1-count] != '_':
                count+=1
            try:
                wav = float(fname[i-count:i])
            except ValueError:
                raise Exception("File name has no valid attribute for wavelength")
        
    return os, pix_dim, wav