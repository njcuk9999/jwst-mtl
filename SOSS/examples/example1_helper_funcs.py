import numpy as np
from astropy.io import fits
import glob
from pathlib import Path


# Helper routines for example1_investigateNoise.py

def check_for_fits_header_value(fname , key , value ):
    if isinstance(value,(list,np.ndarray)):
        return np.any([ check_for_fits_header_value(fname,key,val) for val in value ])
    else:
        with fits.open(fname) as f:
            if key in f[0].header.keys():
                if f[0].header[key] == value:
                    return True
                else:
                    return False
            else:
                raise ValueError("FITS header for "+fname+" does not contain a key labeled '"+key+"'")

def create_noisy_filename( noise , jwstMTLnoise_to_calwebbSteps_MAP=None ):
    
    if jwstMTLnoise_to_calwebbSteps_MAP is None:
        jwstMTLnoise_to_calwebbSteps_MAP = { 'photon':[],'normalize':[] , 'zodibackg':[]
                               , 'flatfield':[] , 'darkframe':['dark_current']
                               , 'nonlinearity':['linearity'] , 'superbias':['superbias']
                               , 'detector':['refpix']
                               }
     
        
    if np.all(list(noise.values())): # if all noise types are desired
        err_str = '--allNoise'
    else:                             # add certain noise types
        err_str = ''
        for key in noise.keys():
            if noise[key] is True:
                err_str += '--'+key
        if err_str == '':
            n_str = '_noNoise'
        else:
            n_str = '_noisy'
        noisy_file_str = 'IDTSOSS_clear'+n_str+err_str+'.fits'
    
    return noisy_file_str




def create_calwebb_config_files( output_path , noise_shopping_lists
                               , calwebb_NIR_TSO_mandatory_steps=None
                               , calwebb_NIR_steps=None
                               , jwstMTLnoise_to_calwebbSteps_MAP=None 
                               , override_calwebb_reffiles=False
                               , user_calwebb_reffiles_dir=None
                               , user_calwebb_reffiles=None
                               , doPrint = False
                               ):
    """ Create configuration files used to customize the
flow of the CALWEBB_DETECTOR1 pipeline

Inputs:

    output_path : string
        - Location of folder where the config files will be saved.
        
    noise_shopping_lists : 2D list of strings
        EXAMPLE: [ [], ['photon'], ['photon','superbias','darkframe'] ]
        - The 1st-dimension length represents the amount of config 
          files you wish to generate.
        - Each sublist represents the noise types the user wishes to
          correct for using that particular config file.
        - Some noise types can't be corrected for by CALWEBB_DETECTOR1
        IMPORTANT: The noise nomenclature for is based off of the 
                   jwst-mtl terminology, which may differ from the
                   CALWEBB terminology.
    
    calwebb_NIR_TSO_mandatory_steps : 1D list of strings
        - List of CALWEBB_DETECTOR1 steps that are always to be
          performed, regardless of what noise is present in the data.
    
    calwebb_NIR_steps : 1D list of strings
        - List of all CALWEBB_DETECTOR1 steps.
        - Not all of these steps are required to process TSOs
    
    jwstMTLnoise_to_calwebbSteps_MAP : dict
        - A one-to-many mapping that links the jwst-mtl noise types
          introduced into a NIRISS simulation with the
          CALWEBB steps that correct them.
    
    override_calwebb_reffiles : boolean
        - Choose whether to override the selection of reference files
          used by the CALWEBB-DETECTOR1 pipeline
    
    user_calwebb_reffiles_dir : string
        - Location of the folder containing the reference files
          that will override the CALWEBB best/default ones
    user_calwebb_reffiles : dict
        - A 1-to-1 mapping of CALWEBB_DETECTOR1 steps and the user-
          defined reference files that are to override the default ones. 
"""


    if output_path is not None: # Create the path if it doesn't exist
        Path(output_path).mkdir(parents=True, exist_ok=True)
    else: # default output_path is the location of the code
        output_path = ''

        # Check to make sure the noise shopping list is
        # in the correct format and type (2D list/array)
    if not isinstance(noise_shopping_lists,(np.ndarray,list)):
        noise_shopping_lists = [[noise_shopping_lists]]

        # Default DMS reference file directory if none was supplied by user
    if override_calwebb_reffiles is True and user_calwebb_reffiles_dir is None:
        user_calwebb_reffiles_dir = '/genesis/jwst/jwst-ref-soss/noise_files/'


        # Initiate the config file name and initial file contents
    base_fname = 'calwebb_tso1'
    base_file_contents = "name = 'Detector1Pipeline'\n"                  \
                       + "class = 'jwst.pipeline.Detector1Pipeline'\n\n" \
                       + "[steps]\n"
    
    if calwebb_NIR_TSO_mandatory_steps is None:
        calwebb_NIR_TSO_mandatory_steps = ['dq_init','saturation','ramp_fit','jump']
    if calwebb_NIR_steps is None:
        calwebb_NIR_steps = ['group_scale','dq_init','saturation','ipc'
                            ,'superbias','refpix','linearity','dark_current'
                            ,'jump','ramp_fit','gain_scale','persistence']
    if jwstMTLnoise_to_calwebbSteps_MAP is None:
        jwstMTLnoise_to_calwebbSteps_MAP = { 'photon':[],'normalize':[] , 'zodibackg':[]
                               , 'flatfield':[] , 'darkframe':['dark_current']
                               , 'nonlinearity':['linearity'] , 'superbias':['superbias']
                               , 'detector':['refpix']
                               }
    cwebb_to_refname = { 'dark_current':'dark' , 'flatfield':'flat'
                       , 'linearity':'linearity' , 'superbias':'superbias'
                       }
    ordered_jwstMTL_noise = ['photon','normalize','zodibackg','flatfield',
                     'darkframe','nonlinearity','superbias','detector']

    
    config_fnames = []
    if doPrint: print('\n')
    for noise_present in noise_shopping_lists:
        
            # Create the config file name based on the noise types
            # that need to be corrected for
        new_fname = output_path + base_fname +'_STEPS'
        no_additional_steps_used = True
        for noise in ordered_jwstMTL_noise:
            if noise in noise_present \
            and len(jwstMTLnoise_to_calwebbSteps_MAP[noise]) != 0:
                new_fname += '--'+noise
                no_additional_steps_used = False
        if no_additional_steps_used is True:
            new_fname += '--NONE'
        new_fname += '.cfg'
        if doPrint: print('Config File created: '+new_fname)
            
        config_fnames.append(new_fname)

            # Create the config file and its base contents
        with open(new_fname, 'w') as new_cfg:
            new_cfg.writelines(base_file_contents)


                # Get the CALWEBB steps needed to correct for the noise present
                # in the data (as specified in 'noise_present')
            jwstMTL_to_calwebbSteps_SUBSET = { k:v for k,v                                     \
                                                   in jwstMTLnoise_to_calwebbSteps_MAP.items() \
                                                   if k in noise_present                       \
                                             }
            calwebb_steps_needed = [ noise for (k,list_in_dict) in jwstMTL_to_calwebbSteps_SUBSET.items() \
                                     for noise in list_in_dict                                            \
                                   ]

                # Specify if a calwebb step should be skipped
                # If performed, option to override the reference file
                # for that given step
            if doPrint: stepz = []
            for step in calwebb_NIR_steps:
                new_cfg.write('\t[['+step+']]\n')

                if step in calwebb_NIR_TSO_mandatory_steps or step in calwebb_steps_needed:
                    new_cfg.write('\tskip = False\n')
                    stepz.append(step)

                        # The previously mentionned reference file override option
                    if override_calwebb_reffiles and user_calwebb_reffiles is not None \
                                                 and step in user_calwebb_reffiles.keys():
                            # If the override reference file located in 'calwebb_reffiles_dir'
                            # is specfied, then use it
                        if user_calwebb_reffiles[step] is not None:
                            new_cfg.write("\toverride_"+cwebb_to_refname[step]+" = '" \
                                          + user_calwebb_reffiles_dir + user_calwebb_reffiles[step]+"'\n" \
                                         )
                            # Otherwise, pick the 'best' one from 'calwebb_reffiles_dir'
                            # This automated routine assumes a certain file name structure
                            # (based off the CRDS file names (ex: jwst_niriss_))
                        else:
                            files = glob.glob(user_calwebb_reffiles_dir+'*'+cwebb_to_refname[step]+'*.fits')
                            if len(files) == 0:
                                raise FileNotFoundError( 'Could not find any CRDS reference file' \
                                                        +'for jwst_niriss_'+cwebb_to_refname[step]+' step' )
                            elif len(files) > 1:
                                useable_files = [ fname for fname in files
                                                  if check_for_fits_header_value(fname,'SUBARRAY'
                                                                                ,['SUBSTRIP256','FULL','GENERIC'])
                                                ]
                                best_file_index = np.argmax([ int(fname[-9:-5]) for fname in useable_files ])
                                best_file = useable_files[best_file_index]
                            else:
                                best_file = files[0]
                            if doPrint: print('For step '+step+', found file: '+best_file)
                            new_cfg.write("\toverride_"+cwebb_to_refname[step]+" = '"+best_file+"'\n")

                else:
                    new_cfg.write('\tskip = True\n')
            
            if doPrint: print('Activated steps: '+str(stepz)+'\n')
    
    return config_fnames