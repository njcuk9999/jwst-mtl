import glob
import io
import os
import yaml
import sys

# you must be in the conda environment 'mirage'

sys.path.insert(0, '/genesis/jwst/bin/mirage/')

from mirage import imaging_simulator
from mirage.yaml import yaml_generator


from jwst.pipeline import Detector1Pipeline

import subprocess as subprocess

import numpy as np

import os.path

from datetime import datetime

import aptxml_to_catalogs as loic


#####################################
# INPUTS
#####################################

pathdir = '/genesis/jwst/userland-ami/loic_mirage_betapic/'
# The .xml file was generated by the APT with the export capability.
xml_name = os.path.join(pathdir, 'betapic_2476_short.xml')
# The .pointing file was generated simultaneously from the APT.
pointing_name = os.path.join(pathdir, 'betapic_2476_short.pointing')

simdata_output_directory = '/genesis/jwst/userland-ami/loic_mirage_betapic/'

pav3 = 000
dates = '2022-05-30'
reffile_defaults = 'crds'
#datatype = 'linear, raw'
datatype = 'raw'

#####################################

# When running this script, first delete all input (and output) files:
# rm -f ?bservation* jw*
# In particular the yaml files.
xxx = subprocess.getoutput('rm -rf '+pathdir+'?bserva* '+pathdir+'jw*')


if False:
    # Here, two things are very important:
    # 1) The target name ('BETA-PIC' or 'PSF-REF') is the target name
    # used in the APT file. Theses catalogues need to refer to those APT names.
    # 2) The various sources can be in different catalogues or in the same.
    # 3) The magnitudes of the sources are defined in that .list file.
    #
    # AB Dor is in field 19 and HD37093 is in field 20 in Kevin's targets.xlsx file
    #catalogues = {'BETA-PIC': {'point_source': os.path.join(ami_example_dir, 'betapic_loic.list')},
    #              'PSF-REF': {'point_source': os.path.join(ami_example_dir, 'betapic_ref.list')}}
    #              'HD-37093': {'point_source': os.path.join(ami_example_dir, 'stars_field19_20_combined_allfilters.list')}}
    catalogues = {'BETA-PIC': {'point_source': os.path.join(pathdir, 'betapic_both.list')},
              'PSF-REF': {'point_source': os.path.join(pathdir, 'betapic_both.list')}}
                #'DISK': {'extended':}} #  I did not find how to call this with extended option

if True:
    # Convert the string observing date to float
    date_object = datetime.strptime(dates, "%Y-%m-%d")
    year, week, day = date_object.isocalendar()
    obs_date_float = year + (week*7 + day)/365.
    print('Catalogues for date: ', obs_date_float)


    # Run apt2cat in aptxml_to_catalog.py to generate the list files
    catalog_list, target_list = loic.apt2cat(xml_name, obs_date_float)

    print('The APT xml file was read and these catalogues and target lists were generated:')
    print(catalog_list)
    print(target_list)

    # Add your planet. If we want to do so, we need to know here which index is our target
    # so that we can read back the catalog file and add a faint companion at the proper
    # relative position. Neil, you should probably add the capability to return an entry
    # index in  the target the make_catalog function.

    # neil.add_source() (separation, position angle, dm) you need the V3PA from pysiaf?

    # Here, we need to remake a catalogues structure as above containing all catalogs returned
    # by apt2cat(). Neil, please do so. I do not know how to create that structure.
    #catalogues = # like above but use the catalog_list
    # Until that is done, use this:
    catalogues = {'BETA-PIC': {'point_source': os.path.join(pathdir, 'betapic_2476_short_BETA-PIC_catalog.list')},
              'PSF-REF': {'point_source': os.path.join(pathdir, 'betapic_2476_short_PSF-REF_catalog.list')}}


if True:

    yam = yaml_generator.SimInput(input_xml=xml_name, pointing_file=pointing_name,
                              catalogs=catalogues, roll_angle=pav3,
                              dates=dates, reffile_defaults=reffile_defaults,
                              verbose=True, output_dir=simdata_output_directory,
                              simdata_output_dir=simdata_output_directory,
                              datatype=datatype)

    yam.create_inputs()

    # Create all files
    yaml_files = glob.glob(os.path.join(simdata_output_directory, 'jw*.yaml'))
    print(yaml_files)

    for file in yaml_files:

        # set astrometric reference file to None to use pysiaf
        with open(file, 'r') as infile:
            yaml_content = yaml.safe_load(infile)
        yaml_content['Reffiles']['astrometric'] = 'None'
        yaml_content['psf_wing_threshold_file'] = 'config'
        modified_file = file.replace('.yaml', '_mod.yaml')
        with io.open(modified_file, 'w') as outfile:
            yaml.dump(yaml_content, outfile, default_flow_style=False)

        t1 = imaging_simulator.ImgSim()
        t1.paramfile = str(modified_file)
        t1.create()


# Call the first stage DMS
if True:

    # Lists all uncal.fits files
    thecall = 'ls -1 '+pathdir+'jw*nis_uncal.fits'
    ls = subprocess.getoutput(thecall)
    uncal_list = np.array(ls.split('\n'))
    # Forge the list of uncal fits files to send to DMS
    #basename_list = np.copy(uncal_list)
    #for i in range(np.size(uncal_list)):
    #    one, two, three = uncal_list[i].partition('_uncal')
    #    basename_list[i] = one+'_uncal.fits'
    # Launch DMS stage 1 and write results in the same path as uncal images
    # with the name default.
    for i, uncal_filename in enumerate(uncal_list):
        result = Detector1Pipeline.call(uncal_list[i] \
                , output_dir = pathdir \
                , save_results=True)

