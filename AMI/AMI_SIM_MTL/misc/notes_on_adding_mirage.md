# Notes on adding Mirage simulations

Examples of scripts to create mirage simulations for AMI:

- https://github.com/spacetelescope/niriss-commissioning/blob/nis019_prep/nis_comm/nis_019/run_ami_simulations_wbadpix.py
- https://github.com/spacetelescope/niriss-commissioning/blob/nis019_prep/nis_comm/nis_019/run_ami_simulations_nobadpix.py

Mostly setting up the yaml files to run through mirage


## 1 setting up mirage yaml files


Make the mirage simulation yaml files:
```
from mirage.yaml import yaml_generator

yam = yaml_generator.SimInput(input_xml=xml_name, pointing_file=pointing_name,
                              catalogs=catalogues, roll_angle=roll_angle,
                              dates=dates, reffile_defaults=reffile_defaults,
                              verbose=True, output_dir=odir,
                              simdata_output_dir=simdata_output_directory,
                              datatype=datatype)

yam.create_inputs()

```

### 1.1 SimInput inputs

- input_xml: The input xml file from APT
- pointing_file: '1093_for_sim_allobs.pointing' - product of APT? One per xml file?
- catalogues: dictionary of names each pointing to a source list. 
              This is defined per target (but can be the same for multiple somehow?)
              Can we generate this using the code for Loic?
- roll_angle: dict of observations XXX (from 0) each having a roll angle, these
              have been manually worked out citing a problem with
              yaml_generator.default_obs_v3pa_on_date(pointing_name, obs1, date='2022-04-01')
              with python 3.7+  - should be able to get this from the xml/generate it?
- dates: set to a single iso date '2022-04-01' - from the xml?
- reffile_defaults: 'crds'  - is there another kind?
- verbose: Set to True
- output_dir: the output directory
- simdata_output_dir: the sim data output directory (not sure how this is 
                      different than output_dir)
- datatype: 'raw'  - is there another kind?


I guess these are saved as jwXXXXXXXXXX.yaml but not sure currently how the
number is definied (xml?)


### 1.2 Edit yaml files (after creation?)

``` 
with open(file, 'r') as infile:
    yaml_content = yaml.safe_load(infile)

# edit key1.key2 = value
yaml_content[key1][key2] = value

with io.open(modified_file, 'w') as outfile:
    yaml.dump(yaml_content, outfile, default_flow_style=False)
      
```

#### 1.2.1 For no badpix

Set global for all observations

- Reffiles.astrometric = None  (uses pysaif?)
- simSignals.psf_wing_threshold_file = config
- simSignals.psfpath = filepath  (non default psf)
- Reffiles.gain = fits file (non default gain map?)
- Reffiles.pixelflat = fits file (non default flat?)
- Reffiles.superbias = fits file (non default super bias)

Set per target

- Reffiles.dark = sim dark file (dark000001/dark00001_uncal.fits or dark000005/dark0000005_uncal.fits)
- Reffiles.linearized_darkfile = None

#### 1.2.2 with badpix

Set global for all observations

- Reffiles.astrometric = None  (uses pysaif?)
- simSignals.psf_wing_threshold_file = config
- simSignals.psfpath = filepath  (non default psf) 

### 1.3 Important notes:
- Mirage currently does not apply the proper motion that is entered in the APT fie. 
  It is therefore important to enter coordinates at the epoch of observation in 
  the APT file. AB Dor is a high proper motion star so we are using 2022.25 
  coordinates in the APT file and the input source list file.

The are using the following non-default reference files to generate the simulation. 
- dark000001_uncal.fits has ngroups = 5 and is used for AB Dor observation that also has ngroups = 5. 
- dark000005_uncal.fits has ngroups = 12 and is used for HD37093 observation that also has ngroups = 12. 
- jwst_niriss_gain_general.fits is the gain file with one value for all pixels 
- jwst_niriss_flat_general.fits is the flat field reference file with no structure 
- jwst_niriss_superbias_sim.fits bias reference file to match the simulation

## 2. Running simulations


For each mirage yaml file it is as simple as this:

```
    from mirage import imaging_simulator
    
    # create data
    t1 = imaging_simulator.ImgSim()
    t1.paramfile = str(filename)
    t1.create()
```


