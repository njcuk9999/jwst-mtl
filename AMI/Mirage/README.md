# Mirage


## Installation

[From here](https://mirage-data-simulator.readthedocs.io/en/latest/install.html#install-from-pypi)

This will create a new conda environment, install required packages (including the jwst pipeline)

```
conda create -n mirage python=3.6 -y
conda activate mirage
pip install healpy==1.12.5
pip install mirage
pip install git+https://github.com/npirzkal/GRISMCONF#egg=grismconf
pip install git+https://github.com/npirzkal/NIRCAM_Gsim#egg=nircam_gsim
pip install git+https://github.com/spacetelescope/jwst#0.15.1
```



## Reference files and environment

[From here](https://mirage-data-simulator.readthedocs.io/en/latest/reference_files.html#reference-files)

Must set the MIRAGE_DATA path:

```
export MIRAGE_DATA="/my_files/jwst/simulations/mirage_data"
```

Must also set the CRDS Environment variables:

``` 
export CRDS_PATH=$HOME/crds_cache
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
```

The can download reference files:

```python
from mirage.reference_files import downloader
download_path = '/path/into/which/files/are/downlaoded/'
downloader.download_reffiles(download_path, instrument='all', dark_type='linearized', skip_darks=False, skip_cosmic_rays=False, skip_psfs=False, skip_grism=False)
```

Edit 2020-06-19: 
Cannot download reference files: [see here](https://github.com/spacetelescope/mirage/issues/498)



## Simulating Image data

[From here](https://mirage-data-simulator.readthedocs.io/en/latest/imaging_simulations.html)

```python
from mirage.imaging_simulator import ImgSim

sim = ImgSim(paramfile='my_yaml_file.yaml')
sim.create()
```

But need to create a yaml config file:

[From here](https://mirage-data-simulator.readthedocs.io/en/latest/example_yaml.html#example-yaml)

NIRCam example:

```
Inst:
  instrument: NIRCam          #Instrument name
  mode: imaging               #Observation mode (e.g. imaging, WFSS, moving_target)
  use_JWST_pipeline: False    #Use pipeline in data transformations

Readout:
  readpatt: DEEP8         #Readout pattern (RAPID, BRIGHT2, etc) overrides nframe,nskip unless it is not recognized
  ngroup: 6               #Number of groups in integration
  nint: 1                 #Number of integrations per exposure
  array_name: NRCB5_FULL  #Name of array (FULL, SUB160, SUB64P, etc)
  filter: F250M           #Filter of simulated data (F090W, F322W2, etc)
  pupil: CLEAR            #Pupil element for simulated data (CLEAR, GRISMC, etc)

Reffiles:                   #Set to None or leave blank if you wish to skip that step
  dark: None                #Dark current integration used as the base
  linearized_darkfile: $MIRAGE_DATA/nircam/darks/linearized/B5/Linearized_Dark_and_SBRefpix_NRCNRCBLONG-DARK-60090141241_1_490_SE_2016-01-09T02h46m50_uncal.fits # Linearized dark ramp to use as input. Supercedes dark above
  badpixmask: crds          # If linearized dark is used, populate output DQ extensions using this file
  superbias: crds           #Superbias file. Set to None or leave blank if not using
  linearity: crds           #linearity correction coefficients
  saturation: crds          #well depth reference files
  gain: crds                #Gain map
  pixelflat: None
  illumflat: None           #Illumination flat field file
  astrometric: crds         #Astrometric distortion file (asdf)
  ipc: crds                 #File containing IPC kernel to apply
  invertIPC: True           #Invert the IPC kernel before the convolution. True or False. Use True if the kernel is designed for the removal of IPC effects, like the JWST reference files are.
  occult: None              #Occulting spots correction image
  pixelAreaMap: crds        #Pixel area map for the detector. Used to introduce distortion into the output ramp.
  subarray_defs:   config   #File that contains a list of all possible subarray names and coordinates
  readpattdefs:    config   #File that contains a list of all possible readout pattern names and associated NFRAME/NSKIP values
  crosstalk:       config   #File containing crosstalk coefficients
  filtpupilcombo:  config   #File that lists the filter wheel element / pupil wheel element combinations. Used only in writing output file
  filter_wheel_positions: config  #File that lists the filter wheel element / pupil wheel element combinations. Used only in writing output file
  flux_cal:        config   #File that lists flux conversion factor and pivot wavelength for each filter. Only used when making direct image outputs to be fed into the grism disperser code.
  filter_throughput: /Users/me/mirage/mirage/config/placeholder.txt #File containing filter throughput curve

nonlin:
  limit: 60000.0        #Upper singal limit to which nonlinearity is applied (ADU)
  accuracy: 0.000001    #Non-linearity accuracy threshold
  maxiter: 10           #Maximum number of iterations to use when applying non-linearity
  robberto:  False      #Use Massimo Robberto type non-linearity coefficients

cosmicRay:
  path: $MIRAGE_DATA/nircam/cosmic_ray_library/    #Path to CR library
  library: SUNMIN                                                              #Type of cosmic rayenvironment (SUNMAX, SUNMIN, FLARE)
  scale: 1.5                                                                           #Cosmic ray rate scaling factor
  suffix: IPC_NIRCam_B5                                            #Suffix of library file names
  seed: 2956411739                                                             #Seed for random number generator

simSignals:
  pointsource: my_point_sources.cat               #File containing a list of point sources to add (x,y locations and magnitudes)
  psfpath: $MIRAGE_DATA/nircam/gridded_psf_library/   #Path to PSF library
  gridded_psf_library_row_padding: 4              # Number of outer rows and columns to avoid when evaluating library. RECOMMEND 4.
  psf_wing_threshold_file: config                 # File defining PSF sizes versus magnitude
  add_psf_wings: True                             # Whether or not to place the core of the psf from the gridded library into an image of the wings before adding.
  psfwfe: predicted                               #PSF WFE value ("predicted" or "requirements")
  psfwfegroup: 0                                  #WFE realization group (0 to 4)
  galaxyListFile: my_galaxies_catalog.list
  extended: None                                 #Extended emission count rate image file name
  extendedscale: 1.0                             #Scaling factor for extended emission image
  extendedCenter: 1024,1024                      #x,y pixel location at which to place the extended image if it is smaller than the output array size
  PSFConvolveExtended: True                      #Convolve the extended image with the PSF before adding to the output image (True or False)
  movingTargetList: None                         #Name of file containing a list of point source moving targets (e.g. KBOs, asteroids) to add.
  movingTargetSersic: None                       #ascii file containing a list of 2D sersic profiles to have moving through the field
  movingTargetExtended: None                     #ascii file containing a list of stamp images to add as moving targets (planets, moons, etc)
  movingTargetConvolveExtended: True             #convolve the extended moving targets with PSF before adding.
  movingTargetToTrack: None                      #File containing a single moving target which JWST will track during observation (e.g. a planet, moon, KBO, asteroid) This file will only be used if mode is set to "moving_target"
  tso_imaging_catalog: None                      #Catalog listing TSO source to be used for imaging TSO simulations
  tso_grism_catalog: None                        #Catalog listing TSO source to be used for grism TSO observations
  zodiacal:  None                                #Zodiacal light count rate image file
  zodiscale:  1.0                                #Zodi scaling factor
  scattered:  None                               #Scattered light count rate image file
  scatteredscale: 1.0                            #Scattered light scaling factor
  bkgdrate: medium                               #Constant background count rate (ADU/sec/pixel in an undispersed image) or "high","medium","low" similar to what is used in the ETC
  poissonseed: 2012872553                        #Random number generator seed for Poisson simulation)
  photonyield: True                              #Apply photon yield in simulation
  pymethod: True                                 #Use double Poisson simulation for photon yield
  expand_catalog_for_segments: False             # Expand catalog for 18 segments and use distinct PSFs
  use_dateobs_for_background: False              # Use date_obs value to determine background. If False, bkgdrate is used.

Telescope:
  ra: 53.1                     #RA of simulated pointing
  dec: -27.8                   #Dec of simulated pointing
  rotation: 0.0                #y axis rotation (degrees E of N)
  tracking: sidereal           #sidereal or non-sidereal

newRamp:
  dq_configfile: config          #config file used by JWST pipeline
  sat_configfile: config         #config file used by JWST pipeline
  superbias_configfile: config   #config file used by JWST pipeline
  refpix_configfile: config      #config file used by JWST pipeline
  linear_configfile: config      #config file used by JWST pipeline

Output:
  file: jw42424024002_01101_00001_nrcb5_uncal.fits   # Output filename
  directory: ./                                # Directory in which to place output files
  datatype: linear,raw                         # Type of data to save. 'linear' for linearized ramp. 'raw' for raw ramp. 'linear,raw' for both
  format: DMS                                  # Output file format Options: DMS, SSR(not yet implemented)
  save_intermediates: False                    # Save intermediate products separately (point source image, etc)
  grism_source_image: False                    # Create an image to be dispersed?
  unsigned: True                               # Output unsigned integers? (0-65535 if true. -32768 to 32768 if false)
  dmsOrient: True                              # Output in DMS orientation (vs. fitswriter orientation).
  program_number: 42424                        # Program Number
  title: Supernovae and Black Holes Near Hyperspatial Bypasses   #Program title
  PI_Name: Doug Adams                          # Proposal PI Name
  Proposal_category: GO                        # Proposal category
  Science_category: Cosmology                  # Science category
  target_name: TARG1                           # Name of target
  target_ra: 53.1001                           # RA of the target, from APT file.
  target_dec: -27.799                          # Dec of the target, from APT file.
  observation_number: '002'                    # Observation Number
  observation_label: Obs2                      # User-generated observation Label
  visit_number: '024'                          # Visit Number
  visit_group: '01'                            # Visit Group
  visit_id: '42424024002'                      # Visit ID
  sequence_id: '1'                             # Sequence ID
  activity_id: '01'                            # Activity ID. Increment with each exposure.
  exposure_number: '00001'                     # Exposure Number
  obs_id: 'V42424024002P0000000001101'         # Observation ID number
  date_obs: '2019-10-15'                       # Date of observation
  time_obs: '06:29:11.852'                     # Time of observation
  obs_template: 'NIRCam Imaging'               # Observation template
  primary_dither_type: NONE                    # Primary dither pattern name
  total_primary_dither_positions: 1            # Total number of primary dither positions
  primary_dither_position: 1                   # Primary dither position number
  subpix_dither_type: 2-POINT-MEDIUM-WITH-NIRISS  #Subpixel dither pattern name
  total_subpix_dither_positions: 2             # Total number of subpixel dither positions
  subpix_dither_position: 2                    # Subpixel dither position number
  xoffset: 344.284                             # Dither pointing offset in x (arcsec)
  yoffset: 466.768                             # Dither pointing offset in y (arcsec)
```