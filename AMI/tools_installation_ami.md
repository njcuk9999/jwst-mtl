# Read Me for AMI 

Take from here: [https://docs.google.com/document/d/1Et1MuJjIYzh47M9_41RmO2ZOidCz9NnErVOG4eqHxQI/edit#](https://docs.google.com/document/d/1Et1MuJjIYzh47M9_41RmO2ZOidCz9NnErVOG4eqHxQI/edit#)

---
## AMI_SIM

## Installing AMI_SIM

Install via git clone:
```bash
git clone https://github.com/anand0xff/ami_sim
```

python requirements:
```
conda install numpy
conda install astropy
pip install --upgrade webbpsf
```
pysynphot (see section [here](#pysynphot))


## Codes

### `driver_binary.py`
This script provides a basic example of how to use the NIRISS-AMI ETC and
binary simulation code.

#### Example:
```bash
driver_binary.py F430M 11.0 2e6 -t simulatedData/ -o 1 -fr 0.16 -dx 2.0 -dy 2.0
```

### `driver_scene.py`
This script simulates the observation of a sky scene with a (Webb)PSF supplied by the user.

Requires a “scene”   -s command
absolute path to oversampled sky scene fits file, normalized to sum to unity
Requires a “psf” (from WebbPSF)   -p command
absolute path to oversampled PSF fits file. Spectral type set in this

#### Example:
```bash
driver_scene.py -t delmoi_scene -o 1  -utr 0  -f F430M -p psf_f430m_oversampled.fits -s tgt_f430m_oversampled.fits  -os 11 -I 4  -G 7  -c 1  -cr 1e10   -v 1 --random_seed 42
```
```
python driver_scene.py --output_absolute_path /My/full/path/directory/ -o 1 -utr 0 -f F430M -p /path/myPSF.fits -s /path/my_scene.fits -os 11 -I 5 -G 7 -c 1 -cr 1e10 -v 1 --random_seed 42
```

### `ami_etc.py`
This script provides access to a 'private' NIRISS-AMI ETC code.

pyami/simcode/
    Directory that contains the engine code

### Questions:
Why do we need a scene that is an odd number of pixels?
Isn’t better to have psf x scene that is a power of 2 the number of pixels?
Why is -utr 1 does not increase the number of images in the cube with driver_scene.py? Because the output product is the processed cube of integrations (CDS - dark - background)*ff, not raw data. 

---

## MIRAGE

[Github](https://github.com/spacetelescope/mirage)
[Install guide](https://mirage-data-simulator.readthedocs.io/en/latest/install.html)
Reference files [here](https://mirage-data-simulator.readthedocs.io/en/latest/reference_files.html#reference-files)
First problem: the reference files for NIRISS amount to 121 GB - need a dedicated server - Most of that is darks.

### Using webbPSF
Jupyter Notebook [here](https://nbviewer.jupyter.org/github/spacetelescope/webbpsf/blob/master/notebooks/WebbPSF_tutorial.ipynb)

---

## PYSYNPHOT

### Installation

```pip install pysynphot```

Download the 6 tar files in  [ftp://archive.stsci.edu/pub/hst/pysynphot/](ftp://archive.stsci.edu/pub/hst/pysynphot/)
into a folder (say `/Volume/mydisk/pysynphot/`)

Or do the following:
```bash
    wget ftp://archive.stsci.edu/pub/hst/pysynphot/synphot1.tar.gz
    wget ftp://archive.stsci.edu/pub/hst/pysynphot/synphot2.tar.gz
    wget ftp://archive.stsci.edu/pub/hst/pysynphot/synphot3.tar.gz
    wget ftp://archive.stsci.edu/pub/hst/pysynphot/synphot4.tar.gz
    wget ftp://archive.stsci.edu/pub/hst/pysynphot/synphot5.tar.gz
    wget ftp://archive.stsci.edu/pub/hst/pysynphot/synphot6.tar.gz
```

Extract:
```bash
    tar -xvf synphot1.tar.gz
    tar -xvf synphot2.tar.gz
    tar -xvf synphot3.tar.gz
    tar -xvf synphot4.tar.gz
    tar -xvf synphot5.tar.gz
    tar -xvf synphot6.tar.gz
```

Within your folder, you now have the sub-sub-sub-folder `/Volum/mydisk/pysynphot/grp/hst/cdbs`
and you need to point the PYSYN_CDBS variable to that sub-sub-sub-folder with the following command :
```bash
export PYSYN_CDBS=/Volumes/mydisk/synphot/grp/hst/cdbs
```


---

## XARA pipeline NRM (Code written by Anthony)

### Dependencies:
Seaborn (conda install seaborn)
Astroquery (conda install astroquery)
Termcolor (conda install termcolor)
Need to create DATA in NRM… and put both fits data cubes (science + calibrator)

```python
run NIRISS_extract_FakeData.py
```
Expects a calibrator and a science data cubes in DATA directory
Can edit NIRISS_extract_FakeData.py to point to the desired fits cubes

NIRISS_extract_FakeData.py creates an OIFITS
Download the gui JMMC LITpro and oifits Explorer to read and visualize those oifits files

When run at the prompt, no image is displayed on screen. Anthony works in Spyder where they do appear.

### Interferometric data explanation

[http://fmillour.com/wp-content/uploads/Manuscrit_final.pdf](http://fmillour.com/wp-content/uploads/Manuscrit_final.pdf)


## The “Sydney” IDL pipeline

Latest working version is here: [https://github.com/AnthonyCheetham/idl_masking/](https://github.com/AnthonyCheetham/idl_masking/)

Or alternative source: [https://github.com/bdawg/](https://github.com/bdawg/)


Consists in 4 scripts to run (see for example idl_masking/process_naco_template.script)
Preprocess the data and put it in a cube (qbe_conica.pro)
Get the visibilities and closure phases (calc_bispect.pro)
Calibrate the data (calibrate_v2_cp_gpi.pro) (that’s when you get your OIFITS file)
Fit a model (binary_gri.pro)

The arts of extracting the measurements is in calc_bispect. The bispectrum is the multiplication of three complex points `FFT(u_i,v_i)*FFT(u_j,v_j)*conj(FFT(u_k,v_k)).`

Calc_bispec.pro calls bispect

---

## ImPLANEIA

Link to github: [https://github.com/anand0xff/ImPlaneIA](https://github.com/anand0xff/ImPlaneIA)

Install:
```bash
git clone https://github.com/anand0xff/ImPlaneIA

cd ImPlaneIA
git branch devmasthdr
git checkout devmasthdr
git pull origin devmasthdr

conda install relic

cd dImPlaneIA
setup.py develop

cd notebooks

```

Then copy files to `ImPlaneIA/example_data/noise`

and run using:
```bash
python NIRISS_AMI_tutorial.py
```





