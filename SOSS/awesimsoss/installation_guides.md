# Read me for SOSS

Sections for every tool

---

## AWESIMSOSS

Simulating NIRISS SOSS time-series data

### Installation

Verified on CentOS but should work on MacOSX as well

In command line:
```bash
conda update conda
conda config --add channels conda-forge
conda config --add channels http://ssb.stsci.edu/astroconda

conda create --name jwst_soss stsci
conda activate jwst_soss
pip install git+https://github.com/spacetelescope/jwst
conda install batman

git clone https://github.com/spacetelescope/awesimsoss
cd awesimsoss
python setup.py develop
cd ..
git clone https://github.com/hover2pi/hotsoss
cd hotsoss
python setup.py develop
cd ..
pip install svo_filters
```

Note: For full functionality requires the JWST pipeline installed (this is done above) - thus you get the pipeline as a by-product.


---

## JWST pipeline

The pileline used by STSci to calibrate the raw data coming from the telescope.

### Installation

Following the instructions provided on the [github page](https://github.com/spacetelescope/jwst/) for end-users (note that the version number at the end of the second line may change):

```
conda create -n jwst_env python=3.7
conda activate jwst_env

pip install git+https://github.com/spacetelescope/jwst@0.14.2

export CRDS_PATH=$HOME/crds_cache
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
```

Of course the last two lines can also be added to your .bashrc file if you're using bash. Or add:

```
setenv CRDS_PATH=$HOME/crds_cache
setenv CRDS_SERVER_URL=https://jwst-crds.stsci.edu
```

to your .cshrc file if you're using C shell instead.

To run the stage 1 pileline on a (simulated) fits file type

```
strun jwst.pipeline.Detector1Pipeline ng4ni3_uncal.fits
```

For more information on running the pipeline from [terminal](https://jwst-pipeline.readthedocs.io/en/latest/jwst/introduction.html#running-from-the-command-line) or from [python](https://jwst-pipeline.readthedocs.io/en/latest/jwst/introduction.html#running-from-within-python) see the documentation.
