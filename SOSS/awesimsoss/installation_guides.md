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
