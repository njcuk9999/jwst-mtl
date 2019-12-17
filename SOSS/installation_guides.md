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

### Example

In python:
```python
# Imports
from awesimsoss import TSO
from hotsoss import STAR_DATA

# Initialize simulation
tso256_clear = TSO(ngrps=3, nints=5, star=STAR_DATA)

# Run it and make a plot
tso256_clear.simulate()
tso256_clear.plot()
```
Note: In Linux this loads up in the default browser (using Bokeh) I need to figure out how to change the default browser for this. 

Note: For full functionality requires the JWST pipeline installed (this is done above) - thus you get the pipeline as a by-product.


---
