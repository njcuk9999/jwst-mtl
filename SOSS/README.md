# Read me for SOSS



## AWESIMSOSS

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
