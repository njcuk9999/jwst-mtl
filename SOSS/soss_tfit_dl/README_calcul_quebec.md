# Installing the DL version of tfit on Calcul Quebec

Use these instructions to install the DL version of tfit on Calcul Quebec.

```bash
module load python/3.9
virtualenv --no-download jwst39
source jwst39/bin/activate
```

```bash
pip install jwst
pip install numpy --no-index
pip install matplotlib --no-index
pip install bottleneck --no-index
pip install packaging --no-index
pip install pyyaml --no-index
pip install jsonschema --no-index
pip install importlib-metadata --no-index
pip install psutil --no-index
pip install requests --no-index
pip install tqdm --no-index
pip install exotic-ld
```

Copy all soss_tfit files to your chosen location.

Then, go to soss_tfit/utils directory and run:
```bash
f2py3 -c tfit5.pyf transitmodel.f keplerian.f ttcor.f occultquad.f mandelagol.f rqsort.f transitdur.f -lpthread -liomp5 --fcompiler=intelem --f90flags='-parallel -mkl -qopenmp' --f77flags='-parallel -mkl -qopenmp'
```

```bash
icc -fPIC -shared -O3 spotrod_tfit.c -o spotrod_tfit.so
```