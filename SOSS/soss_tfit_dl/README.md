# Transit fit

---

## 1. Installation

Download jwst-mtl:

    git clone git@github.com:njcuk9999/jwst-mtl.git

Change to the correct branch this should create a "jwst-mtl" directory
from now on we will refer to this as "{INTROOT}"

You must compile the fortran modules and copy the .so files to a new directory {COMPILED FORTRAN PATH}:

If you have an intel machine this is done as follows:

    cd {INTROOT}/SOSS/soss_tfit/utils/
    f2py3 -c tfit5.pyf transitmodel.f keplerian.f ttcor.f occultquad.f mandelagol.f rqsort.f transitdur.f -lpthread -liomp5 --fcompiler=intelem --f90flags='-parallel -mkl -qopenmp' --f77flags='-parallel -mkl -qopenmp'
    f2py3 -c fittransitmodel3.pyf precision.f90 fittermod.f90 fittransitmodel3.f90 getrhosig.f minpack.f transitmodel.f occultquad.f keplerian.f mandelagol.f ttcor.f -lpthread -liomp5 --fcompiler=intelem --f90flags='-parallel -mkl -qopenmp' --f77flags='-parallel -mkl -qopenmp'

Otherwise, use:

    f2py3 -c tfit5.pyf transitmodel.f keplerian.f ttcor.f occultquad.f mandelagol.f rqsort.f transitdur.f -lgomp --f90flags='-fopenmp' --f77flags='-fopenmp'
    f2py3 -c fittransitmodel3.pyf precision.f90 fittermod.f90 fittransitmodel3.f90 getrhosig.f minpack.f transitmodel.f occultquad.f keplerian.f mandelagol.f ttcor.f -lpthread -liomp5  --f90flags='-parallel -mkl -qopenmp' --f77flags='-parallel -mkl -qopenmp' -lgomp --f90flags=-fopenmp --f77flags=-fopenmp

You must add the following to your bash

    export PYTHONPATH="{COMPILED FORTRAN PATH}":$PYTHONPATH
    export PYTHONPATH="{INTROOT}/SOSS/":$PYTHONPATH
    export PYTHONPATH="{INTROOT}/SOSS/soss_tfit":$PYTHONPATH
    export PATH="{INTROOT}/SOSS/soss_tfit/recipe":$PATH

Currently you have to set the number of threads to use for each fit using the following
Note this is NOT the total number of threads.

    export OMP_NUM_THREADS=N

This value should match "N_fit_threads" in the yaml file.

The total number of threads = N_walker_threads * N_fit_threads

i.e. if N_walker_threads = 3 and N_fit_threads = 8 you need at least 24 threads available.


### 1.1 Install using conda

#### Step 1:

Create a conda environment (python 3.9) - here we use "soss-env" as the profile name

    conda create --name soss-env python=3.9

#### Step 2:

Change directory to the {INTROOT}/SOSS/soss_tfit directory

Pip install requirements:

    conda activate soss-env
    pip install -r requirements.txt

## 1.2 Install using other methods

Either use a venv or other method and install via pip or install modules
manually with the versions provided in {INTROOT}/SOSS/soss_tfit/requirements.txt directory


## 1.3 Notes

Note currently the requirements file has all packages standard and installed for all steps
of the pipeline - this needs cleaning for just the TransitFit code

---

## 2. Using TransitFit

Look at the {INTROOT}/SOSS/soss_tfit/recipes directory


### 2.1 Standard mode

You will need to copy and update a yaml file (i.e. example.yaml) with your parameters

Then you can use the following to run the standard setup:

    python transit_fit.py {yaml_file}

### 2.2 Custom mode

You will need to copy and update a yaml file (i.e. example.yaml) with your parameters

You can then copy and update the top level code (see update transit_fit_example.py)

Then you run with:

    python transit_fit_example.py {yaml_file}

---