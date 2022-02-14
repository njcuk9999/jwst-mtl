# The AMI Simulation wrapper

## 1 AMISIM

Installing AMISIM requires the of cloning the github repository and installing 
some python modules (we handle the python modules as part of the requirements 
to run the AMI simulation wrapper.

To clone AMISIM use the following command:

    git clone git@github.com:anand0xff/ami_sim.git

This will create an ami_sim directory in your current working directory.

Please note down the AMISIM installation location you will need to for the 
yaml file. From this point on we will refer to this as `{AMISIM_ROOT}`.


## 2 Mirage

Mirage is required for the AMI simulation wrapper and so installation will 
be done when installing the AMI simulation wrapper


## 3 The AMI Simulation wrapper

### 3.1 Installation

Installation of the AMI Simulation wrapper requires the cloning of the github 
repository and instllation some python modules.

To clone the AMI Simulation wrapper use the following command:

    git clone git@github.com:njcuk9999/jwst-mtl.git

This should create a directory called `jwst-mtl`. Please note down this 
installation location you will need to for the yaml file. From this point on 
we will refer to this as `{WRAPPER_ROOT}`.

We next recommend a new conda environment (though venv or pip can be used if you prefer).

To add a new conda environment use the following command (requires miniconda or 
anaconda to be installed first):

    conda create --name ami-env python=3.8

Then activate this environment with the following command:

    conda activate ami-env

Next we require you to change to a sub-directory within `jwst-mtl` github 
directory that was created when you cloned the repository above.

    cd jwst-mtl/AMI/AMI_SIM_MTL

Once in this directory you should see a `requirements.txt` file.

Check that you have pip and are in the correct conda environment (if using 
conda) using the following command:

    which pip

It should display something ending in `/envs/ami-env/bion/pip`


You can now install the python modules required for the AMI simulation wrapper 
with the following command:

    pip install -r requirements.txt

Finally we can add the AMI simulation wrapper to the python path by adding the 
following to the `.bashrc`, `.profile`, `.login` or equivalent file. 
Note we assume you are using the bash shell, this command may be different for 
different command line shells.

    export PYTHONPATH={WRAPPER_ROOT}:$PYTHONPATH
    export PATH={WRAPPER_ROOT}/bin/:$PATH

This will allow you to run and call the wrapper from within python or call the 
wrapper function from the command line. If successful you are ready to set 
up the simulation yaml file and then run the simulation wrapper.

### 3.2 The simulation yaml file

You will find an example (commented) yaml file in the `inputs` directory 
(called `example.yaml`). 

It is separated into sections (the simulations and the global parameters). 
Each simulation has its own section and consists of one science observation 
`target` and a set of calibrators to go with that target.

Each target is allowed to have no companions or multiple companions (either 
being planets, disks or bars).

Currently supported simulations (using AMI-SIM) are:

 - target \& calibrator(s)
 - target + planet(s) \& calibrator(s)
 - target + disk(s) \& calibrator(s)
 - target + bar(s) \& calibrator(s)
 - target + planet(s) + disk(s) \& calibrator(s)
 - target + planet(s) + bar(s) \& calibrator(s)
 - target + planet(s) + disk(s) + bar(s) \& calibrator(s)

Note the following parameters MUST be set by the user

 - parameters.general.in_dir
 - parameters.general.out_dir
 - parameters.ami-sim.out_path
 - parameters.ami-sim.install-dir  (using the {AMISIM_ROOT} value)
 - parameters.ami-sim.psf.FXXX.path   (for each filter XXX)

the details are which are in the comments of the yaml file.

Note do not use the following parts of the wrapper (i.e. set them to False)
 
- parameters.mirage.use
- parameters.dms.use
- parameters.ami-cal.use
- parameters.implaneia.use

### 3.3 Running the wrapper

Once installation is complete and a yaml file has been made running is very simple.

From a python script:

    # Import the wrapper
    from ami_mtl.science import wrapper
    
    # Enter the path to the yaml file (can be relative or absolute)
    MY_YAML_FILE = "../inputs/example.yaml"
    
    # Run the simulations
    wrapper.main(WCONFIG=MY_YAML_FILE)


This should produce the simulation(s) as defined in the yaml file.

Otherwise if you have set the $PATH correctly then you should be able to just 
run the following from the command line:

    wrapper.py --config=../inputs/example.yaml


