#!/bin/bash
#SBATCH --account=def-dlafre
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --time=0-1:00
#SBATCH --job-name loicpipe
#SBATCH --output=/home/your_username/scratch/sbatch_outputs/out_sbatch_%A_%a.txt
#SBATCH --mail-type=FAIL
#SBATCH --array=1-1

echo "Starting run_sbatch with ID $SLURM_JOB_ID"

# Set the name of the input file containing the list of yaml files.
# You can change the name of the file or the path to the file.
export INPUT_FILE=input_files.txt

# Here are some options for the FLEXIBLAS variable,
# which is used by the some python libraries to choose the BLAS/LAPACK backend
# (e.g. numpy, scipy, scikit-learn, etc.)
# It is a way to parallelize the computations on the nodes (within numpy, scipy, etc.)
# It can help mainly for matrix/arrays operations.
# NOTE: It is not always useful to set this variable, and it can sometimes slow down the computations.
#       It is recommended to test the performance of your code with and without this variable.
#       with the command seff <job_id> to see the CPU efficiency of your job.
#       It needs to be at least greater than 50%. I generally set it to values <= 4.
# WARNING: If you have some multi-threaded code in your python script,
#          it can lead to oversubscription, so just be aware and test the performance.

# OPTIONS:
# - Use FLEXIBLAS=IMKL to use Intel MKL for BLAS/LAPACK (my favorite)
#   - with the number of threads set by MKL_NUM_THREADS
# - Use FLEXIBLAS=OPENBLAS to use OpenBLAS for BLAS/LAPACK
#   - with the number of threads set by OMP_NUM_THREADS
# - Use FLEXIBLAS=MKL to use Intel MKL for BLAS/LAPACK
#   - with the number of threads set by OMP_NUM_THREADS

# Set the FLAXIBLAS variable
export FLEXIBLAS=IMKL

# Set the number of threads to the number of cpus per task
# (it can be possible to have more threads than cpus, so it can be a multiple of cpus_per_task)
# ex: export OMP_NUM_THREADS=$((2 * SLURM_CPUS_PER_TASK))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load the required modules and activate the virtual environment
module load StdEnv/2023  gcc/12.3 opencv/4.9.0 python
source ~/venvs/jwst_311/bin/activate

# NOTE: You can add jwst-mtl relevant paths to the PYTHONPATH if it is not already done
#       in you .bashrc or .bash_profile or even in the activation script of your virtual environment.
# ex:
# export PYTHONPATH=$PYTHONPATH:/path/to/jwst-mtl/
# export PYTHONPATH=$PYTHONPATH:/path/to/jwst-mtl/SOSS/loicpipe/

# Here we use the `#SBATCH --array=1-N` option to generate multiple jobs,
# for reductions of different datasets or sets of parameters for example.
# This would generate N jobs with SLURM_ARRAY_TASK_ID = 1, ..., N
# and you can access this variable with $SLURM_ARRAY_TASK_ID
echo "Starting python code with SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"

# Read the specific line from the input_file.txt, given by the SLURM_ARRAY_TASK_ID
# where input_files.txt is a file containing the addresses of the yaml files
# for each independent reduction.
yaml_file=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $INPUT_FILE)

# Print the yaml file that will be used for this job
echo "Using yaml file: $yaml_file"

# Pass the line (which is the address of a yaml file) to the python script
python run_loicpipe.py $yaml_file