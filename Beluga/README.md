# Beluga

This directory contains three important files to run the loicpipeline on the Beluga cluster.

## Files

1. `input_files.txt`: This file contains the input YAML files conating the reduction parameters for different datasets or different sets of parameters for the same data set (as you like). Each line in this file represents a separate job that will be launched independently using the `sbatch` command, so you don't have to create a specific job for each one of your reductions.

2. `run_sbatch.sh`: This script is used to submit the `sbatch` jobs. You will need to modify the different parameters in the header of the script (the SBATCH directives) to match your specific needs. The script reads the lines from the `input_files.txt`, corresponding to a YAML file, and launches indepent jobs for each of these YAML files by calling `python run_loicpipe.py <yaml_file>`.

3. `run_loicpipe.py`: This script is used to run the loicpipeline.

## Usage

To use the Beluga directory, follow these steps:

1. Copy the entire Beluga directory to a location outside of the `jwst-mtl` directory. This will allow you to personalize the directory as needed.

2. Navigate to the newly copied Beluga directory.

3. Put YAML files corresponding to different pipeline runs in the `input_files.txt` file. Each line in this file should contain the full path to a YAML file. You can also specify the parameters directly in the file, as shown in the example below:

    ```
    /path/to/your/input_file1.yaml
    /other/path/to/your/input_file2.yaml
    /one/more/path/to/your/input_file3.yaml
    ```

    the SBATCH --array parameter in the `run_sbatch.sh` script indicates whic lines will be read from the `input_files.txt` file. If you want all the lines to be read, the SBATCH --array parameter
    should be set to the number of lines in the `input_files.txt` file. For example, here the file contains 3 lines, so the line in the `run_sbatch.sh` script should be:

    ```bash
    #SBATCH --array=1-3
    ```
    More info here: https://docs.alliancecan.ca/wiki/Job_arrays

    You can have differents input files. Just set the variable `$INPUT_FILE` in `run_sbatch.sh` to the right file name.

4. Run the pipeline by executing the following command:

    ```bash
    sbatch run_sbatch.py
    ```
    To see the status of your jobs, you can use the `squeue` command `sq` which is an alias for `squeue -u <username>`.
    
    You will see something like this:
    ```
    JOBID       USER      ACCOUNT           NAME  ST  TIME_LEFT NODES CPUS TRES_PER_NODE MIN_MEM NODELIST
    47226118_1  adb    def-dlafre_c     loicpipe   R      54:13     1    2        N/A     50G    bl12448
    ```
    The `R` in the `ST` (for status) column means that the job is running. If you see a `PD` instead, it means that the job is pending.

    The JOBID is very useful. It can use for many things:
    - To see the output of the job: `cat slurm-<JOBID>.out`
    - To cancel the job: `scancel <JOBID>`
    - To see the efficiency of the job after it has finished: `seff <JOBID>`. I use it a lot to see if the job is well optimized and adjust the parameters in the `run_sbatch.sh` script, especially the memory (`--mem`) and the cpus (`--cpus-per-task`). Always start with a bit more memory than you think you need so the job doesn't freeze because it runs out of memory, and adjust for later runs.
    - I also use it to save the outputs in `/home/your_username/scratch/sbatch_outputs/out_sbatch_%A_%a.txt` where `%A` is the job id and `%a` is the array id (see --output option). *Don't forget to replace `your_username`.* The scratch directory is a good place to save outputs because it will automatically be deleted after 2 months so you don't have to worry about cleaning up.
    - `sacct` is also a very useful command, like showing the jobs that ran in the last 24h.

    The `sq`command also shows the node on which the job is running. You have access to the node as long as the job is running. You can connect to the node using `ssh bl12448` for example. It could be useful to check if the cpus are being used efficiently with `htop` or `top`.

That's it! You can now run the pipeline on the Beluga cluster.
