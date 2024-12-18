#!/bin/bash
#SBATCH --job-name=optimization_experiment
#SBATCH -A gts-wliao60
#SBATCH --ntasks=1                 # Number of tasks (CPU cores) to request
#SBATCH -N1 --gres=gpu:V100:1                           # Number of nodes and GPUs required
#SBATCH --gres-flags=enforce-binding                # Map CPUs to GPUs
#SBATCH -t 8:00:00                                        # Duration of the job (Ex: 15 mins)
#SBATCH -o slurm_logs/Report-%j.out
#SBATCH -q embers

args_file=$1

# Load any necessary modules (optional)
module load git
module load gcc

# Change to the directory where you want to run your job
source ~/.envs/tml/bin/activate
cd /storage/home/hcoda1/6/ahavrilla3/p-wliao60/alex/repos/Notes/experimentation/optimization

# Your job commands go here
# For example:
srun python experiment.py --mode local --args_file $args_file

# Remember to exit with an appropriate exit code (0 for success, non-zero for failure)
exit 0