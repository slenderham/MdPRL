#!/bin/bash
#SBATCH --job-name=fitattn       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email on job start, end and fault
#SBATCH --mail-user=chong.wang.gr@dartmouth.edu

module purge
module load matlab/r2021b

matlab -nodisplay -nosplash -r Fit_Attention
