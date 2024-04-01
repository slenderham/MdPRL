#!/bin/bash
#SBATCH --job-name=fitattn       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=150:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email on job start, end and fault
#SBATCH --mail-user=chong.wang.gr@dartmouth.edu
#SBATCH --output=slurm_%j.txt

module purge
module load matlab/r2022a

#matlab -nodisplay -nosplash -r Fit_FeatureBased
#echo "Feature Based done"
matlab -nodisplay -nosplash -r Fit_ObjectBased
echo "Object Based done"
#matlab -nodisplay -nosplash -r Fit_FeatureObjectBased
#echo "Feature Object Based done"
matlab -nodisplay -nosplash -r Fit_ConjunctionBased
echo "Conj Based done"
#matlab -nodisplay -nosplash -r Fit_Attention
#echo "Attn done"
