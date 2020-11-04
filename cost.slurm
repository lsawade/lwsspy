#!/bin/bash
#SBATCH --job-name=costgradhess  # create a short name for your job
#SBATCH --output=costgradhess.out
#SBATCH -N1                # node count
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=lsawade@princeton.edu

module purge
module load anaconda3/2020.7 
module load openmpi/gcc/4.0.4/64 
module load cudatoolkit/11.0

conda activate lwsspy

cd /home/lsawade/lwsspy

python lwsspy/seismo/invertcmt/test_depth_optimization.py