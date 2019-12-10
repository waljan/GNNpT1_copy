#!/usr/bin/bash

#SBATCH --mail-user= <mail>
#SBATCH --mail-type=fail,end
#SBATCH --output=./out/optim/slurm-r10-it100-%j.out
#SBATCH --output=./out/optim/slurm-r10-it100-%j.out

#SBATCH --job-name="HP-Opt-100iter"

#SBATCH --partition=gpu
#SBATCH --gres=gpu:gtx1080ti:1

#SBATCH --mem=4000M
#SBATCH --time=14:00:00

#SBATCH --no-requeue

############################################
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate GNNpT1Env

#module load CUDA/10.1.243
############################################

if [ $2 == "base" ]
then
  for FOLD in 0 1 2 3
  do
    python ModelOptimization.py --fold=$FOLD --model=$1 --folder="pT1_dataset/graphs/base-dataset/" --device="cuda" --runs=10 --iterations=100 # --augment
  done
else
  for FOLD in 0 1 2 3
  do
    python ModelOptimization.py --fold=$FOLD --model=$1 --folder="pT1_dataset/graphs/paper-graphs/distance-based_10_13_14_35/" --device="cuda" --runs=10 --iterations=100 #--augment
  done
fi

# run using: bash job_HyperOpt.sh "GraphSAGE" "base"
# or on UBELIX: sbatch job_HyperOpt.sh "GraphSAGE" "base"