#!/usr/bin/bash

#SBATCH --mail-user=jannis.wallau@students.unibe.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=./out/optim/slurm-cnn-r10-it100-%j.out
#SBATCH --output=./out/optim/slurm-cnn-r10-it100-%j.out

#SBATCH --job-name="CNN-HP-Opt-100iter"

#SBATCH --partition=gpu
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --constraint=gtx1080
#SBATCH --mem=4000M
#SBATCH --time=70:00:00

#SBATCH --no-requeue

############################################
source ~/anaconda3/etc/profile.d/conda.sh
#conda activate GNNpT1Env
conda activate GNNEnv
#module load GCC/7.3.0-2.30
#module load GCC/8.3.0
module load GCC/5.4.0-2.26
#module load gcccuda
module load CUDA/10.1.243

#python -c "import torch; print(torch.__version__)"
#python -c "import torch; print(torch.cuda.is_available())"
#echo $PATH
export PATH=/usr/local/cuda/bin:$PATH
#echo $PATH
#echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#echo $LD_LIBRARY_PATH
#nvcc --version
#python -c "import torch; print(torch.version.cuda)"

#export TORCH_CUDA_ARCH_LIST = "6.0 6.1 7.2+PTX 7.5+PTX"

############################################


echo "CNN, fold:" $1
python CNN/CNN_ModelOptimization.py --fold=$1 --device="cuda" --runs=10 --iterations=100 --model="VGG16_bn"


# run using: bash CNN/job_CNN_HyperOpt.sh 0
# or on UBELIX: sbatch CNN/job_CNN_HyperOpt.sh 0
