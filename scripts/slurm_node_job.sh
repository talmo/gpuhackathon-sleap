#!/bin/bash
#SBATCH --job-name=sleap-singularity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
module purge
singularity exec --nv /scratch/gpfs/vineetb/triton-sleap-21.03.sif tritonserver --model-repository=/gpuhackathon-sleap/triton/model_repository --backend-config=tensorflow,version=2