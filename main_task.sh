#!/usr/bin/env bash

#SBATCH --job-name=cattle_act_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --account=COSC021063
#SBATCH --output=log/out/%j.out
#SBATCH --error=log/err/%j.err

cd /user/work/yx25778/CattleDE

module load languages/python/3.12.3
source venv/bin/activate