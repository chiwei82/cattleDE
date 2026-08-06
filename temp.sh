#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --account=COSC021063
#SBATCH --output=log/out/%j.out
#SBATCH --error=log/err/%j.err

set -euo pipefail

cd /user/work/yx25778/cattleDE

module load languages/python/3.12.3
source venv/bin/activate

python prep/pseudo_label.py --conf 0.8 --suffix _highConf