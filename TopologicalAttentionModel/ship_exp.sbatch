#SBATCH --job-name=TopoAMD
#SBATCH --partition=gpu
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=4096
#SBATCH --output=TopoAMD_%J_stdout.txt
#SBATCH --error=TopoAMD_%J_stderr.txt
#SBATCH --time=12:00:00
#SBATCH --mail-user=joseaguilar@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/joseaguilar

module load CUDA
source activate tda-gpu

echo start time
date

/usr/bin/time CUDA_VISIBLE_DEVICES=0 python experiment.py -v --check_gpus -n 20 -b 128 -e 50 -s 51200 -exp_appendix vanilla1 --save_model

echo end time
date


