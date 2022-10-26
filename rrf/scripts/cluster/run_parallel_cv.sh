#!/bin/bash
#SBATCH --ntasks=8 # Request 8 CPUs (spread across as many nodes as scheduler decides)
#SBATCH --mem=16G # Request 16G per node
#SBATCH --constraint=centos7 # Use only centos7 CPUs
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=0-01:00:00 # max time for the sched_mit_sloan_batch partition is 4 days
#SBATCH --output=/home/jtwilde/projects/colorchip/cluster_output/mbi_classify/cv/output_%j.out # %j gives the job id number
#SBATCH --error=/home/jtwilde/projects/colorchip/cluster_error/mbi_classify/cv/error_%j.err # %j gives the job id number
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jtwilde@mit.edu

module load python/3.9.4

cd /home/jtwilde/projects/rrf/rrf/scripts

python ../scripts/run_cv/run_cv.py $1 $2 $3 $4 $SLURM_ARRAY_TASK_ID
