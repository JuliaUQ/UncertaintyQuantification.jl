#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=100
#SBATCH --time=00:01:00
#SBATCH --account=HPC_account_1
#SBATCH --partition=CPU_partition
#SBATCH --ntasks=1
#SBATCH --job-name=my_test_job
#SBATCH --output=sample-%2a/UncertaintyQuantification-%a.out
#SBATCH --error=sample-%2a/UncertaintyQuantification-%a.err
#SBATCH --array=[1-4]



#### EXTRAS ####



echo ========================================================
echo SLURM job: submitted date = `date`
date_start=`date +%s`
echo =========================================================
echo Job output begins
echo -----------------
echo
hostname
echo Running with $SLURM_NTASKS cores



#### RUN COMMAND ####
cd sample-$(printf %02d $SLURM_ARRAY_TASK_ID)
/home/behrensd/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia radius.jl



echo
echo ---------------
echo Job output ends
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================
echo SLURM job: finished date = `date`
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================
