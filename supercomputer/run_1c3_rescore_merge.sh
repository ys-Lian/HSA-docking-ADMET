#!/bin/bash
#$ -N hsa_rescore_merge
#$ -pe smp 1
#$ -l h_rt=1:00:00
#$ -l mem=16G
#$ -o /home/Scratch/hsa_docking/logs_sge/rescore_merge_$JOB_ID.out
#$ -e /home/Scratch/hsa_docking/logs_sge/rescore_merge_$JOB_ID.err
#$ -cwd

mkdir -p /home/Scratch/hsa_docking/logs_sge

echo "======================================================================"
echo "Rescore Merge   Job: $JOB_ID   Host: $(hostname)   $(date)"
echo "======================================================================"

source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate docking

python /home/Scratch/hsa_docking/1c3_rescore_merge.py

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Merge completed at $(date)"
    echo "Results: /home/Scratch/hsa_docking_Yunshan/rescore_wide_summary.csv"
else
    echo "ERROR: Merge failed (exit=$EXIT_CODE)"
    exit $EXIT_CODE
fi
