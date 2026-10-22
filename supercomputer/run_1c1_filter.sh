#!/bin/bash
#$ -N hsa_filter
#$ -pe smp 1
#$ -l h_rt=2:00:00
#$ -l mem=8G
#$ -o logs_sge/filter_$JOB_ID.out
#$ -e logs_sge/filter_$JOB_ID.err
#$ -cwd

mkdir -p logs_sge

echo "======================================================================"
echo "Step 0: Ro5 Filtering + Batch Processing"
echo "Job: $JOB_ID   Host: $(hostname)   $(date)"
echo "======================================================================"

source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
conda activate docking

python 0_filter_and_split.py \
    --ligand_dir ligprep_out/pdbqt \
    --batch_size 50 \
    --outdir     batches \
    --n_pockets  8

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Filtering completed! Please check the total_tasks number in the output above,"
    echo "then modify the THIRD line of run_docking.sh to set TOTAL_TASKS,"
    echo "======================================================================"
else
    echo "ERROR: Filtering failed (exit code $EXIT_CODE)"
    exit $EXIT_CODE
fi
