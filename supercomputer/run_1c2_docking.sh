#!/bin/bash
#$ -N hsa_docking
#$ -t 1-264        # Replace total_tasks output by 0_filter_and_split.py
#$ -pe smp 1
#$ -l h_rt=24:00:00
#$ -l mem=8G              
#$ -o logs_sge/docking_$JOB_ID_$TASK_ID.out
#$ -e logs_sge/docking_$JOB_ID_$TASK_ID.err
#$ -cwd

mkdir -p logs_sge

echo "======================================================================"
echo "Job: $JOB_ID   Task: $SGE_TASK_ID   Host: $(hostname)   $(date)"
echo "BATCH_ID  = $(( ($SGE_TASK_ID - 1) / 8 + 1 ))"
echo "POCKET_ID = $(( ($SGE_TASK_ID - 1) % 8 + 1 ))"
echo "======================================================================"

# ── environment ──────────────────────────────────────────────────────────────────
source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate docking


# Vina executable path
VINA_PATH=~/Scratch/hsa_docking/vina
if [ -f "$VINA_PATH" ] && [ ! -x "$VINA_PATH" ]; then
    chmod +x "$VINA_PATH"
    echo "Vina execution permission has been added: $VINA_PATH"
fi

export OMP_NUM_THREADS=$NSLOTS

# ── docking ──────────────────────────────────────────────────────────────────
echo "Running: python 1c2_hpc_docking.py $SGE_TASK_ID"
python 1c2_hpc_docking.py $SGE_TASK_ID

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task $SGE_TASK_ID completed at $(date)"
else
    echo "ERROR: Task $SGE_TASK_ID failed (exit code $EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi
