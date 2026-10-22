#!/bin/bash
#$ -N hsa_rescore
#$ -t 1-264
#$ -pe smp 1
#$ -l h_rt=48:00:00
#$ -l mem=8G
#$ -o /home/Scratch/hsa_docking/logs_sge/rescore_$JOB_ID_$TASK_ID.out
#$ -e /home/Scratch/hsa_docking/logs_sge/rescore_$JOB_ID_$TASK_ID.err
#$ -cwd

mkdir -p /home/Scratch/hsa_docking/logs_sge
mkdir -p /home/Scratch/hsa_docking/rescore_results

echo "======================================================================"
echo "Rescore: Task=$SGE_TASK_ID  BATCH=$(( ($SGE_TASK_ID-1)/8+1 ))  POCKET=$(( ($SGE_TASK_ID-1)%8+1 ))"
echo "Job: $JOB_ID   Host: $(hostname)   $(date)"
echo "======================================================================"

source /etc/profile.d/modules.sh 2>/dev/null || true
module unload gcc-libs 2>/dev/null || true
module load compilers/gnu/10.2.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate docking

for BIN in ~/Scratch/hsa_docking_Yunshan/vina \
           ~/Scratch/hsa_docking_Yunshan/vina_split \
           ~/gnina; do
    [ -f "$BIN" ] && [ ! -x "$BIN" ] && chmod +x "$BIN"
done

python /home/Scratch/hsa_docking/1c3_rescore_hpc_array.py $SGE_TASK_ID

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task $SGE_TASK_ID completed at $(date)"
else
    echo "ERROR: Task $SGE_TASK_ID failed (exit=$EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi
