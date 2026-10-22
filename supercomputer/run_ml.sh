#!/bin/bash
#$ -N hsa_ml
#$ -t 1-8
#$ -pe smp 4
#$ -l h_rt=48:00:00
#$ -l mem=8G
#$ -o /home/Scratch/hsa_docking/ml/logs/ml_$JOB_ID_$TASK_ID.out
#$ -e /home/Scratch/hsa_docking/ml/logs/ml_$JOB_ID_$TASK_ID.err
#$ -cwd

# Task ID → Model Script Mapping
# 1: LGBM  with_docking      2: LGBM  without_docking
# 3: XGB   with_docking      4: XGB   without_docking
# 5: GB    with_docking      6: GB    without_docking
# 7: DNN   with_docking      8: DNN   without_docking

mkdir -p /home/Scratch/hsa_docking/ml/logs
mkdir -p /home/Scratch/hsa_docking/ml/results

echo "======================================================================"
echo "ML Job: $JOB_ID   Task: $SGE_TASK_ID   Host: $(hostname)   $(date)"
echo "======================================================================"

source /etc/profile.d/modules.sh 2>/dev/null || true
module unload gcc-libs 2>/dev/null || true
module load compilers/gnu/10.2.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate docking

ML_DIR=/home/Scratch/hsa_docking/ml

case $SGE_TASK_ID in
    1) SCRIPT="$ML_DIR/lgbm_with_docking.py";    NAME="LGBM_with" ;;
    2) SCRIPT="$ML_DIR/lgbm_without_docking.py"; NAME="LGBM_without" ;;
    3) SCRIPT="$ML_DIR/xgb_with_docking.py";     NAME="XGB_with" ;;
    4) SCRIPT="$ML_DIR/xgb_without_docking.py";  NAME="XGB_without" ;;
    5) SCRIPT="$ML_DIR/gb_with_docking.py";      NAME="GB_with" ;;
    6) SCRIPT="$ML_DIR/gb_without_docking.py";   NAME="GB_without" ;;
    7) SCRIPT="$ML_DIR/dnn_with_docking.py";     NAME="DNN_with" ;;
    8) SCRIPT="$ML_DIR/dnn_without_docking.py";  NAME="DNN_without" ;;
    *) echo "Unknown task ID: $SGE_TASK_ID"; exit 1 ;;
esac

echo "Running: $NAME"
echo "Script:  $SCRIPT"
echo "----------------------------------------------------------------------"

python "$SCRIPT"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================================"
    echo "$NAME completed at $(date)"
    echo "======================================================================"
else
    echo "ERROR: $NAME failed (exit code $EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi
