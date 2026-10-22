#!/bin/bash
#$ -N step1_prepare
#$ -wd /home/Scratch/hsa_docking
#$ -pe smp 8
#$ -l h_rt=4:00:00
#$ -l mem=16G

echo "=========================================="
echo "Step 1: Prepare features + ligands"
echo "Job ID: $JOB_ID  Host: $(hostname)  Date: $(date)"
echo "=========================================="

source /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate docking

mkdir -p logs

# step 1a
echo "[1a] Running prepare_features..."
python /home/Scratch/hsa_docking/1a_prepare_features.py
if [ $? -ne 0 ]; then
    echo "ERROR: 1a_prepare_features.py failed!"
    exit 1
fi
echo "[1a] Done → open_admet_mppb_cheminform.csv"

# step 1b：PDBQT
echo "[1b] Running prepare_ligands..."
python /home/Scratch/hsa_docking/1b_prepare_ligands.py --in_csv open_admet_mppb_cheminform.csv --outdir ligprep_out --write_pdbqt
if [ $? -ne 0 ]; then
    echo "ERROR: 1b_prepare_ligands.py failed!"
    exit 1
fi
echo "[1b] Done → ligprep_out/pdbqt/"

echo "Step 1 finished at $(date)"
