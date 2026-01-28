#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -J philosophical_advisor_lora
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -o training_%j.log
#SBATCH -e training_%j.err

#####################################################################
# Philosophical Advisor Fine-tuning with Axolotl
# SLURM batch script for HPC cluster
#####################################################################

echo "=========================================="
echo "Philosophical Advisor Fine-tuning"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

#####################################################################
# MODULES — Load Python 3.12 *before* venv creation
#####################################################################
echo ">>> Loading modules"
ml purge
ml load gcc/12.2
ml load cuda/12.2
ml load python/3.12

#####################################################################
# PATHS — update these as needed
#####################################################################
PROJECT_DIR=""
VENV_DIR=""
CONFIG_FILE=""
DATASET_FILE=""

# HuggingFace cache (prevents writing to $HOME and quota issues)
export HUGGING_FACE_CACHE=""
export HF_DATASETS_CACHE="$HUGGING_FACE_CACHE"
export HF_HOME="$HUGGING_FACE_CACHE"
export TRANSFORMERS_CACHE="$HUGGING_FACE_CACHE"
mkdir -p "$HUGGING_FACE_CACHE"

# Axolotl output directory
export OUTPUT_DIR="$PROJECT_DIR/philosophical-advisor-lora"
mkdir -p "$OUTPUT_DIR"

# Set TMPDIR to avoid filling up /tmp
export TMPDIR="/nfs/stak/users/martid24/hpc-share/philosophical-advisor/tmp"
mkdir -p "$TMPDIR"

#####################################################################
# CREATE / ACTIVATE VIRTUAL ENVIRONMENT
#####################################################################
if [ ! -d "$VENV_DIR/bin" ]; then
    echo ">>> Creating virtual environment using Python 3.12"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo ">>> Activated venv at $VENV_DIR"
echo ">>> Python: $(which python)"
echo ">>> Python version: $(python --version)"

#####################################################################
# UPGRADE PIP + BUILD TOOLS
#####################################################################
echo ""
echo ">>> Upgrading pip & installing build tools"
pip install --upgrade pip setuptools wheel build

#####################################################################
# INSTALL AXOLOTL AND DEPENDENCIES
#####################################################################
echo ""
echo ">>> Installing PyTorch with CUDA 12.2 support"
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo ">>> Installing Axolotl and dependencies"
pip install --no-cache-dir packaging ninja
pip install --no-cache-dir axolotl[flash-attn,deepspeed]

# Install additional required packages
echo ""
echo ">>> Installing additional dependencies"
pip install --no-cache-dir transformers accelerate peft datasets bitsandbytes scipy

#####################################################################
# VERIFY INSTALLATION
#####################################################################
echo ""
echo ">>> Verifying installation"

# Check PyTorch and CUDA
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: CUDA not available!")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA verification failed!"
    exit 1
fi

# Check Axolotl
python -c "import axolotl; print(f'Axolotl installed successfully')" || {
    echo "ERROR: Axolotl import failed!"
    exit 1
}

#####################################################################
# VERIFY FILES EXIST
#####################################################################
echo ""
echo ">>> Checking for required files"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✓ Config file found: $CONFIG_FILE"

if [ ! -f "$DATASET_FILE" ]; then
    echo "ERROR: Dataset file not found: $DATASET_FILE"
    exit 1
fi
echo "✓ Dataset file found: $DATASET_FILE"

# Quick dataset validation
DATASET_LINES=$(wc -l < "$DATASET_FILE")
echo "✓ Dataset has $DATASET_LINES lines"

#####################################################################
# DISPLAY TRAINING CONFIGURATION
#####################################################################
echo ""
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Dataset: $DATASET_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "HuggingFace cache: $HUGGING_FACE_CACHE"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time limit: 1 day"
echo "=========================================="

#####################################################################
# RUN TRAINING
#####################################################################
cd "$PROJECT_DIR"

echo ""
echo ">>> Starting training at $(date)"
echo ""

# Use accelerate to launch training
accelerate launch -m axolotl.cli.train "$CONFIG_FILE"

TRAINING_EXIT_CODE=$?

echo ""
echo ">>> Training completed at $(date)"
echo ">>> Exit code: $TRAINING_EXIT_CODE"

#####################################################################
# POST-TRAINING ACTIONS
#####################################################################
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Merge LoRA adapters:"
    echo "   python -m axolotl.cli.merge_lora $CONFIG_FILE --lora_model_dir='$OUTPUT_DIR'"
    echo ""
    echo "2. Test inference:"
    echo "   accelerate launch -m axolotl.cli.inference $CONFIG_FILE --lora_model_dir='$OUTPUT_DIR'"
    echo ""
    echo "Check the log file for detailed training metrics:"
    echo "   training_${SLURM_JOB_ID}.log"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "Check the error log for details:"
    echo "   training_${SLURM_JOB_ID}.err"
    echo ""
    echo "Common issues:"
    echo "- Out of Memory: Reduce micro_batch_size in config"
    echo "- CUDA errors: Check GPU availability"
    echo "- Dataset errors: Verify JSONL format"
    echo ""
fi

# Cleanup temp files if needed
# rm -rf "$TMPDIR"/*

echo ""
echo ">>> Job finished at $(date)"