#!/bin/bash
# Run depth generation in the appropriate Python environment
# Arguments: input_image output_depth [--cube-size N] [--steps N] [--batch-size N] [--no-heal-seams]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================
# CONFIGURE YOUR PYTHON ENVIRONMENT HERE
# ============================================
# Option 1: Conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate marigold

# Option 2: Virtualenv
source /home/trax/Projects/computational-photography/.venv/bin/activate

# Option 3: System Python (if dependencies are installed globally)
# (no activation needed)
# ============================================

# Run the depth processor

/home/trax/Projects/computational-photography/.venv/bin/python "$SCRIPT_DIR/depth_processor.py" "$@"
