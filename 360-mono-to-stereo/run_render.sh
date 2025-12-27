#!/bin/bash
# Run Blender stereo render
# Arguments: color_image depth_image output_image [--displacement N] [--ipd N] [--samples N] [--subdivisions N]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================
# CONFIGURE BLENDER PATH HERE (if not in PATH)
# ============================================
BLENDER_CMD="blender"
# BLENDER_CMD="/path/to/blender/blender"
# ============================================

# Run Blender in background mode with the render script
"$BLENDER_CMD" -b -P "$SCRIPT_DIR/render_stereo.py" -- "$@"
