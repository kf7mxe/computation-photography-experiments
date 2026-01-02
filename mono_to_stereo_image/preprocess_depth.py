#!/usr/bin/env python3
"""
Standalone script to preprocess depth maps before Blender rendering.
This is separate from render_stereo.py to avoid dependency issues with Blender's Python.
"""

import sys
import os
import numpy as np
from PIL import Image, ImageFilter
import cv2

def preprocess_depth_map(depth_path, output_path, smooth_amount=1):
    """
    Preprocess depth map to reduce warping artifacts while preserving object edges.
    - Uses edge-aware filtering to smooth noise without blurring object boundaries
    - Helps reduce geometric distortion in displacement
    """
    print(f"Preprocessing depth map (smooth: {smooth_amount})...")

    depth_img = Image.open(depth_path).convert('L')

    # Convert to numpy for advanced filtering
    depth_np = np.array(depth_img)

    if smooth_amount > 0:
        if smooth_amount == 1:
            # Light smoothing - bilateral filter preserves edges well
            # This removes noise while keeping sharp transitions at object boundaries
            depth_filtered = cv2.bilateralFilter(depth_np, d=5, sigmaColor=30, sigmaSpace=30)

        elif smooth_amount == 2:
            # Medium smoothing - guided filter for better edge preservation
            depth_filtered = cv2.bilateralFilter(depth_np, d=7, sigmaColor=50, sigmaSpace=50)
            # Second pass with smaller kernel
            depth_filtered = cv2.bilateralFilter(depth_filtered, d=5, sigmaColor=30, sigmaSpace=30)

        elif smooth_amount >= 3:
            # Heavy smoothing - multiple passes with edge-aware filtering
            depth_filtered = cv2.bilateralFilter(depth_np, d=9, sigmaColor=75, sigmaSpace=75)
            depth_filtered = cv2.bilateralFilter(depth_filtered, d=7, sigmaColor=50, sigmaSpace=50)
            # Gentle gaussian blur for final smoothing
            depth_filtered = cv2.GaussianBlur(depth_filtered, (5, 5), 1.0)
        else:
            depth_filtered = depth_np

        depth_img = Image.fromarray(depth_filtered)

    # Optional: Enhance contrast slightly to make depth transitions clearer
    # This helps prevent muddy/smudged appearance on objects
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(depth_img)
    depth_img = enhancer.enhance(1.1)  # Subtle contrast boost

    # Save preprocessed depth
    depth_img.save(output_path)
    print(f"Saved preprocessed depth to: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: preprocess_depth.py <input_depth> <output_depth> [smooth_amount]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    smooth = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    preprocess_depth_map(input_path, output_path, smooth)
