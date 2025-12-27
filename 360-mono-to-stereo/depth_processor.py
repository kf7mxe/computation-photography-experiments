#!/usr/bin/env python3
"""
Depth Processor for 360° Images
Generates depth maps from equirectangular panoramas using Marigold AI.
Designed to be called via CLI for environment isolation.
"""

import torch
import numpy as np
import py360convert
from PIL import Image
from diffusers import MarigoldDepthPipeline
import os
import sys
import argparse
import multiprocessing
import cv2


def print_progress(message, step=None, total=None):
    """Print progress in a parseable format for the GUI."""
    if step is not None and total is not None:
        print(f"[PROGRESS] {step}/{total} - {message}", flush=True)
    else:
        print(f"[STATUS] {message}", flush=True)


def heal_seams(image_path, output_path, thickness=15):
    """Apply inpainting to heal visible seams in the depth map."""
    print_progress("Running seam healer...")
    
    img = cv2.imread(image_path, 0)
    h, w = img.shape
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Vertical seams at 0%, 25%, 50%, 75% of width
    seam_locs = [0, w//4, w//2, 3*w//4, w-thickness]
    
    for x in seam_locs:
        cv2.rectangle(mask, (x - thickness, 0), (x + thickness, h), 255, -1)
    
    healed_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    cv2.imwrite(output_path, healed_img)
    print_progress("Seams healed successfully")


def fix_wrap_around(image_path):
    """Fix the wrap-around seam at the edges of the panorama."""
    print_progress("Fixing wrap-around seam...")
    
    img = cv2.imread(image_path, 0)
    h, w = img.shape
    
    # Shift image so the seam is in the center
    img_rolled = np.roll(img, w//2, axis=1)
    
    # Create mask in the center
    mask_rolled = np.zeros((h, w), dtype=np.uint8)
    center_x = w // 2
    mask_rolled[:, center_x-20 : center_x+20] = 255
    
    # Inpaint the center seam
    healed_rolled = cv2.inpaint(img_rolled, mask_rolled, 3, cv2.INPAINT_NS)
    
    # Roll back
    healed_final = np.roll(healed_rolled, -w//2, axis=1)
    cv2.imwrite(image_path, healed_final)
    print_progress("Wrap-around seam fixed")


def load_depth_model():
    """Load the Marigold depth estimation pipeline."""
    print_progress("Loading Marigold model...")
    
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    print_progress(f"Using {num_cores} CPU threads")

    if torch.cuda.is_available():
        dtype = torch.float16
        device = "cuda"
        print_progress("CUDA GPU detected")
    else:
        dtype = torch.float32
        device = "cpu"
        print_progress("Using CPU (slower)")

    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-lcm-v1-0",
        torch_dtype=dtype
    )
    
    pipe.to(device)
    print_progress("Model loaded successfully")
    return pipe


def process_batch(pipe, image_pil_list, cube_size, steps):
    """Run AI inference on a batch of cube face images."""
    
    results = pipe(image_pil_list, num_inference_steps=steps)
    predictions = results.prediction 
    
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    processed_images = []
    
    for i in range(len(predictions)):
        depth_np = predictions[i]
        depth_np = np.squeeze(depth_np)
        
        # Normalize to 0-255
        if depth_np.max() > depth_np.min():
            norm_img = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        else:
            norm_img = depth_np
        
        norm_img = (norm_img * 255).astype(np.uint8)
        
        # Ensure exact cube size
        if norm_img.shape[0] != cube_size:
            norm_pil = Image.fromarray(norm_img)
            norm_pil = norm_pil.resize((cube_size, cube_size), Image.BICUBIC)
            norm_img = np.array(norm_pil)

        # Convert to RGB for py360convert
        norm_img_rgb = np.stack([norm_img]*3, axis=-1)
        processed_images.append(norm_img_rgb)
        
    return processed_images


def process_360_depth(input_path, output_path, cube_size=512, steps=10, 
                      batch_size=6, heal_seams_enabled=True):
    """
    Main processing function for 360° depth map generation.
    
    Args:
        input_path: Path to input equirectangular image
        output_path: Path for output depth map
        cube_size: Size of cube faces (default 512)
        steps: Number of inference steps (default 10)
        batch_size: Batch size for processing (default 6)
        heal_seams_enabled: Whether to apply seam healing (default True)
    """
    
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return False

    # 1. Load Image
    print_progress(f"Loading image: {input_path}")
    img = np.array(Image.open(input_path))
    
    # 2. Slice to Cube Map
    print_progress("Slicing 360° image into cube faces...", 1, 6)
    try:
        cube_faces = py360convert.e2c(img, face_w=cube_size, mode='bilinear', cube_format='dict')
    except Exception as e:
        print(f"[ERROR] Failed to slice image: {e}", file=sys.stderr)
        return False
    
    # 3. Load Model
    pipe = load_depth_model()
    
    face_order = ['F', 'R', 'B', 'L', 'U', 'D']
    face_names = ['Front', 'Right', 'Back', 'Left', 'Up', 'Down']
    
    # Prepare list of images
    print_progress("Preparing cube faces for processing...")
    all_faces_pil = [Image.fromarray(cube_faces[k]) for k in face_order]
    
    processed_faces_map = {}
    
    # 4. Run Batch Processing
    total_batches = (len(all_faces_pil) + batch_size - 1) // batch_size
    
    for batch_idx, i in enumerate(range(0, len(all_faces_pil), batch_size)):
        batch = all_faces_pil[i : i + batch_size]
        batch_keys = face_order[i : i + batch_size]
        batch_names = face_names[i : i + batch_size]
        
        print_progress(f"Processing faces: {', '.join(batch_names)}", batch_idx + 1, total_batches)
        
        batch_results = process_batch(pipe, batch, cube_size, steps)
        
        for key, result_img in zip(batch_keys, batch_results):
            processed_faces_map[key] = result_img

    # 5. Stitch back to equirectangular
    print_progress("Stitching cube faces back to 360° panorama...")
    face_list = [processed_faces_map[k] for k in face_order]
    
    output_w = cube_size * 4
    output_h = cube_size * 2
    
    depth_equi = py360convert.c2e(face_list, h=output_h, w=output_w, cube_format='list')
    
    # 6. Save
    final_img = Image.fromarray(depth_equi).convert("L") 
    final_img.save(output_path)
    print_progress(f"Depth map saved to: {output_path}")
    
    # 7. Optional seam healing
    if heal_seams_enabled:
        heal_seams(output_path, output_path)
        fix_wrap_around(output_path)
    
    print_progress("Depth processing complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate depth maps from 360° equirectangular images using Marigold AI'
    )
    parser.add_argument('input', help='Input 360° image path')
    parser.add_argument('output', help='Output depth map path')
    parser.add_argument('--cube-size', type=int, default=512, 
                        help='Cube face size (default: 512)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Inference steps (default: 10)')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='Batch size for processing (default: 6)')
    parser.add_argument('--no-heal-seams', action='store_true',
                        help='Disable seam healing')
    
    args = parser.parse_args()
    
    success = process_360_depth(
        input_path=args.input,
        output_path=args.output,
        cube_size=args.cube_size,
        steps=args.steps,
        batch_size=args.batch_size,
        heal_seams_enabled=not args.no_heal_seams
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
