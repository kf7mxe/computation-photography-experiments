#!/usr/bin/env python3
"""
Depth Processor for 360° Images
Generates depth maps from equirectangular panoramas using Marigold AI or Depth Anything V2.
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
import gc

# Try importing transformers for Depth Anything
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def print_progress(message, step=None, total=None):
    """Print progress in a parseable format for the GUI."""
    if step is not None and total is not None:
        print(f"[PROGRESS] {step}/{total} - {message}", flush=True)
    else:
        print(f"[STATUS] {message}", flush=True)


def heal_seams(image_path, output_path, thickness=20):
    """Apply inpainting to heal visible seams in the depth map."""
    print_progress("Running seam healer...")

    img = cv2.imread(image_path, 0)
    h, w = img.shape

    mask = np.zeros((h, w), dtype=np.uint8)

    # Vertical seams at 0%, 25%, 50%, 75% of width
    # Increased thickness for better blending across cube face boundaries
    seam_locs = [0, w//4, w//2, 3*w//4, w-thickness]

    for x in seam_locs:
        cv2.rectangle(mask, (x - thickness, 0), (x + thickness, h), 255, -1)

    # Use larger inpaint radius for smoother transitions
    healed_img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, healed_img)
    print_progress("Seams healed successfully")


def fix_wrap_around(image_path):
    """Fix the wrap-around seam at the edges of the panorama."""
    print_progress("Fixing wrap-around seam...")

    img = cv2.imread(image_path, 0)
    h, w = img.shape

    # Shift image so the seam is in the center
    img_rolled = np.roll(img, w//2, axis=1)

    # Create mask in the center with increased width for better blending
    mask_rolled = np.zeros((h, w), dtype=np.uint8)
    center_x = w // 2
    mask_rolled[:, center_x-30 : center_x+30] = 255

    # Inpaint the center seam with larger radius for smoother transition
    healed_rolled = cv2.inpaint(img_rolled, mask_rolled, 7, cv2.INPAINT_TELEA)

    # Roll back
    healed_final = np.roll(healed_rolled, -w//2, axis=1)

    # Apply Gaussian blur to the seam edges for even smoother transition
    blend_width = 40
    left_edge = img[:, :blend_width].astype(float)
    right_edge = img[:, -blend_width:].astype(float)
    healed_left = healed_final[:, :blend_width].astype(float)
    healed_right = healed_final[:, -blend_width:].astype(float)

    # Create alpha blend mask
    alpha = np.linspace(0, 1, blend_width).reshape(1, -1)

    # Blend edges
    healed_final[:, :blend_width] = (healed_left * (1 - alpha) + left_edge * alpha).astype(np.uint8)
    healed_final[:, -blend_width:] = (healed_right * alpha[:, ::-1] + right_edge * (1 - alpha[:, ::-1])).astype(np.uint8)

    cv2.imwrite(image_path, healed_final)
    print_progress("Wrap-around seam fixed")


def get_device_info():
    """Detect available compute device (CUDA, ROCm/HIP, or CPU)."""
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    print_progress(f"Using {num_cores} CPU threads")

    # Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print_progress(f"CUDA GPU detected: {device_name}")
        return "cuda", torch.float16

    # Check for ROCm/HIP (AMD)
    try:
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            print_progress(f"ROCm/HIP detected (version {torch.version.hip})")
            # Force PyTorch to use ROCm
            if torch.cuda.is_available():  # ROCm uses same API as CUDA in PyTorch
                device_name = torch.cuda.get_device_name(0)
                print_progress(f"AMD GPU detected: {device_name}")
                return "cuda", torch.float16  # ROCm uses 'cuda' device in PyTorch
    except:
        pass

    # Fallback to CPU
    print_progress("No GPU detected, using CPU (slower)")
    print_progress("For AMD GPUs: Install ROCm-compatible PyTorch")
    print_progress("For NVIDIA GPUs: Install CUDA-compatible PyTorch")
    return "cpu", torch.float32


def load_marigold_model():
    """Load the Marigold depth estimation pipeline."""
    print_progress("Loading Marigold model...")

    device, dtype = get_device_info()

    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-lcm-v1-0",
        torch_dtype=dtype
    )

    pipe.to(device)

    # Enable memory optimizations
    if device == "cuda":
        try:
            pipe.enable_attention_slicing(1)
            print_progress("Enabled attention slicing for memory efficiency")
        except:
            pass

    print_progress("Marigold model loaded successfully")
    return pipe


def load_depth_anything_model():
    """Load the Depth Anything V2 model."""
    if not TRANSFORMERS_AVAILABLE:
        print("[ERROR] 'transformers' library is missing. Install with: pip install transformers", file=sys.stderr)
        sys.exit(1)

    print_progress("Loading Depth Anything V2 model...")

    device, dtype = get_device_info()

    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

    # Load model with appropriate dtype for memory efficiency
    if device == "cuda":
        model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf",
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )

    model.to(device)

    # Enable gradient checkpointing to save memory
    if device == "cuda":
        try:
            model.gradient_checkpointing_enable()
            print_progress("Enabled gradient checkpointing for memory efficiency")
        except:
            pass

    print_progress("Depth Anything model loaded successfully")
    return image_processor, model


def process_batch(pipe, image_pil_list, cube_size, steps):
    """Run AI inference on a batch of cube face images using Marigold."""

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

    # Clear memory after processing batch
    del results, predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return processed_images


def process_full_panorama(input_path, output_path, processor, model):
    """Process the full 360 panorama directly using Depth Anything."""
    print_progress(f"Loading image for direct processing: {input_path}")

    image = Image.open(input_path).convert("RGB")
    original_size = image.size
    print_progress(f"Original image size: {original_size[0]}x{original_size[1]}")

    # Adaptive resizing based on available memory and image size
    # For very large panoramas (8K+), we need to be more aggressive
    width, height = original_size
    total_pixels = width * height

    if total_pixels > 33_000_000:  # > 8K (7680x4320 = 33M pixels)
        max_dim = 1536
        print_progress(f"Large panorama detected, using max dimension: {max_dim}")
    elif total_pixels > 8_000_000:  # > 4K
        max_dim = 2048
        print_progress(f"4K+ panorama detected, using max dimension: {max_dim}")
    else:
        max_dim = 2560  # Allow higher resolution for smaller images

    inference_size = original_size
    if max(image.size) > max_dim:
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))

        inference_size = (new_width, new_height)
        image = image.resize(inference_size, Image.Resampling.LANCZOS)
        print_progress(f"Resized to {inference_size[0]}x{inference_size[1]} for inference")

    device = model.device

    # Convert to FP16 for memory efficiency on GPU
    if device.type == "cuda":
        inputs = processor(images=image, return_tensors="pt")
        # Move to GPU with FP16
        inputs = {k: v.to(device).half() if v.dtype == torch.float32 else v.to(device)
                  for k, v in inputs.items()}
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)

    # Free the PIL image from memory
    del image
    gc.collect()

    print_progress("Running inference on full panorama...")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Clear inputs immediately
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Interpolate to original size in chunks if very large
    print_progress("Upscaling depth map to original resolution...")

    if total_pixels > 33_000_000:  # Very large image - do two-step upscaling
        # First upscale to intermediate size
        intermediate_size = (original_size[0] // 2, original_size[1] // 2)
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=intermediate_size[::-1],
            mode="bilinear",  # Faster for first pass
            align_corners=False,
        )

        # Move to CPU and free GPU memory
        prediction_cpu = prediction.squeeze().cpu().numpy()
        del predicted_depth, prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Second upscale using CPU (to avoid GPU OOM)
        print_progress("Final upscaling pass (CPU)...")
        prediction_pil = Image.fromarray((prediction_cpu * 255).astype(np.uint8))
        prediction_pil = prediction_pil.resize(original_size, Image.Resampling.LANCZOS)
        output = np.array(prediction_pil).astype(np.float32) / 255.0
        del prediction_cpu, prediction_pil

    else:
        # Single-step upscaling for smaller images
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        output = prediction.squeeze().cpu().numpy()
        del predicted_depth, prediction

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Normalize to 0-255
    print_progress("Normalizing depth values...")
    if output.max() > output.min():
        formatted = (output - output.min()) / (output.max() - output.min())
    else:
        formatted = output

    formatted = (formatted * 255).astype(np.uint8)

    # Save with quality setting
    final_img = Image.fromarray(formatted)
    final_img.save(output_path, quality=95, optimize=True)
    print_progress(f"Depth map saved to: {output_path}")

    # Final cleanup
    del output, formatted, final_img
    gc.collect()

    return True


def process_360_depth(input_path, output_path, args):
    """
    Main processing function for 360° depth map generation.
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return False

    # Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Dispatch based on mode/model
    if args.model == 'depth_anything' and args.mode == 'full':
        try:
            processor, model = load_depth_anything_model()
            success = process_full_panorama(input_path, output_path, processor, model)

            # Clean up model from memory immediately
            del processor, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if success and not args.no_heal_seams:
                 # Just fix wrap around for full panorama, cube seams don't exist
                 fix_wrap_around(output_path)

                 # Apply gentle global smoothing
                 print_progress("Applying global smoothing for consistency...")
                 img_final = cv2.imread(output_path, 0)
                 smoothed = cv2.bilateralFilter(img_final, 5, 25, 25)
                 cv2.imwrite(output_path, smoothed)
                 del img_final, smoothed
                 gc.collect()
                 print_progress("Smoothing complete")

            return success

        except Exception as e:
            print(f"[ERROR] Processing failed: {e}", file=sys.stderr)
            # Emergency cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    # --- Fallback to Marigold/Cube Method ---

    # 1. Load Image
    print_progress(f"Loading image: {input_path}")
    img = np.array(Image.open(input_path))
    cube_size = args.cube_size

    # 2. Slice to Cube Map
    print_progress("Slicing 360° image into cube faces...", 1, 6)
    try:
        cube_faces = py360convert.e2c(img, face_w=cube_size, mode='bilinear', cube_format='dict')
    except Exception as e:
        print(f"[ERROR] Failed to slice image: {e}", file=sys.stderr)
        return False

    # Clear original image from memory
    del img
    gc.collect()

    # 3. Load Model
    pipe = load_marigold_model()

    face_order = ['F', 'R', 'B', 'L', 'U', 'D']
    face_names = ['Front', 'Right', 'Back', 'Left', 'Up', 'Down']

    # Prepare list of images
    print_progress("Preparing cube faces for processing...")
    all_faces_pil = [Image.fromarray(cube_faces[k]) for k in face_order]

    # Clear cube_faces dict from memory
    del cube_faces
    gc.collect()

    processed_faces_map = {}

    # 4. Run Batch Processing
    batch_size = args.batch_size
    total_batches = (len(all_faces_pil) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(all_faces_pil), batch_size)):
        batch = all_faces_pil[i : i + batch_size]
        batch_keys = face_order[i : i + batch_size]
        batch_names = face_names[i : i + batch_size]

        print_progress(f"Processing faces: {', '.join(batch_names)}", batch_idx + 1, total_batches)

        batch_results = process_batch(pipe, batch, cube_size, args.steps)

        for key, result_img in zip(batch_keys, batch_results):
            processed_faces_map[key] = result_img

        # Clear batch from memory
        del batch, batch_results
        gc.collect()

    # Clear model and input faces from memory before stitching
    del pipe, all_faces_pil
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 5. Stitch back to equirectangular
    print_progress("Stitching cube faces back to 360° panorama...")
    face_list = [processed_faces_map[k] for k in face_order]

    output_w = cube_size * 4
    output_h = cube_size * 2

    depth_equi = py360convert.c2e(face_list, h=output_h, w=output_w, cube_format='list')

    # Clear face data from memory
    del face_list, processed_faces_map
    gc.collect()

    # 6. Save
    final_img = Image.fromarray(depth_equi).convert("L")
    final_img.save(output_path)
    print_progress(f"Depth map saved to: {output_path}")

    # Clear depth equirectangular from memory
    del depth_equi, final_img
    gc.collect()

    # 7. Optional seam healing and smoothing
    if not args.no_heal_seams:
        heal_seams(output_path, output_path)
        fix_wrap_around(output_path)

        # Apply gentle global smoothing to reduce any remaining artifacts
        print_progress("Applying global smoothing for consistency...")
        img_final = cv2.imread(output_path, 0)
        # Very light bilateral filter to smooth while preserving edges
        smoothed = cv2.bilateralFilter(img_final, 5, 25, 25)
        cv2.imwrite(output_path, smoothed)
        del img_final, smoothed
        gc.collect()
        print_progress("Smoothing complete")

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_progress("Depth processing complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate depth maps from 360° equirectangular images'
    )
    parser.add_argument('input', help='Input 360° image path')
    parser.add_argument('output', help='Output depth map path')
    parser.add_argument('--cube-size', type=int, default=512,
                        help='Cube face size (default: 512, for cube mode). Lower values use less RAM.')
    parser.add_argument('--steps', type=int, default=10,
                        help='Inference steps (default: 10, for Marigold)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for processing (default: 2). Lower values use less RAM.')
    parser.add_argument('--no-heal-seams', action='store_true',
                        help='Disable seam healing')
    parser.add_argument('--model', type=str, default='marigold', choices=['marigold', 'depth_anything'],
                        help='Depth model to use (default: marigold)')
    parser.add_argument('--mode', type=str, default='cube', choices=['cube', 'full'],
                        help='Processing mode: cube (slicing) or full (direct panorama)')

    args = parser.parse_args()

    try:
        success = process_360_depth(args.input, args.output, args)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Fatal error during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure cleanup always happens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
