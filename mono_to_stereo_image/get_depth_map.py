import torch
import numpy as np
from PIL import Image
from diffusers import MarigoldDepthPipeline
import os
import multiprocessing
import argparse

def load_depth_model():
    """Loads the official Marigold pipeline."""
    print("Loading Marigold Model...")
    
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    print(f" -> Configured PyTorch to use {num_cores} CPU threads.")

    if torch.cuda.is_available():
        dtype = torch.float16
        device = "cuda"
        print(" -> Using CUDA (GPU)")
    else:
        dtype = torch.float32
        device = "cpu"
        print(" -> Using CPU (Warning: Slow)")

    # Load the model
    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-lcm-v1-0",
        torch_dtype=dtype
    )
    
    pipe.to(device)
    return pipe

def process_single_image(input_path, output_path, steps):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    # 1. Load Image
    print(f"Loading image: {input_path}")
    original_img = Image.open(input_path).convert("RGB")
    original_size = original_img.size # (Width, Height)
    
    # 2. Load Model
    pipe = load_depth_model()
    
    # 3. Run Inference
    print(f"Running depth estimation with {steps} steps...")
    with torch.no_grad():
        # processing_resolution=0 means it keeps original resolution or handles it internally
        # You can set processing_resolution=768 to force a size if you run out of RAM
        results = pipe(original_img, num_inference_steps=steps)
    
    depth_pred = results.prediction 
    
    # 4. Process Output (Normalize & Convert)
    # Convert to numpy if it's a tensor
    if torch.is_tensor(depth_pred):
        depth_pred = depth_pred.cpu().numpy()

    # Squeeze out extra dimensions (turning (1, H, W) into (H, W))
    depth_pred = np.squeeze(depth_pred)
    
    # Normalize to 0-255
    min_val = depth_pred.min()
    max_val = depth_pred.max()
    
    if max_val > min_val:
        norm_img = (depth_pred - min_val) / (max_val - min_val)
    else:
        norm_img = depth_pred
    
    depth_uint8 = (norm_img * 255).astype(np.uint8)
    
    # 5. Create PIL Image
    depth_pil = Image.fromarray(depth_uint8)

    # 6. Resize back to original dimensions
    # Marigold might have changed the size during processing. 
    # We ensure the output matches the input exactly.
    if depth_pil.size != original_size:
        print(f" -> Resizing output from {depth_pil.size} back to {original_size}")
        depth_pil = depth_pil.resize(original_size, Image.BICUBIC)

    # 7. Save
    depth_pil.save(output_path)
    print(f"Done! Depth map saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate depth maps from 2D images using Marigold.")
    parser.add_argument("input_start", help="Input image path")
    parser.add_argument("output_path", help="Output depth map path")
    parser.add_argument("--steps", type=int, default=10, help="Inference steps (default: 10)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_single_image(args.input_start, args.output_path, args.steps)