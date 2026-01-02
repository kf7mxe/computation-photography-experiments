import torch
import numpy as np
from PIL import Image
from diffusers import MarigoldDepthPipeline
from transformers import pipeline as hf_pipeline
import os
import multiprocessing
import argparse

DEPTH_MODELS = {
    "marigold-lcm": "prs-eth/marigold-lcm-v1-0",
    "marigold": "prs-eth/marigold-v1-0",
    "depth-anything-v2-small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything-v2-base": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything-v2-large": "depth-anything/Depth-Anything-V2-Large-hf",
}

def load_depth_model(model_name="depth-anything-v2-small"):
    """Loads the specified depth estimation model."""
    print(f"Loading Depth Model: {model_name}...")
    print("(First run may download model files - this can take several minutes)")
    import sys
    sys.stdout.flush()

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

    sys.stdout.flush()

    model_id = DEPTH_MODELS.get(model_name)
    if not model_id:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(DEPTH_MODELS.keys())}")
        return None

    # Load different models based on type
    if "marigold" in model_name:
        print(f" -> Loading Marigold from HuggingFace...")
        sys.stdout.flush()
        pipe = MarigoldDepthPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        )
        pipe.to(device)
        print(" -> Model loaded successfully!")
        sys.stdout.flush()
        return {"type": "marigold", "pipe": pipe}
    else:
        # Depth Anything V2 uses transformers pipeline
        print(f" -> Loading Depth Anything V2 from HuggingFace...")
        print(f" -> This may take a while on first run (downloading ~400MB)...")
        sys.stdout.flush()

        try:
            pipe = hf_pipeline(
                task="depth-estimation",
                model=model_id,
                device=0 if device == "cuda" else -1
            )
            print(" -> Model loaded successfully!")
            sys.stdout.flush()
            return {"type": "depth-anything", "pipe": pipe}
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print("If download failed, try using 'marigold-lcm' instead (smaller, faster)")
            sys.stdout.flush()
            raise

    return pipe

def process_single_image(input_path, output_path, steps, model_name="depth-anything-v2-small"):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    # 1. Load Image
    print(f"Loading image: {input_path}")
    original_img = Image.open(input_path).convert("RGB")
    original_size = original_img.size # (Width, Height)

    # 2. Load Model
    model_info = load_depth_model(model_name)
    if not model_info:
        return

    # 3. Run Inference
    print(f"Running depth estimation...")

    if model_info["type"] == "marigold":
        with torch.no_grad():
            results = model_info["pipe"](original_img, num_inference_steps=steps)
        depth_pred = results.prediction

        # Convert to numpy if it's a tensor
        if torch.is_tensor(depth_pred):
            depth_pred = depth_pred.cpu().numpy()
        depth_pred = np.squeeze(depth_pred)

    else:  # depth-anything
        result = model_info["pipe"](original_img)
        depth_pred = np.array(result["depth"])

    # 4. Process Output (Normalize & Convert)
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

    # 6. Resize back to original dimensions if needed
    if depth_pil.size != original_size:
        print(f" -> Resizing output from {depth_pil.size} back to {original_size}")
        depth_pil = depth_pil.resize(original_size, Image.BICUBIC)

    # 7. Save
    depth_pil.save(output_path)
    print(f"Done! Depth map saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate depth maps from 2D images.")
    parser.add_argument("input_start", help="Input image path")
    parser.add_argument("output_path", help="Output depth map path")
    parser.add_argument("--steps", type=int, default=10, help="Inference steps (only for Marigold models, default: 10)")
    parser.add_argument("--model", type=str, default="depth-anything-v2-small",
                        choices=list(DEPTH_MODELS.keys()),
                        help="Depth model to use (default: depth-anything-v2-small)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_single_image(args.input_start, args.output_path, args.steps, args.model)