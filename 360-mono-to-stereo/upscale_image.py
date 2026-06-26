#!/usr/bin/env python3
"""
AI Image Upscaler using Real-ESRGAN
Upscales images using state-of-the-art AI models.
"""

import argparse
import os
import sys
import gc

def print_progress(message):
    """Print progress in a parseable format for the GUI."""
    print(f"[STATUS] {message}", flush=True)


def check_realesrgan():
    """Check if Real-ESRGAN is installed."""
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        return True
    except ImportError:
        return False


def upscale_image(input_path, output_path, scale=2, model_name='RealESRGAN_x4plus', tile_size=512):
    """
    Upscale an image using Real-ESRGAN.

    Args:
        input_path: Path to input image
        output_path: Path for output image
        scale: Upscale factor (2 or 4)
        model_name: Model to use
        tile_size: Tile size for processing (lower = less RAM)
    """
    import torch
    import numpy as np
    from PIL import Image
    import psutil

    if not check_realesrgan():
        print("[ERROR] Real-ESRGAN not installed. Install with:", file=sys.stderr)
        print("  pip install realesrgan", file=sys.stderr)
        return False

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.archs.srvgg_arch import SRVGGNetCompact

    print_progress(f"Loading upscaler model: {model_name}")

    # Check available memory
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print_progress(f"Available RAM: {available_memory_gb:.1f} GB")

    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
        print_progress("Using GPU for upscaling")
    else:
        device = 'cpu'
        print_progress("Using CPU for upscaling (slower)")

        # Adjust tile size based on available memory for CPU processing
        if available_memory_gb < 4:
            tile_size = 128
            print_progress("Low memory detected, using small tiles (128px)")
        elif available_memory_gb < 8:
            tile_size = 192
            print_progress("Medium memory detected, using medium tiles (192px)")
        else:
            tile_size = min(tile_size, 256)
            print_progress(f"Using tile size: {tile_size}px")

    # Model configurations
    if model_name == 'RealESRGAN_x4plus' or model_name.startswith('RealESRGAN_x4plus'):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    elif model_name == 'RealESRGAN_x4plus_anime' or 'anime' in model_name.lower():
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    elif 'general' in model_name.lower() or 'fast' in model_name.lower():
        # This model uses SRVGGNetCompact architecture, not RRDBNet
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    else:
        # Default to x4plus
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

    # Download model if needed
    model_path = os.path.join(os.path.expanduser('~'), '.cache', 'realesrgan', os.path.basename(model_url))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        print_progress(f"Downloading model (first time only)...")
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        print_progress("Model downloaded successfully")

    # Initialize upscaler with memory-efficient settings
    try:
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=True if device == 'cuda' else False,
            device=device
        )
        print_progress("Upscaler initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize upscaler: {e}", file=sys.stderr)
        return False

    # Load image
    print_progress(f"Loading image: {input_path}")
    try:
        import cv2
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[ERROR] Could not load image: {input_path}", file=sys.stderr)
            return False

        h, w = img.shape[:2]
        print_progress(f"Input size: {w}x{h}")

        # Check if image is too large for available memory
        # Rough estimate: input + output + working memory
        estimated_memory_gb = (w * h * 3 * scale * scale * 8) / (1024**3)
        safe_memory_gb = available_memory_gb * 0.6  # Only use 60% of available RAM

        print_progress(f"Estimated memory needed: {estimated_memory_gb:.1f} GB")
        print_progress(f"Safe available memory: {safe_memory_gb:.1f} GB")

        # Refuse to upscale if it will likely crash
        if estimated_memory_gb > safe_memory_gb:
            print(f"[ERROR] Image too large for upscaling with available memory", file=sys.stderr)
            print(f"[ERROR] Need ~{estimated_memory_gb:.1f} GB, have {safe_memory_gb:.1f} GB safe", file=sys.stderr)
            print(f"[ERROR] Recommendation: Use a smaller input image or get more RAM", file=sys.stderr)
            print(f"[ERROR] Alternative: Upscale on a machine with at least {int(estimated_memory_gb * 2)} GB RAM", file=sys.stderr)
            return False

        # Warn if close to memory limit
        if estimated_memory_gb > safe_memory_gb * 0.8:
            print_progress(f"Warning: Image is very large, using minimal tile size")
            tile_size = 64  # Smallest possible tiles
            upsampler.tile = tile_size
            print_progress(f"Using tile size: {tile_size}px to prevent crash")
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}", file=sys.stderr)
        return False

    # Upscale with progressive tile size reduction on failure
    print_progress(f"Upscaling with {scale}x factor...")
    output = None
    attempts = [(tile_size, "initial"), (128, "smaller"), (96, "very small"), (64, "minimal")]

    for attempt_tile, attempt_name in attempts:
        try:
            upsampler.tile = attempt_tile
            if attempt_tile != tile_size:
                print_progress(f"Retrying with {attempt_name} tiles ({attempt_tile}px)...")

            # Free up memory before attempting
            gc.collect()
            if device == 'cpu':
                import psutil
                # Log memory before upscaling
                mem_before = psutil.virtual_memory().available / (1024**3)
                print_progress(f"Available memory before upscaling: {mem_before:.1f} GB")

            output, _ = upsampler.enhance(img, outscale=scale)

            new_h, new_w = output.shape[:2]
            print_progress(f"Output size: {new_w}x{new_h}")
            break  # Success, exit retry loop

        except Exception as e:
            if attempt_tile == 64:  # Last attempt
                print(f"[ERROR] Upscaling failed even with minimal tiles: {e}", file=sys.stderr)
                return False
            else:
                print_progress(f"Failed with {attempt_tile}px tiles: {str(e)[:100]}")
                # Continue to next smaller tile size
                gc.collect()
                continue

    if output is None:
        print(f"[ERROR] Upscaling failed after all attempts", file=sys.stderr)
        return False

    # Save output
    print_progress(f"Saving upscaled image: {output_path}")
    try:
        # Determine format from extension
        ext = os.path.splitext(output_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif ext == '.png':
            cv2.imwrite(output_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(output_path, output)

        print_progress(f"Upscaled image saved successfully")
    except Exception as e:
        print(f"[ERROR] Failed to save image: {e}", file=sys.stderr)
        return False

    # Cleanup
    del img, output, upsampler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_progress("Upscaling complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='AI Image Upscaler using Real-ESRGAN')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 4],
                        help='Upscale factor (default: 2)')
    parser.add_argument('--model', type=str, default='RealESRGAN_x4plus',
                        help='Model name (default: RealESRGAN_x4plus)')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Tile size for processing (default: 512, lower = less RAM)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    success = upscale_image(
        args.input,
        args.output,
        scale=args.scale,
        model_name=args.model,
        tile_size=args.tile_size
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
