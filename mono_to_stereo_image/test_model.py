#!/usr/bin/env python3
"""
Quick test script to verify depth models load correctly.
Run this before using the GUI to ensure models are downloaded.
"""

import sys
import os

def test_model(model_name):
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}\n")

    try:
        from get_depth_map import load_depth_model

        print("Attempting to load model...")
        model_info = load_depth_model(model_name)

        if model_info:
            print(f"\n✓ SUCCESS: Model '{model_name}' loaded successfully!")
            print(f"  Type: {model_info['type']}")
            return True
        else:
            print(f"\n✗ FAILED: Model '{model_name}' failed to load")
            return False

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Depth Model Test Utility")
    print("This will download models on first run (~400MB for Depth Anything)")
    print()

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        test_model(model_name)
    else:
        # Test the default model
        print("Testing default model: depth-anything-v2-small")
        print("(Pass a model name as argument to test a different model)")
        print()

        success = test_model("depth-anything-v2-small")

        if success:
            print("\n" + "="*60)
            print("Model is ready! You can now use the GUI.")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("Model failed to load. Try 'marigold-lcm' instead:")
            print("  python test_model.py marigold-lcm")
            print("="*60)
            sys.exit(1)
