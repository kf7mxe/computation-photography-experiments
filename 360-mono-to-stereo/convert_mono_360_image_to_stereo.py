import torch
import numpy as np
import py360convert
from PIL import Image
from diffusers import MarigoldDepthPipeline
import os
import multiprocessing

# --- CONFIGURATION ---
INPUT_IMAGE_PATH = "PANO_20200915_191725.jpg"  # Ensure this filename matches your image
OUTPUT_FILENAME = "depth_360_output.jpg"
CUBE_SIZE = 512   
STEPS = 10       
BATCH_SIZE = 6




import cv2 # Ensure you have this imported

def heal_seams(image_path, output_path):
    print(" -> Running Seam Healer...")
    
    # 1. Load the Depth Map
    # We load as grayscale (0)
    img = cv2.imread(image_path, 0)
    h, w = img.shape
    
    # 2. Create a Mask for the Seams
    # We want to tell OpenCV: "Fix pixels at these specific locations"
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # THICKNESS: How wide the blend should be (increase if seams are still visible)
    thickness = 15 
    
    # --- VERTICAL SEAMS ---
    # In a Cube -> Equirectangular conversion, vertical seams usually appear 
    # at 0%, 25%, 50%, and 75% of the image width.
    seam_locs = [0, w//4, w//2, 3*w//4, w-thickness] # Added w-thickness for the wrap-around
    
    for x in seam_locs:
        # Draw a white vertical line on the mask at these locations
        # cv2.rectangle(img, start_point, end_point, color, thickness)
        cv2.rectangle(mask, (x - thickness, 0), (x + thickness, h), 255, -1)
        
    # --- HORIZONTAL SEAMS ---
    # Sometimes horizontal lines appear at the top/bottom boundaries of the main view
    # Usually around 25% and 75% height roughly, but less predictable. 
    # Let's focus on the vertical ones first as they are most noticeable in VR.

    # 3. Apply Inpainting (The Magic Step)
    # NS = Navier-Stokes algorithm (fluid-like smoothing)
    # This looks at pixels *next* to the white lines and flows them into the gap.
    healed_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    
    # 4. Save
    cv2.imwrite(output_path, healed_img)
    print(f" -> Seams healed. Overwrote {output_path}")



def fix_wrap_around(image_path):
    # Load image
    img = cv2.imread(image_path, 0)
    h, w = img.shape
    
    # 1. Cut a slice from the left and append it to the right
    slice_width = 50
    left_slice = img[:, 0:slice_width]
    right_slice = img[:, w-slice_width:w]
    
    # 2. Blend them? 
    # Actually, Inpainting is better. Let's force inpaint on the edges.
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Mark the far left and far right edges as "damaged"
    mask[:, 0:20] = 255
    mask[:, w-20:w] = 255
    
    # Inpaint relies on neighbors. For the edge of an image, it has no neighbor on one side.
    # Trick: We wrap the image using numpy.roll, inpaint the middle, then roll back.
    
    # Shift image so the seam is in the Center
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

def load_depth_model():
    """Loads the official Marigold pipeline."""
    print("Loading Marigold Model...")
    
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    print(f" -> Configured PyTorch to use {num_cores} CPU threads.")

    if torch.cuda.is_available():
        dtype = torch.float16
        device = "cuda"
    else:
        dtype = torch.float32
        device = "cpu"

    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-lcm-v1-0",
        torch_dtype=dtype
    )
    
    pipe.to(device)
    return pipe

def process_batch(pipe, image_pil_list):
    """Runs AI inference on a list of images (Batching)."""
    
    results = pipe(image_pil_list, num_inference_steps=STEPS)
    
    predictions = results.prediction 
    
    # Convert to numpy if it's a tensor
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    processed_images = []
    
    # Iterate through the batch
    for i in range(len(predictions)):
        depth_np = predictions[i]
        
        # --- PARANOID DIMENSION FIX ---
        # Squeeze out '1' dimensions (turning (1, 512, 512) into (512, 512))
        depth_np = np.squeeze(depth_np)
        
        # Normalize to 0-255
        if depth_np.max() > depth_np.min():
            norm_img = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        else:
            norm_img = depth_np
        
        norm_img = (norm_img * 255).astype(np.uint8)
        
        # --- SAFETY RESIZE ---
        # Ensure it is exactly CUBE_SIZE (Marigold sometimes changes resolution)
        if norm_img.shape[0] != CUBE_SIZE:
             # Use PIL to resize quickly
             norm_pil = Image.fromarray(norm_img)
             norm_pil = norm_pil.resize((CUBE_SIZE, CUBE_SIZE), Image.BICUBIC)
             norm_img = np.array(norm_pil)

        # --- RGB CONVERSION ---
        # py360convert sometimes prefers 3 channels. We stack grayscale to RGB.
        # This makes the array shape (512, 512, 3)
        norm_img_rgb = np.stack([norm_img]*3, axis=-1)
        
        processed_images.append(norm_img_rgb)
        
    return processed_images

def process_360_depth(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Input file '{image_path}' not found.")
        return

    # 1. Load Image
    print(f"Loading image: {image_path}")
    img = np.array(Image.open(image_path))
    
    # 2. Slice to Cube Map
    print(f"Slicing 360 image into Cube Map...")
    try:
        cube_faces = py360convert.e2c(img, face_w=CUBE_SIZE, mode='bilinear', cube_format='dict')
    except Exception as e:
        print(f"Error during slicing: {e}")
        return
    
    # 3. Load Model
    pipe = load_depth_model()
    
    face_order = ['F', 'R', 'B', 'L', 'U', 'D']
    
    # Prepare list of images
    print("Preparing batch...")
    all_faces_pil = [Image.fromarray(cube_faces[k]) for k in face_order]
    
    processed_faces_map = {}
    
    # 4. Run Batch Processing
    print(f"Processing faces in batches of {BATCH_SIZE}...")
    
    for i in range(0, len(all_faces_pil), BATCH_SIZE):
        batch = all_faces_pil[i : i + BATCH_SIZE]
        batch_keys = face_order[i : i + BATCH_SIZE]
        
        print(f" -> Running batch for faces: {batch_keys}")
        
        batch_results = process_batch(pipe, batch)
        
        for key, result_img in zip(batch_keys, batch_results):
            processed_faces_map[key] = result_img

    # 5. Stitch back
    print("Stitching back to 360...")
    face_list = [processed_faces_map[k] for k in face_order]
    
    output_w = CUBE_SIZE * 4
    output_h = CUBE_SIZE * 2
    
    # Note: We now have RGB images in the list, so c2e will output an RGB image.
    depth_equi = py360convert.c2e(face_list, h=output_h, w=output_w, cube_format='list')
    
    # 6. Save
    # We save as grayscale since R=G=B
    final_img = Image.fromarray(depth_equi).convert("L") 
    final_img.save(OUTPUT_FILENAME)
    print(f"Done! Saved to {OUTPUT_FILENAME}")
    # heal_seams(OUTPUT_FILENAME, OUTPUT_FILENAME)
    # fix_wrap_around(OUTPUT_FILENAME)
    # print(" -> Wrap-around seam fixed.")

if __name__ == "__main__":
    process_360_depth(INPUT_IMAGE_PATH)