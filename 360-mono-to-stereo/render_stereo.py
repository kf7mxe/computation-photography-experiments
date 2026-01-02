#!/usr/bin/env python3
"""
Stereo 360° Renderer for Blender
Creates stereoscopic 3D renders from a color image and depth map.
Usage: blender -b -P render_stereo.py -- color.jpg depth.jpg output.jpg [options]
"""

import bpy
import sys
import os
import math
import argparse


def print_progress(message):
    """Print progress in a parseable format for the GUI."""
    print(f"[STATUS] {message}", flush=True)


def create_stereo_360(color_path, depth_path, output_path, 
                      displacement_scale=0.20, ipd=0.065, 
                      samples=32, subdivisions=5):
    """
    Create a stereoscopic 360° render from color and depth images.
    
    Args:
        color_path: Path to the color/texture image
        depth_path: Path to the depth map image
        output_path: Path for the output stereo image
        displacement_scale: Strength of 3D displacement (default 0.20)
        ipd: Interocular distance in meters (default 0.065)
        samples: Render samples (default 32)
        subdivisions: Subdivision levels for detail (default 5)
    """
    
    print_progress("Starting stereo render")
    print_progress(f"Color: {color_path}")
    print_progress(f"Depth: {depth_path}")
    print_progress(f"Displacement: {displacement_scale}, IPD: {ipd}m")

    # 1. SETUP SCENE
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Set Device (Try GPU with CUDA/HIP/OptiX, Fallback to CPU)
    gpu_enabled = False
    try:
        preferences = bpy.context.preferences
        cycles_prefs = preferences.addons['cycles'].preferences

        # Try different compute device types in order of preference
        device_types_to_try = ['HIP', 'CUDA', 'OPTIX', 'METAL']

        for compute_type in device_types_to_try:
            try:
                cycles_prefs.compute_device_type = compute_type
                cycles_prefs.get_devices()

                found_gpu = False
                for device in cycles_prefs.devices:
                    if device.type != 'CPU':
                        device.use = True
                        found_gpu = True
                        print_progress(f"Found GPU device: {device.name} ({compute_type})")

                if found_gpu:
                    scene.cycles.device = 'GPU'
                    print_progress(f"GPU rendering enabled with {compute_type}")
                    gpu_enabled = True
                    break
            except:
                continue

        if not gpu_enabled:
            scene.cycles.device = 'CPU'
            print_progress("No GPU found, using CPU rendering (slower)")
            print_progress("For AMD GPUs: Ensure Blender is compiled with HIP/ROCm support")
            print_progress("For NVIDIA GPUs: Ensure CUDA drivers are installed")
    except Exception as e:
        scene.cycles.device = 'CPU'
        print_progress(f"GPU setup failed: {e}")
        print_progress("Using CPU rendering")

    # Render Settings
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False

    # 2. CLEAR SCENE
    print_progress("Preparing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 3. CREATE SPHERE FOR 360° STEREO
    print_progress("Creating displaced sphere...")
    # Use radius of 2.0 for better depth perception at comfortable viewing distance
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=256,  # Increased for smoother 360° viewing
        ring_count=128,  # Increased for better vertical resolution
        radius=2.0,  # Larger radius for better stereo separation
        location=(0, 0, 0)
    )
    sphere = bpy.context.active_object

    # Flip normals so we're looking at the inside of the sphere
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.shade_smooth()

    # Add Subdivision Modifier for smoother displacement
    subsurf = sphere.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.render_levels = subdivisions
    subsurf.levels = min(2, subdivisions)

    # 4. CREATE MATERIAL
    print_progress("Setting up materials...")
    mat = bpy.data.materials.new(name="StereoMat")
    mat.use_nodes = True
    sphere.data.materials.append(mat)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create nodes
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_disp = nodes.new(type='ShaderNodeDisplacement')

    # Load Color Image
    try:
        img_col = bpy.data.images.load(color_path)
        node_tex_color = nodes.new(type='ShaderNodeTexImage')
        node_tex_color.image = img_col
        print_progress(f"Loaded color image: {img_col.size[0]}x{img_col.size[1]}")
    except Exception as e:
        print(f"[ERROR] Could not load color image: {color_path}", file=sys.stderr)
        print(f"[ERROR] {e}", file=sys.stderr)
        return False

    # Load Depth Image
    try:
        img_dep = bpy.data.images.load(depth_path)
        img_dep.colorspace_settings.name = 'Non-Color'
        node_tex_depth = nodes.new(type='ShaderNodeTexImage')
        node_tex_depth.image = img_dep
        print_progress(f"Loaded depth image: {img_dep.size[0]}x{img_dep.size[1]}")
    except Exception as e:
        print(f"[ERROR] Could not load depth image: {depth_path}", file=sys.stderr)
        print(f"[ERROR] {e}", file=sys.stderr)
        return False

    # Add ColorRamp to control depth mapping
    node_colorramp = nodes.new(type='ShaderNodeValToRGB')
    node_colorramp.color_ramp.elements[0].position = 0.0
    node_colorramp.color_ramp.elements[1].position = 1.0

    # DO NOT INVERT: Brighter depth values = farther away = push inward on sphere interior
    # Darker depth values = closer = pop out toward viewer
    # This creates the correct 3D pop-out effect
    node_colorramp.color_ramp.elements[0].color = (0, 0, 0, 1)  # Black at 0 (close/pop-out)
    node_colorramp.color_ramp.elements[1].color = (1, 1, 1, 1)  # White at 1 (far/push-in)

    # Link nodes
    links.new(node_tex_color.outputs['Color'], node_emission.inputs['Color'])
    links.new(node_emission.outputs['Emission'], node_output.inputs['Surface'])
    links.new(node_tex_depth.outputs['Color'], node_colorramp.inputs['Fac'])
    links.new(node_colorramp.outputs['Color'], node_disp.inputs['Height'])
    links.new(node_disp.outputs['Displacement'], node_output.inputs['Displacement'])

    # Displacement settings - significantly increased for stronger 3D effect
    # The displacement scale needs to be much larger for visible depth on a radius=2.0 sphere
    # Using 5x multiplier instead of 2x for dramatic depth perception
    node_disp.inputs['Scale'].default_value = displacement_scale * 5.0
    node_disp.inputs['Midlevel'].default_value = 0.5
    mat.cycles.displacement_method = 'DISPLACEMENT'

    # 5. SETUP CAMERA FOR OMNIDIRECTIONAL STEREO (ODS)
    print_progress("Configuring omnidirectional stereo camera...")
    cam_data = bpy.data.cameras.new(name='StereoCam')
    cam_obj = bpy.data.objects.new(name='StereoCam', object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_obj.location = (0, 0, 0)
    cam_obj.rotation_euler = (math.radians(90), 0, 0)

    # Use panoramic camera with proper 360 stereo settings
    cam_data.type = 'PANO'
    cam_data.panorama_type = 'EQUIRECTANGULAR'

    # Critical: Use PARALLEL stereo convergence mode for 360° viewing
    # This creates proper omnidirectional stereo instead of convergent stereo
    cam_data.stereo.convergence_mode = 'PARALLEL'
    cam_data.stereo.interocular_distance = ipd
    cam_data.stereo.convergence_distance = 2.0  # Match sphere radius for proper parallax

    # 6. RESOLUTION & STEREO SETUP
    w = img_col.size[0]
    h = img_col.size[1]
    
    scene.render.resolution_x = w
    scene.render.resolution_y = h * 2  # Top-Bottom stereo format
    
    scene.render.use_multiview = True
    scene.render.views_format = 'STEREO_3D'
    scene.render.image_settings.views_format = 'STEREO_3D'
    scene.render.image_settings.stereo_3d_format.display_mode = 'TOPBOTTOM'
    
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 95
    scene.render.filepath = output_path

    # 7. RENDER
    print_progress(f"Rendering stereo image ({w}x{h*2})...")
    bpy.ops.render.render(write_still=True)
    print_progress("Render complete!")
    print_progress(f"Output saved to: {output_path}")
    
    return True


def parse_args():
    """Parse command line arguments after the -- separator."""
    argv = sys.argv
    
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
    else:
        args = []
    
    parser = argparse.ArgumentParser(
        description='Render stereoscopic 360° images in Blender'
    )
    parser.add_argument('color', help='Color/texture image path')
    parser.add_argument('depth', help='Depth map image path')
    parser.add_argument('output', help='Output stereo image path')
    parser.add_argument('--displacement', type=float, default=0.20,
                        help='Displacement scale (default: 0.20)')
    parser.add_argument('--ipd', type=float, default=0.065,
                        help='Interocular distance in meters (default: 0.065)')
    parser.add_argument('--samples', type=int, default=32,
                        help='Render samples (default: 32)')
    parser.add_argument('--subdivisions', type=int, default=5,
                        help='Subdivision levels (default: 5)')
    
    return parser.parse_args(args)


if __name__ == "__main__":
    try:
        args = parse_args()
        
        # Normalize paths
        cwd = os.getcwd()
        color_img = args.color if os.path.isabs(args.color) else os.path.join(cwd, args.color)
        depth_img = args.depth if os.path.isabs(args.depth) else os.path.join(cwd, args.depth)
        out_img = args.output if os.path.isabs(args.output) else os.path.join(cwd, args.output)

        success = create_stereo_360(
            color_path=color_img,
            depth_path=depth_img,
            output_path=out_img,
            displacement_scale=args.displacement,
            ipd=args.ipd,
            samples=args.samples,
            subdivisions=args.subdivisions
        )
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        print("\nUsage: blender -b -P render_stereo.py -- color.jpg depth.jpg output.jpg [options]")
        print("Options:")
        print("  --displacement FLOAT  Displacement scale (default: 0.20)")
        print("  --ipd FLOAT          Eye separation in meters (default: 0.065)")
        print("  --samples INT        Render samples (default: 32)")
        print("  --subdivisions INT   Subdivision levels (default: 5)")
        sys.exit(1)