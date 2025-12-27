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
    
    # Set Device (Try GPU, Fallback to CPU)
    try:
        preferences = bpy.context.preferences
        cycles_prefs = preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'
        cycles_prefs.get_devices()
        
        found_gpu = False
        for device in cycles_prefs.devices:
            if device.type != 'CPU':
                device.use = True
                found_gpu = True
        
        if found_gpu:
            scene.cycles.device = 'GPU'
            print_progress("GPU rendering enabled")
        else:
            scene.cycles.device = 'CPU'
            print_progress("Using CPU rendering (slower)")
    except:
        scene.cycles.device = 'CPU'
        print_progress("GPU setup failed, using CPU")

    # Render Settings
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False

    # 2. CLEAR SCENE
    print_progress("Preparing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 3. CREATE SPHERE
    print_progress("Creating displaced sphere...")
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=128, 
        ring_count=64, 
        radius=1, 
        location=(0, 0, 0)
    )
    sphere = bpy.context.active_object
    bpy.ops.object.shade_smooth()

    # Add Subdivision Modifier
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

    # Link nodes
    links.new(node_tex_color.outputs['Color'], node_emission.inputs['Color'])
    links.new(node_emission.outputs['Emission'], node_output.inputs['Surface'])
    links.new(node_tex_depth.outputs['Color'], node_disp.inputs['Height'])
    links.new(node_disp.outputs['Displacement'], node_output.inputs['Displacement'])

    # Displacement settings
    node_disp.inputs['Scale'].default_value = displacement_scale
    node_disp.inputs['Midlevel'].default_value = 0.5
    mat.cycles.displacement_method = 'DISPLACEMENT'

    # 5. SETUP CAMERA
    print_progress("Configuring stereoscopic camera...")
    cam_data = bpy.data.cameras.new(name='StereoCam')
    cam_obj = bpy.data.objects.new(name='StereoCam', object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_obj.location = (0, 0, 0)
    cam_obj.rotation_euler = (math.radians(90), 0, 0)

    cam_data.type = 'PANO'
    cam_data.panorama_type = 'EQUIRECTANGULAR'
    cam_data.stereo.interocular_distance = ipd

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