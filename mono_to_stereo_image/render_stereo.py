import bpy
import sys
import os
import math
import argparse

def create_stereo_optimized(args):
    print(f"--- Starting Optimized Stereo Render ---")

    color_path = args.input_path
    depth_path = args.depth_path
    output_path = args.output_path
    
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # --- MEMORY OPTIMIZATION 1: TILE SIZE ---
    scene.cycles.use_auto_tile = True
    
    # Setup GPU
    try:
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA' 
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            if device.type != 'CPU': device.use = True
        scene.cycles.device = 'GPU'
    except:
        scene.cycles.device = 'CPU'

    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False

    # 2. LOAD IMAGES
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    try:
        img_col = bpy.data.images.load(color_path)
        img_dep = bpy.data.images.load(depth_path)
        img_dep.colorspace_settings.name = 'Non-Color'
    except Exception as e:
        print(f"Error: {e}")
        return

    w = img_col.size[0]
    h = img_col.size[1]
    aspect_ratio = w / h
    print(f" -> Input: {w}x{h} (Aspect: {aspect_ratio:.3f})")

    # 3. CREATE OPTIMIZED PLANE (High resolution for smooth displacement)
    # Higher base resolution = less warping/smudging on displaced geometry
    base_res = 384  # Increased from 256 for even smoother results
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=base_res, y_subdivisions=base_res, size=1)
    plane = bpy.context.active_object
    plane.rotation_euler = (math.radians(90), 0, 0)

    plane.scale[0] = aspect_ratio
    plane.scale[1] = 1.0
    bpy.ops.object.transform_apply(scale=True)

    bpy.ops.object.shade_smooth()

    # Subdivision modifier for final detail
    # Lower levels are fine since base mesh is already high-res
    subsurf = plane.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.render_levels = args.subdivisions
    subsurf.levels = 1  # Viewport level stays low for responsiveness

    # Use adaptive subdivision for better quality on high-detail areas
    subsurf.subdivision_type = 'CATMULL_CLARK'
    subsurf.quality = 4  # Higher quality subdivision

    # 4. MATERIAL (With Edge Pinning)
    mat = bpy.data.materials.new(name="StereoMat")
    mat.use_nodes = True
    plane.data.materials.append(mat)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new('ShaderNodeOutputMaterial')
    node_emit = nodes.new('ShaderNodeEmission')
    node_disp = nodes.new('ShaderNodeDisplacement')
    node_tex_c = nodes.new('ShaderNodeTexImage')
    node_tex_d = nodes.new('ShaderNodeTexImage')
    
    # Edge Pinning logic - IMPROVED to reduce artifacts on foreground objects
    node_uv = nodes.new('ShaderNodeTexCoord')
    node_xyz = nodes.new('ShaderNodeSeparateXYZ')
    node_math_x1 = nodes.new('ShaderNodeMath')
    node_math_x2 = nodes.new('ShaderNodeMath')
    node_math_y1 = nodes.new('ShaderNodeMath')
    node_math_y2 = nodes.new('ShaderNodeMath')
    node_math_mult = nodes.new('ShaderNodeMath')
    node_math_sharp = nodes.new('ShaderNodeMath')
    node_clamp = nodes.new('ShaderNodeMath')
    node_power = nodes.new('ShaderNodeMath')  # NEW: Soften edge transition
    node_mix_depth = nodes.new('ShaderNodeMixRGB')

    node_tex_c.image = img_col
    node_tex_c.extension = 'EXTEND'
    node_tex_d.image = img_dep
    node_tex_d.extension = 'EXTEND'

    # Constants - Reduced border fade for less aggressive pinning
    BORDER_FADE = 0.02  # Reduced from 0.05 - only pin extreme edges

    # Math Setup
    node_math_x1.operation = 'SUBTRACT'
    node_math_x1.inputs[0].default_value = 1.0
    node_math_x2.operation = 'MINIMUM'
    node_math_y1.operation = 'SUBTRACT'
    node_math_y1.inputs[0].default_value = 1.0
    node_math_y2.operation = 'MINIMUM'
    node_math_mult.operation = 'MULTIPLY'
    node_math_sharp.operation = 'DIVIDE'
    node_math_sharp.inputs[1].default_value = BORDER_FADE
    node_power.operation = 'POWER'
    node_power.inputs[1].default_value = 2.0  # Square the mask for smoother falloff
    node_clamp.operation = 'MINIMUM'
    node_clamp.inputs[1].default_value = 1.0
    node_mix_depth.inputs[1].default_value = (0.5, 0.5, 0.5, 1.0)

    # Links - Added power node for smoother transition
    links.new(node_uv.outputs['UV'], node_xyz.inputs['Vector'])
    links.new(node_xyz.outputs['X'], node_math_x1.inputs[1])
    links.new(node_xyz.outputs['X'], node_math_x2.inputs[0])
    links.new(node_math_x1.outputs['Value'], node_math_x2.inputs[1])
    links.new(node_xyz.outputs['Y'], node_math_y1.inputs[1])
    links.new(node_xyz.outputs['Y'], node_math_y2.inputs[0])
    links.new(node_math_y1.outputs['Value'], node_math_y2.inputs[1])
    links.new(node_math_x2.outputs['Value'], node_math_mult.inputs[0])
    links.new(node_math_y2.outputs['Value'], node_math_mult.inputs[1])
    links.new(node_math_mult.outputs['Value'], node_math_sharp.inputs[0])
    links.new(node_math_sharp.outputs['Value'], node_power.inputs[0])  # NEW: Soften with power
    links.new(node_power.outputs['Value'], node_clamp.inputs[0])
    links.new(node_clamp.outputs['Value'], node_mix_depth.inputs['Fac'])
    links.new(node_tex_d.outputs['Color'], node_mix_depth.inputs[2])

    links.new(node_tex_c.outputs['Color'], node_emit.inputs['Color'])
    links.new(node_emit.outputs['Emission'], node_out.inputs['Surface'])
    links.new(node_mix_depth.outputs['Color'], node_disp.inputs['Height'])
    links.new(node_disp.outputs['Displacement'], node_out.inputs['Displacement'])

    node_disp.inputs['Scale'].default_value = args.displacement
    node_disp.inputs['Midlevel'].default_value = 0.5
    mat.cycles.displacement_method = 'DISPLACEMENT'

    # 5. CAMERA
    cam_data = bpy.data.cameras.new(name='StereoCam')
    cam_obj = bpy.data.objects.new(name='StereoCam', object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_obj.rotation_euler = (math.radians(90), 0, 0)

    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.sensor_width = 36.0
    cam_data.sensor_height = 36.0 / aspect_ratio
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(60) 

    SAFETY_MARGIN = 1.15
    half_width = aspect_ratio / 2.0
    dist = half_width / math.tan(cam_data.angle / 2.0)
    final_dist = dist * SAFETY_MARGIN
    cam_obj.location = (0, -final_dist, 0)

    cam_data.stereo.interocular_distance = args.ipd
    cam_data.stereo.convergence_distance = final_dist

    # 6. RENDER
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.use_multiview = True
    scene.render.views_format = 'STEREO_3D'
    scene.render.image_settings.views_format = 'STEREO_3D'
    scene.render.image_settings.stereo_3d_format.display_mode = args.mode
    scene.render.image_settings.stereo_3d_format.use_squeezed_frame = True

    # Use PNG for better stereo compatibility, or set quality for JPEG
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        scene.render.image_settings.file_format = 'JPEG'
        scene.render.image_settings.quality = 95
        # Remove extension - Blender adds it
        filepath = output_path.rsplit('.', 1)[0]
    elif output_path.lower().endswith('.png'):
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.compression = 15
        filepath = output_path.rsplit('.', 1)[0]
    else:
        filepath = output_path
        scene.render.image_settings.file_format = 'PNG'

    scene.render.filepath = filepath

    print(f" -> Rendering Optimized ({w}x{h})...")
    print(f" -> Output path: {filepath}")
    bpy.ops.render.render(write_still=True)

    # Verify file was actually saved
    import glob
    possible_files = glob.glob(f"{filepath}*")
    if possible_files:
        actual_file = possible_files[0]
        print(f" -> SUCCESS: File saved to: {actual_file}")
    else:
        print(f" -> WARNING: Could not find saved file at {filepath}")
        print(f" -> Check Blender's temp directory or output settings")

    print(" -> Done.")

if __name__ == "__main__":
    # Because Blender is weird with sys.argv, we need to extract args after "--"
    # usage: blender -b -P script.py -- [args]
    
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render stereo 3D from 2D + Depth.")
    parser.add_argument("input_path", help="Input color image")
    parser.add_argument("depth_path", help="Input depth map")
    parser.add_argument("output_path", help="Output stereo image")

    parser.add_argument("--displacement", type=float, default=0.2, help="Displacement strength (lower = less warping)")
    parser.add_argument("--ipd", type=float, default=0.065, help="Interpupillary Distance")
    parser.add_argument("--samples", type=int, default=32, help="Render samples")
    parser.add_argument("--subdivisions", type=int, default=2, help="Subdivision levels (2-3 recommended)")
    parser.add_argument("--mode", type=str, default='TOPBOTTOM', choices=['TOPBOTTOM', 'SIDEBYSIDE', 'ANAGLYPH'], help="Stereo mode")

    try:
        args = parser.parse_args(argv)
        create_stereo_optimized(args)
    except Exception as e:
        print(f"Error parsing args: {e}")
        sys.exit(1)