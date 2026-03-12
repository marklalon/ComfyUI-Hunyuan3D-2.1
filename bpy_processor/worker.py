#!/usr/bin/env python3
"""
worker.py - Blender Python subprocess worker (bpy 5.0.1)

This script runs in a separate Python 3.11 environment with bpy installed.
It receives commands via command line arguments and performs mesh operations.

Usage:
    python worker.py <input_obj> <output_obj> --auto-smooth <angle> [--uv-method <method>]

Arguments:
    input_obj       Path to input OBJ file
    output_obj      Path to output OBJ file
    --auto-smooth   Auto smooth angle in degrees (default: 30)
    --uv-method     UV unwrap method: smart_project, lightmap_pack, cube_project, or none
"""

import argparse
import sys
import os
import math


def setup_blender_scene(obj_path: str) -> bool:
    """Setup Blender scene and import OBJ file."""
    import bpy

    # Clear existing mesh objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Import OBJ (built-in since Blender 4.0)
    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
    except Exception as e:
        print(f"ERROR: Failed to import OBJ: {e}", file=sys.stderr)
        return False

    # Get the imported mesh objects
    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not mesh_objects:
        print("ERROR: No mesh objects found after import", file=sys.stderr)
        return False

    # Set active and merge if multiple
    bpy.context.view_layer.objects.active = mesh_objects[0]
    for obj in mesh_objects:
        obj.select_set(True)

    if len(mesh_objects) > 1:
        bpy.ops.object.join()

    return True


def apply_auto_smooth(angle_degrees: float) -> bool:
    """Apply auto smooth via shade_auto_smooth (bpy 5.0+)."""
    import bpy

    obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        return False

    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')

    # Recalculate normals
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.shade_auto_smooth(angle=math.radians(angle_degrees))

    return True


def apply_uv_unwrap(method: str) -> bool:
    """Apply UV unwrapping to the active mesh object."""
    import bpy

    obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        return False

    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    if method == 'smart_project':
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(66.0),
            island_margin=0.02,
            correct_aspect=True,
            scale_to_bounds=False
        )
    elif method == 'lightmap_pack':
        bpy.ops.uv.lightmap_pack(
            PREF_CONTEXT='SEL_FACES',
            PREF_MARGIN_DIV=0.1
        )
    elif method == 'cube_project':
        bpy.ops.uv.cube_project(
            cube_size=1.0,
            correct_aspect=True,
            clip_to_bounds=False,
            scale_to_bounds=True
        )

    bpy.ops.object.mode_set(mode='OBJECT')
    return True


def apply_decimate(ratio: float) -> bool:
    """Apply Decimate modifier to reduce polygon count."""
    import bpy

    obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        return False

    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    mod = obj.modifiers.new(name='Decimate', type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = max(0.0, min(1.0, ratio))

    bpy.ops.object.modifier_apply(modifier=mod.name)

    return True


def export_obj(output_path: str) -> bool:
    """Export the active mesh to OBJ file (built-in since Blender 4.0)."""
    import bpy

    obj = bpy.context.active_object
    if obj is None:
        print("ERROR: No active object to export", file=sys.stderr)
        return False

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    try:
        bpy.ops.wm.obj_export(
            filepath=output_path,
            export_selected_objects=True,
            export_materials=False,
            export_triangulated_mesh=False,
            export_normals=True,
            export_uv=True
        )
    except Exception as e:
        print(f"ERROR: Failed to export OBJ: {e}", file=sys.stderr)
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Blender mesh processor worker')
    parser.add_argument('input_obj', help='Path to input OBJ file')
    parser.add_argument('output_obj', help='Path to output OBJ file')
    parser.add_argument('--auto-smooth', type=float, default=30.0,
                        help='Auto smooth angle in degrees')
    parser.add_argument('--uv-method', default='none',
                        choices=['smart_project', 'lightmap_pack', 'cube_project', 'none'],
                        help='UV unwrap method')
    parser.add_argument('--decimate-ratio', type=float, default=1.0,
                        help='Decimate ratio (0.0-1.0, 1.0 = no decimation)')

    args = parser.parse_args()

    if not os.path.isfile(args.input_obj):
        print(f"ERROR: Input file not found: {args.input_obj}", file=sys.stderr)
        sys.exit(1)

    if not setup_blender_scene(args.input_obj):
        sys.exit(1)

    if args.decimate_ratio < 1.0:
        if not apply_decimate(args.decimate_ratio):
            print("ERROR: Failed to apply decimate modifier", file=sys.stderr)
            sys.exit(1)

    if args.auto_smooth > 0:
        if not apply_auto_smooth(args.auto_smooth):
            print("ERROR: Failed to apply auto smooth", file=sys.stderr)
            sys.exit(1)

    if args.uv_method != 'none':
        if not apply_uv_unwrap(args.uv_method):
            print("ERROR: Failed to apply UV unwrapping", file=sys.stderr)
            sys.exit(1)

    if not export_obj(args.output_obj):
        sys.exit(1)

    print(f"SUCCESS: Processed mesh saved to {args.output_obj}")
    sys.exit(0)


if __name__ == '__main__':
    main()
