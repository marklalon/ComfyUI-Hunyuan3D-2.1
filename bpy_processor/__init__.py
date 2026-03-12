"""
bpy_processor - Blender Python mesh processing module

This module provides mesh processing capabilities using Blender's Python API (bpy).
It runs in a separate Python 3.11 subprocess to support bpy, which is only 
available for Python 3.11.

Usage:
    from bpy_processor import BpyBridge
    
    bridge = BpyBridge()
    result = bridge.process_mesh(mesh, auto_smooth_angle=30.0, uv_method='smart_project')
"""

from .bridge import BpyBridge, BpyEnvironmentError, BpyProcessingError, get_bpy_bridge

__all__ = [
    'BpyBridge',
    'BpyEnvironmentError', 
    'BpyProcessingError',
    'get_bpy_bridge',
]
