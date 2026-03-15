"""
bridge.py - Communication bridge between ComfyUI and bpy subprocess

This module provides a clean interface for calling bpy operations from the
main ComfyUI process (Python 3.12) by spawning a subprocess running Python 3.11
with bpy installed.

Usage:
    from bpy_processor.bridge import BpyBridge
    
    bridge = BpyBridge()
    result_mesh = bridge.process_mesh(
        mesh=input_trimesh,
        auto_smooth_angle=30.0,
        decimate_ratio=1.0
    )
"""

import os
import sys
import tempfile
import subprocess
from typing import Optional, Tuple
from pathlib import Path

import trimesh


def load_trimesh(mesh_path: str, log_info: bool = False):
    """
    加载 mesh 文件并返回 trimesh 对象。
    
    Args:
        mesh_path: mesh 文件路径 (OBJ/GLB/GLTF/STL/PLY/OFF)
        log_info: 如果为 True，打印加载信息
        
    Returns:
        trimesh.Trimesh 对象
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    # Load mesh with process=False to preserve split vertices at UV seams.
    # Default process=True merges vertices at the same position, which destroys
    # per-vertex normals at UV seams (where vertices share position but differ in UV/normal).
    mesh = trimesh.load(mesh_path, process=False)
    
    # Handle Scene objects by concatenating all meshes, preserving file normals
    if isinstance(mesh, trimesh.Scene):
        from hy3dpaint.convert_utils import scene_dump_with_normals
        mesh = scene_dump_with_normals(mesh)
    
    # Ensure vertex normals exist (compute if missing)
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        if log_info:
            print(f"[load_trimesh] No vertex normals found, computing...")
        mesh.compute_vertex_normals()
    
    if log_info:
        has_uv = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        print(f"[load_trimesh] Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, UV: {has_uv}")
    
    return mesh


class BpyEnvironmentError(Exception):
    """Raised when the bpy environment is not properly configured."""
    pass


class BpyProcessingError(Exception):
    """Raised when bpy processing fails."""
    pass


class BpyBridge:
    """
    Bridge for communicating with a bpy subprocess.
    
    This class manages the lifecycle of temporary files and subprocess
    communication for mesh processing operations.
    """
    
    # Default path to Python 3.11 + bpy environment (relative to this file's parent)
    DEFAULT_BPY_ENV_PATH = ".bpy_env"
    
    def __init__(self, bpy_env_path: Optional[str] = None):
        """
        Initialize the bpy bridge.

        Args:
            bpy_env_path: Path to the Python 3.11 virtual environment with bpy.
                         If None, uses the default path ./.bpy_env/
        """
        self.bpy_env_path = bpy_env_path or self._get_default_bpy_env_path()
        self._python_exe = self._find_python_executable()
        
    def _get_default_bpy_env_path(self) -> str:
        """Get the default bpy environment path (sibling to bpy_processor directory)."""
        # .bpy_env should be at the same level as bpy_processor directory
        # i.e., ComfyUI-Hunyuan3D-2.1/.bpy_env/
        base_dir = Path(__file__).parent.parent
        return str(base_dir / self.DEFAULT_BPY_ENV_PATH)
    
    def _find_python_executable(self) -> Optional[str]:
        """Find the Python executable in the bpy environment."""
        bpy_env = Path(self.bpy_env_path)
        
        if sys.platform == 'win32':
            # Windows: check for python.exe in Scripts/ or root
            candidates = [
                bpy_env / "Scripts" / "python.exe",
                bpy_env / "python.exe",
            ]
        else:
            # Unix: check for python in bin/
            candidates = [
                bpy_env / "bin" / "python",
                bpy_env / "bin" / "python3",
                bpy_env / "bin" / "python3.11",
            ]
        
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        
        return None
    
    def get_bpy_version(self) -> Optional[Tuple[int, int, int]]:
        """Get the bpy version. Returns None if unavailable."""
        try:
            result = subprocess.run(
                [self._python_exe, "-c",
                 "import bpy; print('.'.join(map(str, bpy.app.version)))"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return tuple(map(int, result.stdout.strip().split('.')))
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            pass
        return None

    def process_mesh(
        self,
        mesh: trimesh.Trimesh,
        auto_smooth_angle: float = 30.0,
        decimate_ratio: float = 1.0,
    ) -> Tuple[trimesh.Trimesh, str]:
        """
        Process a mesh using bpy operations.

        UV-split vertices are always merged before any processing (required for
        correct auto smooth and decimate results).

        Args:
            mesh: Input trimesh object.
            auto_smooth_angle: Auto smooth angle in degrees (0-180).
            decimate_ratio: Decimate ratio (0.0-1.0, 1.0 = no decimation).

        Returns:
            Tuple of (processed trimesh, path to the output GLB file).
            The caller is responsible for deleting the GLB file when done.

        Raises:
            BpyProcessingError: If processing fails.
        """
        # Handle trimesh Scene by concatenating all meshes
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        # Find bpy_worker.py (in the same directory as this file)
        worker_script = Path(__file__).parent / "worker.py"
        if not worker_script.exists():
            raise BpyProcessingError(f"worker.py not found at {worker_script}")
        
        # Create temporary files for input and output
        temp_input = None
        temp_output = None
        
        try:
            # Create input temp file (GLB format)
            with tempfile.NamedTemporaryFile(
                suffix='.glb', 
                delete=False,
                prefix='bpy_input_'
            ) as f:
                temp_input = f.name
            
            # Create output temp file (GLB format - preserves normals and UV)
            with tempfile.NamedTemporaryFile(
                suffix='.glb', 
                delete=False,
                prefix='bpy_output_'
            ) as f:
                temp_output = f.name
            
            # Export mesh to temp GLB
            mesh.export(temp_input, file_type='glb')
            
            # Build subprocess command
            cmd = [
                self._python_exe,
                str(worker_script),
                temp_input,
                temp_output,
                '--auto-smooth', str(auto_smooth_angle),
                '--decimate-ratio', str(decimate_ratio),
            ]
            
            # Run subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                raise BpyProcessingError(
                    f"bpy processing failed: {error_msg}"
                )
            
            # Load processed mesh
            if not os.path.exists(temp_output):
                raise BpyProcessingError(
                    "bpy processing failed: output file not created"
                )
            
            # GLB format preserves normals and UV correctly
            result_mesh = load_trimesh(temp_output, log_info=True)

            return result_mesh, temp_output

        except subprocess.TimeoutExpired:
            raise BpyProcessingError(
                "bpy processing timed out after 300 seconds"
            )
        except subprocess.SubprocessError as e:
            raise BpyProcessingError(f"Subprocess error: {e}")
        finally:
            # Only clean up input; output is returned to caller for copying
            if temp_input and os.path.exists(temp_input):
                try:
                    os.unlink(temp_input)
                except OSError:
                    pass


# Module-level singleton for convenience
_bpy_bridge_instance: Optional[BpyBridge] = None


def get_bpy_bridge() -> BpyBridge:
    """
    Get the singleton BpyBridge instance.
    
    Returns:
        BpyBridge instance.
    """
    global _bpy_bridge_instance
    if _bpy_bridge_instance is None:
        _bpy_bridge_instance = BpyBridge()
    return _bpy_bridge_instance


