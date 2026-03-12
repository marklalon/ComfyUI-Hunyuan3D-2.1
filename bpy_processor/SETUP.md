# Blender Python (bpy) Environment Setup

This document explains how to set up a Python 3.11 environment with `bpy` (Blender Python API) for the `BlenderMeshProcessor` node.

## Why a Separate Environment?

The `bpy` package on PyPI only supports Python 3.11 (`cp311`). Since ComfyUI uses Python 3.12, we need a separate Python 3.11 environment to use bpy.

The `BlenderMeshProcessor` node communicates with this environment via subprocess, allowing you to use Blender's mesh processing features without affecting your main ComfyUI installation.

## Quick Setup

### Windows

```powershell
# Navigate to the ComfyUI-Hunyuan3D-2.1 directory
cd D:\AI\ComfyUI\custom_nodes\ComfyUI-Hunyuan3D-2.1

# Create a Python 3.11 virtual environment
python3.11 -m venv .bpy_env

# Activate the environment
.\.bpy_env\Scripts\activate

# Install bpy
pip install bpy

# Verify installation
python -c "import bpy; print(f'Blender version: {bpy.app.version_string}')"
```

### Linux / macOS

```bash
# Navigate to the ComfyUI-Hunyuan3D-2.1 directory
cd /path/to/ComfyUI/custom_nodes/ComfyUI-Hunyuan3D-2.1

# Create a Python 3.11 virtual environment
python3.11 -m venv .bpy_env

# Activate the environment
source .bpy_env/bin/activate

# Install bpy
pip install bpy

# Verify installation
python -c "import bpy; print(f'Blender version: {bpy.app.version_string}')"
```

## bpy Versions

Choose the bpy version that matches your needs:

| bpy Version | Blender Version |
|-------------|-----------------|
| bpy-4.0.x   | Blender 4.0     |
| bpy-4.1.x   | Blender 4.1     |
| bpy-4.2.x   | Blender 4.2     |
| bpy-4.3.x   | Blender 4.3     |

To install a specific version:

```bash
pip install bpy==4.2.0
```

## Directory Structure

After setup, your directory should look like this:

```
ComfyUI-Hunyuan3D-2.1/
├── .bpy_env/                   # Python 3.11 + bpy environment
│   ├── Scripts/                # Windows
│   │   ├── python.exe
│   │   └── ...
│   ├── lib/                    # Installed packages
│   │   └── site-packages/
│   │       └── bpy/
│   └── ...
├── bpy_processor/              # bpy processing module
│   ├── __init__.py             # Module exports
│   ├── bridge.py               # Communication bridge
│   ├── worker.py               # Subprocess worker
│   └── SETUP.md                # This file
├── nodes.py                    # Contains BlenderMeshProcessor
└── __init__.py
```

## Testing the Setup

Run this Python script to verify your bpy environment:

```python
# test_bpy_env.py
import subprocess
import os

bpy_env_path = os.path.join(os.path.dirname(__file__), ".bpy_env")
python_exe = os.path.join(bpy_env_path, "Scripts", "python.exe")

result = subprocess.run(
    [python_exe, "-c", "import bpy; print(bpy.app.version_string)"],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print(f"✓ bpy environment working! Blender version: {result.stdout.strip()}")
else:
    print(f"✗ bpy environment error: {result.stderr}")
```

## Features Available

The `BlenderMeshProcessor` node provides:

### AutoSmooth
- Based on angle threshold (0° - 180°)
- Automatically smooths shading across edges below the threshold
- Similar to Blender's "Shade Auto Smooth" feature

### UV Unwrapping Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `smart_project` | Automatic UV projection | General purpose, organic models |
| `lightmap_pack` | Optimized for lightmap baking | Baking, game assets |
| `cube_project` | Cubic projection | Architectural models, hard surface |

## Troubleshooting

### "bpy environment not available" Error

1. Verify the `.bpy_env` directory exists in the extension folder
2. Check that `python.exe` exists at `.bpy_env/Scripts/python.exe` (Windows) or `.bpy_env/bin/python` (Linux/macOS)
3. Verify bpy is installed: `.\.bpy_env\Scripts\python.exe -c "import bpy"`

### ImportError: cannot import name 'bpy'

The bpy package requires specific system libraries. On Linux, you may need:

```bash
sudo apt-get install libxi6 libxxf86vm1 libgl1-mesa-glx
```

### Subprocess Timeout

If processing takes too long, check:
1. The input mesh complexity (large meshes take longer)
2. System resources (bpy needs CPU and memory)

## Alternative: Using Existing Blender Installation

If you have Blender installed, you can use its bundled Python instead of creating a separate environment:

```python
# In bpy_bridge.py, modify the python path:
# bpy_env_path = "C:/Program Files/Blender Foundation/Blender 4.2/4.2/python"
```

However, this is not recommended as:
- Blender's Python may have different configurations
- Blender must be closed when using bpy in subprocess mode
- Version compatibility issues may occur

## Resources

- [bpy on PyPI](https://pypi.org/project/bpy/)
- [bpy Documentation](https://docs.blender.org/api/current/)
- [Blender Python API Guide](https://docs.blender.org/manual/en/latest/advanced/scripting/index.html)
