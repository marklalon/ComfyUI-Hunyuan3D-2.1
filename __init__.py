import sys
import os
import types

_this_dir = os.path.dirname(__file__)
_comfyui_dir = os.path.abspath(os.path.join(_this_dir, "..", ".."))
MODELS_DIR = os.path.join(_comfyui_dir, "models", "hunyuan3d")

# Shape models (DiT + VAE) download directory
os.environ.setdefault("HY3DGEN_MODELS", MODELS_DIR)

# Fix basicsr compatibility with new torchvision (functional_tensor was removed)
if "torchvision.transforms.functional_tensor" not in sys.modules:
    from torchvision.transforms import functional as _F
    sys.modules["torchvision.transforms.functional_tensor"] = _F

sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_this_dir, "hy3dpaint"))
sys.path.insert(0, os.path.join(_this_dir, "hy3dshape"))

from .nodes import LoadHunyuan3DModel, Hunyuan3DShapeGeneration, Hunyuan3DTexureSynthsis, Load3DMesh, ConvertToGLB

NODE_CLASS_MAPPINGS = {
    "LoadHunyuan3DModel": LoadHunyuan3DModel,
    "Hunyuan3DShapeGeneration": Hunyuan3DShapeGeneration,
    "Hunyuan3DTexureSynthsis": Hunyuan3DTexureSynthsis,
    "Load3DMesh": Load3DMesh,
    "ConvertToGLB": ConvertToGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHunyuan3DModel": "Load Hunyuan3D Model",
    "Hunyuan3DShapeGeneration": "Hunyuan3D Shape Generation",
    "Hunyuan3DTexureSynthsis": "Hunyuan3D Texure Synthsis",
    "Load3DMesh": "Load 3D Mesh (OBJ/GLB/STL...)",
    "ConvertToGLB": "Convert to GLB",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
