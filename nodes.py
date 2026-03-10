import os
import tempfile
import torch
import numpy as np
from comfy_api.latest._util import MESH


class LoadHunyuan3DModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "tencent/Hunyuan3D-2.1"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan3D-2.1"

    def load_model(self, model_path, device="cuda"):
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        
        return (model,)


class LoadHunyuan3DImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"default": "assets/demo.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "input_image"
    CATEGORY = "Hunyuan3D-2.1"

    def input_image(self, image_path):
        image = image_path
        return (image,)


class Hunyuan3DShapeGeneration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh_untextured",)
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D-2.1"

    def generate(self, model, image):
        shape_pipeline = model
        trimesh_mesh = shape_pipeline(image=image)[0]

        vertices = torch.from_numpy(np.array(trimesh_mesh.vertices, dtype=np.float32)).unsqueeze(0)
        faces = torch.from_numpy(np.array(trimesh_mesh.faces, dtype=np.int64)).unsqueeze(0)
        mesh_untextured = MESH(vertices, faces)

        return (mesh_untextured,)


class Hunyuan3DTexureSynthsis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mesh_untextured": ("MESH",),
                "max_num_view": ("INT", {"default": 6}),
                "render_size": ("INT", {"default": 512}),
                "texture_size": ("INT", {"default": 1024}),
                "target_count": ("INT", {"default": 40000}),
            }
        }

    RETURN_TYPES = ("TEXURE",)
    RETURN_NAMES = ("mesh_textured",)
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D-2.1"

    def generate(self, image, mesh_untextured, max_num_view, render_size, texture_size, target_count):
        import trimesh
        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        verts = mesh_untextured.vertices[0].cpu().numpy()
        fcs = mesh_untextured.faces[0].cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts, faces=fcs)

        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "hunyuan3d")
        os.makedirs(output_dir, exist_ok=True)
        mesh_path = os.path.join(output_dir, "input_mesh.obj")
        mesh.export(mesh_path)

        paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=max_num_view, render_size=render_size, texture_size=texture_size, target_count=target_count))
        mesh_textured = paint_pipeline(mesh_path, image_path=image)

        return (mesh_textured,)


class Load3DMesh:
    SUPPORTED_EXTENSIONS = ('.obj', '.glb', '.gltf', '.stl', '.ply', '.off')

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "Hunyuan3D-2.1"

    def load_mesh(self, mesh_path):
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        ext = os.path.splitext(mesh_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format '{ext}'. Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}")

        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh", merge_primitives=True)
        vertices = torch.from_numpy(np.array(mesh.vertices, dtype=np.float32)).unsqueeze(0)
        faces = torch.from_numpy(np.array(mesh.faces, dtype=np.int64)).unsqueeze(0)

        return (MESH(vertices, faces),)
