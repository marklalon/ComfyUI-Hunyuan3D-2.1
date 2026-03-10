import os
import tempfile
import torch
import numpy as np
import folder_paths
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
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 4096}),
                "face_count": ("INT", {"default": 40000, "min": 1000, "max": 500000}),
                "simplify_mesh": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("STRING", "TRIMESH", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mesh_textured_path", "mesh_textured", "albedo_texture", "metallic_texture", "roughness_texture")
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D-2.1"

    def generate(self, image, mesh_untextured, texture_size, face_count, simplify_mesh):
        import trimesh
        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        verts = mesh_untextured.vertices[0].cpu().numpy()
        fcs = mesh_untextured.faces[0].cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts, faces=fcs)

        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "hunyuan3d")
        os.makedirs(output_dir, exist_ok=True)
        mesh_path = os.path.join(output_dir, "input_mesh.obj")
        mesh.export(mesh_path)

        paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(texture_size=texture_size, face_count=face_count))
        use_remesh = (simplify_mesh == "enable")
        mesh_textured_path, albedo_texture, metallic_texture, roughness_texture = paint_pipeline(mesh_path, image_path=image, use_remesh=use_remesh)

        # 转换 numpy array 为 ComfyUI IMAGE 格式 (torch tensor, [B, H, W, C])
        # albedo_texture, metallic_texture, roughness_texture 已经是 numpy array [H, W, 3], range [0, 1]
        albedo_tensor = torch.from_numpy(albedo_texture).unsqueeze(0)  # [1, H, W, 3]
        metallic_tensor = torch.from_numpy(metallic_texture).unsqueeze(0) if metallic_texture is not None else torch.zeros(1, 512, 512, 3)
        roughness_tensor = torch.from_numpy(roughness_texture).unsqueeze(0) if roughness_texture is not None else torch.zeros(1, 512, 512, 3)

        # 加载输出的 mesh (trimesh 对象，保留 UV 和材质信息)
        mesh_textured = trimesh.load(mesh_textured_path, force="mesh")

        return (mesh_textured_path, mesh_textured, albedo_tensor, metallic_tensor, roughness_tensor)


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


class ConvertToGLB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "albedo_texture": ("IMAGE",),
                "metallic_texture": ("IMAGE",),
                "roughness_texture": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "mesh/hunyuan3d"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "convert"
    CATEGORY = "Hunyuan3D-2.1"
    OUTPUT_NODE = True

    def convert(self, mesh, albedo_texture, metallic_texture, roughness_texture, filename_prefix):
        from PIL import Image
        from hy3dpaint.convert_utils import create_glb_with_pbr_materials
        
        # 获取输出目录
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        
        # 生成 GLB 文件名
        glb_filename = f"{filename}_{counter:05}.glb"
        glb_path = os.path.join(full_output_folder, glb_filename)
        
        # 转换纹理为 PIL Image
        albedo_np = (albedo_texture[0].cpu().numpy() * 255).astype(np.uint8)
        albedo_img = Image.fromarray(albedo_np)
        
        metallic_np = (metallic_texture[0].cpu().numpy() * 255).astype(np.uint8)
        metallic_img = Image.fromarray(metallic_np)
        
        roughness_np = (roughness_texture[0].cpu().numpy() * 255).astype(np.uint8)
        roughness_img = Image.fromarray(roughness_np)
        
        # 调用 create_glb_with_pbr_materials
        textures_dict = {
            'albedo': albedo_img,
            'metallic': metallic_img,
            'roughness': roughness_img,
        }
        create_glb_with_pbr_materials(mesh, textures_dict, glb_path)
        
        return (glb_path,)
