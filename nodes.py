import os
import shutil
import tempfile
import torch
import numpy as np
from PIL import Image
import folder_paths


def send_progress(message, progress=None, node_id=None, print_to_console=False):
    """发送进度消息到 ComfyUI 界面
    
    Args:
        message: 进度提示文字
        progress: 进度百分比 (0-100)，如果为 None 则只显示文字
        node_id: 节点 ID
    """
    if print_to_console:
        print(f"[Hunyuan3D] {message}{' (' + str(progress) + '%)' if progress else ''}")

    try:
        from server import PromptServer
        if progress is not None:
            PromptServer.instance.send_sync("progress", {
                "value": progress,
                "max": 100,
                "prompt": f"{message} ({progress}%)",
                "node_id": node_id
            })
        else:
            PromptServer.instance.send_sync("progress", {
                "value": 0,
                "max": 0,
                "prompt": message,
                "node_id": node_id
            })
    except Exception:
        print(f"[Hunyuan3D] {message}{' (' + str(progress) + '%)' if progress else ''}")


def tensor_to_pil(image_tensor):
    """将 ComfyUI IMAGE tensor [B, H, W, C] (float 0-1) 转为 PIL Image"""
    img_np = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


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


class Hunyuan3DShapeGeneration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "TRIMESH")
    RETURN_NAMES = ("mesh_path", "mesh_trimesh",)
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D-2.1"
    OUTPUT_NODE = True

    def generate(self, model, image):
        import trimesh
        shape_pipeline = model
        
        send_progress("Generating 3D shape...", 5)
        pil_image = tensor_to_pil(image)
        
        # 创建进度回调函数用于 diffusion sampling
        total_steps = 50  # 默认步数
        def diffusion_callback(step_idx, timestep, latents):
            # 扩散采样进度: 10% - 50%
            progress = int(10 + (step_idx / total_steps) * 40)
            send_progress(f"Diffusion sampling step {step_idx + 1}/{total_steps}", progress)
        
        # 创建进度回调函数用于 volume decoding
        def volume_decode_callback(chunk_idx, total_chunks, resolution=""):
            # 体积解码进度: 50% - 90%
            progress = int(50 + (chunk_idx / total_chunks) * 40)
            res_str = f" [{resolution}]" if resolution else ""
            send_progress(f"Volume decoding{res_str}: {chunk_idx}/{total_chunks}", progress)
        
        send_progress("Running shape generation model...", 10)
        trimesh_mesh = shape_pipeline(
            image=pil_image,
            callback=diffusion_callback,
            callback_steps=1,
            volume_decode_callback=volume_decode_callback
        )[0]

        # 生成完成后将模型移至 CPU 并释放显存，为后续 texture synthesis 腾出空间
        send_progress("Offloading shape model from GPU...", 92)
        try:
            shape_pipeline.to("cpu")
        except Exception:
            pass
        torch.cuda.empty_cache()

        # 保存 mesh 到 ComfyUI output/hunyuan3d_temp 目录
        send_progress("Preparing to save mesh...", 95)
        output_dir = os.path.join(folder_paths.get_output_directory(), "hunyuan3d_temp")
        os.makedirs(output_dir, exist_ok=True)

        mesh_path = os.path.join(output_dir, "input_mesh.obj")
        send_progress("Saving generated mesh...", 98)
        trimesh_mesh.export(mesh_path)
        
        send_progress("Shape generation complete!", 100)

        return (mesh_path, trimesh_mesh)


class Hunyuan3DTexureSynthsis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mesh": ("TRIMESH",),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING", "TRIMESH", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mesh_textured_path", "mesh_textured", "albedo_texture", "metallic_texture", "roughness_texture")
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D-2.1"

    def generate(self, image, mesh, texture_size):
        import trimesh
        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        send_progress("Initializing texture synthesis pipeline...", 5)
        paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(texture_size=texture_size))
        
        send_progress("Preparing image input...", 10)
        pil_image = tensor_to_pil(image)
        
        # 创建进度回调函数
        def paint_progress_callback(stage, progress, message):
            # 阶段进度映射: view_selection(10-15%), multiview(15-50%), enhance(50-65%), bake(65-80%)
            stage_ranges = {
                "view_selection": (10, 15),
                "multiview": (15, 50),
                "enhance": (50, 65),
                "bake": (65, 80),
                "inpaint": (80, 85),
                "save": (85, 90),
            }
            if stage in stage_ranges:
                start, end = stage_ranges[stage]
                actual_progress = int(start + (end - start) * progress / 100)
                send_progress(message, actual_progress)
            else:
                send_progress(message, progress)
        
        send_progress("Generating texture (multiview diffusion)...", 15)
        mesh_textured_path, albedo_texture, metallic_texture, roughness_texture = paint_pipeline(
            mesh=mesh, 
            image_path=pil_image, 
            progress_callback=paint_progress_callback
        )

        # 转换 numpy array 为 ComfyUI IMAGE 格式 (torch tensor, [B, H, W, C])
        # albedo_texture, metallic_texture, roughness_texture 已经是 numpy array [H, W, 3], range [0, 1]
        send_progress("Processing texture output...", 90)
        albedo_tensor = torch.from_numpy(albedo_texture).unsqueeze(0)  # [1, H, W, 3]
        metallic_tensor = torch.from_numpy(metallic_texture).unsqueeze(0) if metallic_texture is not None else torch.zeros(1, 512, 512, 3)
        roughness_tensor = torch.from_numpy(roughness_texture).unsqueeze(0) if roughness_texture is not None else torch.zeros(1, 512, 512, 3)

        # 加载输出的 mesh (trimesh 对象，保留 UV 和材质信息)
        send_progress("Loading textured mesh...", 95)
        mesh_textured = trimesh.load(mesh_textured_path, force="mesh")
        
        send_progress("Texture synthesis complete!", 100)

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

    RETURN_TYPES = ("STRING", "TRIMESH")
    RETURN_NAMES = ("mesh_path", "mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "Hunyuan3D-2.1"
    OUTPUT_NODE = True

    def load_mesh(self, mesh_path):
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        ext = os.path.splitext(mesh_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format '{ext}'. Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}")

        import trimesh
        mesh = trimesh.load(mesh_path)

        return (mesh_path, mesh)


class ConvertToGLB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "mesh/hunyuan3d"}),
            },
            "optional": {
                "albedo_texture": ("IMAGE",),
                "metallic_texture": ("IMAGE",),
                "roughness_texture": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "convert"
    CATEGORY = "Hunyuan3D-2.1"
    OUTPUT_NODE = True

    def convert(self, mesh, filename_prefix, albedo_texture=None, metallic_texture=None, roughness_texture=None):
        from hy3dpaint.convert_utils import create_glb_with_pbr_materials
        
        # 获取输出目录（counter 基于图片文件，不可直接用于 GLB）
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )

        # 找到第一个不与已有 GLB 文件冲突的 counter
        while os.path.exists(os.path.join(full_output_folder, f"{filename}_{counter:05}.glb")):
            counter += 1

        glb_filename = f"{filename}_{counter:05}.glb"
        glb_path = os.path.join(full_output_folder, glb_filename)
        
        # 转换纹理为 PIL Image（仅处理非空的纹理）
        textures_dict = {}
        
        if albedo_texture is not None:
            albedo_np = np.clip(albedo_texture[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
            textures_dict['albedo'] = Image.fromarray(albedo_np)

        if metallic_texture is not None:
            metallic_np = np.clip(metallic_texture[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
            textures_dict['metallic'] = Image.fromarray(metallic_np)

        if roughness_texture is not None:
            roughness_np = np.clip(roughness_texture[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
            textures_dict['roughness'] = Image.fromarray(roughness_np)
        
        # 调用 create_glb_with_pbr_materials
        create_glb_with_pbr_materials(mesh, textures_dict, glb_path)
        
        return (glb_path,)


class RemeshMesh:
    """网格简化/重划分节点，清理网格并简化面数"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "target_count": ("INT", {"default": 40000, "min": 1000, "max": 500000}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "remesh"
    CATEGORY = "Hunyuan3D-2.1"

    def clean_mesh(self, mesh):
        # 基本清理：移除重复顶点、无效面、重新计算法线
        mesh = mesh.process(validate=True)
        
        # 移除未引用的顶点
        mesh.remove_unreferenced_vertices()
        
        # 合并接近的顶点（可修复 non-manifold 问题）
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        
        # 再次移除未引用的顶点（合并后可能产生）
        mesh.remove_unreferenced_vertices()

        return mesh

    def save_mesh_as_obj(self, mesh, save_path):
        """将 trimesh 对象保存为 OBJ 文件（仅顶点、面和法线，无UV和纹理）"""
        from hy3dpaint.DifferentiableRenderer.mesh_utils import save_obj_mesh

        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            mesh.compute_vertex_normals()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_obj_mesh(
            mesh_path=save_path,
            vtx_pos=mesh.vertices,
            pos_idx=mesh.faces,
            vtx_normal=mesh.vertex_normals
        )

    def remesh(self, mesh, target_count):
        import trimesh

        send_progress("Remeshing mesh...", 5)

        # 确保是 Trimesh 对象
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        original_face_num = mesh.faces.shape[0]
        send_progress(f"Original mesh: {original_face_num} faces", 10, print_to_console=True)

        # 清理mesh
        mesh = self.clean_mesh(mesh)
        face_num = mesh.faces.shape[0]
        send_progress(f"Cleaned mesh: {face_num} faces, target: {target_count}", 20, print_to_console=True)

        # 如果面数超过目标，进行简化
        if face_num > target_count:
            send_progress(f"Simplifying mesh from {face_num} to {target_count} faces...", 70, print_to_console=True)
            mesh = mesh.simplify_quadric_decimation(face_count=target_count)
            mesh = self.clean_mesh(mesh)
            send_progress(f"Mesh simplified to {mesh.faces.shape[0]} faces", 90, print_to_console=True)

        # 保存 remesh 后的 mesh 到临时目录
        mesh_remesh_path = os.path.join(
            folder_paths.get_output_directory(), "hunyuan3d_temp", "remeshed.obj"
        )
        self.save_mesh_as_obj(mesh, mesh_remesh_path)
        send_progress(f"Saved remeshed mesh to: {mesh_remesh_path}", 100, print_to_console=True)

        return (mesh,)


class BlenderMeshProcessor:
    """
    Blender 网格处理器 - 使用独立 bpy 环境进行 AutoSmooth 和 UV 展开
    
    该节点通过子进程调用独立 Python 3.11 + bpy 环境，提供 Blender 独有的网格处理功能：
    - AutoSmooth: 基于角度阈值的自动平滑着色
    - UV 展开: Smart Project / Lightmap Pack / Cube Project
    
    使用前需要先创建 bpy 环境，详见 SETUP_BPY.md
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "auto_smooth_angle": ("FLOAT", {
                    "default": 30.0, 
                    "min": 0.0, 
                    "max": 180.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "enable_uv_unwrap": ("BOOLEAN", {"default": False}),
                "uv_method": (["smart_project", "lightmap_pack", "cube_project"], {
                    "default": "smart_project"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D-2.1"

    def process(self, mesh, auto_smooth_angle, enable_uv_unwrap, uv_method):
        from bpy_processor import BpyBridge, BpyProcessingError
        import trimesh

        send_progress("Initializing Blender mesh processor...", 5, print_to_console=True)

        # 确保是 Trimesh 对象
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        bridge = BpyBridge()

        send_progress(f"Using bpy version: {'.'.join(map(str, bridge.get_bpy_version() or (0, 0, 0)))}", 10, print_to_console=True)
        
        # 确定 UV 方法
        actual_uv_method = uv_method if enable_uv_unwrap else 'none'
        
        send_progress(f"Processing: AutoSmooth={auto_smooth_angle}°, UV={actual_uv_method}", 20, print_to_console=True)
        
        try:
            # 调用 bpy 处理
            result_mesh, temp_obj_path = bridge.process_mesh(
                mesh=mesh,
                auto_smooth_angle=auto_smooth_angle,
                uv_method=actual_uv_method
            )

            send_progress(f"Blender processing complete. Vertices: {len(result_mesh.vertices)}, Faces: {len(result_mesh.faces)}", 100, print_to_console=True)

            processed_path = os.path.join(
                folder_paths.get_output_directory(), "hunyuan3d_temp", "blender_processed.obj"
            )
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            shutil.copy2(temp_obj_path, processed_path)
            os.unlink(temp_obj_path)

            return (result_mesh,)
            
        except BpyProcessingError as e:
            raise RuntimeError(f"Blender mesh processing failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Blender processing: {e}")
