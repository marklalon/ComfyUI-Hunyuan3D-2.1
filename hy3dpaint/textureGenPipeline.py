# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch
import copy
import trimesh
import numpy as np
from PIL import Image
from typing import List
from hy3dpaint.DifferentiableRenderer.MeshRender import MeshRender
from hy3dpaint.utils.multiview_utils import multiviewDiffusionNet
from hy3dpaint.utils.pipeline_utils import ViewProcessor
from hy3dpaint.utils.image_super_utils import imageSuperNet
from hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap
import warnings

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)


class Hunyuan3DPaintConfig:
    def __init__(self, texture_size=2048):
        self.device = "cuda"

        _hy3dpaint_dir = os.path.dirname(__file__)
        _comfyui_dir = os.path.abspath(os.path.join(_hy3dpaint_dir, "..", "..", ".."))
        _models_dir = os.path.join(_comfyui_dir, "models", "hunyuan3d")

        self.multiview_cfg_path = os.path.join(_hy3dpaint_dir, "cfgs", "hunyuan-paint-pbr.yaml")
        self.custom_pipeline = os.path.join(_hy3dpaint_dir, "hunyuanpaintpbr")
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.hf_cache_dir = _models_dir
        self.dino_ckpt_path = os.path.join(_models_dir, "facebook", "dinov2-giant")
        self.realesrgan_ckpt_path = os.path.join(_models_dir, "RealESRGAN_x4plus.pth")

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 2048
        self.texture_size = texture_size
        self.max_selected_view_num = 6
        self.resolution = 512
        self.bake_exp = 4
        self.merge_method = "fast"

        # view selection
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)

            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)


class Hunyuan3DPaintPipeline:

    def __init__(self, config=None) -> None:
        self.config = config if config is not None else Hunyuan3DPaintConfig()
        self.models = {}
        self.stats_logs = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
        )
        self.view_processor = ViewProcessor(self.config, self.render)
        self.load_models()

    def load_models(self):
        torch.cuda.empty_cache()
        self.models["super_model"] = imageSuperNet(self.config)
        self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        print("Models Loaded.")

    @torch.no_grad()
    def __call__(self, mesh=None, mesh_path=None, image_path=None, output_mesh_path=None, progress_callback=None):
        """Generate texture for 3D mesh using multiview diffusion
        
        Args:
            mesh: trimesh.Trimesh object (optional, mutually exclusive with mesh_path)
            mesh_path: path to mesh file (optional, mutually exclusive with mesh)
            image_path: image path or PIL Image
            output_mesh_path: output path for textured mesh
            progress_callback: callback function(stage, progress, message) for progress updates
        """
        def report_progress(stage, progress, message):
            if progress_callback:
                progress_callback(stage, progress, message)
        # Ensure image_prompt is a list
        if isinstance(image_path, str):
            image_prompt = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image_prompt = image_path
        if not isinstance(image_prompt, List):
            image_prompt = [image_prompt]
        else:
            image_prompt = image_path

        # Determine path for intermediate files
        if mesh is not None:
            # trimesh object provided directly
            import tempfile
            path = tempfile.gettempdir()
        else:
            path = os.path.dirname(mesh_path)

        if mesh is None:
            mesh = trimesh.load(mesh_path)

        # Output path
        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, f"textured_mesh.obj")

        # Load mesh
        mesh = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh=mesh)

        ########### View Selection #########
        report_progress("view_selection", 0, "Selecting views...")
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )
        report_progress("view_selection", 50, "Rendering normal maps...")
        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)
        report_progress("view_selection", 100, "View selection complete")

        ##########  Style  ###########
        image_caption = "high quality"
        image_style = []
        for image in image_prompt:
            image = image.resize((512, 512))
            if image.mode == "RGBA":
                white_bg = Image.new("RGB", image.size, (255, 255, 255))
                white_bg.paste(image, mask=image.getchannel("A"))
                image = white_bg
            image_style.append(image)
        image_style = [image.convert("RGB") for image in image_style]

        ###########  Multiview  ##########
        report_progress("multiview", 0, "Running multiview diffusion...")
        multiviews_pbr = self.models["multiview_model"](
            image_style,
            normal_maps + position_maps,
            prompt=image_caption,
            custom_view_size=self.config.resolution,
            resize_input=True,
        )
        report_progress("multiview", 100, "Multiview diffusion complete")
        
        ###########  Enhance  ##########
        report_progress("enhance", 0, "Enhancing texture quality...")
        enhance_images = {}
        enhance_images["albedo"] = copy.deepcopy(multiviews_pbr["albedo"])
        enhance_images["mr"] = copy.deepcopy(multiviews_pbr["mr"])

        total_enhance = len(enhance_images["albedo"]) * 2
        enhance_idx = 0
        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = self.models["super_model"](enhance_images["albedo"][i])
            enhance_idx += 1
            report_progress("enhance", int(enhance_idx / total_enhance * 100), f"Enhancing albedo view {i+1}/{len(enhance_images['albedo'])}")
            enhance_images["mr"][i] = self.models["super_model"](enhance_images["mr"][i])
            enhance_idx += 1
            report_progress("enhance", int(enhance_idx / total_enhance * 100), f"Enhancing mr view {i+1}/{len(enhance_images['mr'])}")

        ###########  Bake  ##########
        report_progress("bake", 0, "Baking textures...")
        for i in range(len(enhance_images)):
            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                (self.config.render_size, self.config.render_size)
            )
            enhance_images["mr"][i] = enhance_images["mr"][i].resize((self.config.render_size, self.config.render_size))
        report_progress("bake", 20, "Baking albedo texture...")
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        report_progress("bake", 60, "Baking mr texture...")
        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        report_progress("bake", 100, "Texture baking complete")

        ##########  inpaint  ###########
        report_progress("inpaint", 0, "Inpainting textures...")
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture, force_set=True)
        if "mr" in enhance_images:
            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            self.render.set_texture_mr(texture_mr)
        report_progress("inpaint", 100, "Texture inpainting complete")

        report_progress("save", 0, "Saving mesh...")
        self.render.save_mesh(output_mesh_path, downsample=True)

        # 获取纹理图像 (numpy array, [H, W, 3], range [0, 1])
        albedo_texture = self.render.get_texture()
        metallic_texture, roughness_texture = self.render.get_texture_mr()

        return output_mesh_path, albedo_texture, metallic_texture, roughness_texture
