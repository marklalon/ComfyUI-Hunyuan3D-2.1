import trimesh
import pygltflib
import numpy as np
from PIL import Image
import base64
import io
import tempfile
import os


def combine_metallic_roughness(metallic, roughness, output_path=None):
    """
    将metallic和roughness贴图合并为一张贴图
    GLB格式要求metallic在B通道，roughness在G通道
    
    参数可以是文件路径(str)或PIL Image对象
    """
    # 加载贴图
    if isinstance(metallic, str):
        metallic_img = Image.open(metallic).convert("L")
    else:
        metallic_img = metallic.convert("L") if metallic.mode != "L" else metallic
    
    if isinstance(roughness, str):
        roughness_img = Image.open(roughness).convert("L")
    else:
        roughness_img = roughness.convert("L") if roughness.mode != "L" else roughness

    # 确保尺寸一致
    if metallic_img.size != roughness_img.size:
        roughness_img = roughness_img.resize(metallic_img.size)

    # 创建RGB图像
    width, height = metallic_img.size

    # 转为numpy数组便于操作
    metallic_array = np.array(metallic_img)
    roughness_array = np.array(roughness_img)

    # 创建合并的数组 (R, G, B) = (AO, Roughness, Metallic)
    combined_array = np.zeros((height, width, 3), dtype=np.uint8)
    combined_array[:, :, 0] = 255  # R通道：AO (如果没有AO贴图，设为白色)
    combined_array[:, :, 1] = roughness_array  # G通道：Roughness
    combined_array[:, :, 2] = metallic_array  # B通道：Metallic

    # 转回PIL图像
    combined = Image.fromarray(combined_array)
    
    if output_path:
        combined.save(output_path)
    return combined


def create_glb_with_pbr_materials(mesh, textures_dict, output_path):
    """
    使用pygltflib创建包含完整PBR材质的GLB文件
    
    参数:
        mesh: 文件路径(str)或trimesh对象
        textures_dict: 纹理字典，值可以是文件路径(str)或PIL Image对象
            {
                'albedo': 'path/to/albedo.png' 或 Image对象,
                'metallic': 'path/to/metallic.png' 或 Image对象,
                'roughness': 'path/to/roughness.png' 或 Image对象,
                'normal': 'path/to/normal.png',  # 可选
                'ao': 'path/to/ao.png'  # 可选
            }
        output_path: 输出GLB文件路径
    """
    # 1. 加载mesh
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh)

    # 2. 导出为临时GLB
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as temp_file:
        temp_glb = temp_file.name
    mesh.export(temp_glb)

    try:
        # 3. 加载GLB文件进行材质编辑
        gltf = pygltflib.GLTF2().load(temp_glb)

        # 4. 准备纹理数据
        def image_to_data_uri(image):
            """将图像(PIL Image或路径)转换为data URI"""
            if isinstance(image, str):
                with open(image, "rb") as f:
                    image_data = f.read()
            else:
                # PIL Image 对象
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_data = buffer.getvalue()
            encoded = base64.b64encode(image_data).decode()
            return f"data:image/png;base64,{encoded}"

        # 5. 合并metallic和roughness
        if "metallic" in textures_dict and "roughness" in textures_dict:
            mr_combined = combine_metallic_roughness(textures_dict["metallic"], textures_dict["roughness"])
            textures_dict = dict(textures_dict)  # 复制以避免修改原字典
            textures_dict["metallicRoughness"] = mr_combined

        # 6. 添加图像到GLTF
        images = []
        textures = []

        texture_mapping = {
            "albedo": "baseColorTexture",
            "metallicRoughness": "metallicRoughnessTexture",
            "normal": "normalTexture",
            "ao": "occlusionTexture",
        }

        for tex_type, tex_value in textures_dict.items():
            if tex_type in texture_mapping and tex_value:
                # 添加图像
                image = pygltflib.Image(uri=image_to_data_uri(tex_value))
                images.append(image)

                # 添加纹理
                texture = pygltflib.Texture(source=len(images) - 1)
                textures.append(texture)

        # 7. 创建PBR材质
        pbr_metallic_roughness = pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=1.0, roughnessFactor=1.0
        )

        # 设置纹理索引
        texture_index = 0
        if "albedo" in textures_dict:
            pbr_metallic_roughness.baseColorTexture = pygltflib.TextureInfo(index=texture_index)
            texture_index += 1

        if "metallicRoughness" in textures_dict:
            pbr_metallic_roughness.metallicRoughnessTexture = pygltflib.TextureInfo(index=texture_index)
            texture_index += 1

        # 创建材质
        material = pygltflib.Material(name="PBR_Material", pbrMetallicRoughness=pbr_metallic_roughness)

        # 添加法线贴图
        if "normal" in textures_dict:
            material.normalTexture = pygltflib.NormalTextureInfo(index=texture_index)
            texture_index += 1

        # 添加AO贴图
        if "ao" in textures_dict:
            material.occlusionTexture = pygltflib.OcclusionTextureInfo(index=texture_index)

        # 8. 更新GLTF
        gltf.images = images
        gltf.textures = textures
        gltf.materials = [material]

        # 确保mesh使用材质
        if gltf.meshes:
            for primitive in gltf.meshes[0].primitives:
                primitive.material = 0

        # 9. 将data URI图像转为binary buffer view（GLB二进制块）
        gltf.convert_images(pygltflib.ImageFormat.BUFFERVIEW)

        # 10. 保存最终GLB
        gltf.save(output_path)
        print(f"PBR GLB文件已保存: {output_path}")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_glb):
            os.remove(temp_glb)
