import mimetypes
import trimesh
import pygltflib
import numpy as np
from PIL import Image
import base64
import io


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
        mesh = trimesh.load(mesh, force='mesh')
    
    # Handle trimesh Scene (concatenate all meshes)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Get mesh data
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    
    # Get normals (trimesh always computes vertex_normals automatically)
    normals = mesh.vertex_normals.astype(np.float32)

    # Get UV coordinates if available
    # Flip V coordinate for glTF compatibility (OBJ format uses top-left, glTF uses bottom-left origin)
    uvs = None
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
        uvs = mesh.visual.uv.astype(np.float32)
        uvs[:, 1] = 1.0 - uvs[:, 1]

    # 2. Create GLB directly with pygltflib
    from pygltflib import GLTF2, Buffer, BufferView, Accessor, Mesh, Primitive, Node, Scene
    
    # Create binary buffer
    # Layout: vertices | normals | uvs | indices
    binary = bytearray()
    
    # Vertices
    vertex_byte_offset = len(binary)
    vertex_buffer = vertices.tobytes()
    binary.extend(vertex_buffer)
    
    # Normals
    normal_byte_offset = len(binary)
    normal_buffer = normals.tobytes()
    binary.extend(normal_buffer)
    
    # UVs (if available)
    uv_byte_offset = len(binary)
    uv_buffer = b''
    if uvs is not None:
        uv_buffer = uvs.tobytes()
        binary.extend(uv_buffer)
    
    # Indices
    index_byte_offset = len(binary)
    index_buffer = faces.tobytes()
    binary.extend(index_buffer)
    
    total_byte_length = len(binary)
    
    # Create buffer views
    buffer_views = []
    
    # Vertex buffer view
    buffer_views.append(BufferView(
        buffer=0,
        byteOffset=vertex_byte_offset,
        byteLength=len(vertex_buffer),
        target=34962  # ARRAY_BUFFER
    ))
    
    # Normal buffer view
    buffer_views.append(BufferView(
        buffer=0,
        byteOffset=normal_byte_offset,
        byteLength=len(normal_buffer),
        target=34962  # ARRAY_BUFFER
    ))
    
    # UV buffer view (if available)
    if uvs is not None:
        buffer_views.append(BufferView(
            buffer=0,
            byteOffset=uv_byte_offset,
            byteLength=len(uv_buffer),
            target=34962  # ARRAY_BUFFER
        ))
    
    # Index buffer view
    buffer_views.append(BufferView(
        buffer=0,
        byteOffset=index_byte_offset,
        byteLength=len(index_buffer),
        target=34963  # ELEMENT_ARRAY_BUFFER
    ))
    
    # Create accessors
    accessors = []
    
    # Vertex accessor
    accessors.append(Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=5126,  # FLOAT
        count=len(vertices),
        type="VEC3",
        max=vertices.max(axis=0).tolist(),
        min=vertices.min(axis=0).tolist()
    ))
    
    # Normal accessor
    accessors.append(Accessor(
        bufferView=1,
        byteOffset=0,
        componentType=5126,  # FLOAT
        count=len(normals),
        type="VEC3"
    ))
    
    # UV accessor (if available)
    uv_accessor_index = None
    if uvs is not None:
        accessors.append(Accessor(
            bufferView=2,
            byteOffset=0,
            componentType=5126,  # FLOAT
            count=len(uvs),
            type="VEC2"
        ))
        uv_accessor_index = 2
    
    # Index accessor
    index_accessor_index = 3 if uvs is not None else 2
    accessors.append(Accessor(
        bufferView=index_accessor_index,
        byteOffset=0,
        componentType=5125,  # UNSIGNED_INT
        count=len(faces) * 3,
        type="SCALAR"
    ))
    
    # 3. Prepare textures
    def image_to_data_uri(image):
        """将图像(PIL Image或路径)转换为data URI"""
        if isinstance(image, str):
            with open(image, "rb") as f:
                image_data = f.read()
            mime = mimetypes.guess_type(image)[0] or "image/png"
        else:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            mime = "image/png"
        encoded = base64.b64encode(image_data).decode()
        return f"data:{mime};base64,{encoded}"

    # 合并metallic和roughness
    if "metallic" in textures_dict and "roughness" in textures_dict:
        mr_combined = combine_metallic_roughness(textures_dict["metallic"], textures_dict["roughness"])
        textures_dict = dict(textures_dict)
        textures_dict["metallicRoughness"] = mr_combined

    # 按固定顺序添加图像到GLTF，并记录每种纹理对应的索引
    # 固定顺序确保 tex_type_to_index 与材质赋值保持一致
    TEXTURE_ORDER = ("albedo", "metallicRoughness", "normal", "ao")
    images = []
    textures = []
    tex_type_to_index = {}

    for tex_type in TEXTURE_ORDER:
        tex_value = textures_dict.get(tex_type)
        if tex_value:
            tex_type_to_index[tex_type] = len(images)
            images.append(pygltflib.Image(uri=image_to_data_uri(tex_value)))
            textures.append(pygltflib.Texture(source=len(images) - 1))

    # 4. Create PBR material
    pbr_metallic_roughness = pygltflib.PbrMetallicRoughness(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=1.0,
        roughnessFactor=1.0
    )

    if "albedo" in tex_type_to_index:
        pbr_metallic_roughness.baseColorTexture = pygltflib.TextureInfo(index=tex_type_to_index["albedo"])

    if "metallicRoughness" in tex_type_to_index:
        pbr_metallic_roughness.metallicRoughnessTexture = pygltflib.TextureInfo(index=tex_type_to_index["metallicRoughness"])

    material = pygltflib.Material(name="PBR_Material", pbrMetallicRoughness=pbr_metallic_roughness)

    if "normal" in tex_type_to_index:
        material.normalTexture = pygltflib.NormalTextureInfo(index=tex_type_to_index["normal"])

    if "ao" in tex_type_to_index:
        material.occlusionTexture = pygltflib.OcclusionTextureInfo(index=tex_type_to_index["ao"])

    # 5. Create primitive and mesh
    primitive_attrs = {
        'POSITION': 0,
        'NORMAL': 1,
    }
    if uv_accessor_index is not None:
        primitive_attrs['TEXCOORD_0'] = uv_accessor_index
    
    primitive = Primitive(
        attributes=pygltflib.Attributes(**primitive_attrs),
        indices=index_accessor_index,
        material=0
    )
    
    gltf_mesh = Mesh(primitives=[primitive])
    
    # 6. Create node and scene
    node = Node(mesh=0)
    scene = Scene(nodes=[0])
    
    # 7. Create GLTF
    gltf = GLTF2(
        scene=0,
        scenes=[scene],
        nodes=[node],
        meshes=[gltf_mesh],
        materials=[material],
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[Buffer(byteLength=total_byte_length)],
        images=images,
        textures=textures
    )
    
    # Set binary data
    gltf.set_binary_blob(bytes(binary))
    
    # Convert images to buffer views
    gltf.convert_images(pygltflib.ImageFormat.BUFFERVIEW)
    
    # 8. Save GLB
    gltf.save(output_path)
    print(f"PBR GLB文件已保存: {output_path}")
