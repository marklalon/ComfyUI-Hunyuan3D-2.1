import mimetypes
import trimesh
import pygltflib
import numpy as np
from PIL import Image
import io


def scene_dump_with_normals(scene: trimesh.Scene) -> trimesh.Trimesh:
    """
    等同于 scene.dump(concatenate=True)，但保留各 geometry 中文件的原始法线。

    问题根因：dump() 内部调用 geometry.copy(include_cache=False)，清空了从
    NORMAL accessor 缓存的法线。后续 concatenate() 因缓存为空传入 vertex_normals=None，
    导致 mesh.vertex_normals 在每次访问时从面法线重新计算，在 UV 接缝处产生差异。

    修复：在 dump 前按 scene.graph.nodes_geometry 顺序收集各 geometry 的法线，
    对法线应用对应节点的旋转变换，dump 后恢复到合并后的 mesh 上。
    """
    ordered_normals = []
    for node_name in scene.graph.nodes_geometry:
        T, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        vn = np.array(geom.vertex_normals)  # cache 有效，返回文件法线
        # 对法线应用变换矩阵的旋转部分（inverse transpose）
        R = np.array(T[:3, :3], dtype=np.float64)
        normal_T = np.linalg.inv(R).T
        vn_t = (normal_T @ vn.T).T
        norms = np.linalg.norm(vn_t, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        ordered_normals.append(vn_t / norms)

    mesh = scene.dump(concatenate=True)

    if ordered_normals:
        combined = np.vstack(ordered_normals)
        if len(combined) == len(mesh.vertices):
            mesh.vertex_normals = combined

    return mesh


def extract_texture_from_mesh(mesh):
    """
    从 trimesh 对象中提取纹理
    
    Args:
        mesh: trimesh.Trimesh 对象
        
    Returns:
        dict: 纹理字典 {'albedo': PIL.Image, ...} 或空字典
    """
    textures = {}
    
    if not hasattr(mesh, 'visual'):
        return textures
    
    visual = mesh.visual
    
    # 处理 SimpleMaterial (常用于 OBJ 加载)
    if hasattr(visual, 'material'):
        material = visual.material
        if material is not None:
            # Base color / Diffuse
            if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                try:
                    textures['albedo'] = material.baseColorTexture
                except Exception:
                    pass
            elif hasattr(material, 'diffuse') and material.diffuse is not None:
                # 有些 trimesh 版本使用 diffuse 属性
                if isinstance(material.diffuse, Image.Image):
                    textures['albedo'] = material.diffuse
                    
            # Metallic/Roughness
            if hasattr(material, 'metallicRoughnessTexture') and material.metallicRoughnessTexture is not None:
                try:
                    textures['metallicRoughness'] = material.metallicRoughnessTexture
                except Exception:
                    pass
                    
            # Normal map
            if hasattr(material, 'normalTexture') and material.normalTexture is not None:
                try:
                    textures['normal'] = material.normalTexture
                except Exception:
                    pass
                    
    # 处理 TextureVisuals (常用于带 UV 的 mesh)
    if hasattr(visual, 'image') and visual.image is not None:
        if isinstance(visual.image, Image.Image):
            textures['albedo'] = visual.image
    
    # 处理 SimpleVisual 中的颜色贴图
    if hasattr(visual, 'vertex_attributes'):
        attrs = visual.vertex_attributes
        if hasattr(attrs, 'material') and attrs.material is not None:
            material = attrs.material
            if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                textures['albedo'] = material.baseColorTexture
    
    return textures


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
        mesh = trimesh.load(mesh, force='mesh', process=False)

    # Handle trimesh Scene (concatenate all meshes), preserving file normals
    if isinstance(mesh, trimesh.Scene):
        mesh = scene_dump_with_normals(mesh)

    # Get mesh data
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    
    # Get normals - ensure they exist
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        print("[convert_utils] Warning: No vertex normals found, computing...")
        mesh.compute_vertex_normals()
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
    
    # 3. Prepare textures - embed image bytes directly into binary blob
    def image_to_bytes(image):
        """Convert PIL Image or file path to raw bytes and mime type."""
        if isinstance(image, str):
            with open(image, "rb") as f:
                data = f.read()
            mime = mimetypes.guess_type(image)[0] or "image/png"
        else:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data = buf.getvalue()
            mime = "image/png"
        return data, mime

    # 合并metallic和roughness
    if "metallic" in textures_dict and "roughness" in textures_dict:
        mr_combined = combine_metallic_roughness(textures_dict["metallic"], textures_dict["roughness"])
        textures_dict = dict(textures_dict)
        textures_dict["metallicRoughness"] = mr_combined

    # 按固定顺序将图片字节追加到 binary blob，创建对应的 BufferView
    TEXTURE_ORDER = ("albedo", "metallicRoughness", "normal", "ao")
    images = []
    textures = []
    tex_type_to_index = {}

    for tex_type in TEXTURE_ORDER:
        tex_value = textures_dict.get(tex_type)
        if tex_value:
            img_bytes, mime = image_to_bytes(tex_value)
            # GLB binary blob 要求每个 chunk 4 字节对齐
            pad = (4 - len(binary) % 4) % 4
            binary.extend(b'\x00' * pad)
            img_bv_index = len(buffer_views)
            buffer_views.append(BufferView(
                buffer=0,
                byteOffset=len(binary),
                byteLength=len(img_bytes),
            ))
            binary.extend(img_bytes)
            tex_type_to_index[tex_type] = len(textures)
            images.append(pygltflib.Image(bufferView=img_bv_index, mimeType=mime))
            textures.append(pygltflib.Texture(source=len(images) - 1))

    total_byte_length = len(binary)

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
    
    # Set binary data (includes mesh buffers + embedded image bytes)
    gltf.set_binary_blob(bytes(binary))

    # 8. Save GLB
    gltf.save(output_path)
    print(f"PBR GLB文件已保存: {output_path}")
