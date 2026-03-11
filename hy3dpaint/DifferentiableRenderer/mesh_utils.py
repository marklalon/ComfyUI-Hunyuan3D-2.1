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
import cv2
import numpy as np
from io import StringIO
from typing import Optional, Tuple, Dict, Any


def _safe_extract_attribute(obj: Any, attr_path: str, default: Any = None) -> Any:
    """Extract nested attribute safely from object."""
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return default


def _convert_to_numpy(data: Any, dtype: np.dtype) -> Optional[np.ndarray]:
    """Convert data to numpy array with specified dtype, handling None values."""
    if data is None:
        return None
    return np.asarray(data, dtype=dtype)


def load_mesh(mesh):
    """Load mesh data including vertices, faces, UV coordinates, normals and texture."""
    # Extract vertex positions and face indices
    vtx_pos = _safe_extract_attribute(mesh, "vertices")
    pos_idx = _safe_extract_attribute(mesh, "faces")

    # Extract UV coordinates (reusing face indices for UV indices)
    vtx_uv = _safe_extract_attribute(mesh, "visual.uv")
    uv_idx = pos_idx  # Reuse face indices for UV mapping

    # Extract vertex normals
    vtx_normal = _safe_extract_attribute(mesh, "vertex_normals")

    # Convert to numpy arrays with appropriate dtypes
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)
    vtx_normal = _convert_to_numpy(vtx_normal, np.float32)

    texture_data = None
    return vtx_pos, pos_idx, vtx_uv, uv_idx, vtx_normal, texture_data


def _get_base_path_and_name(mesh_path: str) -> Tuple[str, str]:
    """Get base path without extension and mesh name."""
    base_path = os.path.splitext(mesh_path)[0]
    name = os.path.basename(base_path)
    return base_path, name


def _save_texture_map(
    texture: np.ndarray,
    base_path: str,
    suffix: str = "",
    image_format: str = ".jpg",
    color_convert: Optional[int] = None,
) -> str:
    """Save texture map with optional color conversion."""
    path = f"{base_path}{suffix}{image_format}"
    # Clip to [0, 255] before cast to prevent uint8 overflow/wrap-around
    processed_texture = np.clip(texture * 255, 0, 255).astype(np.uint8)

    if color_convert is not None:
        # If already grayscale (single channel), skip cvtColor to avoid cv2 error
        if processed_texture.ndim == 2 or processed_texture.shape[-1] == 1:
            processed_texture = processed_texture.squeeze()
        else:
            processed_texture = cv2.cvtColor(processed_texture, color_convert)
        cv2.imwrite(path, processed_texture)
    else:
        cv2.imwrite(path, processed_texture[..., ::-1])  # RGB to BGR

    return os.path.basename(path)


def _write_mtl_properties(f, properties: Dict[str, Any]):
    """Write material properties to MTL file."""
    for key, value in properties.items():
        if isinstance(value, (list, tuple)):
            f.write(f"{key} {' '.join(map(str, value))}\n")
        else:
            f.write(f"{key} {value}\n")


def _create_obj_content(
    vtx_pos: np.ndarray, pos_idx: np.ndarray, name: str, vtx_uv: np.ndarray = None, uv_idx: np.ndarray = None, vtx_normal: np.ndarray = None, has_texture: bool = False
) -> str:
    """Create OBJ file content."""
    buffer = StringIO()

    # Write header and vertices
    if has_texture:
        buffer.write(f"mtllib {name}.mtl\n")
    buffer.write(f"o {name}\n")
    np.savetxt(buffer, vtx_pos, fmt="v %.6f %.6f %.6f")
    
    # Write UV coordinates if provided
    if vtx_uv is not None:
        np.savetxt(buffer, vtx_uv, fmt="vt %.6f %.6f")

    # Write vertex normals if provided
    if vtx_normal is not None:
        np.savetxt(buffer, vtx_normal, fmt="vn %.6f %.6f %.6f")

    if has_texture:
        buffer.write("s 0\nusemtl Material\n")

    # Write faces with vectorized formatting
    p = pos_idx + 1   # shape: (F, 3), 1-based
    
    if vtx_normal is not None and vtx_uv is not None and uv_idx is not None:
        # Face format: f v/vt/vn — normal index equals position index (per-vertex normals)
        u = uv_idx + 1
        face_strings = [
            f"f {p[i,0]}/{u[i,0]}/{p[i,0]} {p[i,1]}/{u[i,1]}/{p[i,1]} {p[i,2]}/{u[i,2]}/{p[i,2]}"
            for i in range(len(p))
        ]
    elif vtx_normal is not None:
        # Face format: f v//vn v//vn v//vn
        face_strings = [
            f"f {p[i,0]}//{p[i,0]} {p[i,1]}//{p[i,1]} {p[i,2]}//{p[i,2]}"
            for i in range(len(p))
        ]
    elif vtx_uv is not None and uv_idx is not None:
        # Face format: f v/vt v/vt v/vt
        u = uv_idx + 1
        face_strings = [
            f"f {p[i,0]}/{u[i,0]} {p[i,1]}/{u[i,1]} {p[i,2]}/{u[i,2]}"
            for i in range(len(p))
        ]
    else:
        # Face format: f v v v (vertices only)
        face_strings = [
            f"f {p[i,0]} {p[i,1]} {p[i,2]}"
            for i in range(len(p))
        ]
    buffer.write("\n".join(face_strings) + "\n")

    return buffer.getvalue()


def save_obj_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv=None, uv_idx=None, texture=None, metallic=None, roughness=None, normal=None, vtx_normal=None):
    """Save mesh as OBJ file with optional textures and material.
    
    Args:
        mesh_path: Output path for the OBJ file
        vtx_pos: Vertex positions (N, 3)
        pos_idx: Face indices (F, 3)
        vtx_uv: Optional UV coordinates (N, 2)
        uv_idx: Optional UV indices (F, 3), defaults to pos_idx if not provided
        texture: Optional texture image (H, W, 3)
        metallic: Optional metallic map (H, W)
        roughness: Optional roughness map (H, W)
        normal: Optional normal map (H, W, 3)
        vtx_normal: Optional vertex normals (N, 3)
    """
    # Convert inputs to numpy arrays
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)
    vtx_normal = _convert_to_numpy(vtx_normal, np.float32)

    base_path, name = _get_base_path_and_name(mesh_path)

    # If UV indices not provided but UV coordinates are, use face indices
    if vtx_uv is not None and uv_idx is None:
        uv_idx = pos_idx

    # Determine if we have texture data
    has_texture = texture is not None

    # Create and save OBJ content
    obj_content = _create_obj_content(vtx_pos, pos_idx, name, vtx_uv, uv_idx, vtx_normal, has_texture)
    with open(mesh_path, "w") as obj_file:
        obj_file.write(obj_content)

    # Save texture maps only if texture is provided
    if has_texture:
        texture_maps = {}
        texture_maps["diffuse"] = _save_texture_map(texture, base_path)

        if metallic is not None:
            texture_maps["metallic"] = _save_texture_map(metallic, base_path, "_metallic", ".png", color_convert=cv2.COLOR_RGB2GRAY)
        if roughness is not None:
            texture_maps["roughness"] = _save_texture_map(roughness, base_path, "_roughness", ".png", color_convert=cv2.COLOR_RGB2GRAY)
        if normal is not None:
            texture_maps["normal"] = _save_texture_map(normal, base_path, "_normal", ".png")

        # is_pbr when any PBR map is provided
        _create_mtl_file(base_path, texture_maps, any(x is not None for x in (metallic, roughness, normal)))


def _create_mtl_file(base_path: str, texture_maps: Dict[str, str], is_pbr: bool):
    """Create MTL material file."""
    mtl_path = f"{base_path}.mtl"

    with open(mtl_path, "w") as f:
        f.write("newmtl Material\n")

        if is_pbr:
            # PBR material properties
            properties = {
                "Kd": [0.800, 0.800, 0.800],
                "Ke": [0.000, 0.000, 0.000],
                "Ni": 1.500,
                "d": 1.0,
                "illum": 2,
                "map_Kd": texture_maps["diffuse"],
            }
            _write_mtl_properties(f, properties)

            # Additional PBR maps
            map_configs = [("metallic", "map_Pm"), ("roughness", "map_Pr"), ("normal", "map_Bump -bm 1.0")]

            for texture_key, mtl_key in map_configs:
                if texture_key in texture_maps:
                    f.write(f"{mtl_key} {texture_maps[texture_key]}\n")
        else:
            # Standard material properties
            properties = {
                "Ns": 250.000000,
                "Ka": [0.200, 0.200, 0.200],
                "Kd": [0.800, 0.800, 0.800],
                "Ks": [0.500, 0.500, 0.500],
                "Ke": [0.000, 0.000, 0.000],
                "Ni": 1.500,
                "d": 1.0,
                "illum": 3,
                "map_Kd": texture_maps["diffuse"],
            }
            _write_mtl_properties(f, properties)


def save_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv=None, uv_idx=None, texture=None, metallic=None, roughness=None, normal=None, vtx_normal=None):
    """Save mesh using OBJ format."""
    save_obj_mesh(
        mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic=metallic, roughness=roughness, normal=normal, vtx_normal=vtx_normal
    )
