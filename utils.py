from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from PIL import Image
import random
import torchcubicspline

from read_write_model import read_model, Image as ColmapImage, Camera

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    focal_length: List[float]
    principal_point: List[float]
    image_size: List[int]
    radial_distortion: Optional[List[float]] = None

    def to_matrix(self) -> torch.Tensor:
        """Convert intrinsics to camera matrix."""
        return torch.tensor([
            [self.focal_length[0], 0, self.principal_point[0]],
            [0, self.focal_length[1], self.principal_point[1]],
            [0, 0, 1]
        ], dtype=torch.float32, requires_grad=True)


def load_colmap_model(model_path: Union[str, Path], device: torch.device) -> Tuple[Dict, Dict]:
    """Load cameras and images from COLMAP model."""
    model_path = Path(model_path)
    cameras, images, _ = read_model(str(model_path), '.txt')

    # Convert images to tensors
    img_dict = {}
    for img_id, img in images.items():
        R = torch.tensor(img.qvec2rotmat(), dtype=torch.float32, device=device)
        t = torch.tensor(img.tvec, dtype=torch.float32, device=device)
        img_dict[img_id] = {
            'R': R, 't': t, 'camera_id': img.camera_id, 'name': img.name
        }

    # Convert cameras to tensors
    cam_dict = {}
    for cam_id, cam in cameras.items():
        cam_dict[cam_id] = {
            'model': cam.model,
            'width': cam.width,
            'height': cam.height,
            'params': torch.tensor(cam.params, dtype=torch.float32, device=device)
        }
    return cam_dict, img_dict

def synthetic_generate_masks(
    cameras: Dict,
    images: Dict,
    curve_params: torch.Tensor,
    output_dir: Union[str, Path],
    sigma: float = 2.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """Generate mask as synthetic data."""
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get camera intrinsics (assuming camera_id=1)
    camera = cameras[1]
    intrinsics = CameraIntrinsics(
        focal_length=[camera['params'][0], camera['params'][0]],
        principal_point=[camera['params'][1], camera['params'][2]],
        image_size=[camera['height'], camera['width']],
        radial_distortion=[camera['params'][3], camera['params'][3]]
    )

    # 构造控制点 (参数格式需要调整为[N, 3])
    coords = torch.stack([
        curve_params[0:3],
        torch.stack([curve_params[3], curve_params[4], (curve_params[2] + curve_params[7])/2]),
        curve_params[5:8]
    ]).to(device)
    
    # 创建三次样条
    t_controls = torch.linspace(0, 1, len(coords), device=device)  # 控制点参数
    coeffs = torchcubicspline.natural_cubic_spline_coeffs(t_controls, coords)
    spline = torchcubicspline.NaturalCubicSpline(coeffs)
    
    # 生成插值点 (5000个点)
    t_eval = torch.linspace(0, 1, 5000, device=device)
    curve = spline.evaluate(t_eval).T  # 转置为[3, 100]的格式
    
    # Project and save masks for each image
    for img_id, img_data in images.items():
        uv, valid = world_to_image(intrinsics, img_data['R'], img_data['t'], curve)
        mask = torch.zeros(intrinsics.image_size, device=device)
        if valid.any():
            uv_valid = uv[:, valid].round().long()
            mask[uv_valid[1], uv_valid[0]] = 1.0
            mask = TF.gaussian_blur(
                mask.unsqueeze(0).unsqueeze(0), 
                kernel_size=max(3, 2 * int(3 * sigma) + 1),
                sigma=sigma
            ).squeeze()
            mask = mask / mask.max() if mask.max() > 0 else mask
        
        # Save with Pillow
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(mask_np)
        pil_img.save(output_dir / f'mask_{img_data["name"]}')

def world_to_image(
    intrinsics: CameraIntrinsics,
    R: torch.Tensor,
    t: torch.Tensor,
    points_3d: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D points to 2D image coordinates."""
    points_cam = R @ points_3d + t.unsqueeze(1)
    z = points_cam[2].clamp(min=1e-6)
    x = (points_cam[0] / z * intrinsics.focal_length[0] + intrinsics.principal_point[0])
    y = (points_cam[1] / z * intrinsics.focal_length[1] + intrinsics.principal_point[1])
    valid = (points_cam[2] > 0) & (x >= 0) & (x < intrinsics.image_size[1]) & (y >= 0) & (y < intrinsics.image_size[0])
    return torch.stack([x, y]), valid
