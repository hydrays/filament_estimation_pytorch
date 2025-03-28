#
# filament estimation
#
# 2025-03-04
from typing import Any, Dict, List, Optional, Tuple, Union
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
from utils import CameraIntrinsics, load_colmap_model, world_to_image

class FilamentEstimator:
    def __init__(
        self,
        cameras: Dict,
        images: Dict,
        data_dir: Union[str, Path],
        sigma: float = 10.0,
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.cameras = cameras
        self.images = images
        self.sigma = sigma
        self.lr = learning_rate
        self.n_iters = num_iterations
        self.device = torch.device(device)
        self.intrinsics = self._get_intrinsics()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.masks = self._preload_masks(data_dir/'masks')
        
        self.kernel_size = max(3, 2 * int(3 * self.sigma) + 1)
        self.grid_cache = {}
        for img_id, img_data in self.images.items():
            height, width = self.intrinsics.image_size
            y_grid, x_grid = torch.meshgrid(
                torch.arange(height, device=self.device, dtype=torch.float32),
                torch.arange(width, device=self.device, dtype=torch.float32),
                indexing='ij'
            )
            self.grid_cache[img_id] = (x_grid, y_grid)

    def estimate(self, x0: torch.Tensor) -> Tuple[torch.Tensor, float]:
        x = torch.nn.Parameter(x0.clone().to(self.device))
        optimizer = Adam([x], lr=self.lr)
        
        best_loss = float('inf')
        best_x = x.detach().clone()
        image_ids = list(self.images.keys())

        for iter in range(self.n_iters):
            optimizer.zero_grad()
            
            # 随机选择一张图像
            img_id = random.choice(image_ids)
            img_data = self.images[img_id]
            target = self.masks[img_id]
            width, height = self.intrinsics.image_size[::-1]
            
            # # 曲线生成
            # coords = torch.stack([
            #     x[0:3],
            #     torch.stack([x[3], x[4], (x[2] + x[7])/2]),
            #     x[5:8]
            # ])

            # 构造控制点，调整为[N, 3]格式
            n_points = (len(x) - 2) // 2
            z_start = x[2]
            z_end = x[-1]
            z_intp = torch.linspace(z_start, z_end, n_points, device=self.device)

            # 生成x和y的索引列表
            x_indices = [0] + [3 + 2 * (i - 1) for i in range(1, n_points)]
            y_indices = [1] + [4 + 2 * (i - 1) for i in range(1, n_points)]

            # 提取x和y的值
            xcoords = x[x_indices]
            ycoords = x[y_indices]

            # 组合成控制点坐标
            coords = torch.stack([xcoords, ycoords, z_intp], dim=1)
                        
            # 评估点
            t_controls = torch.linspace(0, 1, len(coords), device=self.device)
            coeffs = torchcubicspline.natural_cubic_spline_coeffs(t_controls, coords)
            spline = torchcubicspline.NaturalCubicSpline(coeffs)
            t_eval = torch.linspace(0, 1, 100, device=self.device)
            curve = spline.evaluate(t_eval).T  # [3, N]

            # ====== 投影计算 ======
            R = img_data['R']
            t = img_data['t']
            
            # 向量化投影
            points_cam = R @ curve + t.unsqueeze(-1)  # [3, N]
            z = points_cam[2].clamp(min=1e-6)
            x_proj = (points_cam[0]/z * self.intrinsics.focal_length[0] + self.intrinsics.principal_point[0])
            y_proj = (points_cam[1]/z * self.intrinsics.focal_length[1] + self.intrinsics.principal_point[1])
            
            # 有效性判断
            valid = (
                (points_cam[2] > 0) &
                (x_proj >= 0) & (x_proj < width) &
                (y_proj >= 0) & (y_proj < height)
            )
            
            # ====== 生成预测mask ======
            grid_x, grid_y = self.grid_cache[img_id]
            pred = torch.zeros_like(target)
            
            if valid.any():
                # 只保留有效点
                xv = x_proj[valid]
                yv = y_proj[valid]
                
                # 向量化距离计算 (使用广播)
                dx = grid_x.unsqueeze(-1) - xv  # [H,W,N_valid]
                dy = grid_y.unsqueeze(-1) - yv
                dist_sq = dx**2 + dy**2
                
                # 高斯权重求和
                weights = torch.exp(-dist_sq / (2 * self.sigma**2))
                pred = weights.sum(dim=-1)  # [H,W]
                
                # 保持原有模糊参数
                pred = TF.gaussian_blur(
                    pred.unsqueeze(0).unsqueeze(0),
                    kernel_size=self.kernel_size,
                    sigma=self.sigma
                ).squeeze()
                
                # 归一化
                pred = pred / pred.max() if pred.max() > 0 else pred

            # ====== 损失计算 ======
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            
            # 更新最佳参数
            with torch.no_grad():
                if loss < best_loss:
                    best_loss = loss.item()
                    best_x.copy_(x.data)
            
            self.logger.info(f"Iter {iter+1}/{self.n_iters} Loss: {loss.item():.4f} Params: {[f'{p.item():.8f}' for p in x.data.cpu().numpy()]}")

        return best_x, best_loss

    def _get_intrinsics(self) -> CameraIntrinsics:
        """Get intrinsics from first camera."""
        cam = self.cameras[1]
        return CameraIntrinsics(
            focal_length=[cam['params'][0], cam['params'][0]],
            principal_point=[cam['params'][1], cam['params'][2]],
            image_size=[cam['height'], cam['width']],
            radial_distortion=[cam['params'][3], cam['params'][3]]
        )
    
    def _preload_masks(self, mask_dir: Union[str, Path]) -> Dict[int, torch.Tensor]:
        """预处理加载所有mask到内存"""
        masks = {}
        mask_dir = Path(mask_dir)
        for img_id, img_data in self.images.items():
            mask_path = mask_dir / f'mask_{img_data["name"]}'
            pil_img = Image.open(mask_path).convert('L')
            mask = torch.from_numpy(np.array(pil_img)).float().to(self.device) / 255.0
            masks[img_id] = mask
        return masks
        