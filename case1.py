#
# test the accuracy and speed of the code
# 

from pathlib import Path
import logging
import torch

from read_write_model import read_model, Image as ColmapImage, Camera
from filament_estimation import FilamentEstimator
from utils import CameraIntrinsics, load_colmap_model, synthetic_generate_masks, world_to_image


def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    data_dir = Path("data/SyntheticCase1/")
    cameras, images = load_colmap_model(data_dir / f'colmap', device)
    
    # 生成掩码
    if 1:
        x_true = torch.tensor([-1.42, 1.25, 3.13, -1.43, 1.22, -1.48, 1.13, 2.99])
        synthetic_generate_masks(cameras, images, x_true, data_dir / f'masks', sigma=10, device=device)

    # 曲线估计    
    estimator = FilamentEstimator(
        cameras=cameras,
        images=images,
        data_dir=data_dir,  # 新增参数
        sigma=10.0,
        learning_rate=0.001,
        num_iterations=1000
    )
    
    x0 = torch.tensor([-1.4333,    1.2356,    3.1422, -1.4489,    1.2133, -1.4733,    1.1400,    2.9956])
    x_est, loss = estimator.estimate(x0)  # 不再需要传递mask_dir
    print(f"Optimized loss: {loss:.8f} Best_params: {[f'{p.item():.4f}' for p in x_est.data.cpu().numpy()]}")

if __name__ == "__main__":
    main()