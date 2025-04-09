from pathlib import Path
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

from read_write_model import read_model, Image as ColmapImage, Camera
from filament_estimation import FilamentEstimator
from utils import CameraIntrinsics, load_colmap_model, synthetic_generate_masks, world_to_image


def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    data_dir = Path("data/SyntheticCase2/")
    cameras, images = load_colmap_model(data_dir / f'colmap', device)

    # 生成掩码
    if 1:
        x_true = torch.tensor([-1.42, 1.25, 3.13, -1.63, 1.22, -1.33, 1.22, -1.48, 1.13, 2.99]).to(device)
        synthetic_generate_masks(cameras, images, x_true, data_dir / f'masks', sigma=10, device=device)

    # 设定 batch_size 列表
    batch_sizes = [2]
    for batch_size in batch_sizes:
        logging.info(f"Running with batch size: {batch_size}")

        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'runs/batch_{batch_size}_{timestamp}'

        # 创建 TensorBoard 写入器
        writer = SummaryWriter(log_dir)

        # 曲线估计
        estimator = FilamentEstimator(
            cameras=cameras,
            images=images,
            data_dir=data_dir,
            sigma=10.0,
            learning_rate=0.001,
            num_iterations=5,
            batch_size=batch_size,
            device=device
        )

        x0 = torch.tensor([-1.42, 1.25, 3.13, -1.43, 1.13, -1.43, 1.22, -1.48, 1.13, 2.99]).to(device)
        x_est, loss = estimator.estimate(x0, writer)

        print(f"Optimized loss: {loss:.8f} Best_params: {[f'{p.item():.4f}' for p in x_est.data.cpu().numpy()]}")

        # 创建保存结果的目录
        result_dir = Path(log_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        # 保存结果到文件
        result_file = result_dir / 'result.txt'
        with open(result_file, 'w') as f:
            f.write(f"Optimized loss: {loss:.8f}\n")
            f.write(f"Best_params: {[f'{p.item():.4f}' for p in x_est.data.cpu().numpy()]}\n")

        # 关闭 TensorBoard 写入器
        writer.close()


if __name__ == "__main__":
    main()