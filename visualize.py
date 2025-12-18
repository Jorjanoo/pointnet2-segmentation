import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import open3d as o3d

from dataset import S3DISDataset
from pointnet2 import PointNet2Seg


def visualize_point_cloud(points, labels, preds=None, save_path=None):
    """
    Визуализирует облако точек с метками и предсказаниями
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Исходные точки с истинными метками
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=labels, cmap='tab20', s=1, alpha=0.6)
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1)
    
    if preds is not None:
        # Предсказания
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                              c=preds, cmap='tab20', s=1, alpha=0.6)
        ax2.set_title('Predictions')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.colorbar(scatter2, ax=ax2)
        
        # Ошибки
        ax3 = fig.add_subplot(133, projection='3d')
        errors = (labels != preds).astype(int)
        scatter3 = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                              c=errors, cmap='Reds', s=1, alpha=0.6)
        ax3.set_title('Errors (Red = Wrong)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        plt.colorbar(scatter3, ax=ax3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to {save_path}')
    else:
        plt.show()


def visualize_with_open3d(points, labels, preds=None, save_path=None):
    """
    Визуализирует облако точек с помощью Open3D
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Цвета на основе меток
    colors = plt.cm.tab20(labels / labels.max())[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
        print(f'Saved point cloud to {save_path}')
    else:
        o3d.visualization.draw_geometries([pcd])


def main():
    parser = argparse.ArgumentParser(description='Visualize PointNet++ predictions')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to S3DIS dataset')
    parser.add_argument('--area', type=str, default='Area_1', help='Area to use')
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points per cloud')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=13, help='Number of classes')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--use_open3d', action='store_true', help='Use Open3D for visualization')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Устройство
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Датасет
    print('Loading dataset...')
    test_dataset = S3DISDataset(
        args.data_dir, area=args.area, split='test',
        num_points=args.num_points
    )
    
    # Модель
    print('Loading model...')
    model = PointNet2Seg(num_classes=args.num_classes, num_points=args.num_points).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'Visualizing {args.num_samples} samples...')
    
    for i in range(min(args.num_samples, len(test_dataset))):
        points, labels = test_dataset[i]
        points_tensor = points.unsqueeze(0).to(device)  # [1, N, 6]
        
        with torch.no_grad():
            # Модель принимает один аргумент [B, N, 6] где 6 = (x, y, z, r, g, b)
            logits = model(points_tensor)  # [1, N, num_classes]
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        
        points_np = points.numpy()
        labels_np = labels.numpy()
        
        if args.use_open3d:
            save_path = os.path.join(args.output_dir, f'sample_{i}.ply')
            visualize_with_open3d(points_np, labels_np, preds, save_path)
        else:
            save_path = os.path.join(args.output_dir, f'sample_{i}.png')
            visualize_point_cloud(points_np, labels_np, preds, save_path)
    
    print(f'Visualizations saved to {args.output_dir}')


if __name__ == '__main__':
    main()

