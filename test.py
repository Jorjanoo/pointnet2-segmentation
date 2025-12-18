import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse

from dataset import S3DISDataset
from pointnet2 import PointNet2Seg
from metrics import calculate_metrics, calculate_iou, calculate_mean_iou


def test(model, test_loader, criterion, device, num_classes, class_names=None):
    """Тестирование модели"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    
    all_preds = []
    all_labels = []
    
    print('Testing...')
    with torch.no_grad():
        for points, labels in test_loader:
            points = points.to(device)
            labels = labels.to(device)
            
            logits = model(points)
            logits = logits.reshape(-1, logits.shape[-1])
            labels_flat = labels.reshape(-1)
            
            loss = criterion(logits, labels_flat)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels_flat).sum().item()
            total_points += labels_flat.size(0)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_flat.cpu().numpy())
    
    # Объединяем все предсказания
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Вычисляем метрики
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * total_correct / total_points
    
    metrics = calculate_metrics(all_preds, all_labels, num_classes)
    iou = calculate_iou(all_preds, all_labels, num_classes)
    mean_iou = calculate_mean_iou(all_preds, all_labels, num_classes)
    
    # Выводим результаты
    print('\n' + '='*60)
    print('TEST RESULTS')
    print('='*60)
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Mean IoU: {mean_iou:.4f}')
    print('\nPer-class IoU:')
    print('-'*60)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    for i in range(num_classes):
        if iou[i] is not None:
            print(f'{class_names[i]:20s}: {iou[i]:.4f}')
        else:
            print(f'{class_names[i]:20s}: N/A (no samples)')
    
    print('\nPer-class Metrics:')
    print('-'*60)
    print(f'{"Class":<20s} {"Precision":<12s} {"Recall":<12s} {"F1-Score":<12s}')
    print('-'*60)
    for i in range(num_classes):
        print(f'{class_names[i]:<20s} {metrics["precision"][i]:<12.4f} '
              f'{metrics["recall"][i]:<12.4f} {metrics["f1"][i]:<12.4f}')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'iou': iou,
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Test PointNet++ for Semantic Segmentation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to S3DIS dataset')
    parser.add_argument('--area', type=str, default='Area_1', help='Area to use')
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_classes', type=int, default=13, help='Number of classes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    # Устройство
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Датасет
    print('Loading test dataset...')
    test_dataset = S3DISDataset(
        args.data_dir, area=args.area, split='test',
        num_points=args.num_points
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Модель
    print('Loading model...')
    model = PointNet2Seg(num_classes=args.num_classes, num_points=args.num_points).to(device)
    
    # Загружаем веса
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    if 'best_iou' in checkpoint:
        print(f'Best IoU during training: {checkpoint["best_iou"]:.4f}')
    
    # Функция потерь
    criterion = nn.CrossEntropyLoss()
    
    # Имена классов для S3DIS
    class_names = [
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
    ]
    
    # Тестирование
    results = test(model, test_loader, criterion, device, args.num_classes, class_names)
    
    print('\n' + '='*60)
    print('Testing completed!')


if __name__ == '__main__':
    main()

