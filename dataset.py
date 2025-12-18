import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class S3DISDataset(Dataset):
    """
    Dataset для S3DIS (Stanford 3D Indoor Spaces)
    Поддерживает загрузку данных в формате (N, 6) где 6 каналов: (x, y, z, r, g, b)
    """
    
    def __init__(self, data_dir, area='Area_1', split='train', num_points=4096, 
                 transform=None, test_size=0.2, val_size=0.1):
        """
        Args:
            data_dir: путь к директории с данными S3DIS
            area: область для загрузки (Area_1, Area_2, etc.)
            split: 'train', 'val' или 'test'
            num_points: количество точек для выборки из каждого облака
            transform: трансформации для аугментации
            test_size: доля тестовой выборки
            val_size: доля валидационной выборки
        """
        self.data_dir = data_dir
        self.area = area
        self.split = split
        self.num_points = num_points
        self.transform = transform
        
        # Загружаем все файлы из указанной области
        self.file_paths = self._load_file_paths()
        
        # Проверяем наличие файлов
        if len(self.file_paths) == 0:
            raise ValueError(
                f"Не найдено файлов данных в {os.path.join(data_dir, area)}. "
                f"Убедитесь, что данные загружены и распакованы."
            )
        
        # Разделяем на train/val/test
        train_files, temp_files = train_test_split(
            self.file_paths, test_size=(test_size + val_size), random_state=42
        )
        val_size_adjusted = val_size / (test_size + val_size)
        val_files, test_files = train_test_split(
            temp_files, test_size=(1 - val_size_adjusted), random_state=42
        )
        
        if split == 'train':
            self.file_paths = train_files
        elif split == 'val':
            self.file_paths = val_files
        else:
            self.file_paths = test_files
        
        print(f"✓ {split}: {len(self.file_paths)} файлов")
    
    def _load_file_paths(self):
        """Загружает пути к файлам данных"""
        file_paths = []
        area_path = os.path.join(self.data_dir, self.area)
        
        if not os.path.exists(area_path):
            return []
        
        # Ищем все файлы с облаками точек (.txt, .npy, .ply)
        for root, dirs, files in os.walk(area_path):
            for file in files:
                if file.endswith('.txt') or file.endswith('.npy') or file.endswith('.ply'):
                    file_paths.append(os.path.join(root, file))
        
        return sorted(file_paths)  # Сортируем для воспроизводимости
    
    def _load_point_cloud(self, file_path):
        """
        Загружает облако точек из файла
        Поддерживает форматы: .ply, .txt, .npy
        """
        if file_path.endswith('.ply'):
            return self._load_ply(file_path)
        elif file_path.endswith('.npy'):
            data = np.load(file_path)
        else:
            # Читаем из текстового файла
            data = np.loadtxt(file_path)
        
        # Проверяем наличие данных
        if len(data) == 0:
            raise ValueError(f"Файл {file_path} пуст или поврежден")
        
        # Проверяем формат данных
        if data.shape[1] >= 7:
            # x, y, z, r, g, b, label
            points = data[:, :6]  # координаты и цвета
            labels = data[:, 6].astype(np.int64)
        elif data.shape[1] >= 4:
            # x, y, z, label
            points = data[:, :3]
            # Добавляем случайные цвета если их нет
            colors = np.random.randint(0, 255, (len(points), 3))
            points = np.concatenate([points, colors], axis=1)
            labels = data[:, 3].astype(np.int64)
        else:
            # Только координаты
            points = data[:, :3]
            colors = np.random.randint(0, 255, (len(points), 3))
            points = np.concatenate([points, colors], axis=1)
            labels = np.zeros(len(points), dtype=np.int64)
        
        return points, labels
    
    def _load_ply(self, file_path):
        """
        Загружает облако точек из PLY файла
        Формат PLY: x, y, z, label (или scalar_Label)
        """
        if HAS_OPEN3D:
            try:
                # Пробуем загрузить через Open3D
                pcd = o3d.io.read_point_cloud(file_path)
                points_xyz = np.asarray(pcd.points)
                
                # Пытаемся получить метки из colors или других атрибутов
                if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
                    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
                else:
                    colors = np.random.randint(0, 255, (len(points_xyz), 3))
                
                # Читаем метки из файла напрямую
                labels = self._read_ply_labels(file_path)
                if labels is None:
                    labels = np.zeros(len(points_xyz), dtype=np.int64)
                
                points = np.concatenate([points_xyz, colors], axis=1)
                return points, labels
            except:
                pass
        
        # Если Open3D не работает, читаем вручную
        return self._read_ply_manual(file_path)
    
    def _read_ply_labels(self, file_path):
        """Читает метки из PLY файла"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Находим начало данных
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() == 'end_header':
                    data_start = i + 1
                    break
            
            # Читаем данные
            labels = []
            for line in lines[data_start:]:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        # Последний столбец - это метка
                        labels.append(int(float(parts[-1])))
            
            return np.array(labels, dtype=np.int64) if labels else None
        except:
            return None
    
    def _read_ply_manual(self, file_path):
        """Читает PLY файл вручную"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Находим начало данных и количество вершин
            num_vertices = 0
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                if line.strip() == 'end_header':
                    data_start = i + 1
                    break
            
            # Читаем данные
            points_xyz = []
            labels = []
            
            for line in lines[data_start:data_start + num_vertices]:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        label = int(float(parts[3]))
                        points_xyz.append([x, y, z])
                        labels.append(label)
            
            points_xyz = np.array(points_xyz)
            labels = np.array(labels, dtype=np.int64)
            
            # Добавляем случайные цвета
            colors = np.random.randint(0, 255, (len(points_xyz), 3))
            points = np.concatenate([points_xyz, colors], axis=1)
            
            return points, labels
        except Exception as e:
            raise ValueError(f"Ошибка загрузки PLY файла {file_path}: {e}")
    
    def _sample_points(self, points, labels, num_points):
        """Выбирает num_points точек из облака"""
        if len(points) >= num_points:
            # Случайная выборка
            indices = np.random.choice(len(points), num_points, replace=False)
            return points[indices], labels[indices]
        else:
            # Дублируем точки если их меньше
            indices = np.random.choice(len(points), num_points, replace=True)
            return points[indices], labels[indices]
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        points, labels = self._load_point_cloud(file_path)
        
        # Выбираем фиксированное количество точек
        points, labels = self._sample_points(points, labels, self.num_points)
        
        # Нормализуем координаты
        points[:, :3] = points[:, :3] - np.mean(points[:, :3], axis=0)
        max_dist = np.max(np.linalg.norm(points[:, :3], axis=1))
        if max_dist > 0:
            points[:, :3] = points[:, :3] / max_dist
        
        # Нормализуем цвета
        points[:, 3:6] = points[:, 3:6] / 255.0
        
        # Применяем трансформации
        if self.transform:
            points = self.transform(points)
        
        # Конвертируем в тензоры
        points = torch.FloatTensor(points)
        labels = torch.LongTensor(labels)
        
        return points, labels

