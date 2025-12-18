import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction Layer для PointNet++
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        """
        Args:
            xyz: координаты точек [B, N, 3]
            points: признаки точек [B, N, C]
        Returns:
            new_xyz: координаты центроидов [B, npoint, 3]
            new_points: новые признаки [B, npoint, mlp[-1]]
        """
        B, N, C = xyz.shape
        S = self.npoint
        
        # Farthest Point Sampling
        fps_idx = self.farthest_point_sample(xyz, S)  # [B, S]
        new_xyz = self.index_points(xyz, fps_idx)  # [B, S, 3]
        
        # Ball Query
        idx = self.ball_query(self.radius, self.nsample, xyz, new_xyz)  # [B, S, nsample]
        
        # Grouping
        grouped_xyz = self.index_points(xyz, idx)  # [B, S, nsample, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, S, nsample, 3]
        
        if points is not None:
            grouped_points = self.index_points(points, idx)  # [B, S, nsample, D]
            grouped_points_norm = torch.cat([grouped_points, grouped_xyz_norm], dim=-1)  # [B, S, nsample, D+3]
        else:
            grouped_points_norm = grouped_xyz_norm  # [B, S, nsample, 3]
        
        grouped_points_norm = grouped_points_norm.permute(0, 3, 2, 1)  # [B, D+3, nsample, S]
        
        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points_norm = F.relu(bn(conv(grouped_points_norm)))
        
        # Max pooling
        new_points = torch.max(grouped_points_norm, 2)[0]  # [B, D', S]
        new_points = new_points.transpose(1, 2)  # [B, S, D']
        
        return new_xyz, new_points
    
    def farthest_point_sample(self, xyz, npoint):
        """Farthest Point Sampling"""
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        
        return centroids
    
    def ball_query(self, radius, nsample, xyz, new_xyz):
        """Ball Query"""
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = self.square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx
    
    def square_distance(self, src, dst):
        """Вычисляет квадрат расстояния между точками"""
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist
    
    def index_points(self, points, idx):
        """Индексирует точки по индексам"""
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


class PointNetFeaturePropagation(nn.Module):
    """
    Feature Propagation Layer для интерполяции признаков обратно на исходные точки
    """
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: координаты точек для интерполяции [B, N, 3]
            xyz2: координаты точек с признаками [B, M, 3]
            points1: признаки для интерполяции [B, N, C1]
            points2: признаки для интерполяции [B, M, C2]
        Returns:
            new_points: интерполированные признаки [B, N, mlp[-1]]
        """
        B, N, C = xyz1.shape
        _, M, _ = xyz2.shape
        
        if M == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = self.square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # Берем 3 ближайших соседа
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(self.index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        new_points = new_points.transpose(1, 2)  # [B, C, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points.transpose(1, 2)  # [B, N, C]
    
    def square_distance(self, src, dst):
        """Вычисляет квадрат расстояния между точками"""
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist
    
    def index_points(self, points, idx):
        """Индексирует точки по индексам"""
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


class PointNet2Seg(nn.Module):
    """
    PointNet++ для семантической сегментации
    """
    def __init__(self, num_classes=13, num_points=4096):
        super(PointNet2Seg, self).__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        
        # Encoder (Set Abstraction Layers)
        # sa1: вход 6 (xyz+rgb) + 3 (xyz_norm) = 9 каналов
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32,
            in_channel=9, mlp=[32, 32, 64]  # 6 (points) + 3 (xyz_norm)
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32,
            in_channel=67, mlp=[64, 64, 128]  # 64 (points) + 3 (xyz_norm)
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32,
            in_channel=131, mlp=[128, 128, 256]  # 128 (points) + 3 (xyz_norm)
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.8, nsample=32,
            in_channel=259, mlp=[256, 256, 512]  # 256 (points) + 3 (xyz_norm)
        )
        
        # Decoder (Feature Propagation Layers)
        self.fp4 = PointNetFeaturePropagation(in_channel=768, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=320, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        
        # Final classification layer
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def forward(self, xyz):
        """
        Args:
            xyz: облако точек [B, N, 6] где 6 = (x, y, z, r, g, b)
        Returns:
            logits: предсказания классов [B, N, num_classes]
        """
        B, N, C = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:, :, :3]
        
        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # Decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        # Classification
        feat = l0_points.transpose(1, 2)  # [B, C, N]
        feat = F.relu(self.bn1(self.conv1(feat)))
        feat = self.drop1(feat)
        logits = self.conv2(feat)
        logits = logits.transpose(1, 2)  # [B, N, num_classes]
        
        return logits

