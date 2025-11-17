"""
雷达数据处理模块
Radar Data Processing Module
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import DBSCAN
import scipy.signal as signal
from dataclasses import dataclass
import json


@dataclass
class RadarPoint:
    """雷达点数据结构"""
    x: float
    y: float
    z: float
    intensity: float
    velocity: float
    timestamp: float


class RadarProcessor:
    """雷达数据处理器"""

    def __init__(self, config: Dict):
        """
        初始化雷达处理器

        Args:
            config: 配置参数字典
        """
        self.config = config
        self.noise_threshold = config.get('noise_threshold', 0.1)
        self.dbscan_eps = config.get('dbscan_eps', 0.5)
        self.dbscan_min_samples = config.get('dbscan_min_samples', 5)

    def load_radar_data(self, file_path: str) -> np.ndarray:
        """
        加载雷达原始数据

        Args:
            file_path: 数据文件路径

        Returns:
            原始雷达数据数组 [range, azimuth, doppler, intensity]
        """
        try:
            # 根据文件格式选择不同的加载方式
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path).values
            elif file_path.endswith('.npy'):
                data = np.load(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                data = np.array(json_data['radar_data'])
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            return data

        except Exception as e:
            print(f"Error loading radar data: {e}")
            return np.array([])

    def polar_to_cartesian(self, polar_data: np.ndarray) -> np.ndarray:
        """
        极坐标转换为笛卡尔坐标

        Args:
            polar_data: 极坐标数据 [range, azimuth, doppler, intensity]

        Returns:
            笛卡尔坐标数据 [x, y, z, intensity, velocity]
        """
        if len(polar_data) == 0:
            return np.array([])

        # 提取极坐标参数
        ranges = polar_data[:, 0]
        azimuths = polar_data[:, 1]
        dopplers = polar_data[:, 2]
        intensities = polar_data[:, 3]

        # 极坐标转笛卡尔坐标 (假设高度为0)
        x = ranges * np.cos(azimuths)
        y = ranges * np.sin(azimuths)
        z = np.zeros_like(x)  # 毫米波雷达通常只有平面信息

        # 组合结果
        cartesian_data = np.column_stack([x, y, z, intensities, dopplers])

        return cartesian_data

    def noise_filter(self, points: np.ndarray) -> np.ndarray:
        """
        噪声过滤

        Args:
            points: 点云数据 [x, y, z, intensity, velocity]

        Returns:
            过滤后的点云数据
        """
        if len(points) == 0:
            return points

        # 1. 强度阈值过滤
        intensity_mask = points[:, 3] > self.noise_threshold

        # 2. 距离阈值过滤 (去除过近的噪声点)
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        distance_mask = distances > 0.5  # 最小距离0.5米

        # 3. 速度阈值过滤 (去除静态杂波)
        velocity_mask = np.abs(points[:, 4]) > 0.1  # 最小速度0.1m/s

        # 应用所有过滤器
        filtered_points = points[intensity_mask & distance_mask & velocity_mask]

        return filtered_points

    def cluster_denoising(self, points: np.ndarray) -> np.ndarray:
        """
        基于聚类的去噪

        Args:
            points: 点云数据

        Returns:
            去噪后的点云数据
        """
        if len(points) < self.dbscan_min_samples:
            return points

        # 只使用空间坐标进行聚类
        spatial_coords = points[:, :3]

        # DBSCAN聚类
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = clustering.fit_predict(spatial_coords)

        # 保留非噪声点 (label != -1)
        valid_mask = labels != -1
        clustered_points = points[valid_mask]

        return clustered_points

    def process_frame(self, raw_data: np.ndarray) -> List[RadarPoint]:
        """
        处理单帧雷达数据

        Args:
            raw_data: 原始雷达数据

        Returns:
            处理后的雷达点列表
        """
        # 极坐标转笛卡尔坐标
        cartesian_points = self.polar_to_cartesian(raw_data)

        # 噪声过滤
        filtered_points = self.noise_filter(cartesian_points)

        # 聚类去噪
        denoised_points = self.cluster_denoising(filtered_points)

        # 转换为RadarPoint对象列表
        radar_points = []
        for point in denoised_points:
            radar_point = RadarPoint(
                x=point[0],
                y=point[1],
                z=point[2],
                intensity=point[3],
                velocity=point[4],
                timestamp=0.0  # 需要从原始数据中获取
            )
            radar_points.append(radar_point)

        return radar_points

    def create_point_cloud(self, radar_points: List[RadarPoint]) -> np.ndarray:
        """
        创建点云数组

        Args:
            radar_points: 雷达点列表

        Returns:
            点云数组 [x, y, z, intensity]
        """
        if not radar_points:
            return np.array([])

        points_array = np.array([
            [point.x, point.y, point.z, point.intensity]
            for point in radar_points
        ])

        return points_array


def main():
    """测试函数"""
    # 配置参数
    config = {
        'noise_threshold': 0.1,
        'dbscan_eps': 0.5,
        'dbscan_min_samples': 5
    }

    # 创建处理器
    processor = RadarProcessor(config)

    # 生成测试数据
    test_data = np.random.rand(100, 4)  # 100个随机点 [range, azimuth, doppler, intensity]
    test_data[:, 0] *= 50  # range: 0-50m
    test_data[:, 1] *= 2 * np.pi  # azimuth: 0-2π
    test_data[:, 2] *= 10  # doppler: -5 to 5 m/s
    test_data[:, 3] = test_data[:, 3] * 0.8 + 0.2  # intensity: 0.2-1.0

    # 处理数据
    radar_points = processor.process_frame(test_data)
    point_cloud = processor.create_point_cloud(radar_points)

    print(f"原始数据点数: {len(test_data)}")
    print(f"处理后点云点数: {len(point_cloud)}")
    print(f"点云形状: {point_cloud.shape}")


if __name__ == "__main__":
    main()