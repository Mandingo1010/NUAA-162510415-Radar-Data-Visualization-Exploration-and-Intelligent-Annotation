"""
雷达-图像融合标注模块
Radar-Image Fusion Annotation Module
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import json
from dataclasses import dataclass


@dataclass
class Detection3D:
    """3D检测结果"""
    center_3d: np.ndarray  # 3D中心点 [x, y, z]
    size_3d: np.ndarray    # 3D尺寸 [dx, dy, dz]
    confidence: float      # 置信度
    class_name: str        # 类别名称
    source: str           # 来源 ('radar', 'image', 'fusion')
    bbox_2d: Optional[List[int]] = None  # 对应的2D边界框 [x, y, w, h]


class RadarImageFusion:
    """雷达-图像融合标注器"""

    def __init__(self, config: Dict):
        """
        初始化融合标注器

        Args:
            config: 配置参数
        """
        self.config = config
        self.camera_matrix = np.array(config.get('camera_matrix'))
        self.extrinsic_matrix = np.array(config.get('extrinsic_matrix'))

        # 融合参数
        self.distance_threshold = config.get('distance_threshold', 2.0)  # 距离阈值(米)
        self.angular_threshold = config.get('angular_threshold', 0.1)   # 角度阈值(弧度)
        self.confidence_weight = config.get('confidence_weight', 0.7)   # 置信度权重

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        将3D点投影到2D图像平面

        Args:
            points_3d: 3D点 [N, 3]

        Returns:
            2D投影点 [N, 2]
        """
        if len(points_3d) == 0:
            return np.array([])

        # 转换为齐次坐标
        points_homo = np.column_stack([points_3d, np.ones(len(points_3d))])

        # 外参变换 (世界坐标系到相机坐标系)
        if self.extrinsic_matrix.shape == (3, 4):
            points_camera = self.extrinsic_matrix @ points_homo.T
        else:
            points_camera = self.extrinsic_matrix @ points_homo.T

        points_camera = points_camera[:3, :].T

        # 过滤相机后面的点
        valid_mask = points_camera[:, 2] > 0
        points_camera = points_camera[valid_mask]

        if len(points_camera) == 0:
            return np.array([])

        # 内参投影
        points_image_homo = self.camera_matrix @ points_camera.T
        points_2d = points_image_homo[:2, :].T / points_image_homo[2, :]

        return points_2d

    def backproject_2d_to_3d(self, bbox_2d: List[int], depth_range: Tuple[float, float] = (5.0, 50.0)) -> np.ndarray:
        """
        将2D边界框反投影到3D射线

        Args:
            bbox_2d: 2D边界框 [x, y, w, h]
            depth_range: 深度范围 (min_depth, max_depth)

        Returns:
            3D射线上的点
        """
        x, y, w, h = bbox_2d
        center_x = x + w // 2
        center_y = y + h // 2

        # 图像中心点
        image_point = np.array([center_x, center_y, 1.0])

        # 反投影到相机坐标系
        camera_point = np.linalg.inv(self.camera_matrix) @ image_point
        camera_ray = camera_point[:2] / camera_point[2]  # 归一化射线方向

        # 生成深度采样点
        depths = np.linspace(depth_range[0], depth_range[1], 20)
        points_3d_camera = []
        for depth in depths:
            point_3d = np.array([camera_ray[0] * depth, camera_ray[1] * depth, depth])
            points_3d_camera.append(point_3d)

        points_3d_camera = np.array(points_3d_camera)

        # 转换到世界坐标系
        if self.extrinsic_matrix.shape == (3, 4):
            extrinsic_inv = np.linalg.inv(np.vstack([self.extrinsic_matrix, [0, 0, 0, 1]]))
        else:
            extrinsic_inv = np.linalg.inv(self.extrinsic_matrix)

        points_homo = np.column_stack([points_3d_camera, np.ones(len(points_3d_camera))])
        points_world_homo = extrinsic_inv @ points_homo.T
        points_3d_world = points_world_homo[:3, :].T

        return points_3d_world

    def associate_detections(self, radar_points: np.ndarray,
                           image_detections: List[Dict]) -> List[Dict]:
        """
        关联雷达点和图像检测结果

        Args:
            radar_points: 雷达点云 [N, 3]
            image_detections: 图像检测结果

        Returns:
            关联结果列表
        """
        if len(radar_points) == 0 or not image_detections:
            return []

        # 将雷达点投影到图像平面
        radar_2d = self.project_3d_to_2d(radar_points)

        associations = []
        for detection in image_detections:
            bbox_2d = detection['bbox']
            center_2d = detection['center']

            # 查找投影点在边界框内的雷达点
            in_box_mask = []
            for point_2d in radar_2d:
                if (bbox_2d[0] <= point_2d[0] <= bbox_2d[0] + bbox_2d[2] and
                    bbox_2d[1] <= point_2d[1] <= bbox_2d[1] + bbox_2d[3]):
                    in_box_mask.append(True)
                else:
                    in_box_mask.append(False)

            in_box_mask = np.array(in_box_mask)
            associated_radar_points = radar_points[in_box_mask]

            if len(associated_radar_points) > 0:
                # 计算关联统计信息
                center_3d = np.mean(associated_radar_points, axis=0)
                distances = np.linalg.norm(associated_radar_points - center_3d, axis=1)

                association = {
                    'image_detection': detection,
                    'radar_points': associated_radar_points,
                    'center_3d': center_3d,
                    'num_points': len(associated_radar_points),
                    'mean_distance': np.mean(distances),
                    'confidence': detection['confidence'] * self.confidence_weight
                }
                associations.append(association)

        return associations

    def create_3d_bbox_from_points(self, points: np.ndarray) -> Dict:
        """
        从点云创建3D边界框

        Args:
            points: 3D点云 [N, 3]

        Returns:
            3D边界框信息
        """
        if len(points) == 0:
            return {}

        # 计算边界框
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords

        return {
            'center': center.tolist(),
            'size': size.tolist(),
            'min_coords': min_coords.tolist(),
            'max_coords': max_coords.tolist(),
            'num_points': len(points)
        }

    def generate_fused_detections(self, radar_points: np.ndarray,
                                 image_detections: List[Dict]) -> List[Detection3D]:
        """
        生成融合检测结果

        Args:
            radar_points: 雷达点云
            image_detections: 图像检测结果

        Returns:
            融合检测结果列表
        """
        # 关联检测
        associations = self.associate_detections(radar_points, image_detections)

        fused_detections = []
        for association in associations:
            # 创建3D边界框
            bbox_3d = self.create_3d_bbox_from_points(association['radar_points'])

            if bbox_3d:
                detection_3d = Detection3D(
                    center_3d=np.array(bbox_3d['center']),
                    size_3d=np.array(bbox_3d['size']),
                    confidence=association['confidence'],
                    class_name=association['image_detection']['class_name'],
                    source='fusion',
                    bbox_2d=association['image_detection']['bbox']
                )
                fused_detections.append(detection_3d)

        # 添加未关联的雷达目标
        unassociated_points = self._find_unassociated_points(radar_points, associations)
        if len(unassociated_points) > 0:
            # 聚类未关联的点
            clusters = self._cluster_points(unassociated_points)
            for cluster_points in clusters:
                bbox_3d = self.create_3d_bbox_from_points(cluster_points)
                if bbox_3d:
                    detection_3d = Detection3D(
                        center_3d=np.array(bbox_3d['center']),
                        size_3d=np.array(bbox_3d['size']),
                        confidence=0.5,  # 默认置信度
                        class_name='unknown_object',
                        source='radar'
                    )
                    fused_detections.append(detection_3d)

        return fused_detections

    def _find_unassociated_points(self, radar_points: np.ndarray,
                                 associations: List[Dict]) -> np.ndarray:
        """
        查找未关联的雷达点

        Args:
            radar_points: 所有雷达点
            associations: 关联结果

        Returns:
            未关联的雷达点
        """
        if not associations:
            return radar_points

        # 收集所有已关联的点
        associated_indices = set()
        for association in associations:
            for i, point in enumerate(radar_points):
                for associated_point in association['radar_points']:
                    if np.allclose(point, associated_point):
                        associated_indices.add(i)
                        break

        # 返回未关联的点
        unassociated_mask = np.ones(len(radar_points), dtype=bool)
        for idx in associated_indices:
            unassociated_mask[idx] = False

        return radar_points[unassociated_mask]

    def _cluster_points(self, points: np.ndarray, eps: float = 1.0,
                        min_samples: int = 3) -> List[np.ndarray]:
        """
        聚类点云

        Args:
            points: 输入点云
            eps: DBSCAN eps参数
            min_samples: DBSCAN min_samples参数

        Returns:
            聚类结果列表
        """
        if len(points) < min_samples:
            return [points] if len(points) > 0 else []

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(points)

        clusters = []
        for label in set(labels):
            if label != -1:  # 忽略噪声点
                cluster_points = points[labels == label]
                clusters.append(cluster_points)

        return clusters

    def refine_annotations(self, detections: List[Detection3D],
                          user_corrections: Optional[List[Dict]] = None) -> List[Detection3D]:
        """
        根据用户校正优化标注结果

        Args:
            detections: 原始检测结果
            user_corrections: 用户校正列表

        Returns:
            优化后的检测结果
        """
        if not user_corrections:
            return detections

        refined_detections = []
        for detection in detections:
            # 查找对应的用户校正
            correction = None
            for corr in user_corrections:
                if corr.get('detection_id') == id(detection):
                    correction = corr
                    break

            if correction:
                # 应用用户校正
                refined_detection = Detection3D(
                    center_3d=np.array(correction.get('center_3d', detection.center_3d)),
                    size_3d=np.array(correction.get('size_3d', detection.size_3d)),
                    confidence=correction.get('confidence', detection.confidence),
                    class_name=correction.get('class_name', detection.class_name),
                    source='manual_corrected',
                    bbox_2d=correction.get('bbox_2d', detection.bbox_2d)
                )
                refined_detections.append(refined_detection)
            else:
                refined_detections.append(detection)

        return refined_detections

    def export_annotations(self, detections: List[Detection3D],
                          output_path: str) -> None:
        """
        导出标注结果

        Args:
            detections: 检测结果列表
            output_path: 输出文件路径
        """
        annotations = []
        for i, detection in enumerate(detections):
            annotation = {
                'id': i,
                'center_3d': detection.center_3d.tolist(),
                'size_3d': detection.size_3d.tolist(),
                'confidence': detection.confidence,
                'class_name': detection.class_name,
                'source': detection.source,
                'bbox_2d': detection.bbox_2d
            }
            annotations.append(annotation)

        output_data = {
            'annotations': annotations,
            'camera_matrix': self.camera_matrix.tolist(),
            'extrinsic_matrix': self.extrinsic_matrix.tolist(),
            'config': self.config
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Annotations exported to {output_path}")

    def load_annotations(self, annotation_path: str) -> List[Detection3D]:
        """
        加载标注文件

        Args:
            annotation_path: 标注文件路径

        Returns:
            检测结果列表
        """
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)

            detections = []
            for ann in data['annotations']:
                detection = Detection3D(
                    center_3d=np.array(ann['center_3d']),
                    size_3d=np.array(ann['size_3d']),
                    confidence=ann['confidence'],
                    class_name=ann['class_name'],
                    source=ann['source'],
                    bbox_2d=ann.get('bbox_2d')
                )
                detections.append(detection)

            return detections

        except Exception as e:
            print(f"Error loading annotations: {e}")
            return []


def main():
    """测试函数"""
    # 配置参数
    config = {
        'camera_matrix': [
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ],
        'extrinsic_matrix': [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ],
        'distance_threshold': 2.0,
        'angular_threshold': 0.1,
        'confidence_weight': 0.7
    }

    # 创建融合标注器
    fusion = RadarImageFusion(config)

    # 生成测试数据
    # 1. 雷达点云
    np.random.seed(42)
    radar_points = np.random.randn(50, 3) * 5
    radar_points[:, 2] = np.abs(radar_points[:, 2]) + 5  # 确保在相机前方

    # 2. 图像检测结果
    image_detections = [
        {
            'bbox': [100, 100, 80, 120],
            'center': [140, 160],
            'confidence': 0.8,
            'class_name': 'car'
        },
        {
            'bbox': [300, 200, 60, 100],
            'center': [330, 250],
            'confidence': 0.7,
            'class_name': 'person'
        }
    ]

    print("Testing radar-image fusion...")

    # 生成融合检测结果
    fused_detections = fusion.generate_fused_detections(radar_points, image_detections)
    print(f"Generated {len(fused_detections)} fused detections")

    # 导出标注
    fusion.export_annotations(fused_detections, "test_annotations.json")

    # 加载标注
    loaded_detections = fusion.load_annotations("test_annotations.json")
    print(f"Loaded {len(loaded_detections)} detections from file")

    print("Radar-image fusion tests completed!")


if __name__ == "__main__":
    main()