"""
图像数据处理模块
Image Data Processing Module
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List, Optional, Dict
import json
import os


class ImageProcessor:
    """图像数据处理器"""

    def __init__(self, config: Dict):
        """
        初始化图像处理器

        Args:
            config: 配置参数字典
        """
        self.config = config
        self.image_size = config.get('image_size', (640, 480))
        self.normalize_mean = config.get('normalize_mean', [0.485, 0.456, 0.406])
        self.normalize_std = config.get('normalize_std', [0.229, 0.224, 0.225])

        # 图像预处理变换
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])

    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像

        Args:
            image_path: 图像文件路径

        Returns:
            RGB图像数组
        """
        try:
            # 使用PIL加载图像并转换为RGB
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            return image_array

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return np.array([])

    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        调整图像尺寸

        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)

        Returns:
            调整尺寸后的图像
        """
        if len(image) == 0:
            return image

        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return resized

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像归一化

        Args:
            image: 输入图像

        Returns:
            归一化后的图像
        """
        if len(image) == 0:
            return image

        # 转换为浮点数并归一化到[0,1]
        normalized = image.astype(np.float32) / 255.0
        return normalized

    def extract_features(self, image: np.ndarray) -> Dict:
        """
        提取图像基本特征

        Args:
            image: 输入图像

        Returns:
            特征字典
        """
        if len(image) == 0:
            return {}

        features = {
            'shape': image.shape,
            'mean_intensity': np.mean(image),
            'std_intensity': np.std(image),
            'min_value': np.min(image),
            'max_value': np.max(image)
        }

        # 计算梯度特征
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Sobel边缘检测
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features.update({
            'edge_strength': np.mean(gradient_magnitude),
            'edge_density': np.sum(gradient_magnitude > 50) / gradient_magnitude.size
        })

        return features

    def undistort_image(self, image: np.ndarray, camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray) -> np.ndarray:
        """
        相机畸变校正

        Args:
            image: 输入图像
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数

        Returns:
            校正后的图像
        """
        if len(image) == 0:
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # 裁剪黑边
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        return undistorted

    def prepare_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        为深度学习模型准备图像数据

        Args:
            image: 输入图像

        Returns:
            预处理后的图像张量
        """
        if len(image) == 0:
            return np.array([])

        # 转换为PIL图像
        pil_image = Image.fromarray(image.astype(np.uint8))

        # 应用变换
        tensor = self.transform(pil_image)

        return tensor.numpy()

    def load_calibration_data(self, calibration_path: str) -> Dict:
        """
        加载相机标定数据

        Args:
            calibration_path: 标定文件路径

        Returns:
            标定参数字典
        """
        try:
            with open(calibration_path, 'r') as f:
                calibration_data = json.load(f)

            # 转换为numpy数组
            if 'camera_matrix' in calibration_data:
                calibration_data['camera_matrix'] = np.array(calibration_data['camera_matrix'])
            if 'dist_coeffs' in calibration_data:
                calibration_data['dist_coeffs'] = np.array(calibration_data['dist_coeffs'])
            if 'radar_to_camera' in calibration_data:
                calibration_data['radar_to_camera'] = np.array(calibration_data['radar_to_camera'])

            return calibration_data

        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return {}


def main():
    """测试函数"""
    # 配置参数
    config = {
        'image_size': (640, 480),
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225]
    }

    # 创建图像处理器
    processor = ImageProcessor(config)

    # 生成测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 处理图像
    resized_image = processor.resize_image(test_image, (320, 240))
    normalized_image = processor.normalize_image(test_image)
    features = processor.extract_features(test_image)

    print(f"原始图像尺寸: {test_image.shape}")
    print(f"调整后尺寸: {resized_image.shape}")
    print(f"归一化后范围: [{normalized_image.min():.3f}, {normalized_image.max():.3f}]")
    print(f"图像特征: {features}")


if __name__ == "__main__":
    main()