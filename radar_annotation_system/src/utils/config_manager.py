"""
配置管理器
Configuration Manager
"""

import json
import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class RadarConfig:
    """雷达配置"""
    noise_threshold: float = 0.1
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5


@dataclass
class ImageConfig:
    """图像配置"""
    image_size: list = None
    normalize_mean: list = None
    normalize_std: list = None

    def __post_init__(self):
        if self.image_size is None:
            self.image_size = [640, 480]
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225]


@dataclass
class CameraConfig:
    """相机配置"""
    matrix: list = None
    extrinsic: list = None

    def __post_init__(self):
        if self.matrix is None:
            self.matrix = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
        if self.extrinsic is None:
            self.extrinsic = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]


@dataclass
class DetectionConfig:
    """检测配置"""
    yolo_enabled: bool = True
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence_threshold: float = 0.5
    sam_enabled: bool = False
    sam_model_path: str = "sam_vit_h_4b8939.pth"
    gpt4v_enabled: bool = False
    gpt4v_api_key: str = ""
    gpt4v_model: str = "gpt-4o-mini"


@dataclass
class FusionConfig:
    """融合配置"""
    distance_threshold: float = 2.0
    angular_threshold: float = 0.1
    confidence_weight: float = 0.7


@dataclass
class VisualizationConfig:
    """可视化配置"""
    window_size: list = None
    background_color: list = None
    point_size: int = 2

    def __post_init__(self):
        if self.window_size is None:
            self.window_size = [800, 600]
        if self.background_color is None:
            self.background_color = [0.1, 0.1, 0.1]


@dataclass
class SystemConfig:
    """系统配置"""
    radar: RadarConfig = None
    image: ImageConfig = None
    camera: CameraConfig = None
    detection: DetectionConfig = None
    fusion: FusionConfig = None
    visualization: VisualizationConfig = None

    def __post_init__(self):
        if self.radar is None:
            self.radar = RadarConfig()
        if self.image is None:
            self.image = ImageConfig()
        if self.camera is None:
            self.camera = CameraConfig()
        if self.detection is None:
            self.detection = DetectionConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = SystemConfig()

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            # 更新配置
            self._update_config_from_dict(data)
            print(f"配置文件加载成功: {config_path}")

        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置")

    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        保存配置文件

        Args:
            config_path: 配置文件路径，默认使用初始化时的路径
        """
        if config_path is None:
            config_path = self.config_path

        if config_path is None:
            raise ValueError("未指定配置文件路径")

        try:
            # 转换为字典
            config_dict = asdict(self.config)

            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

            print(f"配置文件保存成功: {config_path}")

        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def _update_config_from_dict(self, data: Dict[str, Any]) -> None:
        """
        从字典更新配置

        Args:
            data: 配置字典
        """
        if 'radar' in data:
            self._update_dataclass(self.config.radar, data['radar'])

        if 'image' in data:
            self._update_dataclass(self.config.image, data['image'])

        if 'camera' in data:
            self._update_dataclass(self.config.camera, data['camera'])

        if 'detection' in data:
            self._update_dataclass(self.config.detection, data['detection'])

        if 'fusion' in data:
            self._update_dataclass(self.config.fusion, data['fusion'])

        if 'visualization' in data:
            self._update_dataclass(self.config.visualization, data['visualization'])

    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """
        更新数据类对象

        Args:
            obj: 数据类对象
            data: 数据字典
        """
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    def get_config_dict(self) -> Dict[str, Any]:
        """
        获取配置字典

        Returns:
            配置字典
        """
        return asdict(self.config)

    def get_radar_config(self) -> RadarConfig:
        """获取雷达配置"""
        return self.config.radar

    def get_image_config(self) -> ImageConfig:
        """获取图像配置"""
        return self.config.image

    def get_camera_config(self) -> CameraConfig:
        """获取相机配置"""
        return self.config.camera

    def get_detection_config(self) -> DetectionConfig:
        """获取检测配置"""
        return self.config.detection

    def get_fusion_config(self) -> FusionConfig:
        """获取融合配置"""
        return self.config.fusion

    def get_visualization_config(self) -> VisualizationConfig:
        """获取可视化配置"""
        return self.config.visualization

    def update_radar_config(self, **kwargs) -> None:
        """
        更新雷达配置

        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config.radar, key):
                setattr(self.config.radar, key, value)

    def update_image_config(self, **kwargs) -> None:
        """
        更新图像配置

        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config.image, key):
                setattr(self.config.image, key, value)

    def update_camera_config(self, **kwargs) -> None:
        """
        更新相机配置

        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config.camera, key):
                setattr(self.config.camera, key, value)

    def update_detection_config(self, **kwargs) -> None:
        """
        更新检测配置

        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config.detection, key):
                setattr(self.config.detection, key, value)

    def update_fusion_config(self, **kwargs) -> None:
        """
        更新融合配置

        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config.fusion, key):
                setattr(self.config.fusion, key, value)

    def update_visualization_config(self, **kwargs) -> None:
        """
        更新可视化配置

        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config.visualization, key):
                setattr(self.config.visualization, key, value)

    def validate_config(self) -> bool:
        """
        验证配置的有效性

        Returns:
            配置是否有效
        """
        try:
            # 验证雷达配置
            if self.config.radar.noise_threshold < 0:
                print("警告: 雷达噪声阈值应为正数")
                return False

            if self.config.radar.dbscan_eps <= 0:
                print("警告: DBSCAN eps参数应为正数")
                return False

            if self.config.radar.dbscan_min_samples < 1:
                print("警告: DBSCAN min_samples参数应大于0")
                return False

            # 验证图像配置
            if len(self.config.image.image_size) != 2:
                print("警告: 图像尺寸应为长度为2的列表")
                return False

            if len(self.config.image.normalize_mean) != 3:
                print("警告: 归一化均值应为长度为3的列表")
                return False

            if len(self.config.image.normalize_std) != 3:
                print("警告: 归一化标准差应为长度为3的列表")
                return False

            # 验证相机配置
            if len(self.config.camera.matrix) != 3 or any(len(row) != 3 for row in self.config.camera.matrix):
                print("警告: 相机内参矩阵应为3x3矩阵")
                return False

            # 验证检测配置
            if self.config.detection.yolo_confidence_threshold < 0 or self.config.detection.yolo_confidence_threshold > 1:
                print("警告: YOLO置信度阈值应在0-1之间")
                return False

            # 验证融合配置
            if self.config.fusion.distance_threshold <= 0:
                print("警告: 融合距离阈值应为正数")
                return False

            if self.config.fusion.confidence_weight < 0 or self.config.fusion.confidence_weight > 1:
                print("警告: 融合置信度权重应在0-1之间")
                return False

            print("配置验证通过")
            return True

        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

    def create_default_config_file(self, config_path: str) -> None:
        """
        创建默认配置文件

        Args:
            config_path: 配置文件路径
        """
        default_config = SystemConfig()
        self.config = default_config
        self.save_config(config_path)
        print(f"默认配置文件已创建: {config_path}")

    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return json.dumps(asdict(self.config), indent=2, ensure_ascii=False)


def main():
    """测试函数"""
    # 创建配置管理器
    config_manager = ConfigManager()

    # 打印默认配置
    print("默认配置:")
    print(config_manager)

    # 验证配置
    print("\n配置验证结果:", config_manager.validate_config())

    # 更新部分配置
    config_manager.update_radar_config(noise_threshold=0.2, dbscan_eps=0.8)
    config_manager.update_detection_config(yolo_confidence_threshold=0.7)

    print("\n更新后的配置:")
    print(config_manager)

    # 保存配置文件
    try:
        config_manager.save_config("test_config.json")
        print("\n配置文件已保存: test_config.json")

        # 重新加载配置
        new_config_manager = ConfigManager("test_config.json")
        print("\n重新加载的配置:")
        print(new_config_manager)

        # 清理测试文件
        os.remove("test_config.json")
        print("\n测试文件已清理")

    except Exception as e:
        print(f"保存/加载配置文件时出错: {e}")


if __name__ == "__main__":
    main()