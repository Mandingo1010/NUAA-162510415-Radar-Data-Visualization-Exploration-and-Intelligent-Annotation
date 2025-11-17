"""
点云可视化模块
Point Cloud Visualization Module
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Optional, Dict
import cv2


class PointCloudVisualizer:
    """点云可视化器"""

    def __init__(self, config: Dict):
        """
        初始化可视化器

        Args:
            config: 配置参数字典
        """
        self.config = config
        self.window_size = config.get('window_size', (800, 600))
        self.background_color = config.get('background_color', [0.1, 0.1, 0.1])
        self.point_size = config.get('point_size', 2.0)

        # Open3D可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('Radar Point Cloud', self.window_size[0], self.window_size[1])

    def create_point_cloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        创建Open3D点云对象

        Args:
            points: 点云坐标 [N, 3]
            colors: 点云颜色 [N, 3] 或 [N] (强度值)

        Returns:
            Open3D点云对象
        """
        if len(points) == 0:
            return o3d.geometry.PointCloud()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 设置颜色
        if colors is not None:
            if colors.ndim == 1:
                # 强度值转换为颜色
                colors_norm = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
                colors_rgb = plt.cm.jet(colors_norm)[:, :3]
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            else:
                pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def visualize_single_frame(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                              show_axes: bool = True, show_grid: bool = True) -> None:
        """
        可视化单帧点云

        Args:
            points: 点云坐标
            colors: 点云颜色
            show_axes: 是否显示坐标轴
            show_grid: 是否显示网格
        """
        # 清除之前的内容
        self.vis.clear_geometries()

        # 创建点云
        pcd = self.create_point_cloud(points, colors)

        # 添加到可视化器
        self.vis.add_geometry(pcd)

        # 设置渲染选项
        render_opt = self.vis.get_render_option()
        render_opt.background_color = self.background_color
        render_opt.point_size = self.point_size
        render_opt.show_coordinate_frame = show_axes

        # 设置视图控制
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)

        # 运行可视化
        self.vis.run()

    def create_animation_sequence(self, point_sequence: List[np.ndarray],
                                colors_sequence: Optional[List[np.ndarray]] = None,
                                output_path: str = "animation.mp4") -> None:
        """
        创建点云动画序列

        Args:
            point_sequence: 点云序列
            colors_sequence: 颜色序列
            output_path: 输出视频路径
        """
        if not point_sequence:
            print("Empty point sequence")
            return

        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = self.window_size[1], self.window_size[0]
        out = cv2.VideoWriter(output_path, fourcc, 10.0, (w, h))

        print(f"Creating animation with {len(point_sequence)} frames...")

        for i, points in enumerate(point_sequence):
            # 清除并重新渲染
            self.vis.clear_geometries()

            # 创建点云
            colors = colors_sequence[i] if colors_sequence else None
            pcd = self.create_point_cloud(points, colors)
            self.vis.add_geometry(pcd)

            # 渲染当前帧
            self.vis.poll_events()
            self.vis.update_renderer()

            # 捕获当前帧
            image = self.vis.capture_screen_image(do_render=True)
            image_array = np.asarray(image)

            # 转换颜色空间 (RGB to BGR for OpenCV)
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_array

            # 调整尺寸
            image_resized = cv2.resize(image_bgr, (w, h))
            out.write(image_resized)

            if i % 10 == 0:
                print(f"Processed frame {i}/{len(point_sequence)}")

        out.release()
        print(f"Animation saved to {output_path}")

    def plot_with_matplotlib(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                           title: str = "Point Cloud", save_path: Optional[str] = None) -> None:
        """
        使用Matplotlib绘制点云

        Args:
            points: 点云坐标
            colors: 点云颜色
            title: 图表标题
            save_path: 保存路径
        """
        if len(points) == 0:
            print("Empty point cloud")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点云
        if colors is not None:
            if colors.ndim == 1:
                scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                   c=colors, cmap='jet', s=1, alpha=0.6)
                plt.colorbar(scatter, ax=ax, label='Intensity')
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=colors, s=1, alpha=0.6)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.6)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)

        # 设置相等的坐标轴比例
        max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                            points[:, 1].max()-points[:, 1].min(),
                            points[:, 2].max()-points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_with_plotly(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                        title: str = "Interactive Point Cloud") -> None:
        """
        使用Plotly创建交互式点云

        Args:
            points: 点云坐标
            colors: 点云颜色
            title: 图表标题
        """
        if len(points) == 0:
            print("Empty point cloud")
            return

        if colors is not None:
            if colors.ndim == 1:
                fig = go.Figure(data=[go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=colors,
                        colorscale='Jet',
                        showscale=True,
                        colorbar=dict(title="Intensity")
                    ),
                    text=[f'Point {i}' for i in range(len(points))],
                    hovertemplate='X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>'
                )])
            else:
                fig = go.Figure(data=[go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=[f'rgb({c[0]*255:.0f}, {c[1]*255:.0f}, {c[2]*255:.0f})' for c in colors]
                    )
                )])
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(size=2, color='blue')
            )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )

        return fig

    def add_bounding_boxes(self, bounding_boxes: List[Dict]) -> List[o3d.geometry.LineSet]:
        """
        添加3D边界框

        Args:
            bounding_boxes: 边界框列表，每个包含center, size, color

        Returns:
            Open3D线框对象列表
        """
        line_sets = []

        for bbox in bounding_boxes:
            center = bbox['center']  # [x, y, z]
            size = bbox['size']      # [dx, dy, dz]
            color = bbox.get('color', [1, 0, 0])  # 默认红色

            # 计算边界框的8个顶点
            x_min, x_max = center[0] - size[0]/2, center[0] + size[0]/2
            y_min, y_max = center[1] - size[1]/2, center[1] + size[1]/2
            z_min, z_max = center[2] - size[2]/2, center[2] + size[2]/2

            vertices = [
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max]
            ]

            # 定义边
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
            ]

            # 创建线框
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

            line_sets.append(line_set)

        return line_sets

    def close(self):
        """关闭可视化器"""
        self.vis.destroy_window()


def main():
    """测试函数"""
    # 配置参数
    config = {
        'window_size': (800, 600),
        'background_color': [0.1, 0.1, 0.1],
        'point_size': 2.0
    }

    # 创建可视化器
    visualizer = PointCloudVisualizer(config)

    # 生成测试点云
    np.random.seed(42)
    num_points = 1000
    points = np.random.randn(num_points, 3) * 10
    intensities = np.random.rand(num_points)

    print("Testing visualization methods...")

    # Matplotlib可视化
    print("1. Matplotlib visualization")
    visualizer.plot_with_matplotlib(points, intensities, "Test Point Cloud")

    # Plotly可视化
    print("2. Plotly visualization")
    visualizer.plot_with_plotly(points, intensities, "Interactive Test Point Cloud")

    # Open3D可视化 (阻塞式)
    print("3. Open3D visualization (close window to continue)")
    visualizer.visualize_single_frame(points, intensities)

    # 清理
    visualizer.close()
    print("Visualization tests completed!")


if __name__ == "__main__":
    main()