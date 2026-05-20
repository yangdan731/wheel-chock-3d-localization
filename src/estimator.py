"""
止轮器端到端位姿估计器
从 RGB-D 图像输入到 6DoF 位姿输出的完整流水线：
  RGB → YOLOv8-seg 分割 → 掩码+深度反投影 → 点云滤波 → PCA 位姿估计
"""

import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import open3d as o3d
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ==================== 数据结构 ====================

@dataclass
class PoseResult:
    """单帧位姿估计结果"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0    # 度, ZYX 顺序
    pitch: float = 0.0
    yaw: float = 0.0
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.uint8), repr=False)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), repr=False)
    raw_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), repr=False)
    axes: np.ndarray = field(default_factory=lambda: np.eye(3), repr=False)
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)
    success: bool = False


# ==================== 核心类 ====================

class WheelChockPoseEstimator:
    """止轮器端到端位姿估计器"""

    def __init__(
        self,
        model_path: str = "models/best_aug.pt",
        fx: float = 913.0030517578125,
        fy: float = 911.9224243164062,
        cx: float = 635.4708251953125,
        cy: float = 377.90032958984375,
        depth_scale: float = 0.001,
        filter_nb_neighbors: int = 20,
        filter_std_ratio: float = 2.0,
        voxel_size: float = 0.005,
        radius_filter: float = 0.02,
        min_neighbors: int = 10,
    ):
        self.model = YOLO(model_path)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        self.filter_nb_neighbors = filter_nb_neighbors
        self.filter_std_ratio = filter_std_ratio
        self.voxel_size = voxel_size
        self.radius_filter = radius_filter
        self.min_neighbors = min_neighbors

        # 帧间一致性状态
        self._prev_axes: np.ndarray | None = None

    # ==================== 步骤1: 分割 ====================

    def segment(self, rgb: np.ndarray) -> np.ndarray | None:
        """YOLOv8-seg 推理，返回二值掩码"""
        results = self.model(rgb, verbose=False)
        if results[0].masks is None or len(results[0].masks.data) == 0:
            return None
        mask = results[0].masks.data[0].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)
        h, w = rgb.shape[:2]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8)
        return mask

    # ==================== 步骤2: 反投影 ====================

    def back_project(self, depth_raw: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """掩码区域 + 深度图 → 3D点云"""
        if depth_raw.ndim == 3:
            depth_raw = depth_raw.squeeze()
        ys, xs = np.where(mask == 1)
        if len(xs) == 0:
            return None
        depths_mm = depth_raw[ys, xs]
        valid = depths_mm > 0
        xs, ys, depths_mm = xs[valid], ys[valid], depths_mm[valid]
        if len(xs) == 0:
            return None
        z = depths_mm * self.depth_scale
        x = (xs - self.cx) * z / self.fx
        y = (ys - self.cy) * z / self.fy
        return np.stack((x, y, z), axis=-1)

    # ==================== 步骤3: 点云滤波 ====================

    def filter_pointcloud(self, points: np.ndarray) -> np.ndarray | None:
        """组合滤波: 统计滤波 → 半径滤波 → 体素降采样"""
        if len(points) < 10:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 统计滤波
        cl, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.filter_nb_neighbors,
            std_ratio=self.filter_std_ratio,
        )

        # 半径滤波
        cl2, _ = cl.remove_radius_outlier(
            nb_points=self.min_neighbors,
            radius=self.radius_filter,
        )

        # 体素降采样
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(np.asarray(cl2.points))
        pcd_down = pcd_filtered.voxel_down_sample(self.voxel_size)
        points_down = np.asarray(pcd_down.points)

        if len(points_down) < 10:
            return np.asarray(cl2.points)
        return points_down

    # ==================== 步骤4: PCA 姿态估计 ====================

    def estimate_pose(self, points: np.ndarray, prev_axes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """PCA 计算质心 + 主轴方向，返回 (centroid, axes, roll, pitch, yaw) 角度制"""
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        axes = Vt.T

        if np.linalg.det(axes) < 0:
            axes[:, 2] = -axes[:, 2]

        # 帧间一致性：与上一帧保持符号一致
        if prev_axes is not None:
            for i in range(3):
                if np.dot(axes[:, i], prev_axes[:, i]) < 0:
                    axes[:, i] = -axes[:, i]
            if np.linalg.det(axes) < 0:
                axes[:, 2] = -axes[:, 2]

        r = R.from_matrix(axes)
        yaw, pitch, roll = r.as_euler('zyx')
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)

        return centroid, axes, roll_deg, pitch_deg, yaw_deg

    # ==================== 端到端单帧 ====================

    def process(self, rgb: np.ndarray, depth: np.ndarray) -> PoseResult:
        """单帧端到端处理: RGB+深度 → PoseResult"""
        mask = self.segment(rgb)
        if mask is None:
            return PoseResult()

        points = self.back_project(depth, mask)
        if points is None:
            return PoseResult()

        points_filtered = self.filter_pointcloud(points)
        if points_filtered is None:
            return PoseResult()

        centroid, axes, roll, pitch, yaw = self.estimate_pose(points_filtered, self._prev_axes)
        self._prev_axes = axes

        return PoseResult(
            x=centroid[0], y=centroid[1], z=centroid[2],
            roll=roll, pitch=pitch, yaw=yaw,
            mask=mask, points=points_filtered, raw_points=points,
            axes=axes, centroid=centroid,
            success=True,
        )

    def reset_consistency(self) -> None:
        """重置帧间一致性状态（开始新序列时调用）"""
        self._prev_axes = None

    # ==================== 批量处理 ====================

    def process_batch(
        self,
        rgb_dir: str,
        depth_dir: str,
        timestamps_csv: str | None = None,
        progress_callback=None,
    ) -> pd.DataFrame:
        """批量处理目录下所有帧，返回含时间戳的位姿 DataFrame"""
        self.reset_consistency()

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        ts_dict = {}
        if timestamps_csv and os.path.exists(timestamps_csv):
            df_ts = pd.read_csv(timestamps_csv)
            ts_dict = dict(zip(df_ts['frame_idx'], df_ts['rel_sec']))

        results = []
        for idx, fname in enumerate(rgb_files):
            rgb_path = os.path.join(rgb_dir, fname)
            depth_path = os.path.join(depth_dir, fname)

            if not os.path.exists(depth_path):
                continue

            # 提取帧编号
            frame_num = idx
            match = re.search(r'(\d+)', fname)
            if match:
                frame_num = int(match.group(1))
            rel_sec = ts_dict.get(frame_num, np.nan)

            rgb = cv2.imread(rgb_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                results.append([frame_num, rel_sec, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                self._prev_axes = None
                continue

            result = self.process(rgb, depth)

            if result.success:
                results.append([frame_num, rel_sec, result.x, result.y, result.z,
                                result.roll, result.pitch, result.yaw])
            else:
                results.append([frame_num, rel_sec, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                self._prev_axes = None

            if progress_callback:
                progress_callback(idx, len(rgb_files), fname)

        return pd.DataFrame(results, columns=['frame_idx', 'rel_sec', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])

    # ==================== 可视化 ====================

    def visualize(self, rgb: np.ndarray, depth_raw: np.ndarray, result: PoseResult,
                  output_dir: str = "visualization/pipeline", show_3d: bool = True) -> None:
        """生成可视化: 掩码、深度、滤波对比、点云、2D投影、综合面板 + Open3D 6DoF"""
        if not result.success:
            print("位姿估计未成功，跳过可视化")
            return

        os.makedirs(output_dir, exist_ok=True)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw.squeeze()

        mask = result.mask
        # 保存二值掩码
        cv2.imwrite(os.path.join(output_dir, "1_binary_mask.png"), mask * 255)

        # 1) RGB + 掩码叠加
        color_mask = np.zeros_like(rgb)
        color_mask[:, :, 2] = (mask * 255).astype(np.uint8)
        overlay = cv2.addWeighted(rgb, 0.6, color_mask, 0.4, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "2_rgb_mask.png"), overlay)

        # 2) 深度伪彩 + 掩码轮廓
        depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.drawContours(depth_color, contours, -1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(output_dir, "3_depth_colored.png"), depth_color)

        # 3) 深度掩码映射前后对比
        depth_color_full = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_masked = (depth_raw * mask.astype(depth_raw.dtype)).astype(depth_raw.dtype)
        depth_norm_masked = cv2.normalize(depth_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color_masked = cv2.applyColorMap(depth_norm_masked, cv2.COLORMAP_JET)
        h, w = depth_color_full.shape[:2]
        combined_depth = np.hstack((depth_color_full, depth_color_masked))
        cv2.putText(combined_depth, "Before Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_depth, "After Mask", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(output_dir, "4_depth_comparison.png"), combined_depth)

        # 4) 点云 + 质心 + 主轴方向 (matplotlib)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        points = result.points
        centroid = result.centroid
        axes = result.axes
        sample_step = max(1, len(points) // 3000)
        points_sample = points[::sample_step]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2],
                   c='gray', s=1, alpha=0.5)
        ax.scatter(centroid[0], centroid[1], centroid[2],
                   c='red', s=50, marker='o', label='Centroid')
        axis_len = 0.1
        for i, color in enumerate(['r', 'g', 'b']):
            vec = axes[:, i] * axis_len
            ax.quiver(centroid[0], centroid[1], centroid[2], vec[0], vec[1], vec[2],
                      color=color, arrow_length_ratio=0.1, linewidth=2, label=f'Axis {i + 1}')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.legend()
        ax.set_title(f'Pose: ({result.x:.3f}, {result.y:.3f}, {result.z:.3f})m, '
                     f'R={result.roll:.1f} P={result.pitch:.1f} Y={result.yaw:.1f} deg')
        plt.savefig(os.path.join(output_dir, "5_pose.png"), dpi=300)
        plt.close()

        # 6) 滤波前后点云对比
        self._save_filter_comparison(result, output_dir)

        # 7) 2D 位姿投影（将 3D 主轴投影到 RGB 图像上）
        self._save_projected_pose(rgb, result, output_dir)

        # 8) 按离质心距离着色的点云
        self._save_colored_pointcloud(result, output_dir)

        # 9) 综合面板：RGB+掩码 + 2D投影 + 3D点云
        self._save_summary_panel(rgb, result, output_dir)

        # Open3D 交互式 6DoF 可视化
        if show_3d:
            self._visualize_6dof_open3d(points, centroid, axes)

        print(f"可视化已保存至 {output_dir}/")

    def _project_to_2d(self, points_3d: np.ndarray):
        """将 3D 点投影到 2D 像素坐标 (u, v)"""
        z = points_3d[:, 2]
        safe = np.abs(z) > 1e-6
        u = np.full(len(points_3d), np.nan)
        v = np.full(len(points_3d), np.nan)
        u[safe] = self.fx * points_3d[safe, 0] / z[safe] + self.cx
        v[safe] = self.fy * points_3d[safe, 1] / z[safe] + self.cy
        return u, v

    def _save_filter_comparison(self, result: PoseResult, output_dir: str):
        """滤波前后点云对比图"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        raw = result.raw_points
        filtered = result.points
        max_pts = 3000
        step_raw = max(1, len(raw) // max_pts)
        step_filt = max(1, len(filtered) // max_pts)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        s1 = raw[::step_raw]
        ax1.scatter(s1[:, 0], s1[:, 1], s1[:, 2], c='gray', s=0.5, alpha=0.5)
        ax1.set_title(f'Before Filtering ({len(raw)} pts)')
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        s2 = filtered[::step_filt]
        ax2.scatter(s2[:, 0], s2[:, 1], s2[:, 2], c='blue', s=1, alpha=0.8)
        ax2.set_title(f'After Filtering ({len(filtered)} pts)')
        ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_zlabel('Z (m)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "6_filter_comparison.png"), dpi=150)
        plt.close()

    def _save_projected_pose(self, rgb: np.ndarray, result: PoseResult, output_dir: str):
        """将估计的 3D 位姿主轴投影到 2D RGB 图像上"""
        centroid = result.centroid
        axes = result.axes
        axis_len = 0.08  # 投影轴长度（米）

        # 轴端点（相机坐标系）
        endpoints = np.array([centroid + axis_len * axes[:, i] for i in range(3)])
        all_pts = np.vstack([centroid.reshape(1, 3), endpoints])

        u, v = self._project_to_2d(all_pts)
        cu, cv_ = u[0], v[0]  # 质心投影

        out = rgb.copy()
        colors_2d = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: R→X, G→Y, B→Z
        labels = ['X', 'Y', 'Z']

        h, w = rgb.shape[:2]
        # 质心圆点
        if 0 <= cu < w and 0 <= cv_ < h:
            cv2.circle(out, (int(cu), int(cv_)), 5, (0, 255, 255), -1)

        for i in range(3):
            eu, ev = u[i + 1], v[i + 1]
            if np.isnan(eu) or np.isnan(ev):
                continue
            # 裁剪到图像范围内
            eu_clip = np.clip(eu, 0, w - 1)
            ev_clip = np.clip(ev, 0, h - 1)
            if 0 <= cu < w and 0 <= cv_ < h:
                cv2.arrowedLine(out, (int(cu), int(cv_)), (int(eu_clip), int(ev_clip)),
                                colors_2d[i], 2, tipLength=0.15)
            # 标签偏移
            tx = int(eu_clip) + 10 if eu_clip < w - 20 else int(eu_clip) - 20
            ty = int(ev_clip) - 10 if ev_clip > 15 else int(ev_clip) + 20
            cv2.putText(out, labels[i], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors_2d[i], 2)

        # 左上角姿态数值
        cv2.putText(out, f'Pos: ({result.x:.3f}, {result.y:.3f}, {result.z:.3f})m', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out, f'R={result.roll:.1f} P={result.pitch:.1f} Y={result.yaw:.1f} deg', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(output_dir, "7_projected_pose.png"), out)

    def _save_colored_pointcloud(self, result: PoseResult, output_dir: str):
        """按点离质心距离着色的点云图"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        points = result.points
        centroid = result.centroid
        axes = result.axes
        dists = np.linalg.norm(points - centroid, axis=1)

        sample_step = max(1, len(points) // 3000)
        ps = points[::sample_step]
        ds = dists[::sample_step]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=ds, cmap='hot', s=1, alpha=0.8)
        ax.scatter(centroid[0], centroid[1], centroid[2], c='cyan', s=50, marker='o', label='Centroid')
        axis_len = 0.1
        for i, color in enumerate(['r', 'g', 'b']):
            vec = axes[:, i] * axis_len
            ax.quiver(centroid[0], centroid[1], centroid[2], vec[0], vec[1], vec[2],
                      color=color, arrow_length_ratio=0.1, linewidth=2, label=f'Axis {i + 1}')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.legend()
        cbar = plt.colorbar(sc, ax=ax, label='Dist to centroid (m)', shrink=0.5)
        ax.set_title('Point Cloud Colored by Distance to Centroid')
        plt.savefig(os.path.join(output_dir, "8_colored_pointcloud.png"), dpi=300)
        plt.close()

    def _save_summary_panel(self, rgb: np.ndarray, result: PoseResult, output_dir: str):
        """综合面板：RGB+掩码、2D投影、3D点云"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        mask = result.mask
        # 子图1: RGB+掩码（已有文件，直接读取）
        overlay_path = os.path.join(output_dir, "2_rgb_mask.png")
        proj_path = os.path.join(output_dir, "7_projected_pose.png")

        fig = plt.figure(figsize=(18, 6))

        # 左：RGB+掩码叠加
        ax1 = fig.add_subplot(1, 3, 1)
        if os.path.exists(overlay_path):
            img1 = cv2.cvtColor(cv2.imread(overlay_path), cv2.COLOR_BGR2RGB)
            ax1.imshow(img1)
        ax1.set_title('Segmentation')
        ax1.axis('off')

        # 中：2D 投影位姿
        ax2 = fig.add_subplot(1, 3, 2)
        if os.path.exists(proj_path):
            img2 = cv2.cvtColor(cv2.imread(proj_path), cv2.COLOR_BGR2RGB)
            ax2.imshow(img2)
        ax2.set_title('Projected Pose')
        ax2.axis('off')

        # 右：3D 点云 + 位姿
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        points = result.points
        centroid = result.centroid
        axes = result.axes
        sample_step = max(1, len(points) // 2000)
        ps = points[::sample_step]
        ax3.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c='gray', s=0.5, alpha=0.5)
        ax3.scatter(centroid[0], centroid[1], centroid[2], c='red', s=30, marker='o')
        axis_len = 0.1
        for i, color in enumerate(['r', 'g', 'b']):
            vec = axes[:, i] * axis_len
            ax3.quiver(centroid[0], centroid[1], centroid[2], vec[0], vec[1], vec[2],
                       color=color, arrow_length_ratio=0.1, linewidth=2)
        ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
        ax3.set_title(f'3D Pose\n({result.x:.3f}, {result.y:.3f}, {result.z:.3f})m')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "9_summary.png"), dpi=200)
        plt.close()

    def _visualize_6dof_open3d(self, points, centroid, axes, camera_origin=(0, 0, 0), axis_length=0.15):
        """Open3D 交互窗口：点云 + 坐标系箭头"""
        if len(points) > 50000:
            idx = np.random.choice(len(points), 50000, replace=False)
            points_vis = points[idx]
        else:
            points_vis = points

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_vis)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

        geometries = [pcd]

        # 相机原点坐标系
        camera_axes = np.eye(3)
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(3):
            geometries.append(self._create_arrow(camera_origin, camera_axes[:, i], colors[i], length=axis_length))

        # 目标位姿坐标系
        for i in range(3):
            geometries.append(self._create_arrow(centroid, axes[:, i], colors[i], length=axis_length, cylinder_radius=0.003))

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(centroid)
        sphere.paint_uniform_color([1, 0, 0])
        geometries.append(sphere)

        o3d.visualization.draw_geometries(geometries, window_name="6DoF Pose Estimation", width=1024, height=768)

    @staticmethod
    def _create_arrow(origin, direction, color, length=0.1, cylinder_radius=0.005, cone_radius=0.01):
        """创建 Open3D 箭头"""
        direction = direction / np.linalg.norm(direction)
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        if s == 0:
            rot = np.eye(3)
        else:
            c = np.dot(z_axis, direction)
            v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot = np.eye(3) + v_skew + v_skew @ v_skew * (1 - c) / (s * s)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius, cone_radius=cone_radius,
            cylinder_height=length * 0.9, cone_height=length * 0.1)
        arrow.rotate(rot, center=(0, 0, 0))
        arrow.translate(origin)
        arrow.paint_uniform_color(color)
        return arrow

    # ==================== 辅助方法 ====================

    @staticmethod
    def load_intrinsics_from_file(txt_path: str) -> dict[str, float]:
        """从 camera_intrinsics.txt 读取内参"""
        params = {}
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, val = line.split('=')
                    params[key.strip()] = float(val.strip())
        return params

    @staticmethod
    def load_intrinsics_from_extraction(dir_path: str) -> dict[str, float]:
        """从提取目录中的 camera_intrinsics.txt 读取内参"""
        txt_path = os.path.join(dir_path, "camera_intrinsics.txt")
        return WheelChockPoseEstimator.load_intrinsics_from_file(txt_path)
