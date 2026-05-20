"""
止轮器位姿估计 — .bag 视频流端到端流水线
直接从 RealSense .bag 文件读取帧 → 位姿估计 → CSV 输出

用法:
  python scripts/run_bag.py wheelchock_lab.bag -o poses.csv
  python scripts/run_bag.py wheelchock_lab.bag --start 4 --end 22 --vis -o poses.csv
  python scripts/run_bag.py wheelchock_lab.bag --step 2 --max-frames 100 --vis --no-3d
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.estimator import WheelChockPoseEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="从 RealSense .bag 文件端到端估计止轮器位姿")

    parser.add_argument("bag_file", type=str, help="输入 .bag 文件路径")

    # 输出
    parser.add_argument("-o", "--output", type=str, default="poses.csv", help="CSV 输出路径 (默认 poses.csv)")

    # 时间范围 & 采样
    parser.add_argument("--start", type=float, default=None, dest="start_time",
                        help="起始时间（秒，相对视频开始）")
    parser.add_argument("--end", type=float, default=None, dest="end_time",
                        help="结束时间（秒，相对视频开始）")
    parser.add_argument("--step", type=int, default=1, help="每隔 N 帧处理一帧 (默认 1)")
    parser.add_argument("--max-frames", type=int, default=None, dest="max_frames",
                        help="最多处理的帧数")

    # 模型 & 滤波参数
    parser.add_argument("--model", type=str, default="models/best_aug.pt", help="YOLOv8 模型路径")
    parser.add_argument("--filter-neighbors", type=int, default=20)
    parser.add_argument("--filter-std", type=float, default=2.0, dest="filter_std")
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--radius-filter", type=float, default=0.02)
    parser.add_argument("--min-neighbors", type=int, default=10)

    # 可视化
    parser.add_argument("--vis", action="store_true", help="保存每帧可视化图片")
    parser.add_argument("--vis-dir", type=str, default="visualization/bag_pipeline", help="可视化输出目录")
    parser.add_argument("--no-3d", action="store_true", help="不弹出 Open3D 窗口")
    parser.add_argument("--vis-every", type=int, default=10, help="每隔多少帧保存一次可视化 (默认 10，避免 I/O 过多)")

    return parser.parse_args()


class BagPipeline:
    """从 .bag 文件读取帧并端到端估计位姿"""

    def __init__(self, estimator: WheelChockPoseEstimator):
        self.estimator = estimator
        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None
        self._playback: rs.playback | None = None

    def open(self, bag_path: str):
        """打开 .bag 文件，初始化 RealSense pipeline"""
        self._pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
        config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)

        profile = self._pipeline.start(config)

        # 读取相机内参
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_stream.get_intrinsics()
        self.estimator.fx = color_intr.fx
        self.estimator.fy = color_intr.fy
        self.estimator.cx = color_intr.ppx
        self.estimator.cy = color_intr.ppy
        print(f"相机内参: fx={color_intr.fx:.2f}, fy={color_intr.fy:.2f}, "
              f"cx={color_intr.ppx:.2f}, cy={color_intr.ppy:.2f}")

        # 深度比例
        depth_sensor = profile.get_device().first_depth_sensor()
        self.estimator.depth_scale = depth_sensor.get_depth_scale()
        print(f"深度比例: {self.estimator.depth_scale}")

        self._playback = profile.get_device().as_playback()
        self._playback.set_real_time(False)  # 非实时，最快速度处理
        self._align = rs.align(rs.stream.color)

    def close(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

    def read_frame(self) -> tuple[np.ndarray | None, np.ndarray | None, float | None, float | None]:
        """
        读取下一帧对齐的 RGB-D。
        返回 (bgr, depth_mm, timestamp_ms, rel_sec)，EOF 时返回全是 None
        """
        try:
            frames = self._pipeline.wait_for_frames(5000)
        except RuntimeError:
            return None, None, None, None

        if not frames:
            return None, None, None, None

        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return self.read_frame()  # 跳过，读下一帧

        timestamp_ms = color_frame.get_timestamp()

        rgb = np.asanyarray(color_frame.get_data())
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_mm = np.asanyarray(depth_frame.get_data())

        return bgr, depth_mm, timestamp_ms, None  # rel_sec 由调用者计算

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def run(self, bag_path: str, start_time: float | None = None, end_time: float | None = None,
            step: int = 1, max_frames: int | None = None,
            vis: bool = False, vis_dir: str = "visualization/bag_pipeline",
            vis_every: int = 10, show_3d: bool = False, output_csv: str = "poses.csv"):
        """
        运行完整流水线：遍历 .bag 全部帧 → 位姿估计 → 输出 CSV
        """
        self.open(bag_path)
        self.estimator.reset_consistency()

        frame_count = 0    # 已读取帧总数
        saved_count = 0    # 实际处理帧数
        first_timestamp: float | None = None

        results = []
        print("开始处理 .bag 流...")

        pbar = tqdm(unit="frames", desc="处理")
        try:
            while True:
                bgr, depth_mm, timestamp_ms, _ = self.read_frame()
                if bgr is None:
                    break  # EOF

                if first_timestamp is None:
                    first_timestamp = timestamp_ms
                rel_sec = (timestamp_ms - first_timestamp) / 1000.0

                # 时间范围筛选
                if start_time is not None and rel_sec < start_time:
                    frame_count += 1
                    continue
                if end_time is not None and rel_sec > end_time:
                    pbar.write(f"到达结束时间 {end_time}s，停止。")
                    break

                # 步长筛选
                if frame_count % step != 0:
                    frame_count += 1
                    continue

                # 执行端到端位姿估计
                result = self.estimator.process(bgr, depth_mm)

                if result.success:
                    results.append([saved_count, timestamp_ms, rel_sec,
                                    result.x, result.y, result.z,
                                    result.roll, result.pitch, result.yaw])
                else:
                    results.append([saved_count, timestamp_ms, rel_sec,
                                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                    self.estimator._prev_axes = None  # 失败则重置帧间一致性

                # 可视化（采样输出，避免大量 I/O）
                if vis and result.success and saved_count % vis_every == 0:
                    frame_vis_dir = Path(vis_dir) / f"frame_{saved_count:06d}"
                    self.estimator.visualize(bgr, depth_mm, result, str(frame_vis_dir), show_3d=show_3d)

                saved_count += 1
                frame_count += 1
                pbar.update(1)
                pbar.set_postfix_str(f"ok={saved_count} t={rel_sec:.1f}s")

                if max_frames is not None and saved_count >= max_frames:
                    pbar.write(f"达到最大帧数 {max_frames}，停止。")
                    break

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            pbar.close()
            self.close()

        # 输出结果
        df = pd.DataFrame(results, columns=['frame_idx', 'timestamp_ms', 'rel_sec',
                                            'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
        if len(df) > 0:
            df.to_csv(output_csv, index=False)
            valid = df['x'].notna().sum()
            print(f"\n完成: {valid}/{len(df)} 帧成功，结果保存至 {output_csv}")
        else:
            print("未处理任何帧")
        return df


def main():
    args = parse_args()

    if not Path(args.bag_file).exists():
        print(f".bag 文件不存在: {args.bag_file}", file=sys.stderr)
        sys.exit(1)

    estimator = WheelChockPoseEstimator(
        model_path=args.model,
        filter_nb_neighbors=args.filter_neighbors,
        filter_std_ratio=args.filter_std,
        voxel_size=args.voxel_size,
        radius_filter=args.radius_filter,
        min_neighbors=args.min_neighbors,
    )

    with BagPipeline(estimator) as bp:
        bp.run(
            bag_path=args.bag_file,
            start_time=args.start_time,
            end_time=args.end_time,
            step=args.step,
            max_frames=args.max_frames,
            vis=args.vis,
            vis_dir=args.vis_dir,
            vis_every=args.vis_every,
            show_3d=not args.no_3d,
            output_csv=args.output,
        )


if __name__ == "__main__":
    main()
