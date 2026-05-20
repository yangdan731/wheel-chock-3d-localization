"""
止轮器位姿估计端到端流水线 CLI
用法:
  # 单帧处理
  python scripts/run_img.py --rgb path/to/frame.png --depth path/to/depth.png --vis

  # 批量处理
  python scripts/run_img.py --rgb-dir extracted_data/rgb --depth-dir extracted_data/depth -o poses.csv

  # 批量 + 时间戳 + 可视化
  python scripts/run_img.py --rgb-dir extracted_data/rgb --depth-dir extracted_data/depth \
      --timestamps timestamps.csv -o poses.csv --vis

  # 从提取目录自动加载相机内参
  python scripts/run_img.py --rgb-dir extracted_data/rgb --depth-dir extracted_data/depth \
      --intrinsics extracted_data/camera_intrinsics.txt -o poses.csv
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.estimator import WheelChockPoseEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="止轮器端到端位姿估计流水线")

    # 输入模式
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rgb", type=str, help="单帧 RGB 图像路径")
    mode.add_argument("--rgb-dir", type=str, help="批量 RGB 图像目录")

    # 深度
    parser.add_argument("--depth", type=str, help="单帧深度图路径 (与 --rgb 配合)")
    parser.add_argument("--depth-dir", type=str, help="批量深度图目录 (与 --rgb-dir 配合, 默认与 RGB 同名)")

    # 输出
    parser.add_argument("-o", "--output", type=str, default="poses.csv", help="批量模式 CSV 输出路径 (默认 poses.csv)")

    # 模型 & 内参
    parser.add_argument("--model", type=str, default="models/best_aug.pt", help="YOLOv8 模型路径")
    parser.add_argument("--intrinsics", type=str, help="相机内参文件路径 (camera_intrinsics.txt)")

    # 滤波参数
    parser.add_argument("--filter-neighbors", type=int, default=20, help="统计滤波邻域点数 (默认 20)")
    parser.add_argument("--filter-std", type=float, default=2.0, help="统计滤波标准偏差倍数 (默认 2.0)")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="体素降采样大小 (默认 0.005m)")
    parser.add_argument("--radius-filter", type=float, default=0.02, help="半径滤波半径 (默认 0.02m)")
    parser.add_argument("--min-neighbors", type=int, default=10, help="半径滤波最小邻域点数 (默认 10)")

    # 可视化
    parser.add_argument("--vis", action="store_true", help="生成可视化输出")
    parser.add_argument("--vis-dir", type=str, default="visualization/pipeline", help="可视化输出目录")
    parser.add_argument("--no-3d", action="store_true", help="不弹出 Open3D 交互窗口")

    # 时间戳
    parser.add_argument("--timestamps", type=str, help="timestamps.csv 路径 (用于批量模式)")

    return parser.parse_args()


def load_intrinsics(args) -> dict:
    """按优先级加载相机内参：--intrinsics 文件 > --intrinsics 目录 > 默认值"""
    if args.intrinsics:
        return WheelChockPoseEstimator.load_intrinsics_from_file(args.intrinsics)
    return {}


def main():
    args = parse_args()
    intrinsics = load_intrinsics(args)

    kwargs = {
        "model_path": args.model,
        "filter_nb_neighbors": args.filter_neighbors,
        "filter_std_ratio": args.filter_std,
        "voxel_size": args.voxel_size,
        "radius_filter": args.radius_filter,
        "min_neighbors": args.min_neighbors,
    }
    kwargs.update(intrinsics)

    estimator = WheelChockPoseEstimator(**kwargs)

    # ==================== 单帧模式 ====================
    if args.rgb:
        if not args.depth:
            print("单帧模式需要 --depth", file=sys.stderr)
            sys.exit(1)

        rgb = cv2.imread(args.rgb)
        depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
        if rgb is None:
            print(f"无法读取 RGB: {args.rgb}", file=sys.stderr)
            sys.exit(1)
        if depth is None:
            print(f"无法读取深度图: {args.depth}", file=sys.stderr)
            sys.exit(1)

        print(f"处理: {args.rgb}")
        result = estimator.process(rgb, depth)

        if result.success:
            print(f"位置 (m):   x={result.x:.4f}  y={result.y:.4f}  z={result.z:.4f}")
            print(f"姿态 (deg): roll={result.roll:.2f}  pitch={result.pitch:.2f}  yaw={result.yaw:.2f}")
        else:
            print("位姿估计失败：未检测到目标或有效深度数据不足")
            sys.exit(1)

        if args.vis:
            estimator.visualize(rgb, depth, result, args.vis_dir, show_3d=not args.no_3d)

        return

    # ==================== 批量模式 ====================
    if args.rgb_dir:
        depth_dir = args.depth_dir or args.rgb_dir
        if not Path(depth_dir).exists():
            print(f"深度图目录不存在: {depth_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"RGB 目录: {args.rgb_dir}")
        print(f"深度目录: {depth_dir}")
        print(f"模型: {args.model}")
        print()

        rgb_files = sorted([f for f in Path(args.rgb_dir).glob("*.png") if f.name != "camera_intrinsics.txt"])
        total = len(rgb_files)
        if total == 0:
            print("未找到 PNG 图像", file=sys.stderr)
            sys.exit(1)

        pbar = tqdm(total=total, desc="处理帧", unit="frame")

        def update_progress(idx, _total, fname):
            pbar.update(1)
            pbar.set_postfix_str(fname)

        df = estimator.process_batch(
            args.rgb_dir, depth_dir, args.timestamps, progress_callback=update_progress
        )
        pbar.close()

        valid = df['x'].notna().sum()
        print(f"\n完成: {valid}/{len(df)} 帧成功")
        df.to_csv(args.output, index=False)
        print(f"结果已保存至 {args.output}")

        # 批量模式下可选：对第一帧做可视化
        if args.vis and valid > 0:
            first_valid = df[df['x'].notna()].iloc[0]
            frame_idx = int(first_valid['frame_idx'])
            # 找到对应文件
            for f in rgb_files:
                match = re.search(r'(\d+)', f.name)
                if match and int(match.group(1)) == frame_idx:
                    rgb = cv2.imread(str(f))
                    depth_path = Path(depth_dir) / f.name
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if rgb is not None and depth is not None:
                        result = estimator.process(rgb, depth)
                        if result.success:
                            estimator.visualize(rgb, depth, result, args.vis_dir, show_3d=not args.no_3d)
                    break


if __name__ == "__main__":
    main()
