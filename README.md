# 多传感器融合止轮器三维定位系统

基于 Intel RealSense D435i RGB-D 相机和 YOLOv8-seg 实例分割模型，实现铁路止轮器的 2D 分割、3D 点云提取与 6DoF 位姿估计。系统采用特征级融合策略：深度图对齐 + 反投影将分割掩码转为点云，PCA 计算质心与姿态，输出相机坐标系下的 (x, y, z) 坐标和 (roll, pitch, yaw) 欧拉角。

## 项目结构

```
formal_project/
├── src/
│   ├── __init__.py
│   └── estimator.py                # 核心模块：端到端位姿估计类
│
├── scripts/
│   ├── run_img.py                  # CLI：从 PNG 帧推理
│   └── run_bag.py                  # CLI：从 .bag 视频流直接推理
│
├── dataset/
│   ├── wheelchock_dataset/         # YOLOv8-seg 训练数据集
│   │   ├── images/train/  images/val/
│   │   ├── labels/train/  labels/val/
│   │   └── dataset.yaml
│   ├── wheelchock_dataset_augmented/  # 增强后的训练数据
│   └── scripts/
│       ├── offline_augmentation.py # albumentations 离线增强
│       ├── paste_aug.py            # 负样本生成 (OpenCV)
│       └── paste_image.py          # 负样本生成 (PIL)
│
├── models/                         # 训练好的模型权重
│   └── best_aug.pt
│
└── eval/seg_compare/               # HSV颜色阈值 vs YOLOv8-seg 对比实验
    ├── test_image/  test_gt/
    ├── pred_color/  pred_yolo/
    └── scripts/
        ├── hsv_seg.py              # 颜色阈值分割 + 评估
        ├── yolov8_seg.py           # YOLO 批量推理 + 评估
        ├── yolo2mask.py            # YOLO txt → 掩码 PNG
        └── compare_vis.py          # 并排对比可视化
```

## 快速开始

### 1. 从.bag 视频流直接推理

```bash
python scripts/run_bag.py D:\path\to\wheelchock_lab.bag -o poses.csv
```

可选参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-o` | 输出 CSV 路径 | `poses.csv` |
| `--start` / `--end` | 时间范围（秒） | 全部 |
| `--step` | 跳帧间隔 | 1 |
| `--max-frames` | 最大处理帧数 | 无限制 |
| `--vis` | 生成可视化 | 关闭 |
| `--vis-every` | 每 N 帧存一次可视 | 10 |
| `--no-3d` | 不弹 Open3D 窗口 | 关闭 |
| `--model` | 模型路径 | `models/best_aug.pt` |
| `--filter-neighbors` / `--filter-std` / `--voxel-size` | 滤波参数 | 20 / 2.0 / 0.005 |

相机内参和深度比例从 `.bag` 文件自动读取。

### 2. 从提取好的 PNG 帧推理

```bash
# 先提取帧（可选，也可以直接用上面的 bag 模式）
python extract_from_bag_time.py D:\path\to\wheelchock_lab.bag -o test_data

# 单帧
python scripts/run_img.py --rgb test_data/rgb/frame_000247.png \
    --depth test_data/depth/frame_000247.png \
    --intrinsics test_data/camera_intrinsics.txt --vis

# 批量
python scripts/run_img.py --rgb-dir test_data/rgb --depth-dir test_data/depth \
    --intrinsics test_data/camera_intrinsics.txt \
    --timestamps test_data/timestamps.csv -o poses.csv --vis
```

### 3. Python API 调用

```python
from src.estimator import WheelChockPoseEstimator
import cv2

estimator = WheelChockPoseEstimator(model_path="models/best_aug.pt")

# 单帧
rgb = cv2.imread("frame.png")
depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
result = estimator.process(rgb, depth)
if result.success:
    print(f"位置: ({result.x:.3f}, {result.y:.3f}, {result.z:.3f}) m")
    print(f"姿态: R={result.roll:.1f} P={result.pitch:.1f} Y={result.yaw:.1f} deg")
    estimator.visualize(rgb, depth, result, "output/vis/")

# 批量
df = estimator.process_batch("rgb_dir/", "depth_dir/", "timestamps.csv")
df.to_csv("poses.csv", index=False)
```

## 流水线详解

```
RGB图像 ──→ YOLOv8-seg ──→ 二值掩码
                              │
深度图 ───────────────────────┤
                              ↓
              掩码 × 深度 → 针孔反投影 → 3D点云
                              ↓
              统计滤波 → 半径滤波 → 体素降采样
                              ↓
              PCA → 质心(位置) + 主轴(朝向) → 欧拉角
```

### PoseResult 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `x, y, z` | float | 相机坐标系下 3D 位置 (米) |
| `roll, pitch, yaw` | float | ZYX 欧拉角 (度) |
| `mask` | np.ndarray | 二值分割掩码 |
| `points` | np.ndarray | 滤波后点云 (N,3) |
| `raw_points` | np.ndarray | 原始点云（滤波前，用于对比诊断） |
| `axes` | np.ndarray | PCA 主轴矩阵 (3,3) |
| `centroid` | np.ndarray | 点云质心 (3,) |
| `success` | bool | 是否成功 |

## 可视化输出

`--vis` 生成 9 张图（在 `visualization/` 目录下）：

| 文件 | 内容 |
|------|------|
| `1_binary_mask.png` | 二值分割掩码 |
| `2_rgb_mask.png` | RGB + 掩码叠加 + 绿色轮廓 |
| `3_depth_colored.png` | 深度伪彩色 + 掩码轮廓 |
| `4_depth_comparison.png` | 掩码映射前后深度对比 |
| `5_pose.png` | 3D 点云 + 质心 + RGB 主轴箭头 |
| `6_filter_comparison.png` | 滤波前后点云对比 |
| `7_projected_pose.png` | 3D 主轴投影到 2D 图像 |
| `8_colored_pointcloud.png` | 点云按离质心距离着色 |
| `9_summary.png` | 三合一综合面板 |

## 数据准备

### 制作训练数据集

1. 用 Labelme 标注止轮器多边形轮廓
2. 转换为 YOLO 分割格式（每张图一个 `.txt`，包含 `class_id x1 y1 x2 y2 ...` 归一化坐标）
3. 按 80%/20% 划分 train/val，组织为：

```
wheelchock_dataset/
├── dataset.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 数据增强

```bash
cd dataset/scripts
python offline_augmentation.py    # 每张原图生成 5 个增强版本
python paste_aug.py               # 生成负样本
```

## 模型训练

使用 YOLOv8-seg 在 Kaggle 上训练，单类别 `wheelchock`。训练后将 `best.pt` 复制到 `models/best_aug.pt`。

## 对比实验

`eval/seg_compare/` 目录比较 HSV 颜色阈值法与 YOLOv8-seg 的分割性能：

```bash
cd eval/seg_compare/scripts
python hsv_seg.py       # 颜色阈值分割，输出 mIoU/PA/Precision/Recall/F1
python yolov8_seg.py    # YOLO 推理，同上指标
python compare_vis.py   # 生成四合一对比图
```

## CSV 输出格式

| 列 | 说明 |
|----|------|
| `frame_idx` | 帧序号 |
| `timestamp_ms` | 绝对时间戳 (毫秒) |
| `rel_sec` | 相对首帧时间 (秒) |
| `x, y, z` | 3D 位置 (米) |
| `roll, pitch, yaw` | 欧拉角 (度, ZYX) |

## 依赖

- Python 3.10+
- `ultralytics` (YOLOv8, PyTorch)
- `pyrealsense2` (Intel RealSense SDK)
- `opencv-python`, `numpy`, `pandas`
- `open3d` (点云处理)
- `scipy` (旋转矩阵 → 欧拉角)
- `matplotlib` (可视化)
- `tqdm` (进度条)
- `albumentations` (数据增强)

## 运行代码所需bag文件
【超级会员V2】通过百度网盘分享的文件：杨丹
链接：https://pan.baidu.com/s/1ee8A3F3b2x-Iv48PrjVdQQ?pwd=UANg 
复制这段内容打开「百度网盘APP 即可获取」
