# 多传感器融合止轮器三维定位系统

本项目基于 RGB-D 相机（Intel RealSense D435i）和 YOLOv8-seg 实例分割模型，实现铁路车厢连接部件（止轮器）的二维分割、三维点云提取与六自由度（6DoF）位姿估计。系统采用特征级多传感器融合策略，利用深度图对齐与反投影技术将分割掩码转换为三维点云，并通过 PCA 计算质心位置与姿态，最终输出相机坐标系下的三维坐标和欧拉角。

## 项目结构

formal_project/  
├── extracted_data/ # 从 .bag 提取的 RGB 与深度图  
├── seg_compare/ # 颜色阈值法与 YOLOv8-seg 对比实验  
│ ├── test_images/ # 对比测试集原图  
│ ├── test_gt/ # 人工标注的真实掩码  
│ ├── pred_color/ # 颜色方法输出的掩码  
│ ├── pred_yolo/ # YOLO 方法输出的掩码  
│ └── scripts/ # 评估与可视化脚本  
│   ├── compare_vis.py # 生成并排对比图（RGB + GT + 颜色方法 + YOLO）  
│   ├── hsv_seg_mdf.py # 修改后的颜色阈值分割批量处理脚本  
│   ├── yolo2mask.py # 将 YOLO 格式 txt 转换为掩码 PNG  
│   └── yolov8_seg.py # YOLOv8-seg 批量推理脚本  
├── wheelchock_dataset/ # YOLOv8-seg 训练数据集  
│ ├── images/ # 训练/验证图像  
│ ├── labels/ # 对应标注文件  
│ └── dataset.yaml # 数据集配置文件  
├── yolov8_model/ # 训练好的模型权重  
│ └── wheelchock5_best.pt  
├── extract_from_bag.py # 从 RealSense .bag 提取 RGB 和深度图  
├── test_multi_vis.py # 批量生成RGB+掩码叠加图  
└── test_single_frame.py # 单帧完整流程：分割 → 点云提取 → 位姿估计 → 可视化  

## 数据准备

1. 从 .bag 提取图像  
使用 extract_from_bag.py 将 RealSense 录制的 .bag 文件转换为对齐的 RGB 和深度图：  
`python extract_from_bag.py D:\GraduationProjectCode\dataset\real_data\wheelchock.bag -o extracted_data --step 10`  
可选参数 --step（间隔帧数）和 --max_frames（最大帧数）可控制输出数量。  

2. 制作训练数据集  
使用 Labelme 标注止轮器轮廓，保存为 JSON 文件。然后利用 labelme2yolo 或自定义脚本转换为 YOLO 分割格式（每个图像对应一个 .txt 文件，包含类别 ID 和多边形归一化坐标）。按 80% / 20% 划分训练集和验证集，并组织成如下结构：  
wheelchock_dataset/  
├── dataset.yaml  
├── images/  
│   ├── train/  
│   └── val/  
└── labels/  
    ├── train/  
    └── val/  

3. 对比实验测试集  
从机车环境 .bag 中提取20张图像，放入 seg_compare/test_images/。使用 Labelme 标注真实掩码，并转换为 PNG 二值图放入 seg_compare/test_gt/。  

## 模型训练

在kaggle上使用 YOLOv8-seg 训练止轮器分割模型  
训练好的最佳模型保存在 runs/segment/train/weights/best.pt，将其复制到 yolov8_model/wheelchock5_best.pt。  

## 单帧测试与可视化

运行 test_single_frame.py 对指定的一对 RGB 和深度图执行完整流程，并生成以下可视化结果（保存在 visualization/visualization/）：  

1_rgb_mask.png：RGB 图像 + 分割掩码（红色半透明） + 绿色轮廓  
2_depth_colored.png：深度伪彩色图 + 掩码轮廓  
3_pointcloud.png：止轮器三维点云（按深度着色）  
4_pose.png：点云 + 质心（红球） + PCA 主轴（红绿蓝箭头）  
5_6dof.png：相机坐标系（实线）与物体坐标系（虚线）在 3D 空间中的对比  

修改脚本中的 RGB_PATH、DEPTH_PATH 和相机内参即可测试不同帧。  

## 批量处理与对比实验

1. 颜色阈值法批量处理
修改 hsv_seg.py 中的输入/输出路径，运行生成 pred_color/ 目录下的掩码，输出 mIoU、Pixel Accuracy、Precision、Recall、F1 等指标（均值 ± 标准差）。  

2. YOLOv8-seg 批量处理
使用类似 test_single_frame.py 的推理逻辑，批量处理测试集图像，生成 pred_yolo/ 目录，输出 mIoU、Pixel Accuracy、Precision、Recall、F1 等指标（均值 ± 标准差）。  

3. 生成对比图
使用 seg_compare/scripts/compare_vis.py 将原图、真实掩码、两种方法预测掩码并排显示，保存为 comparison.png。  
![img.png](img.png)

## 位姿输出格式 
位置：相机坐标系下的质心坐标 (x, y, z)，单位米。  
姿态：ZYX 顺序的欧拉角 (roll, pitch, yaw)，单位度。  

示例输出：
三维位置: [-0.24323 -0.14048  0.6526]  
姿态:  
Roll : 12.34°  
Pitch: -5.67°  
Yaw  : 28.90°  