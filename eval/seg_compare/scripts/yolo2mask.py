import os
import cv2
import numpy as np


def yolo_txt_to_mask(txt_path, img_width, img_height, output_path):
    """
    将 YOLO 格式的 txt 文件（多边形）转换为二值掩码图像
    txt 每行格式：class_id x1_norm y1_norm x2_norm y2_norm ...
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # 至少需要 class_id + 3个点（6个坐标）
                continue
            # 第一个是 class_id，忽略（因为只有止轮器一类）
            coords = list(map(float, parts[1:]))
            # 将归一化坐标转换为绝对像素坐标
            points = []
            for i in range(0, len(coords), 2):
                x_norm = coords[i]
                y_norm = coords[i + 1]
                x = int(x_norm * img_width)
                y = int(y_norm * img_height)
                points.append([x, y])
            points = np.array(points, dtype=np.int32)
            # 填充多边形
            cv2.fillPoly(mask, [points], 255)
    cv2.imwrite(output_path, mask)
    print(f"Saved mask: {output_path}")

# ==================== 配置 ====================
# 获取当前脚本所在目录（scripts/）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 向上三级到达 formal_project 根目录（scripts -> seg_compare -> eval -> formal_project）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

# 构建路径
txt_dir = os.path.join(project_root, "dataset/wheelchock_dataset", "labels", "train")
image_dir = os.path.join(project_root, "dataset/wheelchock_dataset", "images/train")
output_dir = os.path.join(project_root, "eval", "seg_compare", "test_gt/train")
os.makedirs(output_dir, exist_ok=True)

# txt_dir = os.path.join(project_root, "dataset/wheelchock_dataset", "labels", "val")
# image_dir = os.path.join(project_root, "dataset/wheelchock_dataset", "images/val")
# output_dir = os.path.join(project_root, "eval", "seg_compare", "test_gt/val")
# os.makedirs(output_dir, exist_ok=True)

# 遍历 txt 文件
for txt_file in os.listdir(txt_dir):
    if not txt_file.endswith('.txt'):
        continue
    # 获取对应的图像文件名（假设与 txt 同名，但扩展名为 .png 或 .jpg）
    base_name = os.path.splitext(txt_file)[0]
    # 查找图像文件（支持常见扩展名）
    img_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = os.path.join(image_dir, base_name + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break
    if img_path is None:
        print(f"Warning: No image found for {txt_file}, skipping.")
        continue

    # 读取图像获取尺寸
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Cannot read image {img_path}, skipping.")
        continue
    h, w = img.shape[:2]

    txt_path = os.path.join(txt_dir, txt_file)
    output_path = os.path.join(output_dir, base_name + '.jpg')
    yolo_txt_to_mask(txt_path, w, h, output_path)