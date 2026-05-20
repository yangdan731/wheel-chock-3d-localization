import os
import cv2
import numpy as np
import albumentations as A
from albumentations import DualTransform
import random
import shutil
from tqdm import tqdm

# 抑制 OpenMP 重复初始化警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 配置参数 ====================
INPUT_IMG_DIR = "../../wheelchock_dataset/images/train"  # 原始训练集图像目录
INPUT_LABEL_DIR = "../../wheelchock_dataset/labels/train"  # 原始标注文件目录（txt）
OUTPUT_IMG_DIR = "../../wheelchock_dataset_augmented/images/train"  # 增强后图像输出目录
OUTPUT_LABEL_DIR = "../../wheelchock_dataset_augmented/labels/train"  # 增强后标注输出目录
# NEGATIVE_SAMPLE_DIR = "path/to/red_objects"  # 可选：负样本图片目录（红色物体，不生成标注）

NUM_AUG_PER_IMAGE = 5  # 每张原始图像生成几个增强版本
COPY_ORIGINAL = True  # 是否将原始图像也复制到输出目录（不增强）

# 定义增强序列（使用最新 albumentations 参数）
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.6),  # 去掉 value 参数
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.5),  # 替代 ShiftScaleRotate
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
])


def polygons_to_mask(polygons, img_shape):
    """将多边形列表（绝对像素坐标）转换为二值掩码"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for poly in polygons:
        poly_pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [poly_pts], 1)
    return mask


def mask_to_polygons(mask, min_area=50):
    """从二值掩码中提取多边形轮廓，返回归一化坐标列表"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    h, w = mask.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue
        poly_norm = []
        for point in approx:
            x = point[0][0] / w
            y = point[0][1] / h
            poly_norm.extend([x, y])
        polygons.append(poly_norm)
    return polygons


def parse_yolo_label(txt_path, img_w, img_h):
    """读取YOLO分割标注文件，返回多边形列表（绝对像素坐标）"""
    polygons = []
    if not os.path.exists(txt_path):
        return polygons
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            poly_abs = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_w
                y = coords[i + 1] * img_h
                poly_abs.append([x, y])
            polygons.append(poly_abs)
    return polygons


def save_yolo_label(txt_path, polygons, class_id=0):
    """保存YOLO分割标注文件（归一化坐标）"""
    with open(txt_path, 'w') as f:
        for poly in polygons:
            if len(poly) < 6:
                continue
            line = [str(class_id)] + [f"{coord:.6f}" for coord in poly]
            f.write(' '.join(line) + '\n')


def augment_single_sample(img_path, label_path, output_img_dir, output_label_dir, aug_pipeline, num_aug=5,
                          copy_original=True):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    polygons = parse_yolo_label(label_path, w, h)
    if not polygons:
        print(f"Warning: {img_path} has no polygons, skip augmentation.")
        return

    mask = polygons_to_mask(polygons, img.shape)

    if copy_original:
        cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_original.png"), img)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_label_dir, f"{base_name}_original.txt"))

    for i in range(num_aug):
        augmented = aug_pipeline(image=img, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        new_polygons = mask_to_polygons(aug_mask)
        if not new_polygons:
            continue

        out_img_name = f"{base_name}_aug{i + 1}.png"
        out_txt_name = f"{base_name}_aug{i + 1}.txt"
        cv2.imwrite(os.path.join(output_img_dir, out_img_name), aug_img)
        save_yolo_label(os.path.join(output_label_dir, out_txt_name), new_polygons, class_id=0)

    print(f"Processed {base_name}: generated {num_aug} augmented samples")


def add_negative_samples(neg_dir, output_img_dir):
    if not os.path.exists(neg_dir):
        return
    for fname in os.listdir(neg_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(neg_dir, fname)
            dst = os.path.join(output_img_dir, f"neg_{fname}")
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(dst, img)
                print(f"Added negative sample: {dst}")


def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    img_files = [f for f in os.listdir(INPUT_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(img_files)} original images.")

    for fname in tqdm(img_files, desc="Augmenting"):
        img_path = os.path.join(INPUT_IMG_DIR, fname)
        label_path = os.path.join(INPUT_LABEL_DIR, os.path.splitext(fname)[0] + '.txt')
        augment_single_sample(img_path, label_path, OUTPUT_IMG_DIR, OUTPUT_LABEL_DIR,
                              augmentation_pipeline, NUM_AUG_PER_IMAGE, COPY_ORIGINAL)

    # if NEGATIVE_SAMPLE_DIR and os.path.exists(NEGATIVE_SAMPLE_DIR):
    #     add_negative_samples(NEGATIVE_SAMPLE_DIR, OUTPUT_IMG_DIR)

    print("离线增强完成！")
    print(f"输出目录: {OUTPUT_IMG_DIR} 和 {OUTPUT_LABEL_DIR}")


if __name__ == "__main__":
    main()