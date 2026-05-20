import os
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt


def find_image_file(image_dir, base_name):
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = os.path.join(image_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    for f in os.listdir(image_dir):
        if f.lower().startswith(base_name.lower()) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            return os.path.join(image_dir, f)
    return None


def visualize_sample(gt_name):
    # 使用绝对路径或相对路径（确保目录存在）
    base_dir = os.path.dirname(os.path.dirname(__file__))  # seg_compare 目录
    img_dir = os.path.join(base_dir, "test_image")
    gt_dir = os.path.join(base_dir, "test_gt")
    pred_color_dir = os.path.join(base_dir, "pred_color")
    pred_yolo_dir = os.path.join(base_dir, "pred_yolo")

    base_name = os.path.splitext(gt_name)[0]
    img_path = find_image_file(img_dir, base_name)
    if img_path is None:
        print(f"Error: No image found for base name {base_name}")
        return

    gt_path = os.path.join(gt_dir, gt_name)
    pred_c_path = os.path.join(pred_color_dir, gt_name)
    pred_y_path = os.path.join(pred_yolo_dir, gt_name)

    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred_c = cv2.imread(pred_c_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(pred_c_path) else None
    pred_y = cv2.imread(pred_y_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(pred_y_path) else None

    if img is None or gt is None:
        print(f"Error: Failed to read image or GT for {gt_name}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')

    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(pred_c if pred_c is not None else np.zeros_like(gt), cmap='gray')
    axes[2].set_title("Color Method" if pred_c is not None else "Color Method (No result)")
    axes[2].axis('off')

    axes[3].imshow(pred_y if pred_y is not None else np.zeros_like(gt), cmap='gray')
    axes[3].set_title("YOLOv8-seg" if pred_y is not None else "YOLOv8-seg (No result)")
    axes[3].axis('off')

    plt.tight_layout()
    output_path = os.path.join(base_dir, "comparison.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison image to {output_path}")


if __name__ == "__main__":
    gt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_gt")
    if not os.path.exists(gt_dir):
        print(f"Directory {gt_dir} not found!")
        exit()
    png_files = [f for f in os.listdir(gt_dir) if f.lower().endswith('.png')]
    if not png_files:
        print("No PNG files in test_gt directory!")
        exit()
    sample = png_files[11]  # 0表示第一张图片
    visualize_sample(sample)