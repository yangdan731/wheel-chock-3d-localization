import os
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def find_file_in_subdir(base_dir, subdir, base_name, extensions):
    """在 base_dir/subdir 下查找 base_name + 任意扩展名的文件"""
    target_dir = os.path.join(base_dir, subdir)
    if not os.path.exists(target_dir):
        return None
    for ext in extensions:
        candidate = os.path.join(target_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    # 如果精确匹配失败，尝试模糊匹配（以防大小写或多余字符）
    for f in os.listdir(target_dir):
        name, ext = os.path.splitext(f)
        if name.lower() == base_name.lower() and ext.lower() in extensions:
            return os.path.join(target_dir, f)
    return None

def visualize_sample(dataset='test', idx=0):
    """
    dataset: 'train' 或 'test'
    idx: 该数据集下第几张图片（按文件名排序）
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # seg_compare 目录
    img_dir = os.path.join(base_dir, "test_image", dataset)
    gt_dir = os.path.join(base_dir, "test_gt", dataset)
    pred_color_dir = os.path.join(base_dir, "pred_color", dataset)
    pred_yolo_dir = os.path.join(base_dir, "pred_yolo", "val")

    # 检查目录是否存在
    if not os.path.exists(img_dir):
        print(f"Error: Image directory {img_dir} not found!")
        return
    if not os.path.exists(gt_dir):
        print(f"Error: GT directory {gt_dir} not found!")
        return

    # 获取该数据集下所有图像文件（支持 .jpg, .png, .jpeg）
    img_extensions = ('.jpg', '.jpeg', '.png')
    gt_extensions = ('.jpg', '.png')   # GT 可能是 jpg 或 png
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
    if not img_files:
        print(f"No image files in {img_dir}")
        return
    img_files.sort()  # 排序保证索引一致
    if idx >= len(img_files):
        print(f"Index {idx} out of range (max {len(img_files)-1})")
        return

    img_name = img_files[idx]
    base_name = os.path.splitext(img_name)[0]

    # 构建完整路径（GT 和预测文件可能与图像文件名相同或扩展名不同，需要查找）
    img_path = os.path.join(img_dir, img_name)
    gt_path = find_file_in_subdir(base_dir, os.path.join("test_gt", dataset), base_name, gt_extensions)
    pred_c_path = find_file_in_subdir(base_dir, os.path.join("pred_color", dataset), base_name, gt_extensions)
    pred_y_path = find_file_in_subdir(base_dir, os.path.join("pred_yolo", "val"), base_name, gt_extensions)

    # 读取文件
    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if gt_path else None
    pred_c = cv2.imread(pred_c_path, cv2.IMREAD_GRAYSCALE) if pred_c_path else None
    pred_y = cv2.imread(pred_y_path, cv2.IMREAD_GRAYSCALE) if pred_y_path else None

    if img is None:
        print(f"Failed to read image: {img_path}")
        return
    if gt is None:
        print(f"Warning: Could not read GT for {base_name}, tried path: {gt_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"RGB Image")
    axes[0].axis('off')

    axes[1].imshow(gt if gt is not None else np.zeros_like(img[:,:,0]), cmap='gray')
    axes[1].set_title("Ground Truth" if gt is not None else "GT (missing)")
    axes[1].axis('off')

    axes[2].imshow(pred_c if pred_c is not None else np.zeros_like(img[:,:,0]), cmap='gray')
    axes[2].set_title("Color Method" if pred_c is not None else "Color Method (missing)")
    axes[2].axis('off')

    axes[3].imshow(pred_y if pred_y is not None else np.zeros_like(img[:,:,0]), cmap='gray')
    axes[3].set_title("YOLOv8-seg" if pred_y is not None else "YOLOv8-seg (missing)")
    axes[3].axis('off')

    plt.tight_layout()
    output_path = os.path.join(base_dir, f"comparison_{dataset}_{idx}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison image to {output_path}")

if __name__ == "__main__":
    # 示例：显示测试集第 0 张图片
    visualize_sample(dataset='test', idx=0)
