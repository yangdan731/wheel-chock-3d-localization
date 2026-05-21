import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

def calc_metrics(gt, pred):
    gt_bin = gt > 0
    pred_bin = pred > 0
    iou = jaccard_score(gt_bin.flatten(), pred_bin.flatten(), average='binary', zero_division=0)
    pa = np.mean(gt_bin.flatten() == pred_bin.flatten())
    prec = precision_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    rec = recall_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    f1 = f1_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    return iou, pa, prec, rec, f1

def evaluate_split(model, img_dir, gt_dir, out_dir, split_name):
    """对指定数据集进行推理和评估"""
    os.makedirs(out_dir, exist_ok=True)
    times = []
    metrics = []
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        start = time.perf_counter()
        results = model(img, verbose=False)
        if results[0].masks is None:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, mask)
        # 如果有对应的真实掩码，计算指标
        gt_path = os.path.join(gt_dir, fname)
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                metrics.append(calc_metrics(gt, mask))
    if times:
        avg_time = np.mean(times) * 1000
        print(f"{split_name} average time: {avg_time:.2f} ms")
    else:
        print(f"{split_name}: No images found.")
    if metrics:
        mean = np.mean(metrics, axis=0)
        std = np.std(metrics, axis=0)
        print(f"\n=== YOLO {split_name} Evaluation ===")
        print(f"IoU: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Pixel Acc: {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"Precision: {mean[2]:.4f} ± {std[2]:.4f}")
        print(f"Recall: {mean[3]:.4f} ± {std[3]:.4f}")
        print(f"F1: {mean[4]:.4f} ± {std[4]:.4f}\n")
    else:
        print(f"{split_name}: No ground truth found, skipping metrics.")

if __name__ == "__main__":
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    # 测试集路径
    test_img_dir = os.path.join(project_root, "dataset/wheelchock_dataset", "images", "val")
    test_gt_dir = os.path.join(project_root, "eval", "seg_compare", "test_gt", "val")
    test_out_dir = os.path.join(project_root, "eval", "seg_compare", "pred_yolo", "val")

    # 训练集路径（需要预先准备真实掩码）
    train_img_dir = os.path.join(project_root, "dataset/wheelchock_dataset", "images", "train")
    train_gt_dir = os.path.join(project_root, "eval", "seg_compare", "test_gt", "train")
    train_out_dir = os.path.join(project_root, "eval", "seg_compare", "pred_yolo", "train")

    model_path = os.path.join(project_root, "models", "best_5122.pt")
    model = YOLO(model_path)

    # 评估训练集（如果真实掩码存在）
    if os.path.exists(train_img_dir) and os.path.exists(train_gt_dir):
        evaluate_split(model, train_img_dir, train_gt_dir, train_out_dir, "Train")
    else:
        print("训练集或真实掩码目录不存在，跳过训练集评估。")

    # 评估测试集
    evaluate_split(model, test_img_dir, test_gt_dir, test_out_dir, "Val")