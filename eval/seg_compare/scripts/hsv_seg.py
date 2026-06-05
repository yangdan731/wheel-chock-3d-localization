import os
import cv2
import numpy as np
import time
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

# ------------------------------
# 颜色阈值分割函数
# ------------------------------
def segment_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            pad = 10
            x = max(0, x-pad)
            y = max(0, y-pad)
            w = min(image.shape[1]-x, w+2*pad)
            h = min(image.shape[0]-y, h+2*pad)
            roi_mask = np.zeros_like(mask)
            roi_mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
            return roi_mask
    return mask

# ------------------------------
# 评估函数
# ------------------------------
def calc_metrics(gt, pred):
    gt_bin = gt > 0
    pred_bin = pred > 0
    iou = jaccard_score(gt_bin.flatten(), pred_bin.flatten(), average='binary', zero_division=0)
    pa = np.mean(gt_bin.flatten() == pred_bin.flatten())
    prec = precision_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    rec = recall_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    f1 = f1_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    return iou, pa, prec, rec, f1

# ------------------------------
# 批量处理与评估函数（不输出每张图片时间）
# ------------------------------
def evaluate_split(img_dir, gt_dir, out_dir, split_name):
    os.makedirs(out_dir, exist_ok=True)
    times = []
    metrics = []
    for frame in os.listdir(img_dir):
        if not frame.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        img_path = os.path.join(img_dir,frame)
        img = cv2.imread(img_path)
        if img is None:
            continue
        start = time.perf_counter()
        mask = segment_color(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        out_path = os.path.join(out_dir,frame)
        cv2.imwrite(out_path, mask)
        gt_path = os.path.join(gt_dir,frame)
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
        print(f"\n=== Color {split_name} Evaluation ===")
        print(f"IoU: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Pixel Acc: {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"Precision: {mean[2]:.4f} ± {std[2]:.4f}")
        print(f"Recall: {mean[3]:.4f} ± {std[3]:.4f}")
        print(f"F1: {mean[4]:.4f} ± {std[4]:.4f}")
    else:
        print(f"{split_name}: No ground truth found, skipping metrics.")

# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    train_img_dir = os.path.join(base_dir, "test_image", "train")
    train_gt_dir = os.path.join(base_dir, "test_gt", "train")
    train_out_dir = os.path.join(base_dir, "pred_color", "train")

    test_img_dir = os.path.join(base_dir, "test_image", "test")
    test_gt_dir = os.path.join(base_dir, "test_gt", "test")
    test_out_dir = os.path.join(base_dir, "pred_color", "test")

    print("=" * 50)
    print("Processing Training Set")
    print("=" * 50)
    evaluate_split(train_img_dir, train_gt_dir, train_out_dir, "Train")

    print("\n" + "=" * 50)
    print("Processing Test Set")
    print("=" * 50)
    evaluate_split(test_img_dir, test_gt_dir, test_out_dir, "Test")