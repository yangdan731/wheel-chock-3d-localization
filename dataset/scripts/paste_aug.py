import cv2
import numpy as np
import os
import random

# 路径配置（绝对路径，最稳定）
BG_DIR = r"D:\GraduationProjectCode\formal_project\paste_red\bg"
OBJ_DIR = r"D:\GraduationProjectCode\formal_project\paste_red\object"
SAVE_DIR = r"D:\GraduationProjectCode\formal_project\paste_red\output"

os.makedirs(SAVE_DIR, exist_ok=True)

def random_paste(bg_img, obj_img):
    h_bg, w_bg = bg_img.shape[:2]
    h_obj, w_obj = obj_img.shape[:2]

    # 随机缩放
    scale = random.uniform(0.3, 0.7)
    new_w = int(w_obj * scale)
    new_h = int(h_obj * scale)
    obj_img = cv2.resize(obj_img, (new_w, new_h))

    # 随机位置
    x = random.randint(0, max(0, w_bg - new_w))
    y = random.randint(0, max(0, h_bg - new_h))

    # -------------------------- 修复核心BUG --------------------------
    # 如果没有透明通道，自动创建一个全不透明的 mask
    if obj_img.shape[2] == 3:
        b, g, r = cv2.split(obj_img)
        a = np.ones_like(b) * 255  # 无透明层 → 全不透明
    else:
        b, g, r, a = cv2.split(obj_img)
    # ----------------------------------------------------------------

    mask = a / 255.0
    mask = np.expand_dims(mask, axis=-1)

    roi = bg_img[y:y+new_h, x:x+new_w]
    bg_img[y:y+new_h, x:x+new_w] = (1 - mask) * roi + mask * cv2.merge([b, g, r])
    return bg_img

if __name__ == "__main__":
    bg_list = [os.path.join(BG_DIR, f) for f in os.listdir(BG_DIR) if f.endswith(('.jpg', '.png'))]
    obj_list = [os.path.join(OBJ_DIR, f) for f in os.listdir(OBJ_DIR) if f.endswith('.png')]

    count = 0
    for bg_path in bg_list:
        bg = cv2.imread(bg_path)
        if bg is None:
            print(f"跳过无效背景：{bg_path}")
            continue

        for obj_path in obj_list:
            obj = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            if obj is None:
                continue

            try:
                new_img = random_paste(bg.copy(), obj)
                save_path = os.path.join(SAVE_DIR, f"paste_{count:04d}.jpg")
                cv2.imwrite(save_path, new_img)
                count += 1
            except Exception as e:
                print(f"跳过素材 {obj_path}，原因：{e}")
                continue

    print(f"\n✅ 贴图增强完成！共生成 {count} 张负样本")