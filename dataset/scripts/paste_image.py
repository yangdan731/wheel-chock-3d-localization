import os
import random
from PIL import Image

# ===================== 配置路径 =====================
BG_DIR = "../../paste_red/bg/bg"
OBJECT_DIR = "../../paste_red/object/object"
OUTPUT_DIR = "../../paste_red/output/output"

# 支持的图片格式
SUPPORT_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# 可调节参数
SCALE_FACTOR = 1.0       # 贴图放大倍数
CENTER_FORBID = 0.6      # 中心禁区比例，越大越靠外

# ===================== 核心功能 =====================
def paste_object_to_background():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bg_files = [f for f in os.listdir(BG_DIR) if f.lower().endswith(SUPPORT_FORMATS)]
    obj_files = [f for f in os.listdir(OBJECT_DIR) if f.lower().endswith(SUPPORT_FORMATS)]

    if not bg_files or not obj_files:
        print("错误：背景或贴图文件夹无图片！")
        return

    print(f"找到 {len(bg_files)} 张背景图，{len(obj_files)} 张贴图")
    print("开始合成：贴图放大2倍 + 仅放置在四周...")

    for bg_name in bg_files:
        bg_path = os.path.join(BG_DIR, bg_name)
        output_path = os.path.join(OUTPUT_DIR, bg_name)

        try:
            with Image.open(bg_path).convert("RGBA") as bg_img:
                bg_w, bg_h = bg_img.size

                # 随机选贴图
                obj_name = random.choice(obj_files)
                obj_path = os.path.join(OBJECT_DIR, obj_name)

                with Image.open(obj_path).convert("RGBA") as obj_img:
                    obj_w, obj_h = obj_img.size

                    # 1. 强制放大2倍
                    new_w = int(obj_w * SCALE_FACTOR)
                    new_h = int(obj_h * SCALE_FACTOR)

                    # 安全限制：最大不超过背景的90%，防止溢出报错
                    max_allow_w = int(bg_w * 0.9)
                    max_allow_h = int(bg_h * 0.9)
                    new_w = min(new_w, max_allow_w)
                    new_h = min(new_h, max_allow_h)

                    # 缩放贴图
                    obj_img = obj_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    ow, oh = new_w, new_h

                    # 2. 安全计算四周坐标（修复randrange报错）
                    # 禁区边界
                    forbid_w = int(bg_w * CENTER_FORBID)
                    forbid_h = int(bg_h * CENTER_FORBID)

                    # 随机选择：左侧 / 右侧 / 上侧 / 下侧
                    edge = random.choice(["left", "right", "top", "bottom"])

                    if edge == "left":
                        x = random.randint(0, forbid_w)
                        y = random.randint(0, bg_h - oh)
                    elif edge == "right":
                        x = random.randint(bg_w - ow - forbid_w, bg_w - ow)
                        y = random.randint(0, bg_h - oh)
                    elif edge == "top":
                        x = random.randint(0, bg_w - ow)
                        y = random.randint(0, forbid_h)
                    else: # bottom
                        x = random.randint(0, bg_w - ow)
                        y = random.randint(bg_h - oh - forbid_h, bg_h - oh)

                    # 粘贴透明贴图
                    bg_img.paste(obj_img, (x, y), mask=obj_img)
                    # 保存
                    bg_img.convert("RGB").save(output_path, quality=95)

            print(f"合成成功：{bg_name}")

        except Exception as e:
            print(f"处理图片 {bg_name} 失败：{str(e)}")

    print(f"\n全部完成！已保存至：{OUTPUT_DIR}")

if __name__ == "__main__":
    paste_object_to_background()