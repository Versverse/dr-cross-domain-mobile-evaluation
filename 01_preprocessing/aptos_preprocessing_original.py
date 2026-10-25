import cv2
import os
import numpy as np
import tqdm

# --- 1. 配置路径 (请根据你的实际情况修改) ---
SOURCE_CLASSIFIED_DIR = r"[source]/aptos2019_train_classified"  # <--- 已分类的原始图
TARGET_PREPROCESSED_DIR = r"[source]/aptos2019_train_preprocessed"  # <--- 预处理后保存的位置
IMAGE_SIZE = (224, 224)  # 目标尺寸
CROP_THRESHOLD = 15  # 用于区分黑色背景和视网膜区域的亮度阈值 (可调整)

# --- 2. 确保目标文件夹和子文件夹存在 ---
print(f"确保目标路径存在: {TARGET_PREPROCESSED_DIR}")
for subdir in ["0_No_DR", "1_Mild", "2_Moderate", "3_Severe", "4_Proliferate_DR"]:
    os.makedirs(os.path.join(TARGET_PREPROCESSED_DIR, subdir), exist_ok=True)


# --- 3. 定义核心预处理函数 ---
def crop_and_preprocess_image(image_path, output_size, threshold=CROP_THRESHOLD):
    """
    读取图像，裁剪黑边，缩放尺寸，并应用CLAHE增强对比度。
    """
    # 1. 读取图像 (以彩色方式读取)
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告：无法读取图像 {image_path}")
        return None

    try:
        # --- 步骤 1: 裁剪黑边 (Robust Version) ---
        # a. 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # b. 应用阈值，将低于阈值的像素设为黑色 (0)，高于的设为白色 (255)
        #   这有助于区分非常暗的视网膜区域和纯黑背景
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # c. 查找轮廓
        #    cv2.RETR_EXTERNAL 只查找最外层的轮廓
        #    cv2.CHAIN_APPROX_SIMPLE 压缩轮廓点
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"警告：在 {image_path} 中未找到轮廓，可能图像全黑或阈值太高。跳过裁剪。")
            img_cropped = img  # 如果找不到轮廓，就使用原图进行后续处理
        else:
            # d. 找到面积最大的轮廓 (通常是视网膜区域)
            largest_contour = max(contours, key=cv2.contourArea)

            # e. 计算该轮廓的边界框 (x, y, width, height)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # f. 裁剪原始彩色图像
            #    为了后续缩放不形变，可以裁剪成一个以边界框中心为中心的正方形
            center_x, center_y = x + w // 2, y + h // 2
            side = max(w, h)
            x1 = max(0, center_x - side // 2)
            y1 = max(0, center_y - side // 2)
            x2 = min(img.shape[1], center_x + side // 2)
            y2 = min(img.shape[0], center_y + side // 2)

            # 确保裁剪区域有效
            if y2 > y1 and x2 > x1:
                img_cropped = img[y1:y2, x1:x2]
            else:
                print(f"警告：在 {image_path} 中计算出的裁剪区域无效。跳过裁剪。")
                img_cropped = img  # 使用原图

    except Exception as e:
        print(f"处理 {image_path} 的裁剪步骤时出错: {e}。跳过裁剪。")
        img_cropped = img  # 出错时使用原图进行后续处理

    # --- 步骤 2: 缩放尺寸 ---
    try:
        img_resized = cv2.resize(img_cropped, output_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"处理 {image_path} 的缩放步骤时出错: {e}。返回 None。")
        return None  # 缩放失败通常意味着图像有问题

    # --- 步骤 3: 对比度增强 (CLAHE) ---
    try:
        # 将图像转换到 LAB 色彩空间
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # 对 L (亮度) 通道应用 CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # 合并通道并转回 BGR
        limg = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"处理 {image_path} 的 CLAHE 步骤时出错: {e}。将使用仅缩放后的图像。")
        img_clahe = img_resized  # CLAHE 出错时，至少返回缩放后的结果

    # --- 步骤 4: 归一化 (保留 0-255 范围以便保存) ---
    img_processed = img_clahe

    return img_processed


# --- 4. 遍历所有分类好的图片进行预处理 ---
print("开始预处理图片...")
total_files_processed = 0
total_files_skipped = 0

# 使用 os.walk 遍历所有子文件夹和文件
for root, dirs, files in os.walk(SOURCE_CLASSIFIED_DIR):
    # 获取当前正在处理的类别子文件夹名称 (例如 "0_No_DR")
    subdir = os.path.basename(root)

    # 确保我们只处理 5 个目标类别文件夹
    if subdir in ["0_No_DR", "1_Mild", "2_Moderate", "3_Severe", "4_Proliferate_DR"]:

        target_subdir_path = os.path.join(TARGET_PREPROCESSED_DIR, subdir)
        # 再次确保目标子文件夹存在 (虽然前面已经创建过)
        os.makedirs(target_subdir_path, exist_ok=True)

        print(f"\n正在处理文件夹: {subdir}")

        # 筛选出图片文件
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print(f"  文件夹 {subdir} 中没有找到图片文件。")
            continue

        # 使用 tqdm 显示进度条
        for filename in tqdm.tqdm(image_files, desc=f"  {subdir}", unit="img"):
            source_file_path = os.path.join(root, filename)
            target_file_path = os.path.join(target_subdir_path, filename)

            # 如果目标文件已存在，跳过，不重新处理
            if os.path.exists(target_file_path):
                total_files_skipped += 1
                continue

            # 调用核心预处理函数
            processed_img = crop_and_preprocess_image(source_file_path, IMAGE_SIZE)

            # 如果处理成功，则保存
            if processed_img is not None:
                try:
                    cv2.imwrite(target_file_path, processed_img)
                    total_files_processed += 1
                except Exception as e:
                    print(f"保存处理后的图像 {target_file_path} 时出错: {e}")
            else:
                total_files_skipped += 1  # 处理失败也算跳过

print("\n--- 所有图片预处理完成 ---")
print(f"成功处理并保存: {total_files_processed} 张图片")
print(f"跳过 (已存在或处理失败): {total_files_skipped} 张图片")
print(f"预处理后的图片已保存到: {TARGET_PREPROCESSED_DIR}")