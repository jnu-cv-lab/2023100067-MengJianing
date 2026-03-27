import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. 读取图像并转换为YCbCr ----------------------
# 读取彩色图像（BGR格式）
img = cv2.imread("test.jpg")
if img is None:
    raise ValueError("图片读取失败，请检查路径")

# 转换为YCbCr色彩空间
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(img_ycrcb)  # 拆分Y、Cr、Cb通道

# ---------------------- 2. 对Cb、Cr通道进行下采样（2倍） ----------------------
# 下采样：取偶数行偶数列（步长为2）
Cb_down = Cb[::2, ::2]
Cr_down = Cr[::2, ::2]

# ---------------------- 3. 插值恢复到原尺寸 ----------------------
# 使用双线性插值恢复尺寸
h, w = Y.shape
Cb_up = cv2.resize(Cb_down, (w, h), interpolation=cv2.INTER_LINEAR)
Cr_up = cv2.resize(Cr_down, (w, h), interpolation=cv2.INTER_LINEAR)

# ---------------------- 4. 重建YCbCr并转回RGB ----------------------
img_ycrcb_recon = cv2.merge((Y, Cr_up, Cb_up))
img_rgb_recon = cv2.cvtColor(img_ycrcb_recon, cv2.COLOR_YCrCb2BGR)

# ---------------------- 5. 计算PSNR ----------------------
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # 图像完全一致
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

psnr_value = calculate_psnr(img, img_rgb_recon)
print(f"PSNR 值: {psnr_value:.2f} dB")

# ---------------------- 6. 可视化对比 ----------------------
plt.figure(figsize=(12, 6))

# 原图
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# 重建图
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_rgb_recon, cv2.COLOR_BGR2RGB))
plt.title(f"Reconstructed Image\nPSNR = {psnr_value:.2f} dB")
plt.axis("off")

plt.tight_layout()
