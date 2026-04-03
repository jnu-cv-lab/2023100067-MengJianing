import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# ====================== 1. 预处理 ======================
#自动将彩色图转为灰度图
img_path = "/home/fntq/cv-course/lab01/src/lena.png"
# 直接读取为灰度图
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape
print(f"原图尺寸: {h}×{w}")

# ====================== 2. 下采样（缩小1/2）：两种方式完整实现 ======================
# 方式1：不做预滤波，直接隔行隔列缩小
img_down_direct = img[::2, ::2]

# 方式2：先高斯平滑（低通滤波），再隔行隔列缩小（避免混叠）
img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)  # 5×5高斯核，sigma自动计算
img_down_gauss = img_gaussian[::2, ::2]

# ====================== 3. 图像恢复：两种缩小方式 × 三种插值方法 ======================
h_up, w_up = h, w  # 恢复到原图尺寸

# ---------------------- 对「直接缩小图」做三种插值 ----------------------
# 最近邻内插
img_up_nearest_direct = cv2.resize(img_down_direct, (w_up, h_up), interpolation=cv2.INTER_NEAREST)
# 双线性内插
img_up_bilinear_direct = cv2.resize(img_down_direct, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
# 双三次内插
img_up_bicubic_direct = cv2.resize(img_down_direct, (w_up, h_up), interpolation=cv2.INTER_CUBIC)

# ---------------------- 对「高斯平滑后缩小图」做三种插值 ----------------------
# 最近邻内插
img_up_nearest_gauss = cv2.resize(img_down_gauss, (w_up, h_up), interpolation=cv2.INTER_NEAREST)
# 双线性内插
img_up_bilinear_gauss = cv2.resize(img_down_gauss, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
# 双三次内插
img_up_bicubic_gauss = cv2.resize(img_down_gauss, (w_up, h_up), interpolation=cv2.INTER_CUBIC)

# ====================== 4. 计算所有恢复图的MSE、PSNR（空间域量化指标） ======================
def calc_mse_psnr(original, restored):
    """计算原图与恢复图的MSE和PSNR"""
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        psnr = 100.0  # 完全相同，PSNR无穷大，取100
    else:
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return round(mse, 2), round(psnr, 2)

# ---------------------- 直接缩小的三种插值的MSE/PSNR ----------------------
mse_nearest_direct, psnr_nearest_direct = calc_mse_psnr(img, img_up_nearest_direct)
mse_bilinear_direct, psnr_bilinear_direct = calc_mse_psnr(img, img_up_bilinear_direct)
mse_bicubic_direct, psnr_bicubic_direct = calc_mse_psnr(img, img_up_bicubic_direct)

# ---------------------- 高斯缩小的三种插值的MSE/PSNR ----------------------
mse_nearest_gauss, psnr_nearest_gauss = calc_mse_psnr(img, img_up_nearest_gauss)
mse_bilinear_gauss, psnr_bilinear_gauss = calc_mse_psnr(img, img_up_bilinear_gauss)
mse_bicubic_gauss, psnr_bicubic_gauss = calc_mse_psnr(img, img_up_bicubic_gauss)

# 打印所有结果
print("\n===== 空间域质量评价（直接缩小 + 三种插值） =====")
print(f"最近邻内插: MSE={mse_nearest_direct}, PSNR={psnr_nearest_direct} dB")
print(f"双线性内插: MSE={mse_bilinear_direct}, PSNR={psnr_bilinear_direct} dB")
print(f"双三次内插: MSE={mse_bicubic_direct}, PSNR={psnr_bicubic_direct} dB")

print("\n===== 空间域质量评价（高斯平滑后缩小 + 三种插值） =====")
print(f"最近邻内插: MSE={mse_nearest_gauss}, PSNR={psnr_nearest_gauss} dB")
print(f"双线性内插: MSE={mse_bilinear_gauss}, PSNR={psnr_bilinear_gauss} dB")
print(f"双三次内插: MSE={mse_bicubic_gauss}, PSNR={psnr_bicubic_gauss} dB")

# ====================== 5. 傅里叶变换（FFT）分析： ======================
def fft_process(img):
    """二维FFT + 移中心 + 对数幅度谱，用于显示"""
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)  # 低频移到图像中心
    magnitude = 20 * np.log(1 + np.abs(f_shift))  # 取对数增强显示
    return magnitude

fft_original = fft_process(img)
fft_down_direct = fft_process(img_down_direct)
fft_bilinear_direct = fft_process(img_up_bilinear_direct)

# ====================== 6. DCT变换分析：分块8×8，统计低频能量占比 ======================
def dct_block_analysis(img, block_size=8, low_freq_size=4):
    h, w = img.shape
    total_energy = 0.0
    low_energy_total = 0.0
    dct_sample = None

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size].astype(np.float32)
            dct_block = cv2.dct(block)
            
            if dct_sample is None:
                dct_sample = 20 * np.log(1 + np.abs(dct_block))
            
            block_energy = np.sum(dct_block ** 2)
            total_energy += block_energy
            low_energy = np.sum(dct_block[:low_freq_size, :low_freq_size] ** 2)
            low_energy_total += low_energy

    ratio = round(low_energy_total / total_energy, 4) if total_energy != 0 else 0.0
    return dct_sample, ratio

dct_ori, ratio_ori = dct_block_analysis(img)
dct_nearest_direct, ratio_nearest_direct = dct_block_analysis(img_up_nearest_direct)
dct_bilinear_direct, ratio_bilinear_direct = dct_block_analysis(img_up_bilinear_direct)
dct_bicubic_direct, ratio_bicubic_direct = dct_block_analysis(img_up_bicubic_direct)
dct_nearest_gauss, ratio_nearest_gauss = dct_block_analysis(img_up_nearest_gauss)
dct_bilinear_gauss, ratio_bilinear_gauss = dct_block_analysis(img_up_bilinear_gauss)
dct_bicubic_gauss, ratio_bicubic_gauss = dct_block_analysis(img_up_bicubic_gauss)

print("\n===== DCT 8×8分块 左上角4×4低频能量占比 =====")
print(f"原图: {ratio_ori}")
print("\n--- 直接缩小的三种恢复 ---")
print(f"最近邻: {ratio_nearest_direct}")
print(f"双线性: {ratio_bilinear_direct}")
print(f"双三次: {ratio_bicubic_direct}")
print("\n--- 高斯平滑后缩小的三种恢复 ---")
print(f"最近邻: {ratio_nearest_gauss}")
print(f"双线性: {ratio_bilinear_gauss}")
print(f"双三次: {ratio_bicubic_gauss}")

# ====================== 绘图： ======================
plt.figure(figsize=(18, 12))
plt.subplot(3, 4, 1), plt.imshow(img, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(3, 4, 2), plt.imshow(img_down_direct, cmap="gray"), plt.title("Direct Downscale 1/2"), plt.axis("off")
plt.subplot(3, 4, 3), plt.imshow(img_down_gauss, cmap="gray"), plt.title("Gaussian + Downscale 1/2"), plt.axis("off")
plt.axis("off")

plt.subplot(3, 4, 5), plt.imshow(img_up_nearest_direct, cmap="gray"), plt.title(f"Direct→Nearest\nMSE={mse_nearest_direct}, PSNR={psnr_nearest_direct}dB"), plt.axis("off")
plt.subplot(3, 4, 6), plt.imshow(img_up_bilinear_direct, cmap="gray"), plt.title(f"Direct→Bilinear\nMSE={mse_bilinear_direct}, PSNR={psnr_bilinear_direct}dB"), plt.axis("off")
plt.subplot(3, 4, 7), plt.imshow(img_up_bicubic_direct, cmap="gray"), plt.title(f"Direct→Bicubic\nMSE={mse_bicubic_direct}, PSNR={psnr_bicubic_direct}dB"), plt.axis("off")
plt.axis("off")

plt.subplot(3, 4, 9), plt.imshow(img_up_nearest_gauss, cmap="gray"), plt.title(f"Gaussian→Nearest\nMSE={mse_nearest_gauss}, PSNR={psnr_nearest_gauss}dB"), plt.axis("off")
plt.subplot(3, 4, 10), plt.imshow(img_up_bilinear_gauss, cmap="gray"), plt.title(f"Gaussian→Bilinear\nMSE={mse_bilinear_gauss}, PSNR={psnr_bilinear_gauss}dB"), plt.axis("off")
plt.subplot(3, 4, 11), plt.imshow(img_up_bicubic_gauss, cmap="gray"), plt.title(f"Gaussian→Bicubic\nMSE={mse_bicubic_gauss}, PSNR={psnr_bicubic_gauss}dB"), plt.axis("off")
plt.axis("off")

plt.tight_layout()
plt.savefig("spatial_result_full.png", dpi=300, bbox_inches="tight")
plt.close()

# FFT 图（仅英文标题）
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1), plt.imshow(fft_original, cmap="gray"), plt.title("Original FFT Spectrum"), plt.axis("off")
plt.subplot(1, 3, 2), plt.imshow(fft_down_direct, cmap="gray"), plt.title("Direct Downscale FFT"), plt.axis("off")
plt.subplot(1, 3, 3), plt.imshow(fft_bilinear_direct, cmap="gray"), plt.title("Bilinear Restored FFT"), plt.axis("off")
plt.tight_layout()
plt.savefig("fft_result.png", dpi=300, bbox_inches="tight")
plt.close()

# DCT 图（仅英文标题）
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1), plt.imshow(dct_ori, cmap="gray"), plt.title(f"Original DCT(8×8)\nLow Ratio {ratio_ori}"), plt.axis("off")
plt.subplot(2, 3, 2), plt.imshow(dct_nearest_direct, cmap="gray"), plt.title(f"Direct→Nearest DCT\nRatio {ratio_nearest_direct}"), plt.axis("off")
plt.subplot(2, 3, 3), plt.imshow(dct_bilinear_direct, cmap="gray"), plt.title(f"Direct→Bilinear DCT\nRatio {ratio_bilinear_direct}"), plt.axis("off")
plt.subplot(2, 3, 4), plt.imshow(dct_bicubic_direct, cmap="gray"), plt.title(f"Direct→Bicubic DCT\nRatio {ratio_bicubic_direct}"), plt.axis("off")
plt.tight_layout()
plt.savefig("dct_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n 所有实验结果图已保存：")
print("1. spatial_result_full.png")
print("2. fft_result.png")
print("3. dct_result.png")