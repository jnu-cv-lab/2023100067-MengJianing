
# 导入依赖库，标注各库核心用途
import cv2              # 计算机视觉库，实现滤波、CLAHE等图像处理基础操作
import numpy as np      # 数值计算库，处理图像数组、像素运算、数据格式转换
import matplotlib.pyplot as plt  # 绘图库，绘制图像对比图与灰度直方图
from skimage import data         # 调用库自带标准测试图像，无需外部本地图片
from skimage.metrics import peak_signal_noise_ratio as psnr  # 计算图像峰值信噪比

# 配置matplotlib后端，关闭图形弹窗，仅后台保存图片至本地
plt.switch_backend('Agg')

# 手动实现全局直方图均衡化函数，无现成库函数调用
def manual_hist_eq(img):
    # 获取输入灰度图像的高度h和宽度w
    h, w = img.shape
    # 初始化长度256的整型数组，用于统计0-255各灰度级的像素出现次数
    count = np.zeros(256, dtype=int)
    
    # 双层循环遍历图像每一个像素点
    for i in range(h):
        for j in range(w):
            # 提取当前像素灰度值，对应计数位置加1
            count[img[i,j]] += 1
    
    # 计算图像总像素数量
    total_pixels = h * w
    # 计算各灰度级像素出现的概率
    prob = count / total_pixels
    
    # 计算累积分布函数CDF，逐灰度级累加概率值
    cdf = np.cumsum(prob)
    
    # 生成均衡化灰度映射表，按公式s=255×CDF计算，转换为8位无符号整型
    mapping = (255 * cdf).astype(np.uint8)
    
    # 依据映射表替换原图像像素，返回均衡化后的图像
    return mapping[img]

# 拉普拉斯锐化函数，增强图像边缘与细节
def sharpen(img):
    # 定义3×3拉普拉斯锐化卷积核
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)
    # 执行二维卷积操作，完成图像锐化，输出与原图同深度图像
    return cv2.filter2D(img, -1, kernel)

# 计算图像信息熵，衡量图像信息丰富程度
def calc_entropy(img):
    # 计算图像0-255灰度级的直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    # 直方图归一化，将频次转换为概率值
    hist = hist / hist.sum()
    # 剔除概率为0的项，避免对数计算出现异常
    hist = hist[hist > 0]
    # 按信息熵公式计算并返回结果
    return -np.sum(hist * np.log2(hist))

# 图像批量处理主函数，整合所有算法、绘图、保存与指标计算
def process_image(img, img_name):
    # 调用手动函数，实现全局直方图均衡化
    img_global_eq = manual_hist_eq(img)
    
    # 初始化CLAHE对象，设置对比度限制参数为2.0
    clahe = cv2.createCLAHE(clipLimit=2.0)
    # 对原图执行CLAHE自适应均衡化
    img_clahe = clahe.apply(img)
    
    # 对原图执行3×3窗口均值滤波
    img_mean_3x3 = cv2.blur(img, (3, 3))
    
    # 对原图执行3×3窗口高斯滤波，标准差设为0自动计算
    img_gauss_3x3 = cv2.GaussianBlur(img, (3, 3), 0)
    # 对原图执行5×5窗口高斯滤波，对比不同核大小的滤波效果
    img_gauss_5x5 = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 对原图执行3×3窗口中值滤波
    img_median_3x3 = cv2.medianBlur(img, 3)
    # 对原图执行5×5窗口中值滤波，对比不同核大小的去噪效果
    img_median_5x5 = cv2.medianBlur(img, 5)
    
    # 调用锐化函数，对原图执行边缘增强
    img_sharpen = sharpen(img)
    
    # 组合处理1：先执行3×3中值滤波，再做全局直方图均衡化
    img_filter_then_eq = manual_hist_eq(img_median_3x3)
    # 组合处理2：先做全局直方图均衡化，再执行3×3中值滤波
    img_eq_then_filter = cv2.medianBlur(img_global_eq, 3)

    # 整理所有处理后的图像，存入列表备用
    image_list = [
        img,
        img_global_eq,
        img_clahe,
        img_mean_3x3,
        img_gauss_3x3,
        img_gauss_5x5,
        img_median_3x3,
        img_median_5x5,
        img_sharpen,
        img_filter_then_eq,
        img_eq_then_filter
    ]

    # 对应各图像的标题，用于绘图标注
    title_list = [
        "Original",
        "Global Eq (Manual)",
        "CLAHE",
        "Mean 3x3",
        "Gauss 3x3",
        "Gauss 5x5",
        "Median 3x3",
        "Median 5x5",
        "Sharpen",
        "Filter -> Equal",
        "Equal -> Filter"
    ]

    # 创建绘图画布，采用2行11列合规布局，共22个子图匹配11组图像+直方图
    plt.figure(figsize=(26, 10))
    # 遍历11组处理结果，每组对应1张效果图+1幅直方图
    for index, (current_img, title) in enumerate(zip(image_list, title_list)):
        # 第一行子图：绘制处理后图像，灰度色彩映射，关闭坐标轴
        plt.subplot(2, 11, index + 1)
        plt.imshow(current_img, cmap='gray')
        plt.title(title, fontsize=8)
        plt.axis('off')
        
        # 第二行子图：绘制对应灰度直方图，256个区间，range改为关键字参数修复警告
        plt.subplot(2, 11, index + 1 + 11)
        plt.hist(current_img.flatten(), bins=256, range=[0, 255], color='black')
        plt.xlim(0, 255)
        # 缩小坐标轴字号，避免刻度重叠
        plt.tick_params(labelsize=6)

    # 自动调整子图间距，优化排版
    plt.tight_layout()
    # 高清保存图片，分辨率250DPI，裁剪边缘空白
    plt.savefig(f"{img_name}_result.png", dpi=250, bbox_inches='tight')
    # 关闭画布，释放内存资源
    plt.close()

    # 打印定量评价指标表头
    print(f"\n===== 【{img_name}】定量评价结果 =====")
    print(f"{'处理方法':<22} {'PSNR(信噪比)':<12} {'信息熵'}")
    print("-" * 50)
    # 遍历计算每幅图像的PSNR和信息熵，保留两位小数打印
    for current_img, title in zip(image_list, title_list):
        psnr_value = round(psnr(img, current_img, data_range=255), 2)
        entropy_value = round(calc_entropy(current_img), 2)
        print(f"{title:<22} {psnr_value:<12} {entropy_value}")

# 调用库自带月亮图像，作为低对比度测试图
img_low_contrast = data.moon()

# 调用库自带硬币图像，作为纹理丰富型测试图
img_texture = data.coins()

# 调用库自带月亮图像，用于制作含噪声测试图
img_noisy = data.moon()
# 生成高斯噪声，均值0，标准差20，尺寸与图像一致
noise = np.random.normal(0, 20, img_noisy.shape)
# 噪声叠加到原图，裁剪像素值至0-255有效范围，转换为8位整型
img_noisy = np.clip(img_noisy + noise, 0, 255).astype(np.uint8)

# 调用主函数，依次处理三张测试图像
process_image(img_low_contrast, "1_低对比度月亮")
process_image(img_texture, "2_硬币纹理图")
process_image(img_noisy, "3_带噪声月亮")

# 控制台打印处理完成提示
print("\n全部处理完成，已生成对应结果图片与定量指标") 