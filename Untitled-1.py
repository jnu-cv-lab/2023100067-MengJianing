# 导入需要的库
import cv2          # OpenCV，用于图像读取、处理和显示
import numpy as np  # NumPy，用于数组操作
import os           # 用于处理路径（可选）
import matplotlib.pyplot as plt  # 用于图像显示（替代cv2.imshow）

# ==================== 任务1：读取测试图片 ====================
image_file = "/mnt/c/Users/K/Pictures/test.jpg"  

# 检查文件是否存在（可选）
if not os.path.exists(image_file):
    print("错误：图片文件不存在，请检查路径是否正确！")
    print("提示：在Linux下运行时，请将路径修改为Linux下的实际路径")
    exit()  # 文件不存在则退出

# 使用OpenCV的imread函数读取图片
# 返回一个numpy数组，表示图像数据
img = cv2.imread(image_file)

# 简单检查是否读取成功
if img is None:
    print("错误：无法读取图片，请检查文件路径或文件名！")
    exit()  # 若读取失败则退出程序

# ==================== 任务2：输出图像基本信息 ====================
# 获取图像尺寸（高度、宽度、通道数）
# img.shape返回一个元组 (height, width, channels) 对于彩色图
# 对于灰度图，则只有 (height, width)
h, w = img.shape[:2]          # 高度和宽度
c = img.shape[2] if len(img.shape) == 3 else 1  # 通道数，彩色图为3，灰度图为1
dtype = img.dtype             # 图像的数据类型，如uint8

# 在终端打印信息
print("图像基本信息：")
print("宽度：", w)             # 宽度（列数）
print("长度（高度）：", h)      # 长度（行数）
print("通道数：", c)           # 颜色通道数目
print("数据类型：", dtype)      # 例如uint8

# ==================== 任务3：显示原图（使用 Matplotlib） ====================
# OpenCV 读取的图像是 BGR 格式，而 Matplotlib 默认使用 RGB 格式
# 因此需要将图像从 BGR 转换为 RGB，否则显示颜色会失真
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 创建一个新的图形窗口，并设置窗口大小
plt.figure(figsize=(8, 6))
# 显示转换后的 RGB 图像
plt.imshow(img_rgb)
# 设置窗口标题
plt.title("原始图像")
# 隐藏坐标轴，使显示更干净
plt.axis('off')

# ==================== 任务4：转换为灰度图并显示 ====================
# 使用OpenCV的cvtColor函数将彩色图转换为灰度图
# cv2.COLOR_BGR2GRAY表示BGR颜色空间到灰度空间的转换
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示灰度图
# 创建新的图形窗口
plt.figure(figsize=(8, 6))
# 灰度图需要指定 colormap 为 'gray'，否则默认使用彩色映射
plt.imshow(gray, cmap='gray')
plt.title("灰度图像")
plt.axis('off')

# ==================== 任务5：保存处理结果 ====================
# 将灰度图保存为新文件，文件名为 "gray_image.jpg"
output_file = "gray_image.jpg"
cv2.imwrite(output_file, gray)
print("灰度图已保存为：", output_file)

# ==================== 任务6：用NumPy做一个简单操作 ====================
# 此处选择裁剪图像左上角一块100x100像素的区域，并保存为新文件
# 定义裁剪区域的左上角坐标 (x, y) 和宽高
x_start = 0
y_start = 0
crop_w = 100
crop_h = 100

# 注意：OpenCV中图像数组的索引是 [行, 列] 即 [y, x]
crop_region = img[y_start:y_start+crop_h, x_start:x_start+crop_w]

# 保存裁剪后的图像
crop_file = "crop_image.jpg"
cv2.imwrite(crop_file, crop_region)
print("裁剪图像（左上角100x100区域）已保存为：", crop_file)

# 也可以输出某个像素点的值，作为额外演示
# 例如输出原图左上角第一个像素的BGR值
pixel = img[0, 0]
print("原图左上角像素(0,0)的BGR值：", pixel)
