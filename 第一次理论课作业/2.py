import cv2
import numpy as np

# 计算PSNR（创建一个计算函数，如果mse结果为0，就返回100说明是错的）
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

# 1. 读入彩色图像
img = cv2.imread("/mnt/c/Users/K/Pictures/test.jpg")

# 2. 转换到YCbCr色彩空间，提取Y、Cb、Cr三个通道
#  Y=亮度，Cb=蓝色差，Cr=红色差
#创建一个转换后的新图像
# cv2.cvtColor = OpenCV 里专门用来「转换图片颜色格式」的函数
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# 把转换后的图像拆成 3 个单独通道：Y、Cr、Cb
Y, Cr, Cb = cv2.split(img_ycrcb)

# 3. 对Cb、Cr通道进行下采样（2倍下采样，4:2:0格式）

h, w = Y.shape    #  获取Y通道的高度和宽度（因为Y通道尺寸没有变）
Cb_down = Cb[::2, ::2]# 对Cb通道进行下采样：每隔一行、每隔一列取一个像素：：的用处。图片宽高都变成原来的 1/2
Cr_down = Cr[::2, ::2]

# 4. 用插值方法恢复到原尺寸
Cb_up = cv2.resize(Cb_down, (w, h), interpolation=cv2.INTER_LINEAR)
#Cb_down：变小后的图
#(w, h)：恢复到原来的宽度和高度
#interpolation=cv2.INTER_LINEAR 使用双线性插值来放大
Cr_up = cv2.resize(Cr_down, (w, h), interpolation=cv2.INTER_LINEAR)

# 5. 与原Y通道重建图像，转回RGB,  cv2.merge把原来的Y通道 + 恢复后的Cb、Cr通道 重新合并成一张图
recon_ycrcb = cv2.merge((Y, Cr_up, Cb_up))
recon_img = cv2.cvtColor(recon_ycrcb, cv2.COLOR_YCrCb2BGR)# 把图像从YCbCr转回BGR（OpenCV默认格式）

# 6. 计算PSNR，输出结果
psnr_value = calculate_psnr(img, recon_img)# 调用函数，计算原图和重建图的PSN
print(f"PSNR值: {round(psnr_value, 2)} dB")# 在终端输出PSNR值，保留两位小数

