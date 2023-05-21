import cv2
import numpy as np

# 维纳滤波函数
def wiener_filter(image, noise_auto_corr, image_auto_corr, snr):
    # 计算维纳滤波器的频域表达式
    wiener_filter = np.conj(image_auto_corr) / (np.abs(image_auto_corr) ** 2 + 1/snr/noise_auto_corr)

    # 对图像进行傅里叶变换
    image_freq = np.fft.fft2(image)

    # 进行维纳滤波
    filtered_freq = wiener_filter * image_freq

    # 对滤波后的频域结果进行反傅里叶变换
    filtered_image = np.fft.ifft2(filtered_freq)

    # 返回复原后的图像
    return np.abs(filtered_image)

# 读取原始图像
image = cv2.imread("image.png", 0)  # 读取灰度图像

# 生成高斯噪声
noise = np.random.randn(*image.shape) * 20

# 计算噪声和图像的自相关函数
noise_auto_corr = np.fft.fft2(noise[::-1, ::-1])
image_auto_corr = np.fft.fft2(image[::-1, ::-1])

# 假设已知信噪比为10
snr = 10

# 进行图像复原
restored_image = wiener_filter(image, noise_auto_corr, image_auto_corr, snr)

# 显示图像复原结果
cv2.imshow("Original Image", image)
cv2.imshow("Restored Image", np.uint8(restored_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
