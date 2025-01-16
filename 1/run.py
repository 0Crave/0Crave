import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_convolution(image, kernel):
    """
    图像卷积操作
    
    参数:
        image: numpy.ndarray, 输入图像(可以是彩色或灰度图像)
        kernel: numpy.ndarray, 卷积核
    
    返回:
        output: numpy.ndarray, 卷积后的图像
    """
    # 如果是彩色图像，转换为灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    padding_h = kernel_height // 2
    padding_w = kernel_width // 2
    
    # 创建填充后的图像，处理边界问题
    padded_image = np.zeros((image_height + 2 * padding_h, image_width + 2 * padding_w))
    padded_image[padding_h:padding_h + image_height, padding_w:padding_w + image_width] = image
    
    output = np.zeros_like(image, dtype=np.float32)
    
    # 执行卷积运算
    for i in range(image_height):
        for j in range(image_width):
            output[i, j] = np.sum(
                padded_image[i:i + kernel_height, j:j + kernel_width] * kernel
            )
    
    return output

def manual_histogram(image):
    """
    计算图像直方图
    
    参数:
        image: numpy.ndarray, 输入图像(可以是彩色或灰度图像)
    
    返回:
        如果是彩色图像: list[numpy.ndarray], 包含BGR三个通道的直方图
        如果是灰度图像: numpy.ndarray, 灰度直方图
    """
    if len(image.shape) == 3:
        # 处理彩色图像
        hist_b = np.zeros(256)
        hist_g = np.zeros(256)
        hist_r = np.zeros(256)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist_b[image[i, j, 0]] += 1
                hist_g[image[i, j, 1]] += 1
                hist_r[image[i, j, 2]] += 1
                
        return [hist_b, hist_g, hist_r]
    else:
        # 处理灰度图像
        hist = np.zeros(256)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist[image[i, j]] += 1
        return hist

def manual_texture_features(image):
    """
    计算图像的纹理特征
    
    参数:
        image: numpy.ndarray, 输入图像(可以是彩色或灰度图像)
    
    返回:
        features: numpy.ndarray, 包含以下纹理特征的数组:
        - 平均值
        - 标准差
        - 能量
        - 对比度
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    height, width = image.shape
    features = []
    
    # 计算平均值
    mean = np.sum(image) / (height * width)
    features.append(mean)
    
    # 计算标准差
    variance = 0
    for i in range(height):
        for j in range(width):
            variance += (image[i, j] - mean) ** 2
    std = np.sqrt(variance / (height * width))
    features.append(std)
    
    # 计算能量
    energy = 0
    for i in range(height):
        for j in range(width):
            energy += image[i, j] ** 2
    features.append(energy)
    
    # 计算对比度
    contrast = 0
    for i in range(height-1):
        for j in range(width-1):
            contrast += (image[i, j] - image[i+1, j+1]) ** 2
    features.append(contrast)
    
    return np.array(features)

def process_image(image_path):
    """
    处理图像的主函数，包括边缘检测、滤波、直方图计算和纹理特征提取
    
    参数:
        image_path: str, 输入图像的路径
    
    返回:
        sobel_magnitude: numpy.ndarray, Sobel边缘检测结果
        custom_filtered: numpy.ndarray, 自定义卷积核滤波结果
        color_hist: list[numpy.ndarray], 颜色直方图
        texture_features: numpy.ndarray, 纹理特征
    """
    # 读取图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 定义Sobel算子
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    
    # 执行Sobel边缘检测
    gradient_x = manual_convolution(gray, sobel_x)
    gradient_y = manual_convolution(gray, sobel_y)
    sobel_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))
    
    # 使用给定的卷积核进行滤波
    kernel = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])
    custom_filtered = manual_convolution(gray, kernel)
    custom_filtered = np.uint8(np.clip(custom_filtered, 0, 255))
    
    # 计算并保存所有结果
    color_hist = manual_histogram(img)
    texture_features = manual_texture_features(gray)
    
    # 保存处理结果
    cv2.imwrite('sobel_result.jpg', sobel_magnitude)
    cv2.imwrite('custom_filtered.jpg', custom_filtered)
    
    # 绘制直方图
    plt.figure(figsize=(10, 4))
    colors = ['b', 'g', 'r']
    for i in range(3):
        plt.plot(color_hist[i], color=colors[i])
    plt.title('Color Histogram')
    plt.savefig('color_histogram.png')
    
    np.save('texture_features', texture_features)
    
    return sobel_magnitude, custom_filtered, color_hist, texture_features

def main():
    """
    主函数，用于执行图像处理流程
    """
    # 设置图像路径
    image_path = "/data1/home/chenbocheng/cbc/cv/experiment/1/1.jpg"
    
    try:
        # 执行图像处理
        sobel_result, custom_result, color_hist, texture_features = process_image(image_path)
        print("处理完成！")
        print("纹理特征已保存至 texture_features.npy")
        print("Sobel结果已保存至 sobel_result.jpg")
        print("自定义卷积核结果已保存至 custom_filtered.jpg")
        print("颜色直方图已保存至 color_histogram.png")
        print("纹理特征:", texture_features)
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()