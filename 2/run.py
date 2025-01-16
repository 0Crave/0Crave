import cv2
import numpy as np
import matplotlib.pyplot as plt

def hough_transform_segments(edge_image, rho_resolution=1, theta_resolution=np.pi/180, threshold=50, min_line_length=200, max_line_gap=150):
    """
    实现改进的霍夫变换检测线段。
    
    原理：霍夫变换通过将笛卡尔坐标系中的直线转换为参数空间中的点来检测直线。
    每条直线可以用参数方程 ρ = x*cos(θ) + y*sin(θ) 表示。
    
    参数:
        edge_image (numpy.ndarray): 边缘检测后的二值图像
        rho_resolution (float): ρ的分辨率，表示像素单位的距离精度
        theta_resolution (float): θ的分辨率，表示角度精度（弧度）
        threshold (int): 检测直线所需的最小交点数，值越大检测到的直线越少
        min_line_length (int): 最小线段长度，小于此长度的线段将被忽略
        max_line_gap (int): 同一线段上点之间的最大允许间隔，用于连接断开的线段

    返回:
        line_segments (list): 检测到的线段列表，每个元素为[x1, y1, x2, y2]表示线段的起点和终点坐标
    """
    # 获取图像尺寸
    height, width = edge_image.shape
    
    # 计算对角线长度，用于确定ρ的范围
    diagonal = np.ceil(np.sqrt(height**2 + width**2))
    
    # 创建ρ和θ的数组
    rhos = np.arange(-diagonal, diagonal + 1, rho_resolution)
    thetas = np.arange(0, np.pi, theta_resolution)
    
    # 创建累加器数组，用于存储投票结果
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    
    # 存储边缘点的位置信息，用于后续线段提取
    edge_points = {}  # 键为(rho,theta)，值为对应的边缘点列表
    
    # 获取所有边缘点的坐标
    y_idxs, x_idxs = np.nonzero(edge_image)
    
    # 对每个边缘点进行投票
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        # 计算每个θ对应的ρ值
        for theta_idx in range(len(thetas)):
            theta = thetas[theta_idx]
            # 使用霍夫变换的参数方程计算ρ
            rho = x * np.cos(theta) + y * np.sin(theta)
            # 找到最接近的ρ值的索引
            rho_idx = np.argmin(np.abs(rhos - rho))
            
            # 在累加器中投票
            accumulator[rho_idx, theta_idx] += 1
            
            # 存储边缘点位置
            key = (rho_idx, theta_idx)
            if key not in edge_points:
                edge_points[key] = []
            edge_points[key].append((x, y))
    
    # 提取线段
    line_segments = []
    for rho_idx in range(len(rhos)):
        for theta_idx in range(len(thetas)):
            # 如果累加器中的值超过阈值
            if accumulator[rho_idx, theta_idx] > threshold:
                points = edge_points.get((rho_idx, theta_idx), [])
                
                if len(points) > 0:
                    # 按照x坐标排序点，便于连接线段
                    points = sorted(points, key=lambda p: p[0])
                    
                    # 分组连续的点形成线段
                    current_segment = [points[0]]
                    for j in range(1, len(points)):
                        # 计算相邻点之间的距离
                        dx = points[j][0] - current_segment[-1][0]
                        dy = points[j][1] - current_segment[-1][1]
                        dist = np.sqrt(dx**2 + dy**2)
                        
                        # 如果距离小于最大间隔，将点添加到当前线段
                        if dist <= max_line_gap:
                            current_segment.append(points[j])
                        else:
                            # 如果当前线段足够长，保存它
                            if len(current_segment) >= min_line_length:
                                x1, y1 = current_segment[0]
                                x2, y2 = current_segment[-1]
                                line_segments.append([x1, y1, x2, y2])
                            # 开始新的线段
                            current_segment = [points[j]]
                    
                    # 处理最后一个线段
                    if len(current_segment) >= min_line_length:
                        x1, y1 = current_segment[0]
                        x2, y2 = current_segment[-1]
                        line_segments.append([x1, y1, x2, y2])
    
    return line_segments

def detect_lane_lines(image_path):
    """
    检测图像中的车道线。
    
    步骤：
    1. 读取图像并转换为灰度图
    2. 应用高斯模糊减少噪声
    3. 使用Canny算法检测边缘
    4. 定义感兴趣区域(ROI)
    5. 应用霍夫变换检测线段
    6. 绘制检测到的线段
    
    参数:
        image_path (str): 输入图像的路径
    
    返回:
        result (numpy.ndarray): 标注了检测线段的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊处理，减少噪声
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    edges = cv2.Canny(blur, 30, 100)
    
    # 定义感兴趣区域（ROI）的顶点
    height, width = edges.shape
    roi_vertices = np.array([[(width*0.1, height),      # 左下角
                             (width*0.45, height*0.5),   # 左上角
                             (width*0.55, height*0.5),   # 右上角
                             (width*0.9, height)]],      # 右下角
                           dtype=np.int32)
    
    # 创建掩码并应用
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 使用霍夫变换检测线段
    line_segments = hough_transform_segments(masked_edges,
                                          rho_resolution=1,
                                          theta_resolution=np.pi/180,
                                          threshold=50,
                                          min_line_length=200,
                                          max_line_gap=150)
    
    # 绘制检测到的线段
    result = draw_line_segments(image, line_segments)
    
    return result

def draw_line_segments(image, line_segments):
    """
    在图像上绘制检测到的线段。
    
    参数:
        image (numpy.ndarray): 原始图像
        line_segments (list): 线段列表，每个元素为[x1, y1, x2, y2]
    
    返回:
        result (numpy.ndarray): 绘制了线段的图像
    """
    result = image.copy()
    
    # 设置线条粗细
    line_thickness = 15
    
    for x1, y1, x2, y2 in line_segments:
        # 计算线段长度
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # 只绘制足够长的线段
        if length > 200:
            # 计算单位向量用于延长线段
            dx = (x2-x1)/length
            dy = (y2-y1)/length
            
            # 延长线段（前后各延长30%）
            extend = 0.3
            new_x1 = int(x1 - dx * length * extend)
            new_y1 = int(y1 - dy * length * extend)
            new_x2 = int(x2 + dx * length * extend)
            new_y2 = int(y2 + dy * length * extend)
            
            # 绘制主线
            cv2.line(result, 
                     (new_x1, new_y1), 
                     (new_x2, new_y2), 
                     (0, 255, 0), 
                     line_thickness)
            
            # 绘制额外的线使线条看起来更粗
            for offset in [-2, 2]:
                cv2.line(result, 
                         (new_x1, new_y1 + offset), 
                         (new_x2, new_y2 + offset), 
                         (0, 255, 0), 
                         line_thickness)
    
    return result

def main():
    """
    主函数，执行车道线检测并显示结果。
    """
    # 设置输入图像路径
    image_path = '/data1/home/chenbocheng/cbc/cv/experiment/2/老区.jpg'
    
    # 检测车道线
    result = detect_lane_lines(image_path)
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    # 保存结果
    cv2.imwrite('lane_detection_result.jpg', result)

if __name__ == '__main__':
    main()