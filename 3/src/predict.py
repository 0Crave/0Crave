import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import DigitRecognizer
import os

def segment_characters(image_path):
    """使用投影分析法分割字符"""
    # 读取图像为灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 调整图片大小，保持比例
    height = 280  # 设定固定高度
    ratio = height / img.shape[0]  # 计算缩放比例
    width = int(img.shape[1] * ratio)  # 按比例计算新的宽度
    img = cv2.resize(img, (width, height))  # 调整图片大小
    cv2.imwrite('./debug/1_resized.png', img)  # 保存调整后的图片
    
    # 图像二值化：使用Otsu算法自动找到最佳阈值
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('./debug/2_binary.png', binary)
    
    # 图像去噪：使用形态学操作
    kernel = np.ones((3,3), np.uint8)  # 创建3x3的核
    # 闭运算：先膨胀后腐蚀，填充小孔
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # 开运算：先腐蚀后膨胀，去除小噪点
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('./debug/3_denoised.png', binary)
    
    # 反转图像：确保文字为白色（255），背景为黑色（0）
    binary = cv2.bitwise_not(binary)
    cv2.imwrite('./debug/4_inverted.png', binary)
    
    # 垂直投影：计算每一列的白色像素数量
    v_proj = np.sum(binary, axis=0)
    
    # 可视化垂直投影
    proj_visual = np.zeros((300, width), dtype=np.uint8)  # 创建投影图
    for i, count in enumerate(v_proj):
        # 绘制投影线：从底部向上画线，高度与像素数量成比例
        cv2.line(proj_visual, (i, 299), (i, 299-int(count/100)), 255, 1)
    cv2.imwrite('./debug/5_vertical_projection.png', proj_visual)
    
    # 寻找字符区域
    char_positions = []
    start = None
    min_width = 10  # 最小字符宽度
    min_height = 50  # 最小字符高度
    
    # 通过投影找到字符的起始和结束位置
    for i in range(len(v_proj)):
        if v_proj[i] > 0 and start is None:  # 找到字符起始位置
            start = i
        elif v_proj[i] == 0 and start is not None:  # 找到字符结束位置
            width = i - start
            region = binary[:, start:i]  # 提取当前区域
            h_proj = np.sum(region, axis=1)  # 计算水平投影
            height = np.sum(h_proj > 0)  # 计算字符实际高度
            
            # 判断是否为有效字符：要么宽度够大，要么又窄又高（如数字1）
            if width >= min_width or (width >= 5 and height >= min_height):
                char_positions.append((start, i))
            start = None
    
    # 处理最后一个字符（如果图像末尾有字符）
    if start is not None:
        width = len(v_proj) - start
        region = binary[:, start:]
        h_proj = np.sum(region, axis=1)
        height = np.sum(h_proj > 0)
        if width >= min_width or (width >= 5 and height >= min_height):
            char_positions.append((start, len(v_proj)-1))
    
    # 提取每个字符
    characters = []
    for idx, (start, end) in enumerate(char_positions):
        char_img = binary[:, start:end]  # 提取字符图像
        
        # 获取字符的垂直范围（去除上下空白）
        h_proj = np.sum(char_img, axis=1)
        top = next(i for i, v in enumerate(h_proj) if v > 0)  # 找到顶部
        bottom = len(h_proj) - next(i for i, v in enumerate(h_proj[::-1]) if v > 0)  # 找到底部
        
        # 提取字符（去除上下空白后的图像）
        char_img = binary[top:bottom, start:end]
        
        # 计算宽高比
        height, width = char_img.shape
        aspect_ratio = width / height
        
        # 对窄字符（如数字1）增加水平padding
        if aspect_ratio < 0.5:  # 如果字符太窄
            target_width = int(height * 0.7)  # 目标宽度为高度的0.7倍
            padding_width = (target_width - width) // 2  # 计算需要添加的padding
            # 添加左右padding
            char_img = cv2.copyMakeBorder(
                char_img,
                0, 0,  # 上下不加padding
                padding_width, padding_width,  # 左右加padding
                cv2.BORDER_CONSTANT,
                value=0
            )
        
        # 添加统一的边距
        height, width = char_img.shape
        margin = max(5, int(min(height, width) * 0.2))  # 计算边距
        char_img = cv2.copyMakeBorder(
            char_img,
            margin, margin, margin, margin,  # 四周添加相同的边距
            cv2.BORDER_CONSTANT,
            value=0
        )
        
        # 保存分割出的字符图像
        cv2.imwrite(f'./debug/characters/char_{idx}.png', char_img)
        
        # 在原图上标注字符位置
        if aspect_ratio < 0.5:  # 对于窄字符，显示padding后的区域
            extended_start = max(0, start - padding_width)
            extended_end = min(binary.shape[1], end + padding_width)
            cv2.rectangle(img, (extended_start, top), (extended_end, bottom), (0, 255, 0), 2)
        else:  # 对于普通字符，显示原始区域
            cv2.rectangle(img, (start, top), (end, bottom), (0, 255, 0), 2)
        
        characters.append(char_img)
    
    # 保存标注了字符位置的原图
    cv2.imwrite('./debug/6_detected_regions.png', img)
    
    return characters

def preprocess_character(char_img):
    """预处理单个字符图像"""
    # 调整大小到28x28
    char_img = cv2.resize(char_img, (28, 28))
    
    # 确保是二值图像
    _, char_img = cv2.threshold(char_img, 128, 255, cv2.THRESH_BINARY)
    
    # 转换为PIL Image
    char_img = Image.fromarray(char_img)
    
    # 标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(char_img).unsqueeze(0)

def predict_student_id(image_path):
    """预测学号"""
    # 创建debug目录
    os.makedirs('./debug', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = DigitRecognizer().to(device)
    model.load_state_dict(torch.load(
        './model/saved_model/digit_recognizer_best.pth',
        map_location=device
    ))
    model.eval()
    
    # 分割字符
    characters = segment_characters(image_path)
    
    # 识别每个字符
    predictions = []
    for idx, char_img in enumerate(characters):
        # 预处理字符
        char_tensor = preprocess_character(char_img).to(device)
        
        # 保存预处理后的图
        preprocessed_img = char_tensor.cpu().numpy()[0, 0] * 0.3081 + 0.1307
        preprocessed_img = (preprocessed_img * 255).astype(np.uint8)
        cv2.imwrite(f'./debug/characters/char_{idx}_preprocessed.png', preprocessed_img)
        
        # 预测
        with torch.no_grad():
            output = model(char_tensor)
            prediction = output.argmax(dim=1).item()
            predictions.append(str(prediction))
    
    # 组合结果
    student_id = ''.join(predictions)
    
    # 验证结果
    if len(student_id) < 8:
        print(f"警告：只检测到 {len(student_id)} 个数字")
    
    return student_id

if __name__ == '__main__':
    image_path = './学号.png'
    student_id = predict_student_id(image_path)
    print(f'识别的学号是: {student_id}')