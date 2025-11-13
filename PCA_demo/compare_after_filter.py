#!/usr/bin/env python3
"""
草莓分割结果边缘平滑处理脚本
对分割后的掩码进行边缘平滑，使其更像椭圆，并对比处理前后的效果
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入草莓分割器
import sys
sys.path.append('e:\\Recent Works\\2D-sizing')
from strawberry_segmentation import StrawberrySegmentor

class StrawberryMaskSmoother:
    """草莓掩码边缘平滑器"""
    
    def __init__(self, model_path=None):
        self.segmentor = StrawberrySegmentor("..\weights\yolov11n-seg-086.pt")
    
    def smooth_mask_morphological(self, mask):
        """使用形态学操作平滑掩码"""
        # 将掩码转换为二进制图像
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 创建核来进行形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 闭运算：先膨胀再腐蚀，去除小洞
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 开运算：先腐蚀再膨胀，平滑边缘
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # 转换为浮点掩码
        smoothed_mask = opened.astype(np.float32) / 255.0
        
        return smoothed_mask, binary_mask, opened
    
    def smooth_mask_gaussian(self, mask, kernel_size=15, sigma=3):
        """使用高斯模糊平滑掩码"""
        # 将掩码转换为二进制图像
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(binary_mask, (kernel_size, kernel_size), sigma)
        
        # 阈值处理
        _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # 转换为浮点掩码
        smoothed_mask = thresholded.astype(np.float32) / 255.0
        
        return smoothed_mask, binary_mask, thresholded
    
    def smooth_mask_ellipse_fitting(self, mask):
        """使用椭圆拟合法平滑掩码"""
        # 将掩码转换为二进制图像
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask, binary_mask, binary_mask
        
        # 找到最大的轮廓（假设是草莓）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 椭圆拟合
        if len(largest_contour) >= 5:  # 至少需要5个点来拟合椭圆
            ellipse = cv2.fitEllipse(largest_contour)
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            
            # 创建椭圆掩码
            rows, cols = mask.shape
            smoothed_mask = np.zeros((rows, cols), dtype=np.uint8)
            
            # 绘制填充的椭圆
            cv2.ellipse(smoothed_mask, ellipse, 255, -1)
            
            # 转换为浮点掩码
            smoothed_mask = smoothed_mask.astype(np.float32) / 255.0
            
            return smoothed_mask, binary_mask, smoothed_mask * 255
        
        return mask, binary_mask, binary_mask
    
    def smooth_mask_advanced(self, mask):
        """温和边缘平滑方法：只平滑突出部分，保持整体轮廓"""
        # 步骤1：转换为二进制掩码
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 步骤2：轻微的形态学操作 - 只处理小的突出
        # 使用小核的开运算来去除小的突出
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
        
        # 步骤3：轻微的闭运算来填补小缝隙
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
        
        # 步骤4：局部平滑 - 使用高斯滤波但保持边缘锐利
        # 只对边界区域进行轻微的模糊
        mask_float = closed.astype(np.float32) / 255.0
        
        # 创建掩码区域和边缘区域的分离
        kernel_blur = cv2.GaussianBlur(mask_float, (3, 3), 0.5)
        
        # 只在边缘区域进行混合
        edge_mask = cv2.Canny(closed, 50, 150)
        edge_dilated = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
        
        # 在边缘区域应用轻微的模糊
        result = mask_float.copy()
        result[edge_dilated > 0] = (result[edge_dilated > 0] * 0.7 + 
                                   kernel_blur[edge_dilated > 0] * 0.3)
        
        # 确保结果在合理范围内
        result = np.clip(result, 0, 1)
        
        return result, binary_mask, (result * 255).astype(np.uint8)
    
    def smooth_mask_gentle(self, mask):
        """极温和平滑方法：只处理非常小的突出"""
        # 转换为二进制掩码
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 只使用最小的核进行轻微的开运算
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_tiny)
        
        # 转换为浮点掩码
        smoothed_mask = opened.astype(np.float32) / 255.0
        
        return smoothed_mask, binary_mask, opened
    
    def visualize_comparison(self, image, original_mask, smoothed_mask, method_name, save_path=None):
        """可视化对比结果"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行：原始图像、原始掩码、叠加效果
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原始图像', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 原始掩码
        axes[0, 1].imshow(original_mask, cmap='gray')
        axes[0, 1].set_title('原始分割掩码', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 叠加效果
        image_with_mask = image.copy()
        overlay = np.zeros_like(image)
        overlay[original_mask > 0.5] = [255, 0, 0]  # 红色
        axes[0, 2].imshow(image_with_mask)
        axes[0, 2].imshow(overlay, alpha=0.4)
        axes[0, 2].set_title('原始分割叠加', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # 第二行：平滑方法、平滑掩码、平滑叠加效果
        axes[1, 0].text(0.5, 0.5, f'{method_name}', ha='center', va='center',
                       transform=axes[1, 0].transAxes, fontsize=16, fontweight='bold')
        axes[1, 0].set_title('平滑方法', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 平滑掩码
        axes[1, 1].imshow(smoothed_mask, cmap='gray')
        axes[1, 1].set_title('平滑后掩码', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # 平滑叠加效果
        image_with_smooth = image.copy()
        overlay_smooth = np.zeros_like(image)
        overlay_smooth[smoothed_mask > 0.5] = [0, 255, 0]  # 绿色
        axes[1, 2].imshow(image_with_smooth)
        axes[1, 2].imshow(overlay_smooth, alpha=0.4)
        axes[1, 2].set_title('平滑分割叠加', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 可视化结果已保存到: {save_path}")
        
        # 注释掉plt.show()以避免阻塞脚本执行
        # plt.show()
    
    def process_image(self, image_path, method='advanced', save_path=None):
        """处理单张图片"""
        print(f"正在处理图片: {image_path}")
        
        # 读取原始图片
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取分割结果
        result = self.segmentor.predict(image_path)
        
        if result.masks is None:
            print("✗ 未检测到草莓")
            return
        
        # 处理第一个掩码（假设只有一个大草莓）
        masks = result.masks.data.cpu().numpy()  # 获取所有掩码数据
        original_mask = masks[0]  # 使用第一个掩码
        
        # 确保掩码是正确的类型和形状
        original_mask = original_mask.astype(np.uint8)  # 转换为uint8类型
        
        # 调整掩码尺寸以匹配原始图片
        original_mask_resized = cv2.resize(original_mask, (image.shape[1], image.shape[0]))
        
        # 根据方法选择平滑算法
        method_names = {
            'morphological': '形态学操作平滑',
            'gaussian': '高斯模糊平滑',
            'ellipse': '椭圆拟合平滑',
            'advanced': '温和边缘平滑',
            'gentle': '极温和平滑'
        }
        
        if method == 'morphological':
            smoothed_mask, binary_orig, binary_smooth = self.smooth_mask_morphological(original_mask_resized)
        elif method == 'gaussian':
            smoothed_mask, binary_orig, binary_smooth = self.smooth_mask_gaussian(original_mask_resized)
        elif method == 'ellipse':
            smoothed_mask, binary_orig, binary_smooth = self.smooth_mask_ellipse_fitting(original_mask_resized)
        elif method == 'advanced':
            smoothed_mask, binary_orig, binary_smooth = self.smooth_mask_advanced(original_mask_resized)
        elif method == 'gentle':
            smoothed_mask, binary_orig, binary_smooth = self.smooth_mask_gentle(original_mask_resized)
        else:
            print(f"✗ 未知的平滑方法: {method}")
            return
        
        # 可视化对比
        method_name = method_names.get(method, method)
        
        # 如果没有指定保存路径，生成默认路径
        if save_path is None:
            # 保存到results文件夹
            results_path = Path("..") / "results"
            results_path.mkdir(parents=True, exist_ok=True)  # 确保文件夹存在
            input_path = Path(image_path)
            save_path = results_path / f"comparison_{input_path.stem}_{method}.png"
        
        self.visualize_comparison(image, original_mask_resized, smoothed_mask, method_name, save_path)
        
        # 计算平滑效果指标
        self.calculate_quality_metrics(original_mask_resized, smoothed_mask)
        
        return {
            'original_mask': original_mask_resized,
            'smoothed_mask': smoothed_mask,
            'method': method,
            'method_name': method_name
        }
    
    def calculate_quality_metrics(self, original_mask, smoothed_mask):
        """计算质量指标"""
        # 计算面积变化
        orig_area = np.sum(original_mask > 0.5)
        smooth_area = np.sum(smoothed_mask > 0.5)
        area_change = (smooth_area - orig_area) / orig_area * 100
        
        # 计算周长（轮廓长度）
        orig_contours, _ = cv2.findContours((original_mask * 255).astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smooth_contours, _ = cv2.findContours((smoothed_mask * 255).astype(np.uint8), 
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if orig_contours and smooth_contours:
            orig_perimeter = cv2.arcLength(max(orig_contours, key=cv2.contourArea), True)
            smooth_perimeter = cv2.arcLength(max(smooth_contours, key=cv2.contourArea), True)
            perimeter_change = (smooth_perimeter - orig_perimeter) / orig_perimeter * 100
        else:
            perimeter_change = 0
        
        print(f"\n质量评估:")
        print(f"面积变化: {area_change:.2f}%")
        print(f"周长变化: {perimeter_change:.2f}%")
        print(f"原始面积: {orig_area} 像素")
        print(f"平滑面积: {smooth_area} 像素")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='草莓分割结果边缘平滑处理工具')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图片路径')
    parser.add_argument('--method', type=str, default='gentle',
                       choices=['morphological', 'gaussian', 'ellipse', 'advanced', 'gentle'],
                       help='平滑方法：morphological(形态学)、gaussian(高斯)、ellipse(椭圆拟合)、advanced(温和平滑)、gentle(极温和)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片保存路径')
    parser.add_argument('--model', type=str, default="weights\yolov11n-seg-086.pt",
                       help='模型文件路径（可选）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("草莓分割结果边缘平滑处理工具")
    print("=" * 60)
    print(f"输入图片: {args.input}")
    print(f"平滑方法: {args.method}")
    print("=" * 60)
    
    try:
        # 初始化平滑器
        smoother = StrawberryMaskSmoother()
        
        # 处理图片
        result = smoother.process_image(args.input, args.method, args.output)
        
        if result:
            print("✓ 处理完成！")
        else:
            print("✗ 处理失败！")
            
    except Exception as e:
        print(f"✗ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()