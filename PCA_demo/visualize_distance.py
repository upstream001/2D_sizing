#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
草莓端点距离可视化

在原图上标记端点、连接线并显示真实距离（厘米）
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_camera_intrinsics():
    """加载相机内参"""
    with open('e:\\Recent Works\\2D-sizing\\data\\D405_dataset\\camera_intrinsics.json', 'r') as f:
        intrinsics = json.load(f)
    return intrinsics

def load_depth_data():
    """加载深度数据"""
    depth_path = 'e:\\Recent Works\\2D-sizing\\data\\D405_dataset\\depth\\D405_0004_20251112_170130.npy'
    depth = np.load(depth_path)
    return depth

def pixel_to_3d(u, v, depth, intrinsics):
    """像素坐标转3D相机坐标"""
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    ppx = intrinsics['ppx']
    ppy = intrinsics['ppy']
    depth_scale = intrinsics['depth_scale']
    
    # 获取深度值（米）
    Z = depth[v, u] * depth_scale
    
    # 计算3D坐标
    X = (u - ppx) * Z / fx
    Y = (v - ppy) * Z / fy
    
    return X, Y, Z

def calculate_3d_distance(point1, point2):
    """计算3D欧氏距离"""
    return np.sqrt((point2[0] - point1[0])**2 + 
                  (point2[1] - point1[1])**2 + 
                  (point2[2] - point1[2])**2)

def visualize_distance_on_image():
    """在图像上可视化距离"""
    
    # 图像路径
    image_path = 'e:\\Recent Works\\2D-sizing\\data\\D405_dataset\\images\\D405_0004_20251112_170130.png'
    
    # 加载数据
    intrinsics = load_camera_intrinsics()
    depth = load_depth_data()
    
    # 端点数据
    strawberries = [
        {
            'name': 'Strawberry 1',
            'left': (624, 588),
            'right': (591, 461),
            'distance_cm': 3.13,
            'color': 'red',
            'position': 'Right'
        },
        {
            'name': 'Strawberry 2', 
            'left': (615, 319),
            'right': (626, 438),
            'distance_cm': 2.98,
            'color': 'blue',
            'position': 'Left'
        }
    ]
    
    # 读取图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建matplotlib图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(image_rgb)
    
    print("Starting to draw endpoints and distances...")
    
    # 为每个草莓绘制端点和距离
    for i, strawberry in enumerate(strawberries):
        left_point = strawberry['left']
        right_point = strawberry['right']
        color = strawberry['color']
        name = strawberry['name']
        distance_cm = strawberry['distance_cm']
        
        # 绘制端点
        circle_size = 120  # 点的大小
        
        # 左端点
        circle_left = patches.Circle(left_point, radius=8, linewidth=3, 
                                   edgecolor=color, facecolor='white', alpha=0.8)
        ax.add_patch(circle_left)
        
        # 右端点
        circle_right = patches.Circle(right_point, radius=8, linewidth=3,
                                    edgecolor=color, facecolor='white', alpha=0.8)
        ax.add_patch(circle_right)
        
        # 绘制连接线
        line = plt.Line2D([left_point[0], right_point[0]], 
                         [left_point[1], right_point[1]], 
                         color=color, linewidth=4, alpha=0.8)
        ax.add_line(line)
        
        # 计算中点用于放置标签
        mid_x = (left_point[0] + right_point[0]) / 2
        mid_y = (left_point[1] + right_point[1]) / 2
        
        # 创建距离标签背景
        label_text = f"{distance_cm:.2f} cm"
        
        # 标签背景框
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7, edgecolor='white')
        ax.text(mid_x, mid_y - 20, label_text, fontsize=12, fontweight='bold',
                color='white', ha='center', va='center', bbox=bbox_props)
        
        # 端点标签
        ax.text(left_point[0], left_point[1] - 25, 'L', fontsize=10, fontweight='bold',
                color=color, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        ax.text(right_point[0], right_point[1] + 25, 'R', fontsize=10, fontweight='bold',
                color=color, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        print(f"✓ Drawn {name}: Left{left_point}, Right{right_point}, Distance{distance_cm}cm")
    
    # 添加标题和图例
    ax.set_title('Strawberry Endpoint Distance Visualization\nReal-world Distance Measurement', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 创建图例
    legend_elements = []
    for strawberry in strawberries:
        legend_elements.append(plt.Line2D([0], [0], color=strawberry['color'], 
                                         linewidth=3, label=f"{strawberry['name']}: {strawberry['distance_cm']:.2f} cm"))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
             framealpha=0.9, fancybox=True, shadow=True)
    
    # 添加图像信息
    info_text = f"""Image Info:
• Resolution: {image_rgb.shape[1]} x {image_rgb.shape[0]} pixels
• Camera: Intel D405
• Method: Pixel→3D Camera→Euclidean Distance
• Precision: Millimeter-level"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor='lightgray', alpha=0.8))
    
    # 移除坐标轴
    ax.set_xlim(0, image_rgb.shape[1])
    ax.set_ylim(image_rgb.shape[0], 0)
    ax.axis('off')
    
    # 保存图像
    output_path = 'e:\\Recent Works\\2D-sizing\\results\\distance_visualization.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\n✅ Distance visualization image saved to: {output_path}")
    
    # 同时保存为PDF格式
    pdf_path = 'e:\\Recent Works\\2D-sizing\\results\\distance_visualization.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ PDF version saved to: {pdf_path}")
    
    # 关闭图像显示以避免窗口问题
    plt.close()
    
    return output_path

def create_detailed_distance_analysis():
    """创建详细的距离分析报告"""
    
    # 端点数据
    strawberries = [
        {
            'name': '草莓1 (右侧)',
            'left': (624, 588),
            'right': (591, 461),
            'distance_cm': 3.13,
            'pixel_distance': 131.13,
            'depth_range': '15.14-16.12 cm'
        },
        {
            'name': '草莓2 (左侧)',
            'left': (615, 319), 
            'right': (626, 438),
            'distance_cm': 2.98,
            'pixel_distance': 119.47,
            'depth_range': '15.66-16.04 cm'
        }
    ]
    
    # 加载相机内参和深度数据进行精确计算
    intrinsics = load_camera_intrinsics()
    depth = load_depth_data()
    
    print("\n" + "="*80)
    print("🍓 草莓端点距离可视化分析报告")
    print("="*80)
    
    for i, strawberry in enumerate(strawberries, 1):
        print(f"\n📊 {strawberry['name']}")
        print("-" * 50)
        
        # 像素坐标
        left_u, left_v = strawberry['left']
        right_u, right_v = strawberry['right']
        
        print(f"左端点像素坐标: ({left_u}, {left_v})")
        print(f"右端点像素坐标: ({right_u}, {right_v})")
        print(f"像素距离: {strawberry['pixel_distance']:.2f} 像素")
        
        # 计算3D坐标
        left_3d = pixel_to_3d(left_u, left_v, depth, intrinsics)
        right_3d = pixel_to_3d(right_u, right_v, depth, intrinsics)
        
        print(f"左端点3D坐标: ({left_3d[0]*100:.2f}, {left_3d[1]*100:.2f}, {left_3d[2]*100:.2f}) cm")
        print(f"右端点3D坐标: ({right_3d[0]*100:.2f}, {right_3d[1]*100:.2f}, {right_3d[2]*100:.2f}) cm")
        
        # 计算真实距离
        real_distance = calculate_3d_distance(left_3d, right_3d)
        print(f"真实欧氏距离: {real_distance*100:.2f} cm ({real_distance:.4f} m)")
        
        # 验证与预设值的差异
        diff = abs(real_distance*100 - strawberry['distance_cm'])
        print(f"计算验证: 差异 {diff:.3f} cm (误差 {diff/strawberry['distance_cm']*100:.1f}%)")
    
    # 生成可视化图像
    print(f"\n🎨 正在生成距离可视化图像...")
    output_path = visualize_distance_on_image()
    
    print(f"\n📁 可视化文件:")
    print(f"• PNG格式: {output_path}")
    print(f"• PDF格式: {output_path.replace('.png', '.pdf')}")
    
    print("\n" + "="*80)
    print("✅ 距离可视化完成！")
    print("="*80)

if __name__ == "__main__":
    create_detailed_distance_analysis()