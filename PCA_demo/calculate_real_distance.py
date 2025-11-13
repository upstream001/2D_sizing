#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据深度信息和相机内参计算草莓左右端点间的真实距离

使用流程：
1. 先运行calculate_size.py获取端点坐标
2. 使用深度数据和相机内参转换为真实3D坐标
3. 计算两个端点之间的欧氏距离

作者：Claude
日期：2024
"""

import numpy as np
import json
import sys
import os
from pathlib import Path

# 添加上级目录到路径以便导入工具
sys.path.append(str(Path(__file__).parent.parent))

from distance_utils import load_data, get_pixel_real_distance

class RealDistanceCalculator:
    """真实距离计算器"""
    
    def __init__(self, camera_intrinsics_path, depth_image_path):
        """
        初始化计算器
        
        Args:
            camera_intrinsics_path: 相机内参JSON文件路径
            depth_image_path: 深度图像.npy文件路径
        """
        # 加载数据
        if not load_data(camera_intrinsics_path, depth_image_path):
            raise RuntimeError("无法加载相机内参或深度数据")
        
        print("✓ 真实距离计算器初始化完成")
    
    def get_endpoint_real_distance(self, left_endpoint, right_endpoint):
        """
        计算两个端点之间的真实距离
        
        Args:
            left_endpoint: 左端点坐标 (x, y)
            right_endpoint: 右端点坐标 (x, y)
            
        Returns:
            dict: 包含详细距离信息的字典
        """
        print(f"\n=== 计算端点真实距离 ===")
        print(f"左端点原始坐标: ({left_endpoint[0]:.2f}, {left_endpoint[1]:.2f})")
        print(f"右端点原始坐标: ({right_endpoint[0]:.2f}, {right_endpoint[1]:.2f})")
        
        # 将浮点数坐标转换为整数索引
        left_pixel = (int(round(left_endpoint[0])), int(round(left_endpoint[1])))
        right_pixel = (int(round(right_endpoint[0])), int(round(right_endpoint[1])))
        
        print(f"左端点像素索引: ({left_pixel[0]}, {left_pixel[1]})")
        print(f"右端点像素索引: ({right_pixel[0]}, {right_pixel[1]})")
        
        # 获取左端点的真实距离信息
        left_info = get_pixel_real_distance(left_pixel[0], left_pixel[1])
        if not left_info or not left_info.get('success', False):
            print(f"❌ 无法获取左端点距离信息: {left_info.get('error', '未知错误')}")
            return None
        
        # 获取右端点的真实距离信息
        right_info = get_pixel_real_distance(right_pixel[0], right_pixel[1])
        if not right_info or not right_info.get('success', False):
            print(f"❌ 无法获取右端点距离信息: {right_info.get('error', '未知错误')}")
            return None
        
        # 提取3D坐标
        left_3d = np.array([
            left_info['camera_coord']['X'],
            left_info['camera_coord']['Y'],
            left_info['camera_coord']['Z']
        ])
        
        right_3d = np.array([
            right_info['camera_coord']['X'],
            right_info['camera_coord']['Y'],
            right_info['camera_coord']['Z']
        ])
        
        # 计算欧氏距离
        euclidean_distance = np.linalg.norm(right_3d - left_3d)
        
        # 计算像素距离
        pixel_distance = np.sqrt(
            (right_endpoint[0] - left_endpoint[0])**2 + 
            (right_endpoint[1] - left_endpoint[1])**2
        )
        
        # 计算X、Y、Z方向的差值
        diff_x = right_3d[0] - left_3d[0]
        diff_y = right_3d[1] - left_3d[1]
        diff_z = right_3d[2] - left_3d[2]
        
        result = {
            'left_endpoint': {
                'pixel_coord': left_pixel,
                'camera_coord': left_3d,
                'depth_meters': left_info['depth_meters'],
                'distance_info': left_info
            },
            'right_endpoint': {
                'pixel_coord': right_pixel,
                'camera_coord': right_3d,
                'depth_meters': right_info['depth_meters'],
                'distance_info': right_info
            },
            'distance_analysis': {
                'euclidean_distance_meters': euclidean_distance,
                'pixel_distance': pixel_distance,
                'diff_x_meters': diff_x,
                'diff_y_meters': diff_y,
                'diff_z_meters': diff_z,
                'x_percentage': abs(diff_x) / euclidean_distance * 100,
                'y_percentage': abs(diff_y) / euclidean_distance * 100,
                'z_percentage': abs(diff_z) / euclidean_distance * 100
            }
        }
        
        return result
    
    def print_distance_report(self, result):
        """
        打印详细的距离计算报告
        
        Args:
            result: get_endpoint_real_distance的返回值
        """
        if not result:
            print("❌ 距离计算失败")
            return
        
        print("\n" + "="*60)
        print("草莓端点真实距离计算报告")
        print("="*60)
        
        # 左端点信息
        print(f"\n【左端点信息】")
        print(f"  像素坐标: {result['left_endpoint']['pixel_coord']}")
        print(f"  相机坐标: ({result['left_endpoint']['camera_coord'][0]:+.4f}, {result['left_endpoint']['camera_coord'][1]:+.4f}, {result['left_endpoint']['camera_coord'][2]:+.4f}) 米")
        print(f"  深度距离: {result['left_endpoint']['depth_meters']:.4f} 米")
        
        # 右端点信息
        print(f"\n【右端点信息】")
        print(f"  像素坐标: {result['right_endpoint']['pixel_coord']}")
        print(f"  相机坐标: ({result['right_endpoint']['camera_coord'][0]:+.4f}, {result['right_endpoint']['camera_coord'][1]:+.4f}, {result['right_endpoint']['camera_coord'][2]:+.4f}) 米")
        print(f"  深度距离: {result['right_endpoint']['depth_meters']:.4f} 米")
        
        # 距离分析
        print(f"\n【距离分析】")
        print(f"  欧氏距离: {result['distance_analysis']['euclidean_distance_meters']:.4f} 米")
        print(f"  像素距离: {result['distance_analysis']['pixel_distance']:.2f} 像素")
        print(f"  \n  坐标轴差值:")
        print(f"    X方向差值: {result['distance_analysis']['diff_x_meters']:+.4f} 米")
        print(f"    Y方向差值: {result['distance_analysis']['diff_y_meters']:+.4f} 米")
        print(f"    Z方向差值: {result['distance_analysis']['diff_z_meters']:+.4f} 米")
        
        print(f"\n  各轴向贡献比例:")
        print(f"    X轴贡献: {result['distance_analysis']['x_percentage']:.1f}%")
        print(f"    Y轴贡献: {result['distance_analysis']['y_percentage']:.1f}%")
        print(f"    Z轴贡献: {result['distance_analysis']['z_percentage']:.1f}%")
        
        # 总结
        print(f"\n【总结】")
        print(f"  左端点到相机的距离: {result['left_endpoint']['depth_meters']:.4f} 米")
        print(f"  右端点到相机的距离: {result['right_endpoint']['depth_meters']:.4f} 米")
        print(f"  两端点间的真实距离: {result['distance_analysis']['euclidean_distance_meters']:.4f} 米")
        print("="*60)

def calculate_endpoints_from_size_result(size_result, strawberry_index=0):
    """
    从calculate_size.py的结果中提取端点坐标
    
    Args:
        size_result: calculate_strawberry_size的返回值
        strawberry_index: 草莓索引（从0开始）
        
    Returns:
        tuple: (左端点坐标, 右端点坐标) 或 (None, None) 如果提取失败
    """
    try:
        # 获取指定草莓的线段结果
        if strawberry_index >= len(size_result['line_results_list']):
            print(f"❌ 草莓索引 {strawberry_index} 超出范围 (共有 {len(size_result['line_results_list'])} 个草莓)")
            return None, None
        
        line_data = size_result['line_results_list'][strawberry_index]
        
        if line_data['line_points'] is None:
            print(f"❌ 草莓 {strawberry_index + 1} 没有有效的线段端点")
            return None, None
        
        # 提取端点坐标
        # line_points包含所有线段上的点，前后端点为第一个和最后一个
        endpoints = line_data['line_points']
        if len(endpoints) < 2:
            print(f"❌ 线段端点数量不足: {len(endpoints)}")
            return None, None
        
        left_endpoint = endpoints[0]  # 线段起点（左端点）
        right_endpoint = endpoints[-1]  # 线段终点（右端点）
        
        print(f"✓ 提取到草莓 {strawberry_index + 1} 的端点坐标")
        print(f"  左端点: ({left_endpoint[0]}, {left_endpoint[1]})")
        print(f"  右端点: ({right_endpoint[0]}, {right_endpoint[1]})")
        
        return left_endpoint, right_endpoint
        
    except Exception as e:
        print(f"❌ 提取端点坐标失败: {e}")
        return None, None

def main():
    """主函数"""
    print("=== 草莓端点真实距离计算器 ===")
    
    # 文件路径配置
    camera_intrinsics_path = r"E:\Recent Works\2D-sizing\data\D405_dataset\camera_intrinsics.json"
    depth_image_path = r"E:\Recent Works\2D-sizing\data\D405_dataset\depth\D405_0004_20251112_170130.npy"
    image_path = r"E:\Recent Works\2D-sizing\data\D405_dataset\images\D405_0004_20251112_170130.png"
    
    print(f"使用文件:")
    print(f"  相机内参: {camera_intrinsics_path}")
    print(f"  深度图像: {depth_image_path}")
    print(f"  图像文件: {image_path}")
    
    try:
        # 1. 先运行尺寸分析获取端点坐标
        print(f"\n=== 步骤1: 运行草莓尺寸分析 ===")
        from calculate_size import calculate_strawberry_size
        
        size_result = calculate_strawberry_size(
            image_path, 
            save_results=True, 
            output_dir="e:\\Recent Works\\2D-sizing\\results"
        )
        
        if not size_result:
            print("❌ 尺寸分析失败，无法继续")
            return
        
        # 2. 初始化真实距离计算器
        print(f"\n=== 步骤2: 初始化真实距离计算器 ===")
        calculator = RealDistanceCalculator(camera_intrinsics_path, depth_image_path)
        
        # 3. 逐个计算每个草莓的端点距离
        print(f"\n=== 步骤3: 计算各草莓端点真实距离 ===")
        
        for i in range(size_result['strawberry_count']):
            print(f"\n--- 草莓 {i+1} ---")
            
            # 提取端点坐标
            left_endpoint, right_endpoint = calculate_endpoints_from_size_result(size_result, i)
            
            if left_endpoint is None or right_endpoint is None:
                print(f"❌ 无法获取草莓 {i+1} 的有效端点坐标")
                continue
            
            # 计算真实距离
            distance_result = calculator.get_endpoint_real_distance(left_endpoint, right_endpoint)
            
            if distance_result:
                # 打印详细报告
                calculator.print_distance_report(distance_result)
                
                # 保存结果到文件
                result_file = f"e:\\Recent Works\\2D-sizing\\results\\strawberry_{i+1}_real_distance.txt"
                with open(result_file, 'w', encoding='utf-8') as f:
                    # 重定向输出到文件
                    import io
                    from contextlib import redirect_stdout
                    
                    output = io.StringIO()
                    with redirect_stdout(output):
                        calculator.print_distance_report(distance_result)
                    
                    f.write(output.getvalue())
                
                print(f"\n✓ 详细结果已保存到: {result_file}")
            else:
                print(f"❌ 草莓 {i+1} 距离计算失败")
        
        print(f"\n=== 所有草莓端点真实距离计算完成 ===")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()