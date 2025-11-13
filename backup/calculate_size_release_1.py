#!/usr/bin/env python3
"""
草莓尺寸计算 - 基于PCA分析的尺寸测量
使用find_points_v2.py获取草莓掩码信息和端点数据，进行尺寸计算
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加上级目录到路径以导入strawberry_segmentation
sys.path.append(str(Path(__file__).parent.parent))

# 导入find_points_v2中的类
from PCA_demo.find_points_v2 import StrawberryPointFinder

# 导入真实距离计算相关模块
from distance_utils import load_data, get_pixel_real_distance

def calculate_endpoint_real_distance(left_endpoint, right_endpoint):
    """
    计算两个端点之间的真实距离
    
    Args:
        left_endpoint: 左端点坐标 (x, y)
        right_endpoint: 右端点坐标 (x, y)
        
    Returns:
        dict: 包含详细距离信息的字典
    """
    print(f"  计算端点真实距离:")
    print(f"    左端点: ({left_endpoint[0]:.2f}, {left_endpoint[1]:.2f})")
    print(f"    右端点: ({right_endpoint[0]:.2f}, {right_endpoint[1]:.2f})")
    
    # 将浮点数坐标转换为整数索引
    left_pixel = (int(round(left_endpoint[0])), int(round(left_endpoint[1])))
    right_pixel = (int(round(right_endpoint[0])), int(round(right_endpoint[1])))
    
    # 获取左端点的真实距离信息
    left_info = get_pixel_real_distance(left_pixel[0], left_pixel[1])
    if not left_info or not left_info.get('success', False):
        print(f"    ❌ 无法获取左端点距离信息: {left_info.get('error', '未知错误')}")
        return None
    
    # 获取右端点的真实距离信息
    right_info = get_pixel_real_distance(right_pixel[0], right_pixel[1])
    if not right_info or not right_info.get('success', False):
        print(f"    ❌ 无法获取右端点距离信息: {right_info.get('error', '未知错误')}")
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
        'euclidean_distance': euclidean_distance,
        'pixel_distance': pixel_distance,
        'diff_x': diff_x,
        'diff_y': diff_y,
        'diff_z': diff_z,
        'x_percentage': abs(diff_x) / euclidean_distance * 100,
        'y_percentage': abs(diff_y) / euclidean_distance * 100,
        'z_percentage': abs(diff_z) / euclidean_distance * 100
    }
    
    return result

def calculate_strawberry_size(image_path, save_results=True, output_dir="results", 
                             camera_intrinsics_path=None, depth_image_path=None):
    """
    计算草莓尺寸的主要函数
    
    参数:
        image_path: 输入图像路径
        save_results: 是否保存结果
        output_dir: 输出目录
        camera_intrinsics_path: 相机内参JSON文件路径
        depth_image_path: 深度图像.npy文件路径
    
    返回:
        dict: 包含所有计算结果的字典
    """
    
    # 初始化草莓点查找器，使用与strawberry_segmentation.py相同的模型
    model_path = r"E:\Recent Works\2D-sizing\weights\yolov11m-2c-seg.pt"
    finder = StrawberryPointFinder(model_path)
    
    print("=== 开始草莓尺寸计算 ===")
    
    # 1. 加载深度数据和相机内参（用于真实距离计算）
    print("1. 加载深度数据和相机内参...")
    
    # 设置默认路径
    if camera_intrinsics_path is None:
        camera_intrinsics_path = r"e:\Recent Works\2D-sizing\data\D405_dataset\camera_intrinsics.json"
    
    # 从图像路径推断深度图像路径
    if depth_image_path is None:
        image_name = Path(image_path).stem
        depth_image_name = f"{image_name}.npy"
        depth_image_path = f"e:\\Recent Works\\2D-sizing\\data\\D405_dataset\\depth\\{depth_image_name}"
    
    # 加载数据
    if not load_data(camera_intrinsics_path, depth_image_path):
        print("⚠️ 无法加载深度数据或相机内参，将跳过真实距离计算")
        depth_data_available = False
    else:
        depth_data_available = True
        print("✓ 深度数据和相机内参加载成功")
    
    # 2. 获取掩码信息
    print("2. 获取草莓掩码信息...")
    points_list, image = finder.get_mask_points(image_path, conf_threshold=0.5)
    
    if points_list is None:
        print("❌ 无法获取草莓掩码信息")
        return None
    
    print(f"✓ 成功获取 {len(points_list)} 个草莓的掩码信息")
    
    # 3. 进行PCA分析
    print("3. 进行PCA分析...")
    pca_results_list = finder.perform_pca(points_list)
    
    if not pca_results_list:
        print("❌ PCA分析失败")
        return None
    
    print(f"✓ 成功完成 {len(pca_results_list)} 个草莓的PCA分析")
    
    # 4. 寻找PC1端点（上下端点）
    print("4. 寻找PC1端点...")
    pc1_endpoints_list = []
    
    for i, (points, pca_result) in enumerate(zip(points_list, pca_results_list)):
        top_endpoint, bottom_endpoint, pc1_length = finder.find_pc1_endpoints(points, pca_result)
        
        pc1_endpoints_data = {
            'top_endpoint': top_endpoint,
            'bottom_endpoint': bottom_endpoint,
            'pc1_length': pc1_length,
            'strawberry_id': i + 1
        }
        
        pc1_endpoints_list.append(pc1_endpoints_data)
        print(f"✓ 草莓 {i+1}: PC1长度 = {pc1_length:.2f} 像素")
    
    # 5. 寻找最长垂直线段端点（左右端点）
    print("5. 寻找最长垂直线段端点...")
    line_results_list = []
    
    for i, (points, pca_result) in enumerate(zip(points_list, pca_results_list)):
        line_points, max_length, best_position = finder.find_longest_perpendicular_line(
            points, pca_result, step_size=1, search_range=50
        )
        
        line_results_data = {
            'line_points': line_points,
            'max_length': max_length,
            'best_position': best_position,
            'strawberry_id': i + 1
        }
        
        line_results_list.append(line_results_data)
        
        if line_points is not None:
            print(f"✓ 草莓 {i+1}: 最长垂直线段长度 = {max_length:.2f} 像素")
        else:
            print(f"❌ 草莓 {i+1}: 未找到有效的垂直线段")
    
    # 6. 计算真实距离（如果深度数据可用）
    real_distance_results = []
    if depth_data_available:
        print("6. 计算端点真实距离...")
        
        for i, line_data in enumerate(line_results_list):
            if line_data['line_points'] is not None:
                # line_points是包含两个端点坐标的元组：(point1, point2)
                left_endpoint = line_data['line_points'][0]  # 第一个端点
                right_endpoint = line_data['line_points'][1]  # 第二个端点
                
                # 计算真实距离
                real_distance_info = calculate_endpoint_real_distance(left_endpoint, right_endpoint)
                
                if real_distance_info:
                    real_distance_results.append({
                        'strawberry_id': i + 1,
                        'real_distance_m': real_distance_info['euclidean_distance'],
                        'real_distance_cm': real_distance_info['euclidean_distance'] * 100,
                        'distance_details': real_distance_info
                    })
                    
                    print(f"✓ 草莓 {i+1}: 真实距离 = {real_distance_info['euclidean_distance']*100:.2f} 厘米")
                else:
                    real_distance_results.append({
                        'strawberry_id': i + 1,
                        'real_distance_m': None,
                        'real_distance_cm': None,
                        'distance_details': None
                    })
                    print(f"❌ 草莓 {i+1}: 真实距离计算失败")
            else:
                real_distance_results.append({
                    'strawberry_id': i + 1,
                    'real_distance_m': None,
                    'real_distance_cm': None,
                    'distance_details': None
                })
    else:
        print("6. 跳过真实距离计算（深度数据不可用）")
        for i in range(len(line_results_list)):
            real_distance_results.append({
                'strawberry_id': i + 1,
                'real_distance_m': None,
                'real_distance_cm': None,
                'distance_details': None
            })
    
    # 7. 整理结果数据
    print("7. 整理计算结果...")
    
    results = {
        'image_path': image_path,
        'image': image,
        'strawberry_count': len(points_list),
        'points_list': points_list,
        'pca_results_list': pca_results_list,
        'pc1_endpoints_list': pc1_endpoints_list,
        'line_results_list': line_results_list,
        'real_distance_results': real_distance_results,
        'depth_data_used': depth_data_available,
        'timestamp': np.datetime64('now')
    }
    
    # 8. 输出尺寸信息摘要
    print("\n=== 草莓尺寸计算结果摘要 ===")
    for i, (pc1_data, line_data, real_distance_data) in enumerate(zip(pc1_endpoints_list, line_results_list, real_distance_results)):
        print(f"草莓 {i+1}:")
        print(f"  - PC1方向长度（上下端点距离）: {pc1_data['pc1_length']:.2f} 像素")
        
        if line_data['line_points'] is not None:
            print(f"  - 最长垂直线段长度（左右端点距离）: {line_data['max_length']:.2f} 像素")
            print(f"  - 长宽比: {pc1_data['pc1_length']/line_data['max_length']:.2f}")
            
            # 如果有真实距离数据，显示出来
            if real_distance_data and real_distance_data['real_distance_m'] is not None:
                print(f"  - 左右端点真实距离: {real_distance_data['real_distance_cm']:.2f} 厘米")
                print(f"  - 像素到真实距离转换比: {real_distance_data['real_distance_m']/line_data['max_length']*1000:.4f} 毫米/像素")
            else:
                print(f"  - 左右端点真实距离: 无法计算")
        else:
            print(f"  - 最长垂直线段: 未找到有效线段")
        print()
    
    # 7. 可视化结果
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_size_analysis.png")
        
        print("7. 生成可视化结果...")
        finder.visualize_results(
            points_list, image, pca_results_list, 
            pc1_endpoints_list, line_results_list,
            real_distance_results=real_distance_results,
            save_path=output_path
        )
        
        print(f"✓ 可视化结果已保存到: {output_path}")
    
    return results

def process_folder(folder_path, output_dir="results"):
    """
    批量处理文件夹中的所有图像
    
    参数:
        folder_path: 输入文件夹路径
        output_dir: 输出目录
    """
    
    print(f"=== 批量处理文件夹: {folder_path} ===")
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ 在文件夹 {folder_path} 中未找到图像文件")
        return
    
    print(f"✓ 找到 {len(image_files)} 个图像文件")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个图像
    all_results = []
    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n--- 处理图像 {i}/{len(image_files)}: {image_path.name} ---")
        
        try:
            result = calculate_strawberry_size(str(image_path), save_results=True, output_dir=output_dir)
            if result:
                all_results.append(result)
                print(f"✓ 图像 {image_path.name} 处理成功")
            else:
                print(f"❌ 图像 {image_path.name} 处理失败")
        except Exception as e:
            print(f"❌ 处理图像 {image_path.name} 时发生错误: {e}")
    
    print(f"\n=== 批量处理完成 ===")
    print(f"成功处理 {len(all_results)}/{len(image_files)} 个图像")

def main():
    """主函数"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='草莓尺寸计算工具')
    parser.add_argument('--input', type=str, default=r'E:\Recent Works\2D-sizing\data\D405_dataset\images\D405_0004_20251112_170130.png',
                       help='输入图像路径或文件夹路径')
    parser.add_argument('--output', type=str, default=r'E:\Recent Works\2D-sizing\results',
                       help='输出目录 (默认: results)')
    parser.add_argument('--mode', type=str, choices=['single', 'folder'], 
                       default='single', help='处理模式: single(单文件) 或 folder(文件夹)')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_dir = args.output
    
    if args.mode == 'single':
        # 单文件处理
        if not os.path.exists(input_path):
            print(f"❌ 文件不存在: {input_path}")
            return
        
        result = calculate_strawberry_size(input_path, save_results=True, output_dir=output_dir)
        
        if result:
            print("\n✅ 草莓尺寸计算完成!")
        else:
            print("\n❌ 草莓尺寸计算失败!")
    
    elif args.mode == 'folder':
        # 文件夹批量处理
        if not os.path.exists(input_path):
            print(f"❌ 文件夹不存在: {input_path}")
            return
        
        process_folder(input_path, output_dir)

if __name__ == "__main__":
    main()