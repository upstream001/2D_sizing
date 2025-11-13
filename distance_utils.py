"""
像素距离获取工具函数
提供便捷的像素坐标到真实距离转换功能

使用方法：
    from distance_utils import get_pixel_real_distance, print_distance_info
    
    # 获取单个像素距离
    result = get_pixel_real_distance(640, 360)
    print_distance_info(result)
    
    # 批量获取距离
    coords = [(100, 100), (200, 200), (300, 300)]
    for coord in coords:
        result = get_pixel_real_distance(coord[0], coord[1])
        print_distance_info(result)
"""

import numpy as np
import json

# 全局变量存储相机内参和深度数据
_camera_intrinsics = None
_depth_image = None

def load_data(camera_intrinsics_path=None, depth_image_path=None):
    """
    加载相机内参和深度图像数据
    
    Args:
        camera_intrinsics_path: 相机内参JSON文件路径
        depth_image_path: 深度图像.npy文件路径
        
    Returns:
        bool: 加载是否成功
    """
    global _camera_intrinsics, _depth_image
    
    if camera_intrinsics_path is None:
        camera_intrinsics_path = r"e:\Recent Works\2D-sizing\data\D405_dataset\camera_intrinsics.json"
    
    if depth_image_path is None:
        depth_image_path = r"e:\Recent Works\2D-sizing\data\D405_dataset\depth\D405_0003_20251112_113026.npy"
    
    try:
        # 加载相机内参
        with open(camera_intrinsics_path, 'r') as f:
            _camera_intrinsics = json.load(f)
        
        # 加载深度图像
        _depth_image = np.load(depth_image_path)
        
        print(f"✓ 数据加载成功")
        print(f"  相机内参: {camera_intrinsics_path}")
        print(f"  深度图像: {depth_image_path}")
        print(f"  图像尺寸: {_depth_image.shape[1]} x {_depth_image.shape[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def get_pixel_real_distance(u, v, camera_intrinsics=None, depth_image=None):
    """
    获取指定像素坐标的真实距离信息
    
    Args:
        u, v: 像素坐标 (x, y)
        camera_intrinsics: 相机内参字典，如果为None则使用全局加载的内参
        depth_image: 深度图像数组，如果为None则使用全局加载的深度图像
        
    Returns:
        dict: 包含距离信息的字典，如果失败返回None
    """
    global _camera_intrinsics, _depth_image
    
    # 使用提供的参数或全局变量
    intrinsics = camera_intrinsics or _camera_intrinsics
    depth = depth_image or _depth_image
    
    # 检查数据是否已加载
    if intrinsics is None or depth is None:
        print("✗ 相机内参或深度数据未加载，请先调用 load_data()")
        return None
    
    # 检查像素坐标是否在图像范围内
    if u < 0 or u >= depth.shape[1] or v < 0 or v >= depth.shape[0]:
        return {
            'success': False,
            'error': '像素坐标超出图像范围',
            'pixel_coord': (u, v),
            'image_size': (depth.shape[1], depth.shape[0])
        }
    
    # 获取该像素的原始深度值
    depth_raw = depth[v, u]
    
    if depth_raw == 0:
        return {
            'success': False,
            'error': '该像素位置无有效深度数据',
            'pixel_coord': (u, v),
            'depth_raw': depth_raw
        }
    
    # 转换真实距离（米）
    depth_real = depth_raw * intrinsics['depth_scale']
    
    # 提取相机内参
    fx, fy = intrinsics['fx'], intrinsics['fy']
    ppx, ppy = intrinsics['ppx'], intrinsics['ppy']
    
    # 转换到相机坐标系（右手坐标系）
    # X轴向右，Y轴向下，Z轴向前（相机前方）
    X = (u - ppx) * depth_real / fx
    Y = (v - ppy) * depth_real / fy  
    Z = depth_real
    
    # 计算到原点距离
    distance_to_origin = np.sqrt(X**2 + Y**2 + Z**2)
    
    # 计算与图像中心的像素距离
    center_u, center_v = depth.shape[1] // 2, depth.shape[0] // 2
    pixel_distance_to_center = np.sqrt((u - center_u)**2 + (v - center_v)**2)
    
    return {
        'success': True,
        'pixel_coord': (u, v),
        'image_center': (center_u, center_v),
        'depth_raw': int(depth_raw),
        'depth_meters': depth_real,
        'camera_coord': {
            'X': X,
            'Y': Y, 
            'Z': Z
        },
        'distance_to_origin': distance_to_origin,
        'pixel_distance_to_center': pixel_distance_to_center,
        'depth_scale': intrinsics['depth_scale'],
        'camera_intrinsics': {
            'fx': fx,
            'fy': fy,
            'ppx': ppx,
            'ppy': ppy
        }
    }

def print_distance_info(result, title=None):
    """
    格式化打印距离信息
    
    Args:
        result: get_pixel_real_distance的返回值
        title: 可选的标题
    """
    if not result or not result.get('success', False):
        print(f"{'='*50}")
        if title:
            print(f"{title}")
        else:
            print("像素距离信息")
        print(f"{'='*50}")
        print(f"✗ 获取距离信息失败: {result.get('error', '未知错误')}")
        return
    
    print(f"{'='*60}")
    if title:
        print(f"{title}")
    else:
        print(f"像素 ({result['pixel_coord'][0]}, {result['pixel_coord'][1]}) 的真实距离信息")
    print(f"{'='*60}")
    
    print(f"像素坐标: ({result['pixel_coord'][0]}, {result['pixel_coord'][1]})")
    print(f"图像中心: ({result['image_center'][0]}, {result['image_center'][1]})")
    print(f"到图像中心像素距离: {result['pixel_distance_to_center']:.1f} 像素")
    
    print(f"\n深度信息:")
    print(f"  原始深度值: {result['depth_raw']}")
    print(f"  真实深度: {result['depth_meters']:.4f} 米")
    print(f"  深度缩放因子: {result['depth_scale']:.2e}")
    
    print(f"\n相机坐标系 3D 坐标:")
    print(f"  X (右向): {result['camera_coord']['X']:+.4f} 米")
    print(f"  Y (下向): {result['camera_coord']['Y']:+.4f} 米")
    print(f"  Z (前向): {result['camera_coord']['Z']:+.4f} 米")
    
    print(f"\n距离计算:")
    print(f"  到相机原点距离: {result['distance_to_origin']:.4f} 米")
    
    print(f"\n相机内参:")
    print(f"  fx: {result['camera_intrinsics']['fx']:.2f}")
    print(f"  fy: {result['camera_intrinsics']['fy']:.2f}")
    print(f"  px: {result['camera_intrinsics']['ppx']:.2f}")
    print(f"  py: {result['camera_intrinsics']['ppy']:.2f}")

def point_to_line_distance(point, line_start, line_end):
    """
    计算点到直线的垂直距离
    
    Args:
        point: 点的坐标 (x, y)
        line_start: 直线起点 (x1, y1)
        line_end: 直线终点 (x2, y2)
    
    Returns:
        float: 垂直距离（像素）
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # 向量计算
    A = y1 - y2  # 直线方程 Ax + By + C = 0 的系数
    B = x2 - x1
    C = x1*y2 - x2*y1
    
    # 点到直线的垂直距离公式: |Ax0 + By0 + C| / sqrt(A^2 + B^2)
    distance = abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
    
    return distance

def analyze_image_distances(region=None, step=50):
    """
    分析图像区域的距离分布
    
    Args:
        region: 分析区域 [x1, y1, x2, y2]，None表示全图
        step: 采样步长（像素）
        
    Returns:
        dict: 分析结果统计
    """
    global _camera_intrinsics, _depth_image
    
    if _camera_intrinsics is None or _depth_image is None:
        print("✗ 数据未加载，请先调用 load_data()")
        return None
    
    print(f"{'='*60}")
    print(f"图像距离分布分析")
    print(f"{'='*60}")
    
    if region is None:
        x1, y1, x2, y2 = 0, 0, _depth_image.shape[1], _depth_image.shape[0]
    else:
        x1, y1, x2, y2 = region
    
    print(f"分析区域: ({x1}, {y1}) 到 ({x2}, {y2})")
    print(f"采样步长: {step} 像素")
    
    distances = []
    valid_pixels = 0
    invalid_pixels = 0
    
    for v in range(y1, min(y2, _depth_image.shape[0]), step):
        for u in range(x1, min(x2, _depth_image.shape[1]), step):
            result = get_pixel_real_distance(u, v)
            if result and result.get('success', False):
                distances.append(result['distance_to_origin'])
                valid_pixels += 1
            else:
                invalid_pixels += 1
    
    if not distances:
        print("✗ 没有找到有效的深度数据")
        return None
    
    distances = np.array(distances)
    
    print(f"\n统计结果 (采样点数: {valid_pixels}, 无效点数: {invalid_pixels}):")
    print(f"  最小距离: {distances.min():.4f} 米")
    print(f"  最大距离: {distances.max():.4f} 米")
    print(f"  平均距离: {distances.mean():.4f} 米")
    print(f"  中位数距离: {np.median(distances):.4f} 米")
    print(f"  标准差: {distances.std():.4f} 米")
    print(f"  四分位距 (IQR): {np.percentile(distances, 75) - np.percentile(distances, 25):.4f} 米")
    
    return {
        'region': [x1, y1, x2, y2],
        'step': step,
        'valid_pixels': valid_pixels,
        'invalid_pixels': invalid_pixels,
        'min_distance': distances.min(),
        'max_distance': distances.max(),
        'mean_distance': distances.mean(),
        'median_distance': np.median(distances),
        'std_distance': distances.std(),
        'distances': distances
    }

def main():
    """主函数 - 演示所有功能"""
    print("像素距离获取工具函数演示")
    print("=" * 60)
    
    # 1. 加载数据
    if not load_data():
        return
    
    # 2. 测试单个像素距离获取
    print(f"\n{'='*60}")
    print("测试1: 单个像素距离获取")
    print(f"{'='*60}")
    
    test_coords = [
        (640, 360),    # 图像中心
        (320, 180),    # 左上四分之一  
        (960, 540),    # 右下四分之一
        (1279, 719),   # 右下角
    ]
    
    for u, v in test_coords:
        result = get_pixel_real_distance(u, v)
        print_distance_info(result, f"像素 ({u}, {v})")
        print()
    
    # 3. 批量分析
    print(f"\n{'='*60}")
    print("测试2: 图像区域距离分析")
    print(f"{'='*60}")
    
    # 分析图像中心200x200区域
    center_x, center_y = 640, 360
    region = [center_x - 100, center_y - 100, center_x + 100, center_y + 100]
    
    analyze_image_distances(region, step=20)
    
    print(f"\n{'='*60}")
    print("演示完成!")
    print(f"{'='*60}")
    print("主要函数:")
    print("• load_data() - 加载相机内参和深度数据")
    print("• get_pixel_real_distance(u, v) - 获取单个像素的真实距离")
    print("• print_distance_info(result) - 格式化打印距离信息")
    print("• analyze_image_distances() - 分析图像区域距离分布")

if __name__ == "__main__":
    main()