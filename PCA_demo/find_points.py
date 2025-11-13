#!/usr/bin/env python3
"""
草莓PCA分析 - 找到距离PC1方向最远的两个点
基于PCA分析结果，找到每个草莓上距离第一主成分方向最远的两个点
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入草莓分割器和PCA分析器
import sys
sys.path.append(str(Path(__file__).parent.parent))
from strawberry_segmentation import StrawberrySegmentor

class StrawberryPointFinder:
    """草莓关键点查找器类"""
    
    def __init__(self, model_path=None):
        """初始化查找器"""
        # 初始化草莓分割器
        self.segmentor = StrawberrySegmentor(model_path)
    
    def get_mask_points(self, image_path, conf_threshold=0.25):
        """获取所有草莓掩码的坐标点"""
        
        # 使用分割器进行预测
        result = self.segmentor.predict(image_path, conf_threshold)
        
        if result.masks is None:
            print("✗ 未检测到草莓掩码")
            return None, None
        
        # 读取原始图片
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取所有掩码
        masks = result.masks.data.cpu().numpy()
        
        print(f"✓ 检测到 {len(masks)} 个草莓")
        
        # 存储每个草莓的掩码点
        all_strawberry_points = []
        
        for i, mask in enumerate(masks):
            # 调整掩码大小到原始图片尺寸
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # 获取掩码区域的坐标点
            y_coords, x_coords = np.where(mask_resized > 0.5)
            
            if len(x_coords) == 0:
                print(f"✗ 第 {i+1} 个草莓掩码区域为空")
                continue
            
            # 创建坐标点矩阵
            points = np.column_stack((x_coords, y_coords))
            all_strawberry_points.append(points)
            
            print(f"✓ 第 {i+1} 个草莓获取到 {len(points)} 个掩码点")
        
        if len(all_strawberry_points) == 0:
            print("✗ 所有草莓掩码区域都为空")
            return None, None
        
        return all_strawberry_points, image
    
    def perform_pca(self, points_list):
        """对多个草莓的掩码点进行PCA分析"""
        
        all_pca_results = []
        
        for i, points in enumerate(points_list):
            print(f"正在对第 {i+1} 个草莓进行PCA分析...")
            
            # 中心化数据
            points_centered = points - np.mean(points, axis=0)
            
            # 计算协方差矩阵
            covariance_matrix = np.cov(points_centered.T)
            
            # 计算特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            
            # 按特征值大小排序
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # 计算主成分方向
            principal_components = eigenvectors.T
            
            # 计算方差解释比例
            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            
            # 存储每个草莓的PCA结果
            pca_result = {
                'principal_components': principal_components,
                'explained_variance_ratio': explained_variance_ratio,
                'center': np.mean(points, axis=0),
                'points_count': len(points)
            }
            
            all_pca_results.append(pca_result)
            
            print(f"✓ 第 {i+1} 个草莓PCA分析完成，主成分方差解释比例: {explained_variance_ratio[0]:.3f}, {explained_variance_ratio[1]:.3f}")
        
        return all_pca_results
    
    def find_extreme_points_perpendicular_to_pc1(self, points, pca_result):
        """找到距离PC1方向垂直距离最远的两个点（草莓的左右端点）"""
        
        # 获取第一主成分方向向量
        pc1_direction = pca_result['principal_components'][0]
        center = pca_result['center']
        
        # 计算PC1方向的垂直向量（旋转90度）
        pc1_perpendicular = np.array([-pc1_direction[1], pc1_direction[0]])
        
        # 计算每个点相对于中心点在PC1垂直方向上的投影
        perpendicular_projections = []
        for point in points:
            # 计算点相对于中心点的向量
            vector = point - center
            # 计算在PC1垂直方向上的投影长度
            projection = np.dot(vector, pc1_perpendicular)
            perpendicular_projections.append(projection)
        
        perpendicular_projections = np.array(perpendicular_projections)
        
        # 找到投影最大和最小的两个点（左右端点）
        right_point_idx = np.argmax(perpendicular_projections)
        left_point_idx = np.argmin(perpendicular_projections)
        
        # 获取对应的点坐标
        right_point = points[right_point_idx]
        left_point = points[left_point_idx]
        
        return right_point, left_point, perpendicular_projections[right_point_idx], perpendicular_projections[left_point_idx]
    
    def visualize_extreme_points(self, points_list, image, pca_results_list, save_path=None):
        """可视化所有草莓的极端点（单张图片显示）"""
        
        # 创建单张图片（放大图片尺寸）
        fig = plt.figure(figsize=(16, 12))
        
        # 颜色列表用于区分不同的草莓
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # 显示原始图片
        plt.imshow(image)
        
        for i, (points, pca_results) in enumerate(zip(points_list, pca_results_list)):
            color = colors[i % len(colors)]
            
            # 找到左右端点
            right_point, left_point, right_proj, left_proj = self.find_extreme_points_perpendicular_to_pc1(points, pca_results)
            
            # 绘制所有掩码点（减小点大小）
            plt.scatter(points[:, 0], points[:, 1], c=color, s=0.5, alpha=0.2, label=f'草莓 {i+1} 掩码点')
            
            # 绘制左右端点（减小点大小）
            plt.scatter(right_point[0], right_point[1], c='yellow', s=50, marker='*', 
                       edgecolors='black', linewidth=1, label=f'草莓 {i+1} 右端点')
            plt.scatter(left_point[0], left_point[1], c='magenta', s=50, marker='*', 
                       edgecolors='black', linewidth=1, label=f'草莓 {i+1} 左端点')
            
            # 绘制PC1方向线（减小线宽）
            center = pca_results['center']
            pc1_direction = pca_results['principal_components'][0]
            scale = 150
            plt.arrow(center[0], center[1], 
                     pc1_direction[0] * scale, pc1_direction[1] * scale, 
                     color=color, width=1.5, label=f'草莓 {i+1} PC1方向')
            
            # 添加文本标注（减小字体大小）
            plt.text(right_point[0] + 10, right_point[1] + 10, f'{right_proj:.1f}', 
                    fontsize=8, color='black', alpha=0.8)
            plt.text(left_point[0] + 10, left_point[1] + 10, f'{left_proj:.1f}', 
                    fontsize=8, color='black', alpha=0.8)
        
        plt.title(f'草莓左右端点检测 - 共检测到 {len(points_list)} 个草莓')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 极端点可视化结果已保存到: {save_path}")
        
        plt.show()
        
        return fig
    
    def analyze_image_extreme_points(self, image_path, conf_threshold=0.25, save_path=None):
        """分析单张图片中所有草莓的PC1方向极端点"""
        
        print("=" * 60)
        print("草莓PC1方向极端点分析开始")
        print("=" * 60)
        
        # 获取所有草莓的掩码点
        points_list, image = self.get_mask_points(image_path, conf_threshold)
        
        if points_list is None:
            print("✗ 无法获取掩码点，分析终止")
            return None
        
        # 执行PCA分析
        pca_results_list = self.perform_pca(points_list)
        
        # 分析每个草莓的左右端点
        extreme_points_results = []
        for i, (points, pca_results) in enumerate(zip(points_list, pca_results_list)):
            right_point, left_point, right_proj, left_proj = self.find_extreme_points_perpendicular_to_pc1(points, pca_results)
            
            result = {
                'strawberry_index': i + 1,
                'right_point': right_point,
                'left_point': left_point,
                'right_projection': right_proj,
                'left_projection': left_proj,
                'distance_between_points': np.linalg.norm(right_point - left_point),
                'pca_results': pca_results
            }
            
            extreme_points_results.append(result)
            
            print(f"✓ 草莓 {i+1} 分析完成:")
            print(f"  右端点坐标: ({right_point[0]:.1f}, {right_point[1]:.1f}), 投影值: {right_proj:.1f}")
            print(f"  左端点坐标: ({left_point[0]:.1f}, {left_point[1]:.1f}), 投影值: {left_proj:.1f}")
            print(f"  左右端点间距离: {result['distance_between_points']:.1f} 像素")
        
        # 可视化结果
        fig = self.visualize_extreme_points(points_list, image, pca_results_list, save_path)
        
        print(f"✓ 分析完成，共处理了 {len(extreme_points_results)} 个草莓")
        
        # 返回分析结果
        return {
            'points_list': points_list,
            'image': image,
            'pca_results_list': pca_results_list,
            'extreme_points_results': extreme_points_results,
            'figure': fig
        }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='草莓PC1方向极端点查找脚本')
    parser.add_argument('--input', type=str, default=r"E:\Recent Works\paper-Immature Green Apple Detection\data\strawberry\images\frame_00762.png",
                       help='输入图片路径')
    parser.add_argument('--output', type=str, default=r"E:\Recent Works\paper-Immature Green Apple Detection\results\extreme_points_result.png",
                       help='输出图片路径')
    parser.add_argument('--model', type=str, default=r"E:\Recent Works\paper-Immature Green Apple Detection\weights\yolov11m-2c-seg.pt",
                       help='模型文件路径')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 (0-1)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("草莓PC1方向极端点查找工具")
    print("=" * 60)
    
    try:
        # 初始化查找器
        finder = StrawberryPointFinder(args.model)
        
        # 确保输出目录存在
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 分析图片
        result = finder.analyze_image_extreme_points(
            args.input, 
            args.conf,
            str(output_path)
        )
        
        if result is not None:
            print(f"✓ 分析完成，共检测到 {len(result['extreme_points_results'])} 个草莓")
            print(f"✓ 结果已保存到: {args.output}")
            
            # 打印汇总信息
            print("\n" + "=" * 60)
            print("汇总信息:")
            print("=" * 60)
            for i, res in enumerate(result['extreme_points_results']):
                print(f"草莓 {i+1}:")
                print(f"  右端点: ({res['right_point'][0]:.1f}, {res['right_point'][1]:.1f})")
                print(f"  左端点: ({res['left_point'][0]:.1f}, {res['left_point'][1]:.1f})")
                print(f"  左右端点距离: {res['distance_between_points']:.1f} 像素")
                print(f"  PC1方差解释比例: {res['pca_results']['explained_variance_ratio'][0]:.3f}")
                print()
            
    except Exception as e:
        print(f"✗ 分析过程中出现错误: {e}")

if __name__ == "__main__":
    main()