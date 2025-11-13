#!/usr/bin/env python3
"""
草莓PCA分析 - 寻找垂直PC1方向最长线段
使用草莓分割器获取掩码，进行PCA分析，然后沿PC1方向移动找到垂直PC1方向最长的两个点
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入草莓分割器
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
    
    def find_pc1_endpoints(self, points, pca_result):
        """
        找到PC1方向上的最远端点（上端点和下端点）
        
        参数:
            points: 草莓掩码点坐标
            pca_result: PCA分析结果
        
        返回:
            top_endpoint: PC1方向的"上"端点
            bottom_endpoint: PC1方向的"下"端点
            pc1_length: PC1方向的总长度
        """
        
        # 获取PCA结果
        principal_components = pca_result['principal_components']
        center = pca_result['center']
        
        # PC1方向向量
        pc1_vector = principal_components[0]
        
        # 计算所有点在PC1方向上的投影
        centered_points = points - center
        projections = np.dot(centered_points, pc1_vector)
        
        # 找到PC1方向上的最远点
        min_projection = np.min(projections)
        max_projection = np.max(projections)
        
        # 计算端点坐标
        top_endpoint = center + max_projection * pc1_vector  # 假设这是"上"端点
        bottom_endpoint = center + min_projection * pc1_vector  # 假设这是"下"端点
        
        # PC1方向的总长度
        pc1_length = max_projection - min_projection
        
        return top_endpoint, bottom_endpoint, pc1_length

    def find_longest_perpendicular_line(self, points, pca_result, step_size=1, search_range=50):
        """
        沿PC1方向移动，找到垂直PC1方向最长的线段
        
        参数:
            points: 草莓掩码点坐标
            pca_result: PCA分析结果
            step_size: 沿PC1方向移动的步长
            search_range: 搜索范围（沿PC1方向移动的距离）
        
        返回:
            best_line_points: 最长线段的两个端点
            max_length: 最长线段的长度
            best_position: 沿PC1方向的最佳位置
        """
        
        # 获取PCA结果
        principal_components = pca_result['principal_components']
        center = pca_result['center']
        
        # PC1方向向量（第一主成分）
        pc1_vector = principal_components[0]
        
        # 垂直PC1方向的向量（第二主成分）
        pc2_vector = principal_components[1]
        
        # 归一化向量
        pc1_vector_norm = pc1_vector / np.linalg.norm(pc1_vector)
        pc2_vector_norm = pc2_vector / np.linalg.norm(pc2_vector)
        
        max_length = 0
        best_line_points = None
        best_position = 0
        
        # 沿PC1方向移动
        for t in np.arange(-search_range, search_range + step_size, step_size):
            # 计算当前沿PC1方向的位置
            current_position = center + t * pc1_vector_norm
            
            # 找到沿PC2方向与掩码边界相交的点
            line_points = self.find_intersection_points(points, current_position, pc2_vector_norm)
            
            if line_points is not None:
                # 计算线段长度
                length = np.linalg.norm(line_points[1] - line_points[0])
                
                if length > max_length:
                    max_length = length
                    best_line_points = line_points
                    best_position = t
        
        return best_line_points, max_length, best_position
    
    def find_intersection_points(self, points, start_point, direction_vector):
        """
        找到沿给定方向与掩码边界相交的两个点
        
        参数:
            points: 掩码点坐标
            start_point: 起始点
            direction_vector: 方向向量
        
        返回:
            两个交点坐标，如果没有找到则返回None
        """
        
        # 创建掩码的凸包
        from scipy.spatial import ConvexHull
        
        try:
            # 计算凸包
            hull = ConvexHull(points)
            
            # 获取凸包顶点
            hull_points = points[hull.vertices]
            
            # 计算沿方向向量的射线与凸包的交点
            intersections = []
            
            # 检查所有凸包边
            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % len(hull_points)]
                
                # 计算射线与线段的交点
                intersection = self.line_intersection(start_point, direction_vector, p1, p2 - p1)
                
                if intersection is not None:
                    # 检查交点是否在线段上
                    if self.point_on_segment(intersection, p1, p2):
                        intersections.append(intersection)
            
            # 如果找到两个交点，返回它们
            if len(intersections) >= 2:
                # 按距离起始点的距离排序
                distances = [np.linalg.norm(p - start_point) for p in intersections]
                sorted_indices = np.argsort(distances)
                
                # 取最近的两个交点
                return intersections[sorted_indices[0]], intersections[sorted_indices[1]]
            
        except Exception as e:
            print(f"凸包计算失败: {e}")
            # 如果凸包失败，使用简单的边界框方法
            return self.find_intersection_simple(points, start_point, direction_vector)
        
        return None
    
    def find_intersection_simple(self, points, start_point, direction_vector):
        """
        使用简单方法找到沿给定方向与掩码边界相交的两个点
        
        参数:
            points: 掩码点坐标
            start_point: 起始点
            direction_vector: 方向向量
        
        返回:
            两个交点坐标
        """
        
        # 计算掩码的边界框
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        
        # 计算垂直方向向量
        perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])
        
        # 计算沿方向向量的射线与边界框的交点
        intersections = []
        
        # 检查与四条边的交点
        edges = [
            (np.array([min_x, min_y]), np.array([max_x, min_y])),  # 上边
            (np.array([max_x, min_y]), np.array([max_x, max_y])),  # 右边
            (np.array([max_x, max_y]), np.array([min_x, max_y])),  # 下边
            (np.array([min_x, max_y]), np.array([min_x, min_y]))   # 左边
        ]
        
        for edge_start, edge_end in edges:
            intersection = self.line_intersection(start_point, direction_vector, edge_start, edge_end - edge_start)
            
            if intersection is not None:
                if self.point_on_segment(intersection, edge_start, edge_end):
                    intersections.append(intersection)
        
        # 如果找到两个交点，返回它们
        if len(intersections) >= 2:
            # 按距离起始点的距离排序
            distances = [np.linalg.norm(p - start_point) for p in intersections]
            sorted_indices = np.argsort(distances)
            
            # 取最近的两个交点
            return intersections[sorted_indices[0]], intersections[sorted_indices[1]]
        
        return None
    
    def line_intersection(self, p1, d1, p2, d2):
        """
        计算两条直线的交点
        
        参数:
            p1: 第一条直线的起点
            d1: 第一条直线的方向向量
            p2: 第二条直线的起点
            d2: 第二条直线的方向向量
        
        返回:
            交点坐标，如果没有交点则返回None
        """
        
        # 解线性方程组
        A = np.column_stack((d1, -d2))
        b = p2 - p1
        
        try:
            t = np.linalg.solve(A, b)
            intersection = p1 + t[0] * d1
            return intersection
        except np.linalg.LinAlgError:
            # 直线平行，无交点
            return None
    
    def point_on_segment(self, point, seg_start, seg_end):
        """
        检查点是否在线段上
        
        参数:
            point: 要检查的点
            seg_start: 线段起点
            seg_end: 线段终点
        
        返回:
            True如果点在线段上，否则False
        """
        
        # 检查点是否在线段的边界框内
        min_x = min(seg_start[0], seg_end[0])
        max_x = max(seg_start[0], seg_end[0])
        min_y = min(seg_start[1], seg_end[1])
        max_y = max(seg_start[1], seg_end[1])
        
        if not (min_x <= point[0] <= max_x and min_y <= point[1] <= max_y):
            return False
        
        # 检查点是否在直线上（使用叉积）
        cross_product = (point[1] - seg_start[1]) * (seg_end[0] - seg_start[0]) - \
                       (point[0] - seg_start[0]) * (seg_end[1] - seg_start[1])
        
        return abs(cross_product) < 1e-10
    
    def visualize_results(self, points_list, image, pca_results_list, pc1_endpoints_list, line_results_list, real_distance_results=None, perpendicular_distance_results=None, volume_results=None, save_path=None):
        """可视化所有草莓的PCA结果、PC1端点和找到的最长线段在一张图中"""
        
        # 创建一张图显示所有草莓信息
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 设置全局字体大小
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })
        
        # 颜色列表用于区分不同的草莓
        colors = ['pink']
        
        # 显示原始图片
        ax.imshow(image)
        
        for i, (points, pca_results, pc1_endpoints, line_results) in enumerate(zip(points_list, pca_results_list, pc1_endpoints_list, line_results_list)):
            color = colors[i % len(colors)]
            
            # 绘制掩码点
            ax.scatter(points[:, 0], points[:, 1], c=color, s=0.5, alpha=0.1)
            
            # 获取PCA结果
            principal_components = pca_results['principal_components']
            center = pca_results['center']
            
            # 绘制主成分方向
            scale = 100  # 缩放因子
            
            # 第一主成分（PC1）
            ax.arrow(center[0], center[1], 
                    principal_components[0, 0] * scale, principal_components[0, 1] * scale, 
                    color=color, width=3, alpha=0.9,
                    head_width=8, head_length=10, fc=color, ec=color)
            
            # 第二主成分（PC2）
            ax.arrow(center[0], center[1], 
                    principal_components[1, 0] * scale, principal_components[1, 1] * scale, 
                    color=color, width=3, alpha=0.5, linestyle='--',
                    head_width=6, head_length=8, fc='white', ec=color)
            
            # 绘制中心点
            ax.scatter(center[0], center[1], c=color, s=20, marker='o', 
                      edgecolors='white', linewidths=2, zorder=5)
            
            # 绘制PC1端点（上端点和下端点）
            top_endpoint = pc1_endpoints['top_endpoint']
            bottom_endpoint = pc1_endpoints['bottom_endpoint']
            
            # 绘制PC1方向线段（连接两个端点）
            ax.plot([top_endpoint[0], bottom_endpoint[0]], 
                   [top_endpoint[1], bottom_endpoint[1]], 
                   color=color, linewidth=6, alpha=0.7, linestyle='-', zorder=3)
            
            # 标记上端点（上端点用圆形标记）
            ax.scatter(top_endpoint[0], top_endpoint[1], 
                      c=color, s=30, marker='o', 
                      edgecolors='white', linewidths=2, zorder=7)
            
            # 标记下端点（下端点用圆形标记）
            ax.scatter(bottom_endpoint[0], bottom_endpoint[1], 
                      c=color, s=30, marker='o', 
                      edgecolors='white', linewidths=2, zorder=7)
            
            # 绘制最长线段
            if line_results['line_points'] is not None:
                line_points = line_results['line_points']
                
                # 绘制线段
                ax.plot([line_points[0][0], line_points[1][0]], 
                       [line_points[0][1], line_points[1][1]], 
                       color='yellow', linewidth=4, alpha=0.9, zorder=4)
                
                # 标记端点（左右点，用圆形标记）
                ax.scatter(line_points[0][0], line_points[0][1], 
                          c='yellow', s=20, marker='o', 
                          edgecolors='black', linewidths=1, zorder=6)
                
                ax.scatter(line_points[1][0], line_points[1][1], 
                          c='yellow', s=20, marker='o', 
                          edgecolors='black', linewidths=1, zorder=6)
                
                # 显示真实距离文本（如果可用）
                if real_distance_results and i < len(real_distance_results):
                    real_distance_data = real_distance_results[i]
                    if real_distance_data and real_distance_data['real_distance_cm'] is not None:
                        # 计算线段中点位置
                        mid_x = (line_points[0][0] + line_points[1][0]) / 2
                        mid_y = (line_points[0][1] + line_points[1][1]) / 2
                        
                        # 显示真实距离信息
                        distance_text = f"{real_distance_data['real_distance_cm']:.2f} cm"
                        ax.text(mid_x, mid_y + 20, distance_text, 
                               fontsize=12, fontweight='bold', 
                               ha='center', va='bottom',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                               color='red', zorder=8)
                
                # 绘制垂直距离线段（如果可用）
                if perpendicular_distance_results and i < len(perpendicular_distance_results):
                    perp_data = perpendicular_distance_results[i]
                    if perp_data and 'top_perp_distance_px' in perp_data:
                        # 计算垂足点
                        line_p1 = np.array(line_points[0])
                        line_p2 = np.array(line_points[1])
                        point = np.array(top_endpoint)
                        
                        # 计算垂足
                        line_vec = line_p2 - line_p1
                        line_len_sq = np.dot(line_vec, line_vec)
                        if line_len_sq > 0:
                            t = np.dot(point - line_p1, line_vec) / line_len_sq
                            foot_point = line_p1 + t * line_vec
                            foot_point = tuple(foot_point.astype(int))
                            
                            # 绘制从上端点到垂足的垂线
                            ax.plot([top_endpoint[0], foot_point[0]], 
                                   [top_endpoint[1], foot_point[1]], 
                                   color='green', linewidth=3, linestyle='--', alpha=0.9, zorder=6)
                            
                            # 显示上端点垂直距离
                            if perp_data['top_perp_distance_px'] is not None:
                                top_mid_x = (top_endpoint[0] + foot_point[0]) / 2
                                top_mid_y = (top_endpoint[1] + foot_point[1]) / 2
                                top_distance_text = f"{perp_data['top_perp_distance_cm']:.2f} cm"
                                ax.text(top_mid_x + 15, top_mid_y, top_distance_text, 
                                       fontsize=10, fontweight='bold', 
                                       ha='center', va='bottom',
                                       color='darkgreen', zorder=9)
                        
                        # 计算下端点垂足
                        point = np.array(bottom_endpoint)
                        if line_len_sq > 0:
                            t = np.dot(point - line_p1, line_vec) / line_len_sq
                            foot_point = line_p1 + t * line_vec
                            foot_point = tuple(foot_point.astype(int))
                            
                            # 绘制从下端点到垂足的垂线
                            ax.plot([bottom_endpoint[0], foot_point[0]], 
                                   [bottom_endpoint[1], foot_point[1]], 
                                   color='green', linewidth=3, linestyle='--', alpha=0.9, zorder=6)
                            
                            # 显示下端点垂直距离
                            if perp_data['bottom_perp_distance_px'] is not None:
                                bottom_mid_x = (bottom_endpoint[0] + foot_point[0]) / 2
                                bottom_mid_y = (bottom_endpoint[1] + foot_point[1]) / 2
                                bottom_distance_text = f"{perp_data['bottom_perp_distance_cm']:.2f} cm"
                                ax.text(bottom_mid_x + 15, bottom_mid_y, bottom_distance_text, 
                                       fontsize=10, fontweight='bold', 
                                       ha='center', va='bottom',
                                       color='darkgreen', zorder=9)
                
                # 显示草莓体积信息（如果可用）
                if volume_results and i < len(volume_results):
                    volume_data = volume_results[i]
                    if volume_data and volume_data.get('total_volume_cm3') is not None:
                        # 计算PC1线段中点位置，用于放置体积信息
                        center_x = (top_endpoint[0] + bottom_endpoint[0]) / 2
                        center_y = (top_endpoint[1] + bottom_endpoint[1]) / 2
                        
                        # 显示体积信息（红色）
                        volume_text = f"Volume: {volume_data['total_volume_cm3']:.2f} cm3"
                        ax.text(center_x, center_y-70, volume_text, 
                               fontsize=12, fontweight='bold', 
                               ha='center', va='bottom',
                               color='red')  # 设置字体颜色为红色
                
                # 绘制风筝形状的连接线（连接左右端点和上下端点）
                if line_points is not None:
                    # 连接PC1上端点到最长垂直线段的两个端点
                    ax.plot([top_endpoint[0], line_points[0][0]], 
                           [top_endpoint[1], line_points[0][1]], 
                           color='blue', linewidth=3, alpha=0.8, linestyle='-', zorder=5)
                    
                    ax.plot([top_endpoint[0], line_points[1][0]], 
                           [top_endpoint[1], line_points[1][1]], 
                           color='blue', linewidth=3, alpha=0.8, linestyle='-', zorder=5)
                    
                    # 连接PC1下端点到最长垂直线段的两个端点
                    ax.plot([bottom_endpoint[0], line_points[0][0]], 
                           [bottom_endpoint[1], line_points[0][1]], 
                           color='blue', linewidth=3, alpha=0.8, linestyle='-', zorder=5)
                    
                    ax.plot([bottom_endpoint[0], line_points[1][0]], 
                           [bottom_endpoint[1], line_points[1][1]], 
                           color='blue', linewidth=3, alpha=0.8, linestyle='-', zorder=5)
        
        # 设置标题
        ax.set_title('草莓PCA分析：PC1方向端点与最长垂直线段（含风筝形状连接）', fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        
        # 创建图例
        legend_elements = [
            plt.Line2D([0], [0], color='gray', lw=3, label='PC1主成分方向'),
            plt.Line2D([0], [0], color='red', lw=6, alpha=0.7, label='PC1方向线段（端点间）'),
            plt.scatter([], [], c='red', s=30, marker='o', edgecolors='white', linewidths=2, label='PC1端点'),
            plt.Line2D([0], [0], color='yellow', lw=4, label='最长垂直线段'),
            plt.scatter([], [], c='yellow', s=20, marker='o', edgecolors='black', linewidths=1, label='垂直线段端点'),
            plt.Line2D([0], [0], color='green', lw=3, linestyle='--', label='垂直距离线段'),
            plt.Line2D([0], [0], color='blue', lw=3, label='风筝形状连接线'),
        ]
        
        # 如果有真实距离数据，在图例中添加说明
        if real_distance_results and any(data and data['real_distance_cm'] is not None for data in real_distance_results):
            legend_elements.append(plt.Line2D([0], [0], color='none', label='真实距离标注（厘米）'))
        
        # 如果有垂直距离数据，在图例中添加说明
        if perpendicular_distance_results and any(data and 'top_perp_distance_cm' in data for data in perpendicular_distance_results):
            legend_elements.append(plt.Line2D([0], [0], color='none', label='垂直距离标注（厘米）'))
        
        # 如果有体积数据，在图例中添加说明
        if volume_results and any(data and data.get('total_volume_cm3') is not None for data in volume_results):
            legend_elements.append(plt.Line2D([0], [0], color='none', label='草莓体积（立方厘米）'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                 frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 结果已保存到: {save_path}")
        
        plt.show()
        
        return fig
    
    def visualize_original_with_connections(self, image, pca_results_list, pc1_endpoints_list, line_results_list, real_distance_results=None, save_path=None):
        """
        只显示原图和风筝形状连接线的简洁可视化
        
        参数:
            image: 原始图像
            pca_results_list: PCA分析结果列表
            pc1_endpoints_list: PC1端点信息列表
            line_results_list: 最长垂直线段结果列表
            save_path: 保存路径（可选）
        
        返回:
            matplotlib图形对象
        """
        
        # 创建图形和坐标轴
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # 显示原始图像
        ax.imshow(image)
        
        # 遍历每个草莓，绘制风筝形状连接线
        for i, (pca_results, pc1_endpoints, line_results) in enumerate(zip(pca_results_list, pc1_endpoints_list, line_results_list)):
            # 获取颜色
            color = plt.cm.tab10(i)
            
            # 获取PC1端点
            top_endpoint = pc1_endpoints['top_endpoint']
            bottom_endpoint = pc1_endpoints['bottom_endpoint']
            
            # 获取最长垂直线段端点
            if line_results['line_points'] is not None:
                line_points = line_results['line_points']
                
                # 只绘制风筝形状的连接线（蓝色）
                # 连接PC1上端点到最长垂直线段的两个端点
                ax.plot([top_endpoint[0], line_points[0][0]], 
                       [top_endpoint[1], line_points[0][1]], 
                       color='blue', linewidth=3, alpha=0.9, linestyle='-', zorder=3)
                
                ax.plot([top_endpoint[0], line_points[1][0]], 
                       [top_endpoint[1], line_points[1][1]], 
                       color='blue', linewidth=3, alpha=0.9, linestyle='-', zorder=3)
                
                # 连接PC1下端点到最长垂直线段的两个端点
                ax.plot([bottom_endpoint[0], line_points[0][0]], 
                       [bottom_endpoint[1], line_points[0][1]], 
                       color='blue', linewidth=3, alpha=0.9, linestyle='-', zorder=3)
                
                ax.plot([bottom_endpoint[0], line_points[1][0]], 
                       [bottom_endpoint[1], line_points[1][1]], 
                       color='blue', linewidth=3, alpha=0.9, linestyle='-', zorder=3)
                
                # 显示真实距离文本（如果可用）
                if real_distance_results and i < len(real_distance_results):
                    real_distance_data = real_distance_results[i]
                    if real_distance_data and real_distance_data['real_distance_cm'] is not None:
                        # 计算线段中点位置
                        mid_x = (line_points[0][0] + line_points[1][0]) / 2
                        mid_y = (line_points[0][1] + line_points[1][1]) / 2
                        
                        # 显示真实距离信息
                        distance_text = f"{real_distance_data['real_distance_cm']:.2f} cm"
                        ax.text(mid_x, mid_y + 20, distance_text, 
                               fontsize=12, fontweight='bold', 
                               ha='center', va='bottom',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                               color='red', zorder=8)
        
        # 设置标题
        ax.set_title('原始图像 - 风筝形状连接线', fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            # 生成简洁版本的保存路径
            base_name = os.path.splitext(save_path)[0]
            extension = os.path.splitext(save_path)[1]
            simple_save_path = f"{base_name}_simple{extension}"
            
            plt.savefig(simple_save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 简洁版结果已保存到: {simple_save_path}")
        
        plt.show()
        
        return fig
    
    def get_next_available_filename(self, base_path):
        """获取下一个可用的文件名，避免重复"""
        base_name = os.path.splitext(base_path)[0]
        extension = os.path.splitext(base_path)[1]
        
        # 如果基础文件名不存在，直接返回
        if not os.path.exists(base_path):
            return base_path
        
        # 如果存在，则添加序号
        counter = 1
        while True:
            new_path = f"{base_name}_{counter}{extension}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1
    
    def analyze_image(self, image_path, conf_threshold=0.25, save_path=None):
        """分析单张图片中的所有草莓"""
        
        print("=" * 60)
        print("草莓PCA分析 - 寻找最长垂直线段")
        print("=" * 60)
        
        # 获取所有草莓的掩码点
        points_list, image = self.get_mask_points(image_path, conf_threshold)
        
        if points_list is None:
            print("✗ 无法获取掩码点，分析终止")
            return None
        
        # 执行PCA分析
        pca_results_list = self.perform_pca(points_list)
        
        # 为每个草莓寻找PC1端点和最长垂直线段
        pc1_endpoints_list = []
        line_results_list = []
        
        for i, (points, pca_results) in enumerate(zip(points_list, pca_results_list)):
            # 寻找PC1方向的端点
            print(f"正在为第 {i+1} 个草莓寻找PC1方向端点...")
            top_endpoint, bottom_endpoint, pc1_length = self.find_pc1_endpoints(points, pca_results)
            
            pc1_endpoints = {
                'top_endpoint': top_endpoint,
                'bottom_endpoint': bottom_endpoint,
                'pc1_length': pc1_length
            }
            
            pc1_endpoints_list.append(pc1_endpoints)
            
            print(f"✓ 第 {i+1} 个草莓PC1方向长度: {pc1_length:.2f} 像素")
            
            # 寻找最长垂直线段
            print(f"正在为第 {i+1} 个草莓寻找最长垂直线段...")
            line_points, max_length, best_position = self.find_longest_perpendicular_line(points, pca_results)
            
            line_results = {
                'line_points': line_points,
                'max_length': max_length,
                'best_position': best_position
            }
            
            line_results_list.append(line_results)
            
            if line_points is not None:
                print(f"✓ 第 {i+1} 个草莓找到最长垂直线段，长度: {max_length:.2f} 像素")
            else:
                print(f"✗ 第 {i+1} 个草莓未找到有效线段")
        
        # 如果提供了保存路径，自动处理文件名重复问题
        if save_path:
            save_path = self.get_next_available_filename(save_path)
        
        # 可视化结果（包含PC1端点和最长垂直线段）
        fig = self.visualize_results(points_list, image, pca_results_list, pc1_endpoints_list, line_results_list, save_path=save_path)
        
        # 生成简洁版本的可视化结果（只显示原图和连接线）
        print("正在生成简洁版可视化结果...")
        simple_fig = self.visualize_original_with_connections(image, pca_results_list, pc1_endpoints_list, line_results_list, save_path)
        
        print(f"✓ 分析完成，共处理了 {len(pca_results_list)} 个草莓")
        
        # 返回分析结果
        return {
            'points_list': points_list,
            'image': image,
            'pca_results_list': pca_results_list,
            'line_results_list': line_results_list,
            'figure': fig
        }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='草莓PCA分析 - 寻找最长垂直线段')
    parser.add_argument('--input', type=str, default=r"E:\Recent Works\2D-sizing\data\strawberry\images\frame_00762.png",
                       help='输入图片路径或文件夹路径')
    parser.add_argument('--output', type=str, default=r"E:\Recent Works\2D-sizing\results",
                       help='输出文件夹路径（仅对文件夹输入有效）')
    parser.add_argument('--model', type=str, default=r"E:\Recent Works\2D-sizing\weights\yolov11m-2c-seg.pt",
                       help='模型文件路径（可选，默认使用训练好的模型或预训练模型）')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 (0-1)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("草莓PCA分析 - 寻找最长垂直线段工具")
    print("=" * 60)
    
    try:
        # 初始化查找器
        finder = StrawberryPointFinder(args.model)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 分析单张图片，自动处理文件名重复
            # 为单张图片生成正确的保存路径
            output_path = Path(args.output)
            if output_path.is_dir():
                # 如果输出路径是文件夹，则在其中创建具体的文件路径
                save_path = output_path / f"pca_line_result_{input_path.stem}.png"
            else:
                # 如果输出路径是文件路径，直接使用
                save_path = args.output
            
            result = finder.analyze_image(
                str(input_path), 
                args.conf,
                str(save_path)
            )
            
            if result is not None:
                print(f"✓ 单张图片分析完成，共检测到 {len(result['pca_results_list'])} 个草莓")
                print(f"✓ 结果已保存到: {save_path}")
                
        else:
            print("✗ 输入路径不存在或不是文件")
            
    except Exception as e:
        print(f"✗ 分析过程中出现错误: {e}")

if __name__ == "__main__":
    main()