#!/usr/bin/env python3
"""
草莓像素面积计算器
根据 find_points_v2.py 获取的四个关键点计算草莓的像素面积
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse
import sys

# 设置中文字体支持
import matplotlib
import matplotlib.font_manager as fm
import warnings

# 抑制字体警告
warnings.filterwarnings("ignore", category=UserWarning)

def setup_chinese_font():
    """设置中文字体"""
    # 获取系统中可用的字体列表
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 按优先级尝试设置中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Noto Sans CJK', 'Source Han Sans']
    
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 设置字体参数
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    if selected_font:
        matplotlib.rcParams['font.family'] = [selected_font] + matplotlib.rcParams['font.family']
        print(f"✓ 使用中文字体: {selected_font}")
    else:
        print("⚠️  未找到中文字体，将使用英文标签")
        print("💡 如需显示中文，请安装中文字体（如SimHei.ttf）或在代码中使用英文标签")
    
    return selected_font is not None

# 初始化字体设置
has_chinese_font = setup_chinese_font()

class StrawberryAreaCalculator:
    """草莓面积计算器类"""
    
    def __init__(self):
        """初始化面积计算器"""
        pass
    
    def polygon_area(self, points):
        """
        使用鞋带公式计算多边形面积
        
        参数:
            points: 多边形顶点坐标，形状为 (n, 2)
        
        返回:
            多边形面积
        """
        if len(points) < 3:
            return 0.0
        
        # 确保点按顺序排列
        points = np.array(points)
        
        # 使用鞋带公式计算面积
        n = len(points)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i, 0] * points[j, 1]
            area -= points[j, 0] * points[i, 1]
        
        return abs(area) / 2.0
    
    def triangle_area(self, p1, p2, p3):
        """
        使用向量叉积计算三角形面积
        
        参数:
            p1, p2, p3: 三角形的三个顶点坐标
        
        返回:
            三角形面积
        """
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        
        # 计算两个边的向量
        v1 = p2 - p1
        v2 = p3 - p1
        
        # 计算叉积的模长
        cross_product = np.cross(v1, v2)
        
        return abs(cross_product) / 2.0
    
    def quadrilateral_area(self, quad_points):
        """
        计算四边形面积，将四边形分割为两个三角形
        
        参数:
            quad_points: 四边形四个顶点的坐标，顺序为 [PC1上端点, PC1下端点, 左端点, 右端点]
        
        返回:
            四边形面积
        """
        if len(quad_points) != 4:
            return 0.0
        
        # 将四边形分割为两个三角形
        # 三角形1: PC1上端点, 左端点, 右端点
        # 三角形2: PC1下端点, 左端点, 右端点
        
        pc1_top, pc1_bottom, left_point, right_point = quad_points
        
        # 计算两个三角形的面积
        area1 = self.triangle_area(pc1_top, left_point, right_point)
        area2 = self.triangle_area(pc1_bottom, left_point, right_point)
        
        total_area = area1 + area2
        
        return total_area
    
    def convex_hull_area(self, points):
        """
        计算点集的凸包面积
        
        参数:
            points: 点集坐标
        
        返回:
            凸包面积
        """
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(points)
            return hull.volume  # 在2D中，volume实际上就是面积
        except:
            return 0.0
    
    def calculate_strawberry_area_from_analysis(self, analysis_result, strawberry_index=0):
        """
        从 find_points_v2.py 的分析结果计算草莓面积
        
        参数:
            analysis_result: analyze_image 方法返回的分析结果
            strawberry_index: 草莓索引（默认第一个）
        
        返回:
            面积信息字典
        """
        if analysis_result is None:
            return None
        
        line_results_list = analysis_result['line_results_list']
        pca_results_list = analysis_result['pca_results_list']
        points_list = analysis_result['points_list']
        
        if strawberry_index >= len(line_results_list):
            return None
        
        # 获取第 i 个草莓的数据
        line_results = line_results_list[strawberry_index]
        pca_results = pca_results_list[strawberry_index]
        original_points = points_list[strawberry_index]
        
        # 获取最长线段的两个端点（左右端点）
        line_points = line_results['line_points']
        
        if line_points is None:
            return None
        
        left_point = line_points[0]
        right_point = line_points[1]
        
        # 计算 PC1 方向的两个端点
        pc1_endpoints = self.find_pc1_endpoints(original_points, pca_results)
        
        if pc1_endpoints is None:
            return None
        
        pc1_top = pc1_endpoints['top']
        pc1_bottom = pc1_endpoints['bottom']
        
        # 计算四边形面积
        quad_points = [pc1_top, pc1_bottom, left_point, right_point]
        quadrilateral_area = self.quadrilateral_area(quad_points)
        
        # 计算原始掩码的实际面积
        actual_mask_area = len(original_points)
        
        # 计算凸包面积
        hull_area = self.convex_hull_area(original_points)
        
        return {
            'strawberry_index': strawberry_index,
            'quadrilateral_area': quadrilateral_area,
            'actual_mask_area': actual_mask_area,
            'hull_area': hull_area,
            'pc1_endpoints': pc1_endpoints,
            'line_endpoints': {
                'left': left_point,
                'right': right_point
            },
            'points': quad_points
        }
    
    def find_pc1_endpoints(self, points, pca_results):
        """
        找到 PC1 方向上的两个端点
        
        参数:
            points: 草莓掩码点
            pca_results: PCA 分析结果
        
        返回:
            包含上下端点的字典
        """
        principal_components = pca_results['principal_components']
        center = pca_results['center']
        
        # PC1 方向向量
        pc1_vector = principal_components[0]
        pc1_vector_norm = pc1_vector / np.linalg.norm(pc1_vector)
        
        # 计算每个点到 PC1 方向的投影
        centered_points = points - center
        projections = np.dot(centered_points, pc1_vector_norm)
        
        # 找到最远的两个点作为端点
        min_projection_idx = np.argmin(projections)
        max_projection_idx = np.argmax(projections)
        
        pc1_bottom = points[min_projection_idx]  # PC1 方向下端点
        pc1_top = points[max_projection_idx]     # PC1 方向上端点
        
        return {
            'top': pc1_top,
            'bottom': pc1_bottom
        }
    
    def visualize_area_calculation(self, analysis_result, area_result, save_path=None):
        """
        可视化面积计算结果
        
        参数:
            analysis_result: 分析结果
            area_result: 面积计算结果
            save_path: 保存路径
        """
        image = analysis_result['image']
        strawberry_index = area_result['strawberry_index']
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：原始图像和关键点
        ax1.imshow(image)
        
        # 绘制关键点
        pc1_top = area_result['pc1_endpoints']['top']
        pc1_bottom = area_result['pc1_endpoints']['bottom']
        left_point = area_result['line_endpoints']['left']
        right_point = area_result['line_endpoints']['right']
        
        # 绘制四边形
        quad_points = area_result['points']
        quad_array = np.array(quad_points + [quad_points[0]])  # 闭合多边形
        
        # 根据字体可用性选择标签语言
        if has_chinese_font:
            quad_label = '面积计算四边形'
            pc1_top_label = 'PC1上端点'
            pc1_bottom_label = 'PC1下端点'
            left_label = '垂直线段左端点'
            right_label = '垂直线段右端点'
            title = f'草莓{strawberry_index + 1} - 面积计算关键点'
        else:
            quad_label = 'Quadrilateral for Area Calculation'
            pc1_top_label = 'PC1 Upper Endpoint'
            pc1_bottom_label = 'PC1 Lower Endpoint'
            left_label = 'Vertical Left Endpoint'
            right_label = 'Vertical Right Endpoint'
            title = f'Strawberry {strawberry_index + 1} - Key Points for Area Calculation'
        
        # 绘制四边形边界
        ax1.plot(quad_array[:, 0], quad_array[:, 1], 'b-', linewidth=3, alpha=0.8, label=quad_label)
        
        # 确保四个关键点都能正确显示，增大标记点的大小和zorder
        # PC1上下端点（红色，大圆点）
        ax1.scatter(pc1_top[0], pc1_top[1], c='red', s=20, marker='o', 
                   edgecolors='white', linewidths=3, label=pc1_top_label, zorder=6)
        ax1.scatter(pc1_bottom[0], pc1_bottom[1], c='red', s=20, marker='o', 
                   edgecolors='white', linewidths=3, label=pc1_bottom_label, zorder=6)
        
        # 左右端点（绿色，大圆点）
        ax1.scatter(left_point[0], left_point[1], c='green', s=20, marker='o', 
                   edgecolors='white', linewidths=3, label=left_label, zorder=6)
        ax1.scatter(right_point[0], right_point[1], c='green', s=20, marker='o', 
                   edgecolors='white', linewidths=3, label=right_label, zorder=6)
        
        # 添加风筝形状连接线：左右两个点分别和上下两个点连接
        # PC1上端点到左右端点
        ax1.plot([pc1_top[0], left_point[0]], [pc1_top[1], left_point[1]], 
                'cyan', linewidth=3, alpha=0.8, zorder=4)
        ax1.plot([pc1_top[0], right_point[0]], [pc1_top[1], right_point[1]], 
                'cyan', linewidth=3, alpha=0.8, zorder=4)
        
        # PC1下端点到左右端点
        ax1.plot([pc1_bottom[0], left_point[0]], [pc1_bottom[1], left_point[1]], 
                'cyan', linewidth=3, alpha=0.8, zorder=4)
        ax1.plot([pc1_bottom[0], right_point[0]], [pc1_bottom[1], right_point[1]], 
                'cyan', linewidth=3, alpha=0.8, zorder=4)
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        # 更新图例以包含风筝形状连接线
        if has_chinese_font:
            pc1_endpoints_label = 'PC1 端点'
            vertical_endpoints_label = '垂直线段端点'
            kite_connections_label = '风筝形状连接线'
        else:
            pc1_endpoints_label = 'PC1 Endpoints'
            vertical_endpoints_label = 'Vertical Line Endpoints'
            kite_connections_label = 'Kite-shaped Connections'
        
        legend_handles = [
            plt.Line2D([0], [0], color='blue', linewidth=3, label=quad_label),
            plt.scatter([], [], c='red', s=150, marker='o', edgecolors='white', 
                       linewidths=3, label=pc1_endpoints_label),
            plt.scatter([], [], c='green', s=150, marker='o', edgecolors='white', 
                       linewidths=3, label=vertical_endpoints_label),
            plt.Line2D([0], [0], color='cyan', linewidth=3, label=kite_connections_label)
        ]
        ax1.legend(handles=legend_handles)
        ax1.axis('off')
        
        # 右图：面积信息
        ax2.axis('off')
        
        # 根据字体可用性选择文本语言
        if has_chinese_font:
            area_info_text = f"""
草莓{strawberry_index + 1} 面积分析结果

四边形面积: {area_result['quadrilateral_area']:.2f} 像素²
实际掩码面积: {area_result['actual_mask_area']} 像素²
凸包面积: {area_result['hull_area']:.2f} 像素²

面积差异分析:
四边形 vs 实际: {abs(area_result['quadrilateral_area'] - area_result['actual_mask_area']):.2f}
四边形 vs 凸包: {abs(area_result['quadrilateral_area'] - area_result['hull_area']):.2f}

比例关系:
四边形/实际: {area_result['quadrilateral_area']/area_result['actual_mask_area']:.3f}
四边形/凸包: {area_result['quadrilateral_area']/area_result['hull_area']:.3f}
            """
        else:
            area_info_text = f"""
Strawberry {strawberry_index + 1} Area Analysis Results

Quadrilateral Area: {area_result['quadrilateral_area']:.2f} pixels²
Actual Mask Area: {area_result['actual_mask_area']} pixels²
Convex Hull Area: {area_result['hull_area']:.2f} pixels²

Area Difference Analysis:
Quadrilateral vs Actual: {abs(area_result['quadrilateral_area'] - area_result['actual_mask_area']):.2f}
Quadrilateral vs Hull: {abs(area_result['quadrilateral_area'] - area_result['hull_area']):.2f}

Proportion Analysis:
Quadrilateral/Actual: {area_result['quadrilateral_area']/area_result['actual_mask_area']:.3f}
Quadrilateral/Hull: {area_result['quadrilateral_area']/area_result['hull_area']:.3f}
            """
        
        ax2.text(0.1, 0.9, area_info_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 面积计算可视化结果已保存到: {save_path}")
        
        plt.show()
        
        return fig
    
    def batch_calculate_areas(self, analysis_results):
        """
        批量计算多个草莓的面积
        
        参数:
            analysis_results: 分析结果列表
        
        返回:
            面积结果列表
        """
        if not isinstance(analysis_results, list):
            analysis_results = [analysis_results]
        
        all_area_results = []
        
        for i, analysis_result in enumerate(analysis_results):
            if analysis_result is not None:
                line_results_list = analysis_result['line_results_list']
                
                for j in range(len(line_results_list)):
                    area_result = self.calculate_strawberry_area_from_analysis(analysis_result, j)
                    if area_result:
                        all_area_results.append(area_result)
        
        return all_area_results

def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='草莓像素面积计算器')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图片路径')
    parser.add_argument('--output', type=str, default=r"E:\Recent Works\2D-sizing\results",
                       help='输出文件夹路径')
    parser.add_argument('--model', type=str, default=r"E:\Recent Works\2D-sizing\weights\yolov11n-seg-086.pt",
                       help='模型文件路径')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--strawberry-index', type=int, default=0,
                       help='要计算的草莓索引（默认第一个）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("草莓像素面积计算器")
    print("=" * 60)
    
    try:
        # 导入草莓点查找器
        current_dir = Path(__file__).parent
        sys.path.append(str(current_dir))
        from find_points_v2 import StrawberryPointFinder
        
        # 初始化点查找器和面积计算器
        point_finder = StrawberryPointFinder(args.model)
        area_calculator = StrawberryAreaCalculator()
        
        # 分析图片
        print("正在分析图片...")
        analysis_result = point_finder.analyze_image(args.input, args.conf)
        
        if analysis_result is None:
            print("✗ 图片分析失败")
            return
        
        # 根据字体可用性选择输出语言
        if has_chinese_font:
            calculating_msg = f"正在计算第 {args.strawberry_index + 1} 个草莓的面积..."
            error_msg = "✗ 面积计算失败"
        else:
            calculating_msg = f"Calculating area for strawberry {args.strawberry_index + 1}..."
            error_msg = "✗ Area calculation failed"
        
        print(calculating_msg)
        area_result = area_calculator.calculate_strawberry_area_from_analysis(
            analysis_result, args.strawberry_index
        )
        
        if area_result is None:
            print(error_msg)
            return
        
        # 根据字体可用性选择输出语言 - 现在area_result已经可用
        if has_chinese_font:
            results_title = f"草莓 {args.strawberry_index + 1} 面积计算结果"
            quad_area = f"四边形面积: {area_result['quadrilateral_area']:.2f} 像素²"
            actual_area = f"实际掩码面积: {area_result['actual_mask_area']} 像素²"
            hull_area = f"凸包面积: {area_result['hull_area']:.2f} 像素²"
            diff = f"面积差异: {abs(area_result['quadrilateral_area'] - area_result['actual_mask_area']):.2f} 像素²"
            ratio = f"面积比例: {area_result['quadrilateral_area']/area_result['actual_mask_area']:.3f}"
        else:
            results_title = f"Strawberry {args.strawberry_index + 1} Area Calculation Results"
            quad_area = f"Quadrilateral Area: {area_result['quadrilateral_area']:.2f} pixels²"
            actual_area = f"Actual Mask Area: {area_result['actual_mask_area']} pixels²"
            hull_area = f"Convex Hull Area: {area_result['hull_area']:.2f} pixels²"
            diff = f"Area Difference: {abs(area_result['quadrilateral_area'] - area_result['actual_mask_area']):.2f} pixels²"
            ratio = f"Area Ratio: {area_result['quadrilateral_area']/area_result['actual_mask_area']:.3f}"
        
        # 显示结果
        print("\n" + "=" * 40)
        print(results_title)
        print("=" * 40)
        print(quad_area)
        print(actual_area)
        print(hull_area)
        print(diff)
        print(ratio)
        
        # 保存可视化结果
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = Path(args.input)
        save_path = output_dir / f"area_calculation_{input_path.stem}_strawberry_{args.strawberry_index + 1}.png"
        
        area_calculator.visualize_area_calculation(analysis_result, area_result, str(save_path))
        
        print(f"\n✓ 面积计算完成，结果已保存到: {save_path}")
        
    except Exception as e:
        print(f"✗ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()