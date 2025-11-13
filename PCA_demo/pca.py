#!/usr/bin/env python3
"""
草莓PCA分析脚本
使用草莓分割器获取掩码，然后对掩码区域进行PCA分析
可视化两个主要方向向量
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

class StrawberryPCAnalyzer:
    """草莓PCA分析器类"""
    
    def __init__(self, model_path=None):
        """初始化分析器"""
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
    
    def visualize_pca_results(self, points_list, image, pca_results_list, save_path=None):
        """可视化所有草莓的PCA结果"""
        
        # 根据草莓数量确定子图布局
        num_strawberries = len(points_list)
        
        # 创建足够大的图形
        fig = plt.figure(figsize=(5 * num_strawberries, 12))
        
        # 颜色列表用于区分不同的草莓
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, (points, pca_results) in enumerate(zip(points_list, pca_results_list)):
            color = colors[i % len(colors)]
            
            # 子图1: 原始图片和当前草莓掩码
            plt.subplot(3, num_strawberries, i + 1)
            plt.imshow(image)
            plt.scatter(points[:, 0], points[:, 1], c=color, s=1, alpha=0.3)
            plt.title(f'草莓 {i+1} 掩码区域')
            plt.axis('off')
            
            # 子图2: 当前草莓的PCA主成分方向
            plt.subplot(3, num_strawberries, num_strawberries + i + 1)
            plt.imshow(image)
            
            # 获取PCA结果
            principal_components = pca_results['principal_components']
            center = pca_results['center']
            
            # 绘制主成分方向
            scale = 100  # 缩放因子
            
            # 第一主成分
            plt.arrow(center[0], center[1], 
                     principal_components[0, 0] * scale, principal_components[0, 1] * scale, 
                     color='red', width=3, label='PC1')
            
            # 第二主成分
            plt.arrow(center[0], center[1], 
                     principal_components[1, 0] * scale, principal_components[1, 1] * scale, 
                     color='blue', width=3, label='PC2')
            
            plt.scatter(center[0], center[1], c='green', s=50, marker='o', label='中心点')
            plt.title(f'草莓 {i+1} PCA方向')
            plt.legend()
            plt.axis('off')
            
            # 子图3: 当前草莓的方差解释比例
            plt.subplot(3, num_strawberries, 2 * num_strawberries + i + 1)
            explained_variance_ratio = pca_results['explained_variance_ratio']
            labels = ['PC1', 'PC2']
            bar_colors = ['red', 'blue']
            plt.bar(labels, explained_variance_ratio, color=bar_colors)
            plt.title(f'草莓 {i+1} 方差解释比例')
            plt.ylabel('方差解释比例')
            plt.ylim(0, 1)
            
            # 在柱状图上添加数值标签
            for j, v in enumerate(explained_variance_ratio):
                plt.text(j, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 结果已保存到: {save_path}")
        
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
        print("草莓PCA分析开始")
        print("=" * 60)
        
        # 获取所有草莓的掩码点
        points_list, image = self.get_mask_points(image_path, conf_threshold)
        
        if points_list is None:
            print("✗ 无法获取掩码点，分析终止")
            return None
        
        # 执行PCA分析
        pca_results_list = self.perform_pca(points_list)
        
        # 如果提供了保存路径，自动处理文件名重复问题
        if save_path:
            save_path = self.get_next_available_filename(save_path)
        
        # 可视化结果
        fig = self.visualize_pca_results(points_list, image, pca_results_list, save_path)
        
        print(f"✓ 分析完成，共处理了 {len(pca_results_list)} 个草莓")
        
        # 返回分析结果
        return {
            'points_list': points_list,
            'image': image,
            'pca_results_list': pca_results_list,
            'figure': fig
        }
    
    def analyze_multiple_strawberries(self, folder_path, conf_threshold=0.25, output_folder=None):
        """分析多个草莓图片（处理每张图片中的所有草莓）"""
        
        folder_path = Path(folder_path)
        
        # 检查文件夹是否存在
        if not folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print("✗ 文件夹中没有找到图片文件")
            return []
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 创建输出文件夹
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = folder_path / "pca_analysis_results"
            output_path.mkdir(exist_ok=True)
        
        # 分析每张图片中的所有草莓
        results = []
        for i, image_file in enumerate(image_files):
            print(f"\n分析图片 {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # 分析图片中的所有草莓，自动处理文件名重复
                save_path = output_path / f"pca_result_{image_file.stem}.png"
                result = self.analyze_image(str(image_file), conf_threshold, str(save_path))
                
                if result is not None:
                    results.append({
                        'file': image_file.name,
                        'result': result,
                        'pca_components': [r['principal_components'] for r in result['pca_results_list']],
                        'variance_ratio': [r['explained_variance_ratio'] for r in result['pca_results_list']]
                    })
                
            except Exception as e:
                print(f"✗ 分析图片 {image_file.name} 时出错: {e}")
                results.append({
                    'file': image_file.name,
                    'error': str(e)
                })
        
        # 生成综合分析报告
        self.generate_comprehensive_report(results, output_path)
        
        return results
    
    def generate_comprehensive_report(self, results, output_path):
        """生成综合分析报告"""
        
        report_path = output_path / "pca_analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("草莓PCA分析综合报告\n")
            f.write("=" * 60 + "\n\n")
            
            successful_analyses = [r for r in results if 'result' in r]
            failed_analyses = [r for r in results if 'error' in r]
            
            f.write(f"总处理图片数: {len(results)}\n")
            f.write(f"成功分析数: {len(successful_analyses)}\n")
            f.write(f"失败分析数: {len(failed_analyses)}\n\n")
            
            if successful_analyses:
                # 计算平均方差解释率
                avg_variance_ratio = np.mean([r['variance_ratio'] for r in successful_analyses], axis=0)
                
                f.write("PCA分析统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"平均第一主成分解释方差: {avg_variance_ratio[0]:.3f}\n")
                f.write(f"平均第二主成分解释方差: {avg_variance_ratio[1]:.3f}\n")
                f.write(f"平均累计解释方差: {np.sum(avg_variance_ratio):.3f}\n\n")
                
                f.write("详细分析结果:\n")
                f.write("-" * 40 + "\n")
                
                for result in successful_analyses:
                    f.write(f"文件: {result['file']}\n")
                    f.write(f"  第一主成分解释方差: {result['variance_ratio'][0]:.3f}\n")
                    f.write(f"  第二主成分解释方差: {result['variance_ratio'][1]:.3f}\n")
                    f.write(f"  累计解释方差: {np.sum(result['variance_ratio']):.3f}\n\n")
            
            if failed_analyses:
                f.write("失败分析:\n")
                f.write("-" * 40 + "\n")
                for result in failed_analyses:
                    f.write(f"文件: {result['file']}\n")
                    f.write(f"  错误: {result['error']}\n\n")
        
        print(f"✓ 综合分析报告已保存到: {report_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='草莓PCA分析脚本')
    parser.add_argument('--input', type=str, default=r"E:\Recent Works\paper-Immature Green Apple Detection\data\strawberry\images\frame_00243.png",
                       help='输入图片路径或文件夹路径')
    parser.add_argument('--output', type=str, default=r"E:\Recent Works\paper-Immature Green Apple Detection\results",
                       help='输出文件夹路径（仅对文件夹输入有效）')
    parser.add_argument('--model', type=str, default=r"E:\Recent Works\paper-Immature Green Apple Detection\weights\yolov11n-seg-086.pt",
                       help='模型文件路径（可选，默认使用训练好的模型或预训练模型）')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 (0-1)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("草莓PCA分析工具")
    print("=" * 60)
    
    try:
        # 初始化分析器
        analyzer = StrawberryPCAnalyzer(args.model)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 分析单张图片，自动处理文件名重复
            # 为单张图片生成正确的保存路径
            output_path = Path(args.output)
            if output_path.is_dir():
                # 如果输出路径是文件夹，则在其中创建具体的文件路径
                save_path = output_path / f"pca_result_{input_path.stem}.png"
            else:
                # 如果输出路径是文件路径，直接使用
                save_path = args.output
            
            result = analyzer.analyze_image(
                str(input_path), 
                args.conf,
                str(save_path)
            )
            
            if result is not None:
                print(f"✓ 单张图片分析完成，共检测到 {len(result['pca_results_list'])} 个草莓")
                print(f"✓ 结果已保存到: {save_path}")
                
        elif input_path.is_dir():
            # 分析文件夹
            results = analyzer.analyze_multiple_strawberries(
                str(input_path), 
                args.conf,
                args.output
            )
            
            successful_analyses = [r for r in results if 'result' in r]
            print(f"✓ 批量分析完成！成功分析 {len(successful_analyses)} 张图片")
            
        else:
            print("✗ 输入路径不存在")
            
    except Exception as e:
        print(f"✗ 分析过程中出现错误: {e}")

if __name__ == "__main__":
    main()