#!/usr/bin/env python3
"""
草莓图片分割推理脚本
使用YOLOv11模型对草莓图片进行实例分割
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class StrawberrySegmentor:
    """草莓分割器类"""
    
    def __init__(self, model_path=None):
        """初始化分割器"""
        # 设置默认模型路径
        if model_path is None:
            # 检查是否有训练好的模型
            trained_model = Path("runs/segment/yolov11_papple_seg/weights/best.pt")
            if trained_model.exists():
                model_path = str(trained_model)
                print(f"✓ 使用训练好的模型: {model_path}")
            else:
                # 使用预训练模型
                model_path = "weights/yolo11n-seg.pt"
                print(f"✓ 使用预训练模型: {model_path}")
        
        # 加载模型
        try:
            self.model = YOLO(model_path)
            print(f"✓ 模型加载成功")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise
        
        # 设置类别名称（草莓）
        self.class_names = {0: 'strawberry'}
    
    def predict(self, image_path, conf_threshold=0.25, iou_threshold=0.7):
        """对单张图片进行预测"""
        
        # 检查图片是否存在
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        print(f"正在处理图片: {image_path}")
        
        # 使用模型进行预测
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            save=False,
            verbose=False
        )
        
        return results[0]  # 返回第一个结果（单张图片）
    
    def visualize_results(self, image_path, result, save_path=None):
        """可视化分割结果"""
        
        # 读取原始图片
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建可视化图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示原始图片
        ax1.imshow(image)
        ax1.set_title('原始图片')
        ax1.axis('off')
        
        # 显示分割结果
        ax2.imshow(image)
        
        # 绘制检测框和分割掩码
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            
            # 创建汇总掩码图
            combined_mask = np.zeros_like(image)
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # 提取边界框信息
                x1, y1, x2, y2, conf, cls = box
                
                # 绘制分割掩码
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_binary = mask_resized > 0.5
                
                # 创建彩色掩码（为每个草莓使用稍微不同的蓝色）
                color_mask = np.zeros_like(image)
                intensity = min(255, 180 + i * 20)  # 每个草莓蓝色深度不同
                color_mask[mask_binary, 2] = intensity  # 蓝色通道
                
                # 将当前掩码合并到汇总掩码中
                combined_mask[mask_binary] = color_mask[mask_binary]
                
                # 绘制边界框
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='blue', facecolor='none'
                )
                ax2.add_patch(rect)
                
                # 添加置信度标签
                ax2.text(x1, y1-10, f'草莓: {conf:.2f}', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.7),
                        fontsize=8, color='white')
            
            # 一次性叠加所有掩码
            ax2.imshow(combined_mask, alpha=0.6)
        
        ax2.set_title('分割结果')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # 保存或显示结果
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 结果已保存到: {save_path}")
        
        plt.show()
    
    def process_folder(self, folder_path, output_folder=None, conf_threshold=0.25):
        """处理整个文件夹的图片"""
        
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
            return
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 创建输出文件夹
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = folder_path / "segmentation_results"
            output_path.mkdir(exist_ok=True)
        
        # 处理每张图片
        results = []
        for i, image_file in enumerate(image_files):
            print(f"处理图片 {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # 进行预测
                result = self.predict(str(image_file), conf_threshold)
                
                # 保存可视化结果
                save_path = output_path / f"result_{image_file.stem}.png"
                self.visualize_results(str(image_file), result, str(save_path))
                
                # 记录结果
                if result.masks is not None:
                    num_strawberries = len(result.masks)
                    results.append({
                        'file': image_file.name,
                        'strawberries': num_strawberries,
                        'confidence': result.boxes.conf.mean().item() if result.boxes.conf.numel() > 0 else 0
                    })
                else:
                    results.append({
                        'file': image_file.name,
                        'strawberries': 0,
                        'confidence': 0
                    })
                
            except Exception as e:
                print(f"✗ 处理图片 {image_file.name} 时出错: {e}")
                results.append({
                    'file': image_file.name,
                    'strawberries': 0,
                    'confidence': 0,
                    'error': str(e)
                })
        
        # 生成结果报告
        self.generate_report(results, output_path)
        
        return results
    
    def generate_report(self, results, output_path):
        """生成处理报告"""
        
        report_path = output_path / "segmentation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("草莓分割结果报告\n")
            f.write("=" * 50 + "\n\n")
            
            total_images = len(results)
            total_strawberries = sum(r['strawberries'] for r in results)
            avg_confidence = np.mean([r['confidence'] for r in results if r['confidence'] > 0])
            
            f.write(f"总处理图片数: {total_images}\n")
            f.write(f"检测到草莓总数: {total_strawberries}\n")
            f.write(f"平均置信度: {avg_confidence:.3f}\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 50 + "\n")
            
            for result in results:
                f.write(f"文件: {result['file']}\n")
                f.write(f"  草莓数量: {result['strawberries']}\n")
                f.write(f"  平均置信度: {result['confidence']:.3f}\n")
                if 'error' in result:
                    f.write(f"  错误: {result['error']}\n")
                f.write("\n")
        
        print(f"✓ 处理报告已保存到: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='草莓图片分割推理脚本')
    parser.add_argument('--input', type=str, default=r"E:\Recent Works\2D-sizing\data\D405_dataset\images\D405_0006_20251112_170143.png",
                       help='输入图片路径或文件夹路径')
    parser.add_argument('--output', type=str, default=r"E:\Recent Works\2D-sizing\results\segmentation_result.png",
                       help='输出文件路径（仅对单张图片有效）')
    parser.add_argument('--model', type=str, default=r"E:\Recent Works\2D-sizing\weights\yolov11m-2c-seg.pt",
                       help='模型文件路径（可选，默认使用训练好的模型或预训练模型）')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 (0-1)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IoU阈值 (0-1)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("草莓图片分割推理工具")
    print("=" * 60)
    
    try:
        # 初始化分割器
        segmentor = StrawberrySegmentor(args.model)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 处理单张图片
            result = segmentor.predict(str(input_path), args.conf, args.iou)
            
            # 可视化结果
            # 确保输出目录存在
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            segmentor.visualize_results(str(input_path), result, str(output_path))
            
            # 显示统计信息
            if result.masks is not None:
                num_strawberries = len(result.masks)
                avg_confidence = result.boxes.conf.mean().item() if result.boxes.conf.numel() > 0 else 0
                print(f"✓ 检测到 {num_strawberries} 个草莓")
                print(f"✓ 平均置信度: {avg_confidence:.3f}")
            else:
                print("✗ 未检测到草莓")
                
        elif input_path.is_dir():
            # 处理文件夹
            results = segmentor.process_folder(
                str(input_path), 
                args.output, 
                args.conf
            )
            
            # 显示总体统计
            total_strawberries = sum(r['strawberries'] for r in results)
            print(f"✓ 处理完成！共检测到 {total_strawberries} 个草莓")
            
        else:
            print("✗ 输入路径不存在")
            
    except Exception as e:
        print(f"✗ 处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()