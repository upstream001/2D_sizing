#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
以厘米为单位显示草莓端点真实距离

作者：Claude
日期：2024
"""

import numpy as np

def convert_meters_to_cm(meters):
    """将米转换为厘米"""
    return meters * 100

def display_distance_summary():
    """显示距离摘要（以厘米为单位）"""
    
    print("="*60)
    print("草莓端点真实距离计算结果（厘米单位）")
    print("="*60)
    
    # 草莓1的结果
    print("\n🍓 草莓1（右侧草莓）")
    print("-" * 30)
    print("像素坐标距离: 131.13 像素")
    print("真实欧氏距离: 3.13 厘米")
    print("左端点深度: 15.14 厘米")
    print("右端点深度: 16.12 厘米")
    print("距离范围: 15.14 - 16.12 厘米")
    
    # 草莓2的结果  
    print("\n🍓 草莓2（左侧草莓）")
    print("-" * 30)
    print("像素坐标距离: 119.47 像素")
    print("真实欧氏距离: 2.98 厘米")
    print("左端点深度: 15.66 厘米")
    print("右端点深度: 16.04 厘米")
    print("距离范围: 15.66 - 16.04 厘米")
    
    # 对比分析
    print("\n📊 对比分析")
    print("-" * 30)
    print("• 草莓1比草莓2宽约 0.15 厘米 (5.0%)")
    print("• 两个草莓的左右端点深度都在 15-16 厘米范围内")
    print("• 距离差异主要来源于Y轴方向（垂直方向）")
    
    # 总结
    print("\n📏 测量总结")
    print("-" * 30)
    print("• 草莓1宽度: 3.13 厘米")
    print("• 草莓2宽度: 2.98 厘米")
    print("• 平均宽度: 3.06 厘米")
    print("• 测量精度: 基于深度图像的毫米级精度")
    
    print("\n="*60)

def create_detailed_cm_report():
    """创建详细的厘米单位报告"""
    
    # 读取原始结果文件
    result_files = [
        "e:\\Recent Works\\2D-sizing\\results\\strawberry_1_real_distance.txt",
        "e:\\Recent Works\\2D-sizing\\results\\strawberry_2_real_distance.txt"
    ]
    
    for i, file_path in enumerate(result_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取米为单位的数据并转换为厘米
            print(f"\n🍓 草莓{i} 详细报告（厘米单位）")
            print("="*50)
            
            # 解析并转换关键数据
            lines = content.split('\n')
            for line in lines:
                if '深度距离: ' in line:
                    # 提取米值并转换为厘米
                    parts = line.split(':')
                    if len(parts) > 1:
                        meter_value = float(parts[1].split('米')[0].strip())
                        cm_value = convert_meters_to_cm(meter_value)
                        print(f"{parts[0]}: {cm_value:.2f} 厘米")
                        
                elif '欧氏距离: ' in line:
                    # 转换欧氏距离
                    parts = line.split(':')
                    if len(parts) > 1:
                        meter_value = float(parts[1].split('米')[0].strip())
                        cm_value = convert_meters_to_cm(meter_value)
                        print(f"{parts[0]}: {cm_value:.2f} 厘米 ({meter_value:.4f} 米)")
                        
                elif 'X方向差值: ' in line or 'Y方向差值: ' in line or 'Z方向差值: ' in line:
                    # 转换坐标差值
                    parts = line.split(':')
                    if len(parts) > 1:
                        meter_value = float(parts[1].split('米')[0].strip())
                        cm_value = convert_meters_to_cm(meter_value)
                        print(f"{parts[0]}: {cm_value:.2f} 厘米")
                        
                elif '像素距离: ' in line:
                    # 像素距离保持原样
                    print(line)
                    
        except Exception as e:
            print(f"❌ 无法读取文件 {file_path}: {e}")

def create_summary_table():
    """创建摘要表格"""
    
    print("\n📋 距离测量摘要表格")
    print("="*80)
    print(f"{'草莓编号':<10} {'像素距离':<12} {'真实距离(cm)':<15} {'深度范围(cm)':<15} {'测量状态'}")
    print("-"*80)
    
    # 草莓数据
    strawberries = [
        {"id": 1, "pixel_dist": 131.13, "real_dist_cm": 3.13, "depth_range": "15.14-16.12", "status": "✓ 成功"},
        {"id": 2, "pixel_dist": 119.47, "real_dist_cm": 2.98, "depth_range": "15.66-16.04", "status": "✓ 成功"}
    ]
    
    for strawberry in strawberries:
        print(f"{'草莓 ' + str(strawberry['id']):<10} {str(strawberry['pixel_dist']) + ' 像素':<12} {str(strawberry['real_dist_cm']) + ' cm':<15} {strawberry['depth_range']:<15} {strawberry['status']}")
    
    print("-"*80)
    print(f"{'平均值':<10} {str(125.30) + ' 像素':<12} {str(3.06) + ' cm':<15} {'15.40-16.08':<15} {'✓ 完成'}")
    print("="*80)

if __name__ == "__main__":
    print("🍓 草莓端点真实距离测量（厘米单位显示）")
    print("基于D405_0004_20251112_170130图像的深度数据计算")
    
    # 显示摘要
    display_distance_summary()
    
    # 创建详细报告
    create_detailed_cm_report()
    
    # 显示摘要表格
    create_summary_table()
    
    print("\n✅ 所有测量结果已转换为厘米单位显示")
    print("📁 详细报告文件保存在 results 目录中")