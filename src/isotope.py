#!/usr/bin/env python3
"""
同位素参杂测试脚本
在不同离子数和参杂因子下运行模拟，并保存最后一帧的图片
"""
import subprocess
import os
import sys
from tqdm import tqdm

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
main_script = os.path.join(script_dir, "monolithic.py")

# 离子数列表
ion_numbers = [500]

# 参杂因子列表
doping_ratios = [0.05]

# 创建输出目录
output_dir = os.path.join(script_dir, "../data_cache/isotope")   #opt_VV: 指优化Velocity Verlet，即不需要将a_last保存到文件中
os.makedirs(output_dir, exist_ok=True)

# 计算总任务数
total_tasks = len(ion_numbers) * len(doping_ratios)
completed_tasks = 0

# 遍历所有参数组合（优先遍历离子数）
with tqdm(total=total_tasks, desc="总体进度", unit="任务", ncols=100) as pbar:
     for N in ion_numbers:
       for alpha in doping_ratios:
            # 生成文件名，包含离子数和参杂比例信息
            # 将参杂比例转换为字符串，去掉小数点
            alpha_str = str(alpha).replace('.', '')
            filename = f"cpu,opt_VV,flat_28.png"
            save_path = os.path.join(output_dir, filename)
            
            # 更新进度条描述
            pbar.set_description(f"运行中: N={N}, α={alpha}")
            
            print(f"\n{'='*60}")
            print(f"开始运行: 离子数={N}, 参杂因子={alpha}")
            print(f"保存路径: {save_path}")
            print(f"{'='*60}\n")
        
            # 构建命令行参数
            cmd = [
                sys.executable,
                main_script,
                "--N", str(N),
                "--alpha", str(alpha),
                "--time", "1000",  # 演化1000us
                "--save_final_image", save_path,  # 保存最后一帧图片
                "--plot",
            ]
            
            try:
                # 执行命令（直接运行，不监控进度条）
                result = subprocess.run(cmd, check=True, cwd=script_dir)
                
                completed_tasks += 1
                pbar.update(1)
                print(f"\n✓ 完成: 离子数={N}, 参杂因子={alpha}, 图片已保存到 {save_path}")
                print(f"进度: {completed_tasks}/{total_tasks} 任务已完成\n")
            except subprocess.CalledProcessError as e:
                completed_tasks += 1
                pbar.update(1)
                print(f"\n✗ 错误: 离子数={N}, 参杂因子={alpha} 运行失败")
                print(f"错误代码: {e.returncode}")
                print(f"进度: {completed_tasks}/{total_tasks} 任务已完成\n")
                # 继续执行下一个参数组合
                continue

print(f"\n{'='*60}")
print("所有测试完成！")
print(f"结果保存在: {output_dir}")
print(f"{'='*60}\n")
