#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU训练监控脚本
实时监控GPU、CPU和内存使用情况
"""

import torch
import time
import psutil
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def get_gpu_info():
    """获取GPU信息"""
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory_used = torch.cuda.memory_allocated(i) / 1e9
            gpu_memory_cached = torch.cuda.memory_reserved(i) / 1e9
            gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            gpu_utilization = gpu_memory_used / gpu_memory_total * 100
            
            gpu_info.append({
                'id': i,
                'name': gpu_name,
                'memory_used': gpu_memory_used,
                'memory_cached': gpu_memory_cached,
                'memory_total': gpu_memory_total,
                'utilization': gpu_utilization
            })
    return gpu_info

def get_system_info():
    """获取系统信息"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used': memory.used / 1e9,
        'memory_total': memory.total / 1e9,
        'disk_percent': disk.percent,
        'disk_used': disk.used / 1e9,
        'disk_total': disk.total / 1e9
    }

def print_header():
    """打印监控表头"""
    print("\n" + "="*80)
    print("🖥️  ECAgent GPU 训练监控")
    print("="*80)

def print_gpu_status(gpu_info):
    """打印GPU状态"""
    if not gpu_info:
        print("❌ 没有检测到GPU设备")
        return
    
    print("\n📱 GPU 状态:")
    for gpu in gpu_info:
        print(f"   GPU {gpu['id']}: {gpu['name']}")
        print(f"   ├─ 显存使用: {gpu['memory_used']:.1f}GB / {gpu['memory_total']:.1f}GB ({gpu['utilization']:.1f}%)")
        print(f"   └─ 缓存显存: {gpu['memory_cached']:.1f}GB")

def print_system_status(system_info):
    """打印系统状态"""
    print("\n💻 系统状态:")
    print(f"   CPU 使用率: {system_info['cpu_percent']:.1f}%")
    print(f"   内存使用: {system_info['memory_used']:.1f}GB / {system_info['memory_total']:.1f}GB ({system_info['memory_percent']:.1f}%)")
    print(f"   磁盘使用: {system_info['disk_used']:.1f}GB / {system_info['disk_total']:.1f}GB ({system_info['disk_percent']:.1f}%)")

def print_training_tips():
    """打印训练提示"""
    print("\n💡 训练提示:")
    print("   - 如果GPU利用率低，可以增加batch_size")
    print("   - 如果显存不足，可以启用梯度检查点或使用量化")
    print("   - 按 Ctrl+C 停止监控")

def monitor_training(refresh_interval=10):
    """监控训练过程"""
    print_header()
    
    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 清屏 (仅在终端中)
            if sys.stdout.isatty():
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                print_header()
            
            print(f"\n⏰ 监控时间: {current_time}")
            
            # 获取GPU信息
            gpu_info = get_gpu_info()
            print_gpu_status(gpu_info)
            
            # 获取系统信息
            system_info = get_system_info()
            print_system_status(system_info)
            
            # 打印提示
            if refresh_interval == 10:  # 只在首次显示
                print_training_tips()
                refresh_interval = 10
            
            print(f"\n🔄 下次刷新: {refresh_interval}秒后")
            print("-" * 80)
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
    except Exception as e:
        print(f"\n❌ 监控出错: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ECAgent GPU训练监控")
    parser.add_argument("--interval", "-i", type=int, default=10, 
                       help="刷新间隔(秒), 默认10秒")
    parser.add_argument("--once", action="store_true", 
                       help="只显示一次状态信息")
    
    args = parser.parse_args()
    
    if args.once:
        # 只显示一次状态
        print_header()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n⏰ 当前时间: {current_time}")
        
        gpu_info = get_gpu_info()
        print_gpu_status(gpu_info)
        
        system_info = get_system_info()
        print_system_status(system_info)
    else:
        # 持续监控
        monitor_training(args.interval)

if __name__ == "__main__":
    main()