#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU环境下的ECAgent模型训练脚本
支持QLoRA量化训练和多GPU分布式训练
"""

import os
import sys
from pathlib import Path
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def update_env_config():
    """更新环境配置为GPU模式"""
    os.environ.update({
        "USE_GPU": "True",
        "CUDA_VISIBLE_DEVICES": "0",  # 指定使用的GPU
        "MAX_SAMPLES": "500",  # GPU环境可以处理更多样本
        "MAX_EPOCHS": "3",
        "BATCH_SIZE": "4",  # GPU可以使用更大的批次大小
        "LEARNING_RATE": "2e-4",
        "USE_QUANTIZATION": "True",  # 启用4bit量化
        "GRADIENT_CHECKPOINTING": "True",  # 启用梯度检查点
        "DATALOADER_NUM_WORKERS": "4"
    })

def check_gpu_environment():
    """检查GPU环境"""
    if not torch.cuda.is_available():
        print("❌ CUDA不可用！请检查GPU驱动和PyTorch安装")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU设备")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return True

def optimize_gpu_settings():
    """优化GPU性能设置"""
    # GPU内存优化设置
    torch.backends.cudnn.benchmark = True  # 优化卷积性能
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32
    torch.backends.cudnn.allow_tf32 = True
    
    print("✅ GPU性能优化设置已启用")

def main():
    """主函数"""
    print("🚀 开始ECAgent GPU训练模式...")
    
    # 检查GPU环境
    if not check_gpu_environment():
        return
    
    # 优化GPU设置
    optimize_gpu_settings()
    
    # 更新环境配置
    update_env_config()
    
    try:
        from train_ecommerce_faq import load_and_process_ecommerce_faq, train_ecommerce_model
        
        # 加载和处理数据
        print("📊 加载和处理数据...")
        data_path = load_and_process_ecommerce_faq(
            output_path="./data/ecommerce_faq_gpu_train.json",
            max_samples=int(os.environ.get("MAX_SAMPLES", 500))
        )
        
        # 开始GPU训练
        print("🎯 开始GPU模型训练...")
        print(f"📋 训练配置:")
        print(f"   - 样本数量: {os.environ.get('MAX_SAMPLES')}")
        print(f"   - 训练轮次: {os.environ.get('MAX_EPOCHS')}")
        print(f"   - 批次大小: {os.environ.get('BATCH_SIZE')}")
        print(f"   - 学习率: {os.environ.get('LEARNING_RATE')}")
        print(f"   - 量化训练: {os.environ.get('USE_QUANTIZATION')}")
        
        model_path = train_ecommerce_model(
            data_path=data_path,
            output_dir="./models/gpu_fine_tuned",
            num_epochs=int(os.environ.get("MAX_EPOCHS", 3)),
            batch_size=int(os.environ.get("BATCH_SIZE", 4)),
            learning_rate=float(os.environ.get("LEARNING_RATE", 2e-4)),
            use_quantization=os.environ.get("USE_QUANTIZATION", "True").lower() == "true"
        )
        
        print(f"✅ GPU训练完成！")
        print(f"📁 模型保存路径: {model_path}")
        print("\n🧪 测试建议:")
        print("1. 运行前端测试: python simple_gradio_test.py")
        print("2. 性能基准测试: python benchmark_gpu_model.py")
        print("3. 模型质量评估: python evaluate_model.py")
        print("4. GPU监控: python monitor_gpu.py")
        
    except Exception as e:
        print(f"❌ GPU训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()