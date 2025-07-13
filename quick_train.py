#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速训练脚本 - 适用于GPU服务器
"""

import sys
from pathlib import Path
import logging
sys.path.insert(0, str(Path(__file__).parent))

from train_ecommerce_faq import load_and_process_ecommerce_faq, train_ecommerce_model, update_env_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quick_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def quick_train():
    """
    快速训练配置 - 适合GPU服务器测试
    """
    logger.info("开始快速训练模式（GPU服务器）...")
    
    try:
        # 1. 加载少量数据进行快速测试
        data_path = load_and_process_ecommerce_faq(
            output_path="./data/ecommerce_faq_quick.json",
            max_samples=200  # 使用200个样本进行快速训练
        )
        
        # 2. 快速训练配置
        model_path = train_ecommerce_model(
            data_path=data_path,
            model_name="Qwen/Qwen-7B-Chat",
            output_dir="./models/fine_tuned_quick",
            num_epochs=1,  # 只训练1个epoch
            batch_size=4,  # GPU服务器可以使用更大的批次
            learning_rate=3e-4,
            use_quantization=True  # 启用量化节省GPU内存
        )
        
        # 3. 更新配置
        update_env_config(model_path)
        
        logger.info("=" * 60)
        logger.info("快速训练完成！")
        logger.info(f"模型保存位置: {model_path}")
        logger.info("可以通过以下方式测试：")
        logger.info("1. 重启 ECAgent 服务")
        logger.info("2. 或直接运行 python simple_start.py")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"快速训练失败: {e}")
        raise

def full_train():
    """
    完整训练配置 - 适合GPU服务器正式训练
    """
    logger.info("开始完整训练模式（GPU服务器）...")
    
    try:
        # 1. 加载完整数据集
        data_path = load_and_process_ecommerce_faq(
            output_path="./data/ecommerce_faq_full.json",
            max_samples=None  # 使用完整数据集
        )
        
        # 2. 完整训练配置
        model_path = train_ecommerce_model(
            data_path=data_path,
            model_name="Qwen/Qwen-7B-Chat",
            output_dir="./models/fine_tuned_ecommerce_faq",
            num_epochs=3,
            batch_size=4,
            learning_rate=2e-4,
            use_quantization=True
        )
        
        # 3. 更新配置
        update_env_config(model_path)
        
        logger.info("=" * 60)
        logger.info("完整训练完成！")
        logger.info(f"模型保存位置: {model_path}")
        logger.info("已自动更新 .env 配置文件")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"完整训练失败: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ECAgent 训练脚本")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="训练模式：quick (快速测试) 或 full (完整训练)")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        quick_train()
    else:
        full_train() 