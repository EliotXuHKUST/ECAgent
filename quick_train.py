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
    快速训练配置 - 适合CPU环境测试
    """
    logger.info("开始快速训练模式（CPU环境）...")
    
    try:
        # 修改环境配置以适应CPU环境
        update_env_config({
            "USE_GPU": False,
            "MAX_SAMPLES": 20,  # 减少样本数量以加快训练
            "MAX_EPOCHS": 1     # 减少训练轮数
        })
        
        logger.info("开始加载 Andyrasika/Ecommerce_FAQ 数据集...")
        # 数据处理 - 使用更少的样本进行快速训练
        data_path = load_and_process_ecommerce_faq(
            output_path="./data/ecommerce_faq_quick.json",
            max_samples=20  # 使用20个样本进行快速训练
        )
        
        logger.info("开始模型微调...")
        model_path = train_ecommerce_model(
            data_path=data_path,
            output_dir="./models/quick_tuned",
            num_epochs=1,
            batch_size=1,
            learning_rate=5e-5,
            use_quantization=False  # 在CPU环境下禁用量化
        )
        
        # 3. 更新配置
        update_env_config(model_path)
        
        logger.info(f"快速训练完成！模型保存在: {model_path}")
        logger.info("可以使用以下方式测试模型:")
        logger.info("1. python test_quick_model.py")
        logger.info("2. python gradio_app.py")
        
        return model_path
        
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