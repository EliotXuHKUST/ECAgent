#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 Andyrasika/Ecommerce_FAQ 数据集训练 ECAgent 模型
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from datasets import load_dataset
    from models.fine_tuning.train import ECommerceFineTuner
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: 依赖包未安装: {e}")
    print("请运行: pip install -r requirements.txt")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_and_process_ecommerce_faq(
    output_path: str = "./data/ecommerce_faq_train.json",
    max_samples: int = None
) -> str:
    """
    加载并处理 Andyrasika/Ecommerce_FAQ 数据集
    """
    logger.info("开始加载 Andyrasika/Ecommerce_FAQ 数据集...")
    
    try:
        # 加载数据集
        dataset = load_dataset("Andyrasika/Ecommerce_FAQ")
        
        # 获取训练数据
        train_data = dataset['train']
        logger.info(f"数据集加载成功，总计 {len(train_data)} 条记录")
        
        # 转换数据格式
        formatted_data = []
        for i, item in enumerate(train_data):
            if max_samples and i >= max_samples:
                break
                
            # 获取问题和答案
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            
            if not question or not answer:
                logger.warning(f"跳过空记录: {item}")
                continue
            
            # 增强答案格式，使其更像专业客服
            enhanced_answer = f"您好！{answer}"
            if not enhanced_answer.endswith(('。', '！', '？')):
                enhanced_answer += "。"
            enhanced_answer += "如有其他问题，请随时咨询。"
            
            formatted_data.append({
                "question": question,
                "answer": enhanced_answer
            })
        
        logger.info(f"数据处理完成，有效记录 {len(formatted_data)} 条")
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存处理后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据保存至: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        raise


def train_ecommerce_model(
    data_path: str,
    model_name: str = "Qwen/Qwen-7B-Chat",
    output_dir: str = "./models/fine_tuned_ecommerce_faq",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    use_quantization: bool = True
):
    """
    训练电商客服模型
    """
    logger.info("开始模型微调...")
    
    try:
        # 创建微调器
        fine_tuner = ECommerceFineTuner(
            model_name=model_name,
            output_dir=output_dir
        )
        
        # 开始训练
        fine_tuner.train(
            train_data_path=data_path,
            validation_split=0.1,
            use_quantization=use_quantization,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            save_steps=100,
            logging_steps=50,
            warmup_steps=50,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            eval_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        logger.info("模型微调完成！")
        
        # 测试模型
        logger.info("测试微调后的模型...")
        test_questions = [
            "如何申请退货？",
            "订单什么时候发货？",
            "如何查看物流信息？",
            "支付失败怎么办？"
        ]
        
        for question in test_questions:
            response = fine_tuner.generate_response(question)
            logger.info(f"问题: {question}")
            logger.info(f"回答: {response}")
            logger.info("-" * 50)
        
        return output_dir
        
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise


def update_env_config(model_path: str):
    """
    更新环境配置以启用微调模型
    """
    logger.info("更新环境配置...")
    
    env_file = ".env"
    if not os.path.exists(env_file):
        # 从模板创建 .env 文件
        if os.path.exists("env_template.txt"):
            with open("env_template.txt", 'r', encoding='utf-8') as f:
                content = f.read()
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            logger.warning("未找到 env_template.txt，请手动创建 .env 文件")
            return
    
    # 更新配置
    with open(env_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in lines:
        if line.startswith('MODEL_FINE_TUNED_MODEL_PATH='):
            updated_lines.append(f'MODEL_FINE_TUNED_MODEL_PATH={model_path}\n')
        elif line.startswith('MODEL_USE_FINE_TUNED='):
            updated_lines.append('MODEL_USE_FINE_TUNED=true\n')
        else:
            updated_lines.append(line)
    
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    logger.info("环境配置更新完成")


def main():
    """
    主函数：完整的训练流程
    """
    if not DEPENDENCIES_AVAILABLE:
        print("错误：依赖包未安装")
        print("请运行：pip install -r requirements.txt")
        return
    
    logger.info("开始 ECAgent 电商客服模型训练...")
    
    try:
        # 1. 加载和处理数据
        data_path = load_and_process_ecommerce_faq(
            output_path="./data/ecommerce_faq_train.json",
            max_samples=1000  # 限制样本数量以加快训练
        )
        
        # 2. 训练模型
        model_path = train_ecommerce_model(
            data_path=data_path,
            model_name="Qwen/Qwen-7B-Chat",
            output_dir="./models/fine_tuned_ecommerce_faq",
            num_epochs=3,
            batch_size=2,
            learning_rate=2e-4,
            use_quantization=True  # 启用量化，适用于GPU服务器
        )
        
        # 3. 更新环境配置
        update_env_config(model_path)
        
        logger.info("=" * 60)
        logger.info("训练完成！")
        logger.info(f"模型保存位置: {model_path}")
        logger.info("已自动更新 .env 配置文件")
        logger.info("现在可以重启 ECAgent 使用微调后的模型")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"训练过程失败: {e}")
        raise


if __name__ == "__main__":
    main() 