#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复训练脚本的 pad_token 问题
"""

import sys
from pathlib import Path
import logging
sys.path.insert(0, str(Path(__file__).parent))

try:
    from train_ecommerce_faq import load_and_process_ecommerce_faq, update_env_config
    from models.fine_tuning.train import ECommerceFineTuner
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: 依赖包未安装: {e}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixed_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FixedECommerceFineTuner(ECommerceFineTuner):
    """修复了 pad_token 问题的微调器"""
    
    def setup_model_and_tokenizer(self, use_quantization: bool = True):
        """设置模型和tokenizer - 修复版本"""
        self.logger.info(f"Loading model: {self.model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='right'
        )
        
        # 修复pad_token问题
        self._fix_pad_token()
        
        # 量化配置
        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # 加载量化模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # 准备模型进行量化训练
                from peft import prepare_model_for_kbit_training
                self.model = prepare_model_for_kbit_training(self.model)
                
            except ImportError:
                self.logger.warning("量化依赖不可用，使用普通模型")
                use_quantization = False
        
        if not use_quantization:
            # 加载普通模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # 如果添加了新的特殊token，需要调整模型嵌入层
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.logger.info(f"Resized model embeddings to {len(self.tokenizer)}")
        
        self.logger.info("Model and tokenizer loaded successfully")
    
    def _fix_pad_token(self):
        """修复pad_token问题"""
        self.logger.info("开始修复 pad_token 问题...")
        
        # 方法1: 使用eos_token作为pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
        
        # 方法2: 确保pad_token_id正确设置
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.info(f"Set pad_token_id to eos_token_id: {self.tokenizer.pad_token_id}")
        
        # 方法3: 为Qwen模型添加特殊处理
        if "Qwen" in self.model_name and self.tokenizer.pad_token is None:
            # 尝试添加特殊的pad_token
            special_tokens = {'pad_token': '<|endoftext|>'}
            self.tokenizer.add_special_tokens(special_tokens)
            self.logger.info("Added special pad_token for Qwen model")
        
        # 方法4: 最后检查，如果还是没有，强制设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token or "[PAD]"
            self.logger.info(f"Force set pad_token to: {self.tokenizer.pad_token}")
        
        # 最终确认
        self.logger.info(f"Final tokenizer pad_token: {self.tokenizer.pad_token}")
        self.logger.info(f"Final tokenizer pad_token_id: {self.tokenizer.pad_token_id}")
        
        # 测试padding功能
        try:
            test_texts = ["测试文本1", "测试文本2"]
            tokens = self.tokenizer(test_texts, padding=True, return_tensors="pt")
            self.logger.info("✅ Padding test passed!")
        except Exception as e:
            self.logger.error(f"Padding test failed: {e}")

# 修复的训练函数
def fixed_train_ecommerce_model(
    data_path: str,
    model_name: str = "Qwen/Qwen-7B-Chat",
    output_dir: str = "./models/fine_tuned_ecommerce_faq",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    use_quantization: bool = True
):
    """修复版本的训练函数"""
    logger.info("开始模型微调（修复版本）...")
    
    try:
        # 使用修复版本的微调器
        fine_tuner = FixedECommerceFineTuner(
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
            try:
                response = fine_tuner.generate_response(question)
                logger.info(f"问题: {question}")
                logger.info(f"回答: {response}")
                logger.info("-" * 50)
            except Exception as e:
                logger.error(f"测试问题失败: {e}")
        
        return output_dir
        
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise

def main():
    """主函数：使用修复版本的训练流程"""
    if not DEPENDENCIES_AVAILABLE:
        print("错误：依赖包未安装")
        print("请运行：pip install -r requirements.txt datasets tiktoken einops transformers_stream_generator")
        return
    
    logger.info("开始 ECAgent 电商客服模型训练（修复版本）...")
    
    try:
        # 1. 加载和处理数据
        data_path = load_and_process_ecommerce_faq(
            output_path="./data/ecommerce_faq_train.json",
            max_samples=1000  # 限制样本数量以加快训练
        )
        
        # 2. 使用修复版本的训练函数
        model_path = fixed_train_ecommerce_model(
            data_path=data_path,
            model_name="Qwen/Qwen-7B-Chat",
            output_dir="./models/fine_tuned_ecommerce_faq_fixed",
            num_epochs=3,
            batch_size=2,
            learning_rate=2e-4,
            use_quantization=True
        )
        
        # 3. 更新环境配置
        update_env_config(model_path)
        
        logger.info("=" * 60)
        logger.info("修复版本训练完成！")
        logger.info(f"模型保存位置: {model_path}")
        logger.info("已自动更新 .env 配置文件")
        logger.info("现在可以重启 ECAgent 使用微调后的模型")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"训练过程失败: {e}")
        raise

if __name__ == "__main__":
    main() 