"""
模型微调脚本
支持QLoRA和LoRA微调，适用于电商客服场景
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

# 处理依赖包可能未安装的情况
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback
    )
    from peft import (
        LoraConfig, get_peft_model,
        TaskType, prepare_model_for_kbit_training,
        PeftModel
    )
    from datasets import Dataset, load_dataset
    import bitsandbytes as bnb
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: Fine-tuning dependencies not available. Please install requirements.txt")

from config.settings import get_settings


class ECommerceFineTuner:
    """电商客服模型微调器"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen-7B-Chat",
                 output_dir: str = "./models/fine_tuned"):
        
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("Fine-tuning dependencies not available")
        
        self.settings = get_settings()
        self.model_name = model_name
        self.output_dir = output_dir
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 模型组件
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_model_and_tokenizer(self, use_quantization: bool = True):
        """设置模型和tokenizer"""
        self.logger.info(f"Loading model: {self.model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='right'
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 量化配置
        if use_quantization:
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
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # 加载普通模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora_config(self, 
                         r: int = 8,
                         lora_alpha: int = 32,
                         lora_dropout: float = 0.1,
                         target_modules: List[str] = None):
        """设置LoRA配置"""
        
        if target_modules is None:
            # 根据模型类型选择目标模块
            if "Qwen" in self.model_name:
                # Qwen模型的attention层名称
                target_modules = [
                    "c_attn", "c_proj", "w1", "w2"
                ]
            else:
                # 默认目标模块（适用于大多数模型）
                target_modules = [
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
        
        self.logger.info(f"Using target modules: {target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
        # 应用LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        self.peft_model.print_trainable_parameters()
        
        self.logger.info("LoRA configuration applied successfully")
    
    def load_dataset(self, data_path: str, validation_split: float = 0.1) -> tuple:
        """加载训练数据集"""
        self.logger.info(f"Loading dataset from: {data_path}")
        
        if data_path.endswith('.json'):
            # 加载JSON格式数据
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.jsonl'):
            # 加载JSONL格式数据
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # 转换数据格式
        formatted_data = []
        for item in data:
            # 构建电商客服对话格式
            if 'instruction' in item and 'output' in item:
                # 指令格式
                conversation = f"用户：{item['instruction']}\n客服：{item['output']}"
            elif 'input' in item and 'output' in item:
                # 输入输出格式
                conversation = f"用户：{item['input']}\n客服：{item['output']}"
            elif 'question' in item and 'answer' in item:
                # 问答格式
                conversation = f"用户：{item['question']}\n客服：{item['answer']}"
            else:
                self.logger.warning(f"Skipping item with unknown format: {item}")
                continue
            
            formatted_data.append({"text": conversation})
        
        self.logger.info(f"Loaded {len(formatted_data)} training examples")
        
        # 创建数据集
        dataset = Dataset.from_list(formatted_data)
        
        # 分割训练和验证集
        if validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        else:
            train_dataset = dataset
            eval_dataset = None
        
        return train_dataset, eval_dataset
    
    def tokenize_function(self, examples):
        """tokenize函数"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def create_training_arguments(self,
                                num_train_epochs: int = 3,
                                per_device_train_batch_size: int = 4,
                                gradient_accumulation_steps: int = 4,
                                learning_rate: float = 2e-4,
                                save_steps: int = 500,
                                logging_steps: int = 100,
                                warmup_steps: int = 100,
                                max_steps: int = -1,
                                weight_decay: float = 0.01,
                                lr_scheduler_type: str = "cosine",
                                **kwargs) -> TrainingArguments:
        """创建训练参数"""
        
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if kwargs.get('eval_dataset') else "no",
            eval_steps=save_steps if kwargs.get('eval_dataset') else None,
            save_total_limit=3,
            load_best_model_at_end=True if kwargs.get('eval_dataset') else False,
            metric_for_best_model="eval_loss" if kwargs.get('eval_dataset') else None,
            greater_is_better=False,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            lr_scheduler_type=lr_scheduler_type,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            run_name=f"ecommerce_finetune_{int(time.time())}" if 'time' in globals() else "ecommerce_finetune",
            **kwargs
        )
    
    def train(self, 
              train_data_path: str,
              validation_split: float = 0.1,
              use_quantization: bool = True,
              **training_kwargs):
        """训练模型"""
        
        try:
            # 1. 设置模型和tokenizer
            self.setup_model_and_tokenizer(use_quantization=use_quantization)
            
            # 2. 设置LoRA配置
            self.setup_lora_config()
            
            # 3. 加载数据集
            train_dataset, eval_dataset = self.load_dataset(
                train_data_path, 
                validation_split=validation_split
            )
            
            # 4. tokenize数据
            train_dataset = train_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            
            if eval_dataset:
                eval_dataset = eval_dataset.map(
                    self.tokenize_function,
                    batched=True,
                    remove_columns=eval_dataset.column_names
                )
            
            # 5. 数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # 6. 创建训练参数
            training_args = self.create_training_arguments(
                eval_dataset=eval_dataset,
                **training_kwargs
            )
            
            # 7. 创建callbacks
            callbacks = []
            if eval_dataset:
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
            
            # 8. 创建trainer
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                callbacks=callbacks
            )
            
            # 9. 开始训练
            self.logger.info("Starting training...")
            trainer.train()
            
            # 10. 保存模型
            self.logger.info("Saving model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # 11. 保存训练信息
            self.save_training_info(training_args, train_dataset, eval_dataset)
            
            self.logger.info(f"Training completed! Model saved to {self.output_dir}")
            
            return trainer
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_training_info(self, training_args, train_dataset, eval_dataset):
        """保存训练信息"""
        info = {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "training_args": training_args.to_dict(),
            "train_dataset_size": len(train_dataset),
            "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
            "timestamp": time.time() if 'time' in globals() else 0
        }
        
        info_path = os.path.join(self.output_dir, "training_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def load_fine_tuned_model(self, peft_model_path: str):
        """加载微调后的模型"""
        self.logger.info(f"Loading fine-tuned model from: {peft_model_path}")
        
        # 加载基础模型
        if self.model is None:
            self.setup_model_and_tokenizer(use_quantization=False)
        
        # 加载PEFT模型
        self.peft_model = PeftModel.from_pretrained(self.model, peft_model_path)
        
        self.logger.info("Fine-tuned model loaded successfully")
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """使用微调模型生成回复"""
        if self.peft_model is None:
            raise ValueError("Model not loaded. Please train or load a fine-tuned model first.")
        
        # 格式化输入
        formatted_prompt = f"用户：{prompt}\n客服："
        
        # tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.peft_model.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取客服回复部分
        if "客服：" in response:
            response = response.split("客服：")[-1].strip()
        
        return response
    
    def evaluate_model(self, test_data_path: str) -> Dict[str, float]:
        """评估模型性能"""
        if self.peft_model is None:
            raise ValueError("Model not loaded. Please train or load a fine-tuned model first.")
        
        # 加载测试数据
        test_dataset, _ = self.load_dataset(test_data_path, validation_split=0)
        
        # 简单评估（可以扩展为更复杂的评估指标）
        results = {
            "total_examples": len(test_dataset),
            "avg_response_length": 0,
            "generation_time": 0
        }
        
        total_length = 0
        start_time = time.time() if 'time' in globals() else 0
        
        for i, example in enumerate(test_dataset):
            if i >= 10:  # 只评估前10个样本作为示例
                break
            
            # 提取用户问题
            text = example["text"]
            if "用户：" in text:
                user_question = text.split("用户：")[1].split("客服：")[0].strip()
                response = self.generate_response(user_question)
                total_length += len(response)
        
        end_time = time.time() if 'time' in globals() else 0
        
        results["avg_response_length"] = total_length / min(10, len(test_dataset))
        results["generation_time"] = end_time - start_time
        
        return results


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """创建示例数据集"""
    sample_data = []
    
    # 示例对话数据
    examples = [
        {"instruction": "如何申请退货？", "output": "您好！申请退货很简单：1. 登录您的账户 2. 找到相应订单 3. 点击申请退货 4. 填写退货原因 5. 提交申请。我们会在1-2个工作日内审核您的申请。如有其他问题请随时咨询。"},
        {"instruction": "订单什么时候发货？", "output": "您好！订单发货时间根据商品类型和库存情况而定：1. 现货商品：通常在付款后24小时内发货 2. 预售商品：按照商品页面显示的发货时间 3. 定制商品：5-7个工作日。您可以在订单详情页查看具体发货时间。"},
        {"instruction": "如何查看物流信息？", "output": "您好！查看物流信息的方法：1. 登录账户进入\"我的订单\" 2. 找到相应订单点击\"查看物流\" 3. 或者直接在快递公司官网输入运单号查询。如果物流信息长时间未更新，请联系我们协助处理。"},
        {"instruction": "有什么优惠活动吗？", "output": "您好！我们经常举办各种优惠活动：1. 新用户注册即享受优惠券 2. 会员专享折扣 3. 节假日促销活动 4. 满减优惠。建议您关注我们的官方公众号获取最新活动信息。"},
        {"instruction": "支付失败怎么办？", "output": "您好！支付失败可能的原因和解决方法：1. 银行卡余额不足 - 请确认余额充足 2. 网络问题 - 请检查网络连接 3. 银行系统维护 - 请稍后重试 4. 支付限额 - 请联系银行调整限额。如仍无法解决，请联系客服协助。"}
    ]
    
    # 扩展样本数据
    for i in range(num_samples):
        sample = examples[i % len(examples)].copy()
        sample_data.append(sample)
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample dataset created at: {output_path}")


if __name__ == "__main__":
    # 创建示例数据集
    create_sample_dataset("./data/sample_train.json", num_samples=20)
    
    # 微调示例
    if DEPENDENCIES_AVAILABLE:
        try:
            fine_tuner = ECommerceFineTuner(
                model_name="Qwen/Qwen-7B-Chat",
                output_dir="./models/fine_tuned_ecommerce"
            )
            
            # 训练模型
            fine_tuner.train(
                train_data_path="./data/sample_train.json",
                validation_split=0.2,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                save_steps=50,
                logging_steps=10
            )
            
            # 测试生成
            response = fine_tuner.generate_response("如何申请退货？")
            print(f"Generated response: {response}")
            
        except Exception as e:
            print(f"Fine-tuning failed: {e}")
    else:
        print("Dependencies not available for fine-tuning") 