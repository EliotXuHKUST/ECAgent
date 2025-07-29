# ECAgent GPU环境部署指南

## 📋 目录
- [系统要求](#系统要求)
- [环境准备](#环境准备)
- [依赖安装](#依赖安装)
- [GPU配置](#gpu配置)
- [模型训练](#模型训练)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

## 🔧 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐 RTX 3090/4090, A100, V100 等)
- **显存**: 
  - 最低要求: 12GB (7B模型 + 4bit量化)
  - 推荐配置: 24GB+ (7B模型全精度训练)
  - 理想配置: 40GB+ (支持更大批次和更复杂模型)
- **内存**: 32GB+ 系统内存
- **存储**: 100GB+ 可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 20.04+) / Windows 10+ / macOS
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8+ 或 12.x
- **Docker**: 可选，用于容器化部署

## 🚀 环境准备

### 1. CUDA安装验证
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi

# 验证PyTorch CUDA支持
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
```

### 2. 创建虚拟环境
```bash
# 创建虚拟环境
python -m venv ecagent_gpu_env
source ecagent_gpu_env/bin/activate  # Linux/macOS
# 或 ecagent_gpu_env\Scripts\activate  # Windows

# 升级pip
pip install --upgrade pip
```

## 📦 依赖安装

### 1. 安装PyTorch (GPU版本)
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证安装
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 安装项目依赖
```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装额外GPU优化依赖
pip install flash-attn --no-build-isolation  # 可选，提升注意力计算效率
pip install deepspeed  # 可选，支持分布式训练
pip install ninja  # 编译加速
```

### 3. 验证关键组件
```bash
# 验证transformers
python -c "from transformers import AutoTokenizer; print('Transformers OK')"

# 验证PEFT
python -c "from peft import LoraConfig; print('PEFT OK')"

# 验证bitsandbytes
python -c "import bitsandbytes as bnb; print('Bitsandbytes OK')"

# 验证数据集加载
python -c "from datasets import load_dataset; print('Datasets OK')"
```

## ⚙️ GPU配置

### 1. 创建GPU训练脚本
```bash
# 复制并修改快速训练脚本
cp quick_train.py gpu_train.py
```

### 2. GPU训练配置文件
创建 `gpu_train.py`:

```python
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

def main():
    """主函数"""
    print("🚀 开始ECAgent GPU训练模式...")
    
    # 检查GPU环境
    if not check_gpu_environment():
        return
    
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
        
    except Exception as e:
        print(f"❌ GPU训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

### 3. 多GPU配置 (可选)
对于多GPU环境，创建 `multi_gpu_train.py`:

```python
#!/usr/bin/env python3
"""
多GPU分布式训练脚本
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup_distributed():
    """设置分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    dist.barrier()
    return True, rank, world_size, gpu

# 启动命令:
# torchrun --nproc_per_node=2 multi_gpu_train.py
```

## 🏃 模型训练

### 1. 快速验证训练
```bash
# 小规模测试 (约5-10分钟)
python gpu_train.py

# 自定义参数训练
CUDA_VISIBLE_DEVICES=0 MAX_SAMPLES=100 MAX_EPOCHS=1 python gpu_train.py
```

### 2. 完整训练流程
```bash
# 全量数据训练 (可能需要几小时)
CUDA_VISIBLE_DEVICES=0 \
MAX_SAMPLES=5000 \
MAX_EPOCHS=5 \
BATCH_SIZE=8 \
LEARNING_RATE=1e-4 \
python gpu_train.py
```

### 3. 多GPU训练 (如果有多个GPU)
```bash
# 使用2个GPU
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 multi_gpu_train.py

# 使用4个GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 multi_gpu_train.py
```

## ⚡ 性能优化

### 1. 内存优化配置
在训练脚本中添加:

```python
# GPU内存优化设置
torch.backends.cudnn.benchmark = True  # 优化卷积性能
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32
torch.backends.cudnn.allow_tf32 = True

# 梯度累积 (当GPU内存不够时)
gradient_accumulation_steps = 4
effective_batch_size = batch_size * gradient_accumulation_steps
```

### 2. 量化配置优化
```python
# 4bit量化配置 (节省显存)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # 使用bfloat16提升性能
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_quant_storage=torch.uint8
)
```

### 3. LoRA参数优化
```python
# 高性能LoRA配置
lora_config = LoraConfig(
    r=16,  # 增加rank提升性能
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

## 📊 监控和基准测试

### 1. 创建GPU监控脚本
创建 `monitor_gpu.py`:

```python
#!/usr/bin/env python3
"""GPU训练监控脚本"""
import torch
import time
import psutil
from datetime import datetime

def monitor_training():
    """监控训练过程中的GPU使用情况"""
    while True:
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # GPU信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory_used = torch.cuda.memory_allocated(i) / 1e9
                gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                gpu_utilization = gpu_memory_used / gpu_memory_total * 100
                
                print(f"[{current_time}] GPU {i}: {gpu_memory_used:.1f}GB/{gpu_memory_total:.1f}GB ({gpu_utilization:.1f}%)")
        
        # CPU和内存信息
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        print(f"[{current_time}] CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")
        print("-" * 50)
        
        time.sleep(10)

if __name__ == "__main__":
    monitor_training()
```

### 2. 创建性能基准测试
创建 `benchmark_gpu_model.py`:

```python
#!/usr/bin/env python3
"""GPU模型性能基准测试"""
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def benchmark_inference():
    """基准测试推理性能"""
    model_path = "./models/gpu_fine_tuned"
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 测试用例
    test_queries = [
        "如何退货？",
        "这个商品有保修吗？",
        "支持货到付款吗？",
        "快递多久能到？",
        "可以使用优惠券吗？"
    ]
    
    print("🧪 开始推理性能测试...")
    
    total_time = 0
    for i, query in enumerate(test_queries):
        start_time = time.time()
        
        inputs = tokenizer(f"用户：{query}\n客服：", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        
        print(f"Query {i+1}: {inference_time:.2f}s")
        print(f"Response: {response}")
        print("-" * 50)
    
    avg_time = total_time / len(test_queries)
    print(f"平均推理时间: {avg_time:.2f}s")
    print(f"吞吐量: {1/avg_time:.2f} queries/second")

if __name__ == "__main__":
    benchmark_inference()
```

## 🐛 常见问题

### 1. GPU内存不足
```bash
# 错误: CUDA out of memory
# 解决方案:
# 1. 减少batch_size
BATCH_SIZE=1 python gpu_train.py

# 2. 启用梯度检查点
GRADIENT_CHECKPOINTING=True python gpu_train.py

# 3. 使用4bit量化
USE_QUANTIZATION=True python gpu_train.py
```

### 2. CUDA版本不匹配
```bash
# 检查CUDA版本兼容性
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# 重新安装对应版本的PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 多GPU训练问题
```bash
# 检查分布式训练环境
python -c "import torch.distributed as dist; print('Distributed available')"

# 设置正确的环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=2
export RANK=0
```

### 4. 性能优化检查
```bash
# 检查是否启用了性能优化
python -c "
import torch
print(f'CUDNN Benchmark: {torch.backends.cudnn.benchmark}')
print(f'TF32 Matmul: {torch.backends.cuda.matmul.allow_tf32}')
print(f'TF32 CUDNN: {torch.backends.cudnn.allow_tf32}')
"
```

## 📈 推荐训练配置

### 单GPU配置 (RTX 3090/4090)
```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_SAMPLES=1000 \
MAX_EPOCHS=3 \
BATCH_SIZE=4 \
LEARNING_RATE=2e-4 \
USE_QUANTIZATION=True \
python gpu_train.py
```

### 高端GPU配置 (A100/V100)
```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_SAMPLES=5000 \
MAX_EPOCHS=5 \
BATCH_SIZE=8 \
LEARNING_RATE=1e-4 \
USE_QUANTIZATION=False \
python gpu_train.py
```

### 多GPU配置
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
multi_gpu_train.py \
--max_samples=10000 \
--epochs=5 \
--batch_size=16
```

---

## 🎯 快速开始

1. **验证环境**: `python -c "import torch; print(torch.cuda.is_available())"`
2. **安装依赖**: `pip install -r requirements.txt`
3. **快速训练**: `python gpu_train.py`
4. **测试模型**: `python simple_gradio_test.py`
5. **性能基准**: `python benchmark_gpu_model.py`

**预计训练时间**:
- 快速测试 (100样本): 5-10分钟
- 标准训练 (1000样本): 30-60分钟  
- 完整训练 (5000样本): 2-4小时

**内存需求**:
- 4bit量化: 8-12GB GPU内存
- 16bit半精度: 16-20GB GPU内存
- 32bit全精度: 28-32GB GPU内存