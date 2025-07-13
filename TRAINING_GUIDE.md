# ECAgent 模型训练指南

## 概述
使用 `Andyrasika/Ecommerce_FAQ` 数据集训练 ECAgent 电商客服模型。

## 环境要求
- Python 3.8+
- CUDA 支持的 GPU
- 至少 16GB GPU 内存（推荐 24GB+）

## 安装依赖
```bash
pip install -r requirements.txt
pip install datasets tiktoken einops transformers_stream_generator
```

## 训练选项

### 1. 快速训练（推荐先运行）
```bash
python quick_train.py --mode quick
```
- 使用 200 个样本
- 1 个训练轮次
- 适合测试环境和验证设置

### 2. 完整训练
```bash
python quick_train.py --mode full
```
- 使用完整数据集（79条记录）
- 3 个训练轮次
- 适合正式训练

### 3. 自定义训练
```bash
python train_ecommerce_faq.py
```
- 完全自定义的训练配置
- 1000 个样本限制
- 可以修改脚本中的参数

## 训练配置

### 默认参数
- **模型**: Qwen/Qwen-7B-Chat
- **量化**: 启用（4-bit QLoRA）
- **学习率**: 2e-4
- **批次大小**: 2-4
- **LoRA 参数**: r=8, alpha=32, dropout=0.1
- **目标模块**: Qwen专用（c_attn, c_proj, w1, w2）

### 内存优化
- 使用 4-bit 量化
- 梯度累积（steps=8）
- 启用 gradient checkpointing

## 数据处理

脚本会自动：
1. 从 HuggingFace 下载 `Andyrasika/Ecommerce_FAQ` 数据集
2. 转换为训练格式
3. 增强答案质量（添加礼貌用语）
4. 保存到 `./data/` 目录

## 输出文件

训练完成后会生成：
- `./models/fine_tuned_ecommerce_faq/` - 模型文件
- `./data/ecommerce_faq_train.json` - 处理后的训练数据
- `training.log` - 训练日志
- 自动更新的 `.env` 配置文件

## 使用微调模型

训练完成后，脚本会自动更新 `.env` 配置：
```bash
MODEL_FINE_TUNED_MODEL_PATH=./models/fine_tuned_ecommerce_faq
MODEL_USE_FINE_TUNED=true
```

重启 ECAgent 服务：
```bash
python simple_start.py
```

## 故障排除

### 1. CUDA 内存不足
- 减少 batch_size
- 增加 gradient_accumulation_steps
- 使用更小的模型

### 2. 依赖包问题
```bash
pip install --upgrade transformers torch
```

### 3. 模型下载失败
- 检查网络连接
- 设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 训练监控

查看训练进度：
```bash
tail -f training.log
```

## 性能建议

### 快速测试
- batch_size=1
- num_epochs=1
- 使用少量样本

### 高质量训练
- batch_size=4
- num_epochs=3-5
- 使用完整数据集

### 内存受限
- 启用量化
- 减少 batch_size
- 增加 gradient_accumulation_steps

## 训练时间估算

基于 V100 GPU：
- 快速训练：5-10分钟
- 完整训练：30-60分钟

## 验证训练效果

训练完成后，脚本会自动测试几个问题：
- 如何申请退货？
- 订单什么时候发货？
- 如何查看物流信息？
- 支付失败怎么办？

## 进阶配置

修改 `train_ecommerce_faq.py` 中的参数：
```python
model_path = train_ecommerce_model(
    data_path=data_path,
    model_name="Qwen/Qwen-7B-Chat",  # 可更换其他模型
    output_dir="./models/fine_tuned_ecommerce_faq",
    num_epochs=5,                    # 增加训练轮次
    batch_size=8,                    # 增加批次大小
    learning_rate=1e-4,              # 调整学习率
    use_quantization=True
)
```

## 注意事项

1. 确保有足够的磁盘空间（模型约 4-7GB）
2. 首次运行会下载模型，需要良好的网络连接
3. 量化已经启用，适合GPU服务器训练
4. 训练完成后记得备份模型文件 