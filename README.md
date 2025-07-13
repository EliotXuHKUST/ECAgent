# ECAgent - 电商客服助手系统

基于LangChain和大语言模型的智能电商客服系统，具备多轮对话能力、知识检索能力、安全合规以及易部署等特点。

## 🌟 特性

- **🤖 智能对话**: 基于大语言模型的多轮对话能力
- **📚 知识检索**: RAG检索增强生成，精准回答用户问题
- **🛡️ 安全合规**: 多层安全过滤，敏感词检测，内容审核
- **💾 会话管理**: 完整的会话管理和记忆功能
- **🎨 友好界面**: 基于Gradio的美观用户界面
- **📊 系统监控**: 完整的系统统计和监控功能
- **🐳 容器化部署**: 支持Docker一键部署
- **🔧 模型微调**: 支持QLoRA/LoRA微调，适应业务场景

## 📋 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio 前端   │───▶│  LangChain 核心  │───▶│  LLM Provider   │
│   (聊天界面)    │    │   (Chain 编排)  │    │  (Qwen/ChatGLM) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ VectorStore     │
                       │ (Chroma/FAISS)  │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Memory & Tools  │
                       │ (对话记忆+工具) │
                       └─────────────────┘
```

## 🚀 快速开始

### 方式一：自动脚本启动（推荐）

```bash
# 克隆项目
git clone <your-repo-url>
cd ECAgent

# 运行启动脚本
chmod +x scripts/start.sh
./scripts/start.sh
```

### 方式二：手动安装

1. **环境要求**
   - Python 3.9+
   - CUDA 11.8+ (可选，用于GPU加速)
   - Docker (可选，用于容器化部署)

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据**
   ```bash
   # 创建必要目录
   mkdir -p data/knowledge_base
   
   # 将FAQ、产品手册等文档放入知识库目录
   cp your_faq.txt data/knowledge_base/
   ```

4. **启动服务**
   ```bash
   # 启动API服务
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
   
   # 启动前端服务
   python frontend/gradio_app.py
   ```

### 方式三：Docker部署

```bash
# 进入部署目录
cd deploy

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

## 📁 项目结构

```
ECAgent/
├── api/                     # API服务
│   ├── main.py             # FastAPI主入口
│   └── __init__.py
├── chains/                  # LangChain链
│   ├── rag_chain.py        # RAG检索链
│   ├── security_chain.py   # 安全过滤链
│   └── __init__.py
├── config/                  # 配置文件
│   ├── settings.py         # 系统配置
│   ├── prompts.py          # 提示词模板
│   └── sensitive_words.txt # 敏感词列表
├── frontend/                # 前端界面
│   ├── gradio_app.py       # Gradio应用
│   └── __init__.py
├── memory/                  # 记忆管理
│   ├── conversation_memory.py
│   └── __init__.py
├── models/                  # 模型相关
│   ├── llm_config.py       # LLM配置
│   ├── fine_tuning/        # 微调脚本
│   └── __init__.py
├── vectorstores/            # 向量存储
│   ├── knowledge_base.py   # 知识库构建
│   └── __init__.py
├── deploy/                  # 部署配置
│   ├── docker/
│   ├── docker-compose.yml
│   └── nginx.conf
├── scripts/                 # 脚本
│   ├── start.sh            # 启动脚本
│   └── stop.sh             # 停止脚本
├── data/                    # 数据目录
│   ├── knowledge_base/     # 知识库
│   └── sessions/           # 会话数据
└── requirements.txt         # 依赖文件
```

## 🛠️ 配置说明

### 环境变量配置

主要配置项可通过环境变量设置：

```bash
# 模型配置
export MODEL_LLM_MODEL_NAME="Qwen/Qwen-7B-Chat"
export MODEL_EMBEDDING_MODEL_NAME="BAAI/bge-base-zh"
export MODEL_LLM_DEVICE="auto"

# API配置
export API_API_HOST="0.0.0.0"
export API_API_PORT=8000

# 安全配置
export SECURITY_ENABLE_AUDIT_LOG=true
export SECURITY_SENSITIVE_WORDS_PATH="./config/sensitive_words.txt"

# 数据配置
export DATA_KNOWLEDGE_BASE_PATH="./data/knowledge_base"
export DATA_LOG_LEVEL="INFO"
```

### 知识库准备

支持多种文档格式：

```bash
data/knowledge_base/
├── faq/
│   ├── 售前FAQ.txt
│   ├── 售后FAQ.txt
│   └── 退换货FAQ.txt
├── products/
│   ├── 产品手册.pdf
│   └── 产品规格.csv
└── policies/
    ├── 服务条款.txt
    └── 隐私政策.txt
```

### 微调数据格式

```json
[
    {
        "instruction": "如何申请退货？",
        "input": "",
        "output": "您好！申请退货很简单：1. 登录您的账户 2. 找到相应订单 3. 点击申请退货 4. 填写退货原因 5. 提交申请。我们会在1-2个工作日内审核您的申请。"
    }
]
```

## 🔧 使用说明

### 1. 聊天界面

访问 `http://localhost:7860` 使用聊天界面：

- **智能客服**：主要的对话界面
- **系统管理**：查看系统状态、搜索知识库
- **安全管理**：管理敏感词、查看审计日志

### 2. API接口

访问 `http://localhost:8000/docs` 查看API文档：

- `POST /chat` - 聊天接口
- `GET /stats` - 系统统计
- `POST /session/start` - 开始会话
- `GET /security/stats` - 安全统计

### 3. 模型微调

```bash
# 运行微调脚本
python models/fine_tuning/train.py

# 或者自定义微调
python -c "
from models.fine_tuning.train import ECommerceFineTuner
trainer = ECommerceFineTuner()
trainer.train('data/train.json')
"
```

## 📊 系统监控

### 访问监控面板

- **Grafana**: `http://localhost:3000` (admin/admin)
- **Prometheus**: `http://localhost:9090`
- **API统计**: `http://localhost:8000/stats`

### 主要监控指标

- 系统状态和健康度
- 会话数量和活跃度
- 模型推理时间
- 安全过滤统计
- 知识库查询性能

## 🛡️ 安全特性

### 多层安全防护

1. **输入过滤**：敏感词检测和过滤
2. **内容审核**：基于规则和LLM的内容审核
3. **输出格式化**：统一的客服用语格式
4. **审计日志**：完整的操作记录
5. **会话管理**：安全的会话超时机制

### 敏感词管理

```bash
# 添加敏感词
curl -X POST "http://localhost:8000/security/sensitive-words" \
  -H "Content-Type: application/json" \
  -d '{"word": "新敏感词"}'

# 查看敏感词列表
curl "http://localhost:8000/security/sensitive-words"
```

## 🔄 维护操作

### 备份数据

```bash
# 备份知识库
cp -r data/knowledge_base backup/

# 备份会话数据
cp -r data/sessions backup/

# 备份配置
cp -r config backup/
```

### 清理日志

```bash
# 使用停止脚本清理
./scripts/stop.sh --clean-logs

# 手动清理
rm -rf logs/*.log
```

### 更新系统

```bash
# 停止服务
./scripts/stop.sh

# 更新代码
git pull

# 重新启动
./scripts/start.sh
```

## 🚨 故障排除

### 常见问题

1. **模型加载失败**
   ```bash
   # 检查显存
   nvidia-smi
   
   # 使用CPU模式
   export MODEL_LLM_DEVICE="cpu"
   ```

2. **知识库为空**
   ```bash
   # 检查文档路径
   ls -la data/knowledge_base/
   
   # 重建知识库
   python -c "from vectorstores.knowledge_base import create_knowledge_base; create_knowledge_base('data/knowledge_base', force_rebuild=True)"
   ```

3. **服务无法启动**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep :8000
   
   # 查看日志
   tail -f logs/app.log
   ```

### 性能优化

1. **GPU加速**
   ```bash
   # 确保CUDA可用
   python -c "import torch; print(torch.cuda.is_available())"
   
   # 设置GPU设备
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **内存优化**
   ```bash
   # 使用量化模型
   export MODEL_USE_QUANTIZATION=true
   
   # 减少批处理大小
   export MODEL_BATCH_SIZE=1
   ```

## 📝 开发指南

### 扩展功能

1. **添加新的Chain**
   ```python
   # 在chains/目录下创建新文件
   from langchain.chains import LLMChain
   
   class CustomChain:
       def __init__(self, llm):
           self.llm = llm
   ```

2. **添加新的工具**
   ```python
   # 在tools/目录下创建新工具
   from langchain.tools import BaseTool
   
   class CustomTool(BaseTool):
       name = "custom_tool"
       description = "工具描述"
   ```

### 测试

```bash
# 运行测试
python -m pytest tests/

# 测试特定模块
python -m pytest tests/test_api.py
```

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📞 支持

如果您遇到问题或有建议，请：

1. 查看 [FAQ](docs/FAQ.md)
2. 搜索 [Issues](https://github.com/your-repo/ECAgent/issues)
3. 创建新的 Issue
4. 联系开发团队

---

**ECAgent** - 让电商客服更智能，让用户体验更美好！ 🚀 