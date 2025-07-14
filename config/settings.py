"""
ECAgent 配置管理模块
"""

import os
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """配置类"""
    
    # 基础配置
    environment: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=True, description="是否启用调试模式")
    
    # LLM配置
    model_llm_model_name: str = Field(default="Qwen/Qwen-7B-Chat", description="LLM模型名称")
    model_llm_device: str = Field(default="auto", description="LLM设备")
    model_llm_max_tokens: int = Field(default=512, description="LLM最大token数")
    model_llm_temperature: float = Field(default=0.1, description="LLM温度")
    
    # 嵌入模型配置
    model_embedding_model_name: str = Field(default="BAAI/bge-base-zh", description="嵌入模型名称")
    model_embedding_device: str = Field(default="cpu", description="嵌入模型设备")
    
    # 微调配置
    model_fine_tuned_model_path: Optional[str] = Field(default=None, description="微调模型路径")
    model_use_fine_tuned: bool = Field(default=False, description="是否使用微调模型")
    
    # 向量存储配置
    vector_chroma_persist_directory: str = Field(default="./chroma_db", description="Chroma持久化目录")
    vector_chroma_collection_name: str = Field(default="ecommerce_kb", description="Chroma集合名称")
    vector_retrieval_top_k: int = Field(default=5, description="检索top-k数量")
    vector_retrieval_score_threshold: float = Field(default=0.5, description="检索分数阈值")
    vector_chunk_size: int = Field(default=500, description="文档分割大小")
    vector_chunk_overlap: int = Field(default=50, description="文档重叠大小")
    
    # API配置
    api_api_host: str = Field(default="0.0.0.0", description="API主机")
    api_api_port: int = Field(default=8000, description="API端口")
    api_api_workers: int = Field(default=1, description="API工作进程数")
    api_cors_origins: List[str] = Field(default=["*"], description="CORS允许的源")
    api_cors_methods: List[str] = Field(default=["*"], description="CORS允许的方法")
    api_cors_headers: List[str] = Field(default=["*"], description="CORS允许的头部")
    
    # 前端配置
    frontend_gradio_host: str = Field(default="0.0.0.0", description="Gradio主机")
    frontend_gradio_port: int = Field(default=7860, description="Gradio端口")
    frontend_gradio_share: bool = Field(default=False, description="是否共享Gradio链接")
    frontend_ui_title: str = Field(default="ECAgent 电商客服助手", description="前端界面标题")
    frontend_ui_description: str = Field(default="智能电商客服系统", description="前端界面描述")
    
    # 安全配置
    security_sensitive_words_path: str = Field(default="./config/sensitive_words.txt", description="敏感词文件路径")
    security_session_timeout: int = Field(default=3600, description="会话超时时间（秒）")
    security_max_session_turns: int = Field(default=10, description="最大会话轮数")
    security_enable_audit_log: bool = Field(default=True, description="是否启用审计日志")
    security_audit_log_path: str = Field(default="./logs/audit.log", description="审计日志路径")
    
    # 数据配置
    data_knowledge_base_path: str = Field(default="./data/knowledge_base", description="知识库路径")
    data_train_data_path: str = Field(default="./data/train.json", description="训练数据路径")
    data_eval_data_path: str = Field(default="./data/eval.json", description="评估数据路径")
    data_log_level: str = Field(default="INFO", description="日志级别")
    data_log_file: str = Field(default="./logs/app.log", description="日志文件路径")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False,
    }

    # 添加property支持嵌套访问
    @property
    def api(self):
        """API配置访问器"""
        class _Api:
            def __init__(self, settings):
                self.api_host = settings.api_api_host
                self.api_port = settings.api_api_port
                self.api_workers = settings.api_api_workers
                self.cors_origins = settings.api_cors_origins
                self.cors_methods = settings.api_cors_methods
                self.cors_headers = settings.api_cors_headers
        return _Api(self)
    
    @property
    def model(self):
        """模型配置访问器"""
        class _Model:
            def __init__(self, settings):
                self.llm_model_name = settings.model_llm_model_name
                self.llm_device = settings.model_llm_device
                self.llm_max_tokens = settings.model_llm_max_tokens
                self.llm_temperature = settings.model_llm_temperature
                self.embedding_model_name = settings.model_embedding_model_name
                self.embedding_device = settings.model_embedding_device
                self.fine_tuned_model_path = settings.model_fine_tuned_model_path
                self.use_fine_tuned = settings.model_use_fine_tuned
        return _Model(self)
    
    @property
    def vector_store(self):
        """向量存储配置访问器"""
        class _VectorStore:
            def __init__(self, settings):
                self.chroma_persist_directory = settings.vector_chroma_persist_directory
                self.chroma_collection_name = settings.vector_chroma_collection_name
                self.retrieval_top_k = settings.vector_retrieval_top_k
                self.retrieval_score_threshold = settings.vector_retrieval_score_threshold
                self.chunk_size = settings.vector_chunk_size
                self.chunk_overlap = settings.vector_chunk_overlap
        return _VectorStore(self)
    
    @property
    def security(self):
        """安全配置访问器"""
        class _Security:
            def __init__(self, settings):
                self.sensitive_words_path = settings.security_sensitive_words_path
                self.session_timeout = settings.security_session_timeout
                self.max_session_turns = settings.security_max_session_turns
                self.enable_audit_log = settings.security_enable_audit_log
                self.audit_log_path = settings.security_audit_log_path
        return _Security(self)
    
    @property
    def data(self):
        """数据配置访问器"""
        class _Data:
            def __init__(self, settings):
                self.knowledge_base_path = settings.data_knowledge_base_path
                self.train_data_path = settings.data_train_data_path
                self.eval_data_path = settings.data_eval_data_path
                self.log_level = settings.data_log_level
                self.log_file = settings.data_log_file
        return _Data(self)
    
    @property
    def frontend(self):
        """前端配置访问器"""
        class _Frontend:
            def __init__(self, settings):
                self.gradio_host = settings.frontend_gradio_host
                self.gradio_port = settings.frontend_gradio_port
                self.gradio_share = settings.frontend_gradio_share
                self.ui_title = settings.frontend_ui_title
                self.ui_description = settings.frontend_ui_description
        return _Frontend(self)


# 全局设置实例
settings = Settings()


def get_settings() -> Settings:
    """获取设置实例"""
    return settings


def update_settings(**kwargs) -> None:
    """更新配置"""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


def load_settings_from_file(config_file: str) -> None:
    """从文件加载配置"""
    global settings
    if os.path.exists(config_file):
        # 正确地从指定文件加载配置
        old_env_file = os.environ.get('ENV_FILE')
        os.environ['ENV_FILE'] = config_file
        settings = Settings()
        if old_env_file:
            os.environ['ENV_FILE'] = old_env_file
        else:
            os.environ.pop('ENV_FILE', None)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")


def reload_settings() -> None:
    """重新加载配置"""
    global settings
    settings = Settings() 