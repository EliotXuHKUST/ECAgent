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
    
    # 前端配置
    frontend_gradio_host: str = Field(default="0.0.0.0", description="Gradio主机")
    frontend_gradio_port: int = Field(default=7860, description="Gradio端口")
    frontend_gradio_share: bool = Field(default=False, description="是否共享Gradio链接")
    
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
        settings = Settings() 