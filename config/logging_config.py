"""
统一日志配置模块
避免多处重复调用logging.basicConfig
"""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from config.settings import get_settings

# 全局标志，确保只初始化一次
_logging_initialized = False


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    force_reinit: bool = False
) -> None:
    """
    设置统一的日志配置
    
    Args:
        log_level: 日志级别，默认从settings获取
        log_file: 日志文件路径，默认从settings获取
        force_reinit: 是否强制重新初始化
    """
    global _logging_initialized
    
    if _logging_initialized and not force_reinit:
        return
    
    settings = get_settings()
    
    # 使用传入参数或配置文件中的值
    log_level = log_level or settings.data.log_level
    log_file = log_file or settings.data.log_file
    
    # 创建日志目录
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 清除现有的handlers（如果强制重新初始化）
    if force_reinit:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置根日志级别
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.root.setLevel(numeric_level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        # 使用RotatingFileHandler避免日志文件过大
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
    
    _logging_initialized = True
    
    # 记录初始化完成
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    获取logger实例，确保日志已初始化
    
    Args:
        name: logger名称，通常使用__name__
        
    Returns:
        配置好的logger实例
    """
    # 确保日志已初始化
    if not _logging_initialized:
        setup_logging()
    
    return logging.getLogger(name)


def is_logging_initialized() -> bool:
    """检查日志是否已初始化"""
    return _logging_initialized


def reset_logging() -> None:
    """重置日志配置"""
    global _logging_initialized
    _logging_initialized = False
    
    # 清除所有handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 重置日志级别
    logging.root.setLevel(logging.WARNING) 