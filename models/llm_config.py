"""
LLM和嵌入模型配置模块
支持Qwen、ChatGLM等模型的配置和加载
"""
import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path

# 由于依赖包可能还未安装，我们先定义接口
try:
    from langchain.llms import HuggingFacePipeline
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: Some dependencies are not available. Please install requirements.txt")

from config.settings import get_settings


class LLMConfig:
    """LLM配置管理类"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_cache = {}
        self.tokenizer_cache = {}
    
    def get_device(self, preferred_device: str = "auto") -> str:
        """获取可用设备"""
        if preferred_device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return preferred_device
    
    def get_model_path(self, model_name: str) -> str:
        """获取模型路径"""
        # 支持本地路径和HuggingFace模型名称
        if os.path.exists(model_name):
            return model_name
        
        # 检查本地模型缓存
        local_models_dir = "./models/weights"
        local_path = os.path.join(local_models_dir, model_name.replace("/", "_"))
        if os.path.exists(local_path):
            return local_path
        
        # 返回HuggingFace模型名称
        return model_name
    
    def load_tokenizer(self, model_name: str, **kwargs) -> Optional[Any]:
        """加载tokenizer"""
        if not DEPENDENCIES_AVAILABLE:
            return None
        
        cache_key = f"{model_name}_{hash(str(kwargs))}"
        if cache_key in self.tokenizer_cache:
            return self.tokenizer_cache[cache_key]
        
        try:
            model_path = self.get_model_path(model_name)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs
            )
            
            # 设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.tokenizer_cache[cache_key] = tokenizer
            return tokenizer
        
        except Exception as e:
            print(f"Error loading tokenizer for {model_name}: {e}")
            return None
    
    def load_model(self, 
                   model_name: str, 
                   device: str = "auto",
                   torch_dtype: Optional[torch.dtype] = None,
                   **kwargs) -> Optional[Any]:
        """加载模型"""
        if not DEPENDENCIES_AVAILABLE:
            return None
        
        cache_key = f"{model_name}_{device}_{torch_dtype}_{hash(str(kwargs))}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        try:
            device = self.get_device(device)
            model_path = self.get_model_path(model_name)
            
            # 设置torch_dtype
            if torch_dtype is None:
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                **kwargs
            )
            
            self.model_cache[cache_key] = model
            return model
        
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def create_pipeline(self, 
                       model_name: str,
                       device: str = "auto",
                       max_new_tokens: int = 512,
                       temperature: float = 0.1,
                       do_sample: bool = True,
                       **kwargs) -> Optional[Any]:
        """创建模型推理pipeline"""
        if not DEPENDENCIES_AVAILABLE:
            return None
        
        try:
            device = self.get_device(device)
            
            # 加载tokenizer和model
            tokenizer = self.load_tokenizer(model_name)
            model = self.load_model(model_name, device)
            
            if tokenizer is None or model is None:
                return None
            
            # 创建pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                device=0 if device == "cuda" else -1,
                **kwargs
            )
            
            return pipe
        
        except Exception as e:
            print(f"Error creating pipeline for {model_name}: {e}")
            return None


class EmbeddingConfig:
    """嵌入模型配置管理类"""
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_cache = {}
    
    def get_embedding_model(self, 
                           model_name: str,
                           device: str = "cpu",
                           **kwargs) -> Optional[Any]:
        """获取嵌入模型"""
        if not DEPENDENCIES_AVAILABLE:
            return None
        
        cache_key = f"{model_name}_{device}_{hash(str(kwargs))}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True},
                **kwargs
            )
            
            self.embedding_cache[cache_key] = embeddings
            return embeddings
        
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            return None


# 全局配置实例
llm_config = LLMConfig()
embedding_config = EmbeddingConfig()


def get_llm(model_name: Optional[str] = None, 
           device: str = "auto",
           max_new_tokens: int = 512,
           temperature: float = 0.1,
           **kwargs) -> Optional[Any]:
    """获取LLM实例"""
    if not DEPENDENCIES_AVAILABLE:
        print("Dependencies not available. Please install requirements.txt")
        return None
    
    settings = get_settings()
    
    # 使用配置中的模型名称
    if model_name is None:
        model_name = settings.model.llm_model_name
    
    # 确保model_name不为None
    if not model_name:
        print("No model name provided")
        return None
    
    # 检查是否使用微调模型
    if settings.model.use_fine_tuned and settings.model.fine_tuned_model_path:
        return get_fine_tuned_llm(
            model_path=settings.model.fine_tuned_model_path,
            base_model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
    
    # 创建pipeline
    pipe = llm_config.create_pipeline(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs
    )
    
    if pipe is None:
        return None
    
    # 创建LangChain LLM包装器
    try:
        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
        )
        return llm
    except Exception as e:
        print(f"Error creating LangChain LLM wrapper: {e}")
        return None


def get_embeddings(model_name: Optional[str] = None,
                  device: str = "cpu",
                  **kwargs) -> Optional[Any]:
    """获取嵌入模型实例"""
    if not DEPENDENCIES_AVAILABLE:
        print("Dependencies not available. Please install requirements.txt")
        return None
    
    settings = get_settings()
    
    # 使用配置中的模型名称
    if model_name is None:
        model_name = settings.model.embedding_model_name
    
    # 确保model_name不为None
    if not model_name:
        print("No embedding model name provided")
        return None
    
    return embedding_config.get_embedding_model(
        model_name=model_name,
        device=device,
        **kwargs
    )


def get_fine_tuned_llm(model_path: str, 
                      base_model_name: str,
                      device: str = "auto",
                      max_new_tokens: int = 512,
                      temperature: float = 0.1,
                      **kwargs) -> Optional[Any]:
    """获取微调后的LLM实例"""
    if not DEPENDENCIES_AVAILABLE:
        print("Dependencies not available. Please install requirements.txt")
        return None
    
    try:
        device = llm_config.get_device(device)
        
        # 加载基础模型
        base_model = llm_config.load_model(base_model_name, device)
        if base_model is None:
            return None
        
        # 加载微调模型
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # 加载tokenizer
        tokenizer = llm_config.load_tokenizer(base_model_name)
        if tokenizer is None:
            return None
        
        # 创建pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=0 if device == "cuda" else -1,
            **kwargs
        )
        
        # 创建LangChain LLM包装器
        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
        )
        
        return llm
    
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return None


def validate_model_availability(model_name: str) -> bool:
    """验证模型是否可用"""
    if not DEPENDENCIES_AVAILABLE:
        return False
    
    try:
        # 尝试加载tokenizer来验证模型
        tokenizer = llm_config.load_tokenizer(model_name)
        return tokenizer is not None
    except Exception:
        return False


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """获取可用的模型列表"""
    models = {
        "qwen": {
            "name": "Qwen/Qwen-7B-Chat",
            "type": "causal_lm",
            "language": "zh",
            "license": "commercial",
            "size": "7B"
        },
        "chatglm3": {
            "name": "THUDM/chatglm3-6b",
            "type": "causal_lm", 
            "language": "zh",
            "license": "commercial",
            "size": "6B"
        },
        "baichuan": {
            "name": "baichuan-inc/Baichuan2-7B-Chat",
            "type": "causal_lm",
            "language": "zh",
            "license": "commercial",
            "size": "7B"
        }
    }
    
    # 检查模型可用性
    available_models = {}
    for key, model_info in models.items():
        if validate_model_availability(model_info["name"]):
            available_models[key] = model_info
    
    return available_models


def get_model_info(model_name: str) -> Dict[str, Any]:
    """获取模型信息"""
    available_models = get_available_models()
    
    for key, model_info in available_models.items():
        if model_info["name"] == model_name:
            return model_info
    
    return {
        "name": model_name,
        "type": "unknown",
        "language": "unknown",
        "license": "unknown",
        "size": "unknown"
    }


# 导出主要函数
__all__ = [
    "get_llm",
    "get_embeddings", 
    "get_fine_tuned_llm",
    "validate_model_availability",
    "get_available_models",
    "get_model_info",
    "LLMConfig",
    "EmbeddingConfig"
] 