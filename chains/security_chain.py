"""
安全过滤链模块
实现敏感词过滤、内容合规检查、输出格式化等安全功能
"""
import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from filelock import FileLock

# 处理依赖包可能未安装的情况
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseOutputParser
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: LangChain dependencies not available. Please install requirements.txt")

from config.settings import get_settings
from config.prompts import PromptTemplates


class SecurityFilterChain:
    """安全过滤链"""
    
    def __init__(self, 
                 llm: Optional[Any] = None,
                 sensitive_words_path: str = None):
        self.settings = get_settings()
        self.llm = llm
        self.prompt_templates = PromptTemplates()
        
        # 敏感词列表
        self.sensitive_words = []
        self.sensitive_words_path = sensitive_words_path or self.settings.security.sensitive_words_path
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化
        self._load_sensitive_words()
        self._init_chains()
        
        # 审计日志
        self.audit_enabled = self.settings.security.enable_audit_log
        self.audit_log_path = self.settings.security.audit_log_path
        
        # 创建审计日志目录
        if self.audit_enabled:
            Path(self.audit_log_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _load_sensitive_words(self):
        """加载敏感词列表"""
        try:
            if os.path.exists(self.sensitive_words_path):
                with open(self.sensitive_words_path, 'r', encoding='utf-8') as f:
                    self.sensitive_words = [
                        line.strip() for line in f.readlines() 
                        if line.strip() and not line.startswith('#')
                    ]
                self.logger.info(f"Loaded {len(self.sensitive_words)} sensitive words")
            else:
                self.logger.warning(f"Sensitive words file not found: {self.sensitive_words_path}")
                self.sensitive_words = []
        except Exception as e:
            self.logger.error(f"Error loading sensitive words: {e}")
            self.sensitive_words = []
    
    def _init_chains(self):
        """初始化处理链"""
        if not DEPENDENCIES_AVAILABLE or self.llm is None:
            self.content_review_chain = None
            self.format_chain = None
            self.intent_chain = None
            return
        
        try:
            # 内容审核链
            self.content_review_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_templates.get_security_review_prompt(),
                verbose=False
            )
            
            # 格式化链
            self.format_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_templates.get_format_response_prompt(),
                verbose=False
            )
            
            # 意图识别链
            self.intent_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_templates.get_intent_classification_prompt(),
                verbose=False
            )
            
            self.logger.info("Security chains initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing security chains: {e}")
            self.content_review_chain = None
            self.format_chain = None
            self.intent_chain = None
    
    def filter_sensitive_words(self, text: str) -> Tuple[str, List[str]]:
        """过滤敏感词"""
        if not text:
            return text, []
        
        filtered_text = text
        detected_words = []
        
        for word in self.sensitive_words:
            if word.lower() in filtered_text.lower():
                # 使用正则表达式进行精确匹配
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                if pattern.search(filtered_text):
                    detected_words.append(word)
                    filtered_text = pattern.sub('*' * len(word), filtered_text)
        
        return filtered_text, detected_words
    
    def check_compliance(self, text: str) -> Dict[str, Any]:
        """合规性检查"""
        if not text:
            return {
                "is_compliant": True,
                "review_result": "文本为空",
                "confidence": 1.0
            }
        
        # 基础规则检查
        basic_check = self._basic_compliance_check(text)
        if not basic_check["is_compliant"]:
            return basic_check
        
        # 使用LLM进行深度检查
        if self.content_review_chain is not None:
            try:
                result = self.content_review_chain.invoke({"text": text})
                review_text = result.get("text", "").strip()
                
                is_compliant = "通过" in review_text or "合规" in review_text
                
                return {
                    "is_compliant": is_compliant,
                    "review_result": review_text,
                    "confidence": 0.8 if is_compliant else 0.9
                }
            except Exception as e:
                self.logger.error(f"Error in LLM compliance check: {e}")
                return {
                    "is_compliant": True,
                    "review_result": "LLM检查失败，使用基础检查",
                    "confidence": 0.5
                }
        
        # 如果LLM不可用，使用基础检查
        return basic_check
    
    def _basic_compliance_check(self, text: str) -> Dict[str, Any]:
        """基础合规性检查"""
        # 检查是否包含个人信息模式
        personal_info_patterns = [
            r'\b\d{15,18}\b',  # 身份证号
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # 银行卡号
            r'\b1[3-9]\d{9}\b',  # 手机号
            r'\b\w+@\w+\.\w+\b',  # 邮箱
            r'\b\d{6}\b'  # 验证码
        ]
        
        for pattern in personal_info_patterns:
            if re.search(pattern, text):
                return {
                    "is_compliant": False,
                    "review_result": "不通过：可能包含个人敏感信息",
                    "confidence": 0.9
                }
        
        # 检查是否包含不当承诺
        promise_keywords = [
            "100%", "绝对", "保证", "一定", "必然", "肯定能", "无条件"
        ]
        
        for keyword in promise_keywords:
            if keyword in text:
                return {
                    "is_compliant": False,
                    "review_result": f"不通过：包含不当承诺词汇 '{keyword}'",
                    "confidence": 0.8
                }
        
        # 检查长度限制
        if len(text) > 1000:
            return {
                "is_compliant": False,
                "review_result": "不通过：回复内容过长",
                "confidence": 0.7
            }
        
        return {
            "is_compliant": True,
            "review_result": "基础检查通过",
            "confidence": 0.7
        }
    
    def format_response(self, text: str) -> str:
        """格式化回复"""
        if not text:
            return "很抱歉，我无法为您提供回复。"
        
        # 使用LLM格式化
        if self.format_chain is not None:
            try:
                result = self.format_chain.invoke({"text": text})
                formatted_text = result.get("text", "").strip()
                
                if formatted_text:
                    return formatted_text
            except Exception as e:
                self.logger.error(f"Error in LLM formatting: {e}")
        
        # 基础格式化
        return self._basic_format_response(text)
    
    def _basic_format_response(self, text: str) -> str:
        """基础格式化"""
        # 清理文本
        cleaned_text = text.strip()
        
        # 添加问候语（如果没有）
        if not any(greeting in cleaned_text for greeting in ["您好", "你好", "Hello"]):
            cleaned_text = "您好！" + cleaned_text
        
        # 添加结尾礼貌用语（如果没有）
        polite_endings = ["谢谢", "感谢", "如有其他问题", "还有什么", "随时咨询"]
        if not any(ending in cleaned_text for ending in polite_endings):
            cleaned_text += "\n\n如有其他问题，请随时咨询。"
        
        return cleaned_text
    
    def classify_intent(self, text: str) -> str:
        """分类用户意图"""
        if not text:
            return "其他"
        
        # 使用LLM分类
        if self.intent_chain is not None:
            try:
                result = self.intent_chain.invoke({"question": text})
                intent = result.get("text", "").strip()
                
                # 验证意图类别
                valid_intents = ["咨询", "投诉", "退换货", "物流", "支付", "优惠", "其他"]
                if intent in valid_intents:
                    return intent
            except Exception as e:
                self.logger.error(f"Error in intent classification: {e}")
        
        # 基础关键词分类
        return self._basic_intent_classification(text)
    
    def _basic_intent_classification(self, text: str) -> str:
        """基础意图分类"""
        text_lower = text.lower()
        
        # 投诉相关
        if any(word in text_lower for word in ["投诉", "抱怨", "不满", "差评", "退款"]):
            return "投诉"
        
        # 退换货相关
        if any(word in text_lower for word in ["退货", "换货", "退换", "不要了"]):
            return "退换货"
        
        # 物流相关
        if any(word in text_lower for word in ["物流", "配送", "快递", "发货", "到货"]):
            return "物流"
        
        # 支付相关
        if any(word in text_lower for word in ["支付", "付款", "扣款", "充值", "余额"]):
            return "支付"
        
        # 优惠相关
        if any(word in text_lower for word in ["优惠", "折扣", "活动", "券", "促销"]):
            return "优惠"
        
        # 默认为咨询
        return "咨询"
    
    def process(self, 
                text: str,
                user_input: str = None,
                session_id: str = None) -> Dict[str, Any]:
        """完整的安全过滤流程"""
        try:
            # 记录开始时间
            start_time = datetime.now()
            
            # 1. 敏感词过滤
            filtered_text, detected_words = self.filter_sensitive_words(text)
            
            # 2. 合规性检查
            compliance_result = self.check_compliance(filtered_text)
            
            if not compliance_result["is_compliant"]:
                final_text = "很抱歉，该问题不便回答。如需帮助，请联系人工客服。"
                result = {
                    "success": False,
                    "filtered_text": final_text,
                    "original_text": text,
                    "detected_words": detected_words,
                    "compliance_check": compliance_result,
                    "reason": compliance_result["review_result"]
                }
            else:
                # 3. 格式化回复
                formatted_text = self.format_response(filtered_text)
                
                result = {
                    "success": True,
                    "filtered_text": formatted_text,
                    "original_text": text,
                    "detected_words": detected_words,
                    "compliance_check": compliance_result
                }
            
            # 4. 意图识别（如果提供了用户输入）
            if user_input:
                result["intent"] = self.classify_intent(user_input)
            
            # 5. 审计日志
            if self.audit_enabled:
                self._log_audit(
                    user_input=user_input,
                    original_output=text,
                    filtered_output=result["filtered_text"],
                    session_id=session_id,
                    detected_words=detected_words,
                    compliance_result=compliance_result,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in security processing: {e}")
            return {
                "success": False,
                "filtered_text": "很抱歉，处理过程中出现错误。请稍后再试。",
                "original_text": text,
                "error": str(e)
            }
    
    def _log_audit(self, **kwargs):
        """记录审计日志"""
        try:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": kwargs.get("session_id"),
                "user_input": kwargs.get("user_input"),
                "original_output": kwargs.get("original_output"),
                "filtered_output": kwargs.get("filtered_output"),
                "detected_words": kwargs.get("detected_words", []),
                "compliance_result": kwargs.get("compliance_result", {}),
                "processing_time": kwargs.get("processing_time", 0)
            }
            
            # 写入审计日志文件
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error writing audit log: {e}")
    
    def add_sensitive_word(self, word: str) -> bool:
        """添加敏感词（带文件锁保护）"""
        try:
            if word not in self.sensitive_words:
                self.sensitive_words.append(word)
                
                # 使用文件锁保护并发写入
                lock_file = self.sensitive_words_path + ".lock"
                with FileLock(lock_file):
                    with open(self.sensitive_words_path, 'a', encoding='utf-8') as f:
                        f.write(f'\n{word}')
                
                self.logger.info(f"Added sensitive word: {word}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error adding sensitive word: {e}")
            return False
    
    def remove_sensitive_word(self, word: str) -> bool:
        """移除敏感词（带文件锁保护）"""
        try:
            if word in self.sensitive_words:
                self.sensitive_words.remove(word)
                
                # 使用文件锁保护并发写入
                lock_file = self.sensitive_words_path + ".lock"
                with FileLock(lock_file):
                    with open(self.sensitive_words_path, 'w', encoding='utf-8') as f:
                        for w in self.sensitive_words:
                            f.write(f'{w}\n')
                
                self.logger.info(f"Removed sensitive word: {word}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing sensitive word: {e}")
            return False
    
    def get_sensitive_words(self) -> List[str]:
        """获取敏感词列表"""
        return self.sensitive_words.copy()
    
    def reload_sensitive_words(self) -> bool:
        """重新加载敏感词"""
        try:
            self._load_sensitive_words()
            return True
        except Exception as e:
            self.logger.error(f"Error reloading sensitive words: {e}")
            return False
    
    def get_audit_logs(self, 
                      session_id: str = None,
                      start_time: str = None,
                      end_time: str = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """获取审计日志"""
        try:
            if not os.path.exists(self.audit_log_path):
                return []
            
            logs = []
            with open(self.audit_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            log_entry = json.loads(line.strip())
                            
                            # 过滤条件
                            if session_id and log_entry.get("session_id") != session_id:
                                continue
                            
                            if start_time and log_entry.get("timestamp") < start_time:
                                continue
                            
                            if end_time and log_entry.get("timestamp") > end_time:
                                continue
                            
                            logs.append(log_entry)
                            
                            if len(logs) >= limit:
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
            return logs
            
        except Exception as e:
            self.logger.error(f"Error reading audit logs: {e}")
            return []
    
    def get_security_stats(self) -> Dict[str, Any]:
        """获取安全统计信息"""
        try:
            stats = {
                "sensitive_words_count": len(self.sensitive_words),
                "audit_enabled": self.audit_enabled,
                "chains_available": {
                    "content_review": self.content_review_chain is not None,
                    "format_chain": self.format_chain is not None,
                    "intent_chain": self.intent_chain is not None
                }
            }
            
            # 审计日志统计
            if self.audit_enabled and os.path.exists(self.audit_log_path):
                with open(self.audit_log_path, 'r', encoding='utf-8') as f:
                    log_count = sum(1 for line in f if line.strip())
                stats["audit_log_count"] = log_count
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting security stats: {e}")
            return {}


def create_security_chain(llm: Optional[Any] = None,
                         sensitive_words_path: str = None) -> SecurityFilterChain:
    """便捷函数：创建安全过滤链"""
    return SecurityFilterChain(llm=llm, sensitive_words_path=sensitive_words_path)


# 导出主要类和函数
__all__ = [
    "SecurityFilterChain",
    "create_security_chain"
] 