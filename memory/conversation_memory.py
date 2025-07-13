"""
对话记忆管理模块
实现会话管理、记忆存储、超时处理等功能
"""
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# 处理依赖包可能未安装的情况
try:
    from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.schema.messages import SystemMessage
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: LangChain dependencies not available. Please install requirements.txt")

from config.settings import get_settings


class ECommerceConversationMemory:
    """电商对话记忆管理器"""
    
    def __init__(self, 
                 max_turns: int = None,
                 session_timeout: int = None,
                 persist_directory: str = None):
        self.settings = get_settings()
        
        # 配置参数
        self.max_turns = max_turns or self.settings.security_max_session_turns
        self.session_timeout = session_timeout or self.settings.security_session_timeout
        self.persist_directory = persist_directory or "./data/sessions"
        
        # 内存存储
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 创建持久化目录
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 加载已有会话
        self._load_sessions()
    
    def _load_sessions(self):
        """加载已有会话"""
        try:
            sessions_file = Path(self.persist_directory) / "sessions.json"
            if sessions_file.exists():
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.sessions = data.get('sessions', {})
                    self.session_metadata = data.get('metadata', {})
                
                self.logger.info(f"Loaded {len(self.sessions)} sessions")
        except Exception as e:
            self.logger.error(f"Error loading sessions: {e}")
            self.sessions = {}
            self.session_metadata = {}
    
    def _save_sessions(self):
        """保存会话到文件"""
        try:
            sessions_file = Path(self.persist_directory) / "sessions.json"
            data = {
                'sessions': self.sessions,
                'metadata': self.session_metadata,
                'saved_at': time.time()
            }
            
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving sessions: {e}")
    
    def create_session(self, 
                      session_id: str,
                      user_id: str = None,
                      metadata: Dict[str, Any] = None) -> bool:
        """创建新会话"""
        try:
            current_time = time.time()
            
            self.sessions[session_id] = {
                'messages': [],
                'created_at': current_time,
                'last_active': current_time,
                'user_id': user_id,
                'turn_count': 0,
                'is_active': True
            }
            
            self.session_metadata[session_id] = {
                'intent_history': [],
                'topics': [],
                'satisfaction_score': None,
                'escalated': False,
                'resolved': False,
                **(metadata or {})
            }
            
            self.logger.info(f"Created session {session_id}")
            self._save_sessions()
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating session {session_id}: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话"""
        if session_id not in self.sessions:
            return None
        
        # 检查会话是否过期
        if self._is_session_expired(session_id):
            self._expire_session(session_id)
            return None
        
        return self.sessions[session_id]
    
    def add_message(self, 
                   session_id: str,
                   user_message: str,
                   ai_response: str,
                   metadata: Dict[str, Any] = None) -> bool:
        """添加消息到会话"""
        try:
            session = self.get_session(session_id)
            if session is None:
                # 自动创建会话
                self.create_session(session_id)
                session = self.sessions[session_id]
            
            current_time = time.time()
            
            # 添加用户消息
            user_msg = {
                'type': 'human',
                'content': user_message,
                'timestamp': current_time,
                'metadata': metadata or {}
            }
            
            # 添加AI回复
            ai_msg = {
                'type': 'ai',
                'content': ai_response,
                'timestamp': current_time,
                'metadata': metadata or {}
            }
            
            session['messages'].extend([user_msg, ai_msg])
            session['last_active'] = current_time
            session['turn_count'] += 1
            
            # 保持窗口大小
            if len(session['messages']) > self.max_turns * 2:
                session['messages'] = session['messages'][-self.max_turns * 2:]
            
            self.logger.debug(f"Added message to session {session_id}")
            self._save_sessions()
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding message to session {session_id}: {e}")
            return False
    
    def get_chat_history(self, session_id: str) -> List[Tuple[str, str]]:
        """获取聊天历史（格式化为元组列表）"""
        session = self.get_session(session_id)
        if session is None:
            return []
        
        messages = session['messages']
        chat_history = []
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                
                if human_msg['type'] == 'human' and ai_msg['type'] == 'ai':
                    chat_history.append((human_msg['content'], ai_msg['content']))
        
        return chat_history
    
    def get_recent_messages(self, session_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """获取最近的消息"""
        session = self.get_session(session_id)
        if session is None:
            return []
        
        messages = session['messages']
        return messages[-count:] if messages else []
    
    def clear_session(self, session_id: str) -> bool:
        """清除会话"""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
            
            self.logger.info(f"Cleared session {session_id}")
            self._save_sessions()
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing session {session_id}: {e}")
            return False
    
    def _is_session_expired(self, session_id: str) -> bool:
        """检查会话是否过期"""
        if session_id not in self.sessions:
            return True
        
        session = self.sessions[session_id]
        current_time = time.time()
        last_active = session.get('last_active', 0)
        
        return (current_time - last_active) > self.session_timeout
    
    def _expire_session(self, session_id: str):
        """过期会话"""
        if session_id in self.sessions:
            self.sessions[session_id]['is_active'] = False
            self.logger.info(f"Expired session {session_id}")
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        expired_sessions = []
        
        for session_id in list(self.sessions.keys()):
            if self._is_session_expired(session_id):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.clear_session(session_id)
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """获取会话统计信息"""
        session = self.get_session(session_id)
        if session is None:
            return {}
        
        metadata = self.session_metadata.get(session_id, {})
        
        return {
            'session_id': session_id,
            'user_id': session.get('user_id'),
            'created_at': session.get('created_at'),
            'last_active': session.get('last_active'),
            'turn_count': session.get('turn_count', 0),
            'message_count': len(session.get('messages', [])),
            'is_active': session.get('is_active', True),
            'duration': time.time() - session.get('created_at', 0),
            'metadata': metadata
        }
    
    def get_all_sessions_stats(self) -> Dict[str, Any]:
        """获取所有会话统计信息"""
        total_sessions = len(self.sessions)
        active_sessions = sum(1 for s in self.sessions.values() if s.get('is_active', True))
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'expired_sessions': total_sessions - active_sessions,
            'session_timeout': self.session_timeout,
            'max_turns': self.max_turns
        }
    
    def update_session_metadata(self, 
                              session_id: str,
                              metadata: Dict[str, Any]) -> bool:
        """更新会话元数据"""
        try:
            if session_id not in self.session_metadata:
                self.session_metadata[session_id] = {}
            
            self.session_metadata[session_id].update(metadata)
            self._save_sessions()
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating session metadata {session_id}: {e}")
            return False
    
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """获取会话元数据"""
        return self.session_metadata.get(session_id, {})
    
    def search_sessions(self, 
                       user_id: str = None,
                       start_time: float = None,
                       end_time: float = None,
                       active_only: bool = False) -> List[str]:
        """搜索会话"""
        matching_sessions = []
        
        for session_id, session in self.sessions.items():
            # 按用户ID过滤
            if user_id and session.get('user_id') != user_id:
                continue
            
            # 按时间范围过滤
            created_at = session.get('created_at', 0)
            if start_time and created_at < start_time:
                continue
            if end_time and created_at > end_time:
                continue
            
            # 按活跃状态过滤
            if active_only and not session.get('is_active', True):
                continue
            
            matching_sessions.append(session_id)
        
        return matching_sessions
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """导出会话数据"""
        session = self.get_session(session_id)
        if session is None:
            return None
        
        metadata = self.get_session_metadata(session_id)
        stats = self.get_session_stats(session_id)
        
        return {
            'session_data': session,
            'metadata': metadata,
            'stats': stats,
            'exported_at': time.time()
        }
    
    def backup_sessions(self, backup_path: str = None) -> bool:
        """备份会话数据"""
        try:
            if backup_path is None:
                backup_path = f"./data/backups/sessions_backup_{int(time.time())}.json"
            
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
            
            backup_data = {
                'sessions': self.sessions,
                'metadata': self.session_metadata,
                'stats': self.get_all_sessions_stats(),
                'backup_time': time.time()
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Sessions backed up to {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error backing up sessions: {e}")
            return False


class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        self.memory = ECommerceConversationMemory()
        self.logger = logging.getLogger(__name__)
    
    def start_session(self, 
                     session_id: str,
                     user_id: str = None,
                     metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """开始新会话"""
        success = self.memory.create_session(session_id, user_id, metadata)
        
        if success:
            return {
                'status': 'success',
                'session_id': session_id,
                'message': '会话创建成功'
            }
        else:
            return {
                'status': 'error',
                'message': '会话创建失败'
            }
    
    def process_message(self, 
                       session_id: str,
                       user_message: str,
                       ai_response: str,
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理消息"""
        success = self.memory.add_message(session_id, user_message, ai_response, metadata)
        
        if success:
            stats = self.memory.get_session_stats(session_id)
            return {
                'status': 'success',
                'session_stats': stats,
                'message': '消息处理成功'
            }
        else:
            return {
                'status': 'error',
                'message': '消息处理失败'
            }
    
    def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """获取对话上下文"""
        chat_history = self.memory.get_chat_history(session_id)
        recent_messages = self.memory.get_recent_messages(session_id)
        stats = self.memory.get_session_stats(session_id)
        metadata = self.memory.get_session_metadata(session_id)
        
        return {
            'chat_history': chat_history,
            'recent_messages': recent_messages,
            'session_stats': stats,
            'metadata': metadata
        }
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """结束会话"""
        success = self.memory.clear_session(session_id)
        
        if success:
            return {
                'status': 'success',
                'message': '会话结束成功'
            }
        else:
            return {
                'status': 'error',
                'message': '会话结束失败'
            }
    
    def cleanup_sessions(self) -> Dict[str, Any]:
        """清理过期会话"""
        try:
            before_count = len(self.memory.sessions)
            self.memory.cleanup_expired_sessions()
            after_count = len(self.memory.sessions)
            
            return {
                'status': 'success',
                'cleaned_count': before_count - after_count,
                'remaining_count': after_count,
                'message': '会话清理完成'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'会话清理失败: {str(e)}'
            }


# 全局会话管理器实例
session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """获取会话管理器实例"""
    return session_manager


# 导出主要类和函数
__all__ = [
    "ECommerceConversationMemory",
    "SessionManager", 
    "get_session_manager"
] 