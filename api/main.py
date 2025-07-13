"""
FastAPI主服务
集成RAG检索、安全过滤、对话记忆等模块
"""
import uuid
import time
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# 处理依赖包可能未安装的情况
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: FastAPI dependencies not available. Please install requirements.txt")

from config.settings import get_settings
from models.llm_config import get_llm, get_embeddings
from chains.rag_chain import ECommerceRAGChain
from chains.security_chain import SecurityFilterChain
from memory.conversation_memory import get_session_manager
from vectorstores.knowledge_base import KnowledgeBaseBuilder


# 请求/响应模型
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI回复")
    session_id: str = Field(..., description="会话ID")
    sources: Optional[List[str]] = Field(None, description="参考来源")
    filtered: bool = Field(False, description="是否被过滤")
    intent: Optional[str] = Field(None, description="用户意图")
    processing_time: float = Field(..., description="处理时间")


class SessionInfo(BaseModel):
    session_id: str
    user_id: Optional[str]
    created_at: float
    last_active: float
    turn_count: int
    message_count: int
    is_active: bool
    duration: float


class SystemStats(BaseModel):
    api_status: str
    llm_available: bool
    vectorstore_available: bool
    security_enabled: bool
    total_sessions: int
    active_sessions: int


# 全局变量
rag_chain: Optional[ECommerceRAGChain] = None
security_chain: Optional[SecurityFilterChain] = None
session_manager = None
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    await startup_event()
    yield
    # 关闭时清理
    await shutdown_event()


# 创建FastAPI应用
if DEPENDENCIES_AVAILABLE:
    app = FastAPI(
        title="ECAgent API",
        description="电商客服助手API服务",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=settings.api.cors_methods,
        allow_headers=settings.api.cors_headers,
    )
else:
    app = None


async def startup_event():
    """应用启动事件"""
    global rag_chain, security_chain, session_manager
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, settings.data.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting ECAgent API service...")
        
        # 初始化LLM和嵌入模型
        logger.info("Initializing models...")
        llm = get_llm()
        embeddings = get_embeddings()
        
        # 初始化知识库
        logger.info("Building knowledge base...")
        kb_builder = KnowledgeBaseBuilder(embeddings)
        vectorstore = kb_builder.build_knowledge_base(
            data_path=settings.data.knowledge_base_path,
            persist_directory=settings.vector_store.chroma_persist_directory,
            force_rebuild=False
        )
        
        # 初始化RAG链
        logger.info("Initializing RAG chain...")
        rag_chain = ECommerceRAGChain(
            llm=llm,
            vectorstore=vectorstore,
            embeddings=embeddings
        )
        
        # 初始化安全过滤链
        logger.info("Initializing security chain...")
        security_chain = SecurityFilterChain(
            llm=llm,
            sensitive_words_path=settings.security.sensitive_words_path
        )
        
        # 获取会话管理器
        session_manager = get_session_manager()
        
        logger.info("ECAgent API service started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start ECAgent API service: {e}")
        raise


async def shutdown_event():
    """应用关闭事件"""
    logger = logging.getLogger(__name__)
    logger.info("Shutting down ECAgent API service...")
    
    # 清理过期会话
    if session_manager:
        session_manager.cleanup_sessions()
    
    logger.info("ECAgent API service stopped")


def get_rag_chain() -> ECommerceRAGChain:
    """获取RAG链实例"""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    return rag_chain


def get_security_chain() -> SecurityFilterChain:
    """获取安全过滤链实例"""
    if security_chain is None:
        raise HTTPException(status_code=503, detail="Security chain not initialized")
    return security_chain


def get_session_manager_instance():
    """获取会话管理器实例"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    return session_manager


# API路由
if DEPENDENCIES_AVAILABLE:
    
    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "message": "ECAgent API is running"
        }
    
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        request: ChatRequest,
        rag_chain: ECommerceRAGChain = Depends(get_rag_chain),
        security_chain: SecurityFilterChain = Depends(get_security_chain),
        session_mgr = Depends(get_session_manager_instance)
    ):
        """聊天接口"""
        start_time = time.time()
        
        try:
            # 生成或使用现有会话ID
            session_id = request.session_id or str(uuid.uuid4())
            
            # 获取对话历史
            context = session_mgr.get_conversation_context(session_id)
            chat_history = context.get("chat_history", [])
            
            # 执行RAG查询
            rag_result = rag_chain.query(
                question=request.message,
                chat_history=chat_history if chat_history else None
            )
            
            # 提取答案和来源
            raw_answer = rag_result.get("answer", "很抱歉，我无法为您提供相关信息。")
            sources = []
            
            # 处理来源文档
            source_documents = rag_result.get("source_documents", [])
            if source_documents:
                for doc in source_documents[:3]:  # 只取前3个来源
                    if hasattr(doc, 'metadata') and doc.metadata:
                        source = doc.metadata.get('source', '')
                        if source:
                            sources.append(source)
            
            # 安全过滤
            security_result = security_chain.process(
                text=raw_answer,
                user_input=request.message,
                session_id=session_id
            )
            
            final_answer = security_result.get("filtered_text", raw_answer)
            filtered = not security_result.get("success", True)
            intent = security_result.get("intent", "其他")
            
            # 保存对话到会话
            session_mgr.process_message(
                session_id=session_id,
                user_message=request.message,
                ai_response=final_answer,
                metadata={
                    "sources": sources,
                    "intent": intent,
                    "filtered": filtered,
                    "user_id": request.user_id,
                    **(request.metadata or {})
                }
            )
            
            processing_time = time.time() - start_time
            
            return ChatResponse(
                response=final_answer,
                session_id=session_id,
                sources=sources,
                filtered=filtered,
                intent=intent,
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in chat endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/session/start")
    async def start_session(
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_mgr = Depends(get_session_manager_instance)
    ):
        """开始新会话"""
        try:
            session_id = str(uuid.uuid4())
            result = session_mgr.start_session(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.delete("/session/{session_id}")
    async def end_session(
        session_id: str,
        session_mgr = Depends(get_session_manager_instance)
    ):
        """结束会话"""
        try:
            result = session_mgr.end_session(session_id)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/session/{session_id}", response_model=SessionInfo)
    async def get_session_info(
        session_id: str,
        session_mgr = Depends(get_session_manager_instance)
    ):
        """获取会话信息"""
        try:
            context = session_mgr.get_conversation_context(session_id)
            stats = context.get("session_stats", {})
            
            if not stats:
                raise HTTPException(status_code=404, detail="Session not found")
            
            return SessionInfo(
                session_id=stats.get("session_id", session_id),
                user_id=stats.get("user_id"),
                created_at=stats.get("created_at", 0),
                last_active=stats.get("last_active", 0),
                turn_count=stats.get("turn_count", 0),
                message_count=stats.get("message_count", 0),
                is_active=stats.get("is_active", True),
                duration=stats.get("duration", 0)
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/session/cleanup")
    async def cleanup_sessions(
        background_tasks: BackgroundTasks,
        session_mgr = Depends(get_session_manager_instance)
    ):
        """清理过期会话"""
        try:
            result = session_mgr.cleanup_sessions()
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/stats", response_model=SystemStats)
    async def get_system_stats(
        rag_chain: ECommerceRAGChain = Depends(get_rag_chain),
        security_chain: SecurityFilterChain = Depends(get_security_chain),
        session_mgr = Depends(get_session_manager_instance)
    ):
        """获取系统统计信息"""
        try:
            # RAG统计
            rag_stats = rag_chain.get_stats()
            
            # 安全统计
            security_stats = security_chain.get_security_stats()
            
            # 会话统计
            session_stats = session_mgr.memory.get_all_sessions_stats()
            
            return SystemStats(
                api_status="healthy",
                llm_available=rag_stats.get("llm_available", False),
                vectorstore_available=rag_stats.get("document_count", 0) > 0,
                security_enabled=security_stats.get("audit_enabled", False),
                total_sessions=session_stats.get("total_sessions", 0),
                active_sessions=session_stats.get("active_sessions", 0)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/knowledge-base/stats")
    async def get_knowledge_base_stats(
        rag_chain: ECommerceRAGChain = Depends(get_rag_chain)
    ):
        """获取知识库统计信息"""
        try:
            stats = rag_chain.get_stats()
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/knowledge-base/search")
    async def search_knowledge_base(
        query: str,
        top_k: int = 5,
        rag_chain: ECommerceRAGChain = Depends(get_rag_chain)
    ):
        """搜索知识库"""
        try:
            results = rag_chain.search_documents(query, top_k)
            
            # 格式化结果
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                    "source": doc.metadata.get('source', '') if hasattr(doc, 'metadata') else ''
                })
            
            return {
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/security/stats")
    async def get_security_stats(
        security_chain: SecurityFilterChain = Depends(get_security_chain)
    ):
        """获取安全统计信息"""
        try:
            stats = security_chain.get_security_stats()
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/security/sensitive-words")
    async def get_sensitive_words(
        security_chain: SecurityFilterChain = Depends(get_security_chain)
    ):
        """获取敏感词列表"""
        try:
            words = security_chain.get_sensitive_words()
            return {
                "sensitive_words": words,
                "count": len(words)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/security/sensitive-words")
    async def add_sensitive_word(
        word: str,
        security_chain: SecurityFilterChain = Depends(get_security_chain)
    ):
        """添加敏感词"""
        try:
            success = security_chain.add_sensitive_word(word)
            return {
                "success": success,
                "message": "敏感词添加成功" if success else "敏感词已存在"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.delete("/security/sensitive-words/{word}")
    async def remove_sensitive_word(
        word: str,
        security_chain: SecurityFilterChain = Depends(get_security_chain)
    ):
        """移除敏感词"""
        try:
            success = security_chain.remove_sensitive_word(word)
            return {
                "success": success,
                "message": "敏感词移除成功" if success else "敏感词不存在"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/security/audit-logs")
    async def get_audit_logs(
        session_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        security_chain: SecurityFilterChain = Depends(get_security_chain)
    ):
        """获取审计日志"""
        try:
            logs = security_chain.get_audit_logs(
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            return {
                "logs": logs,
                "count": len(logs)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def create_app():
    """创建应用实例"""
    if not DEPENDENCIES_AVAILABLE:
        raise RuntimeError("Required dependencies not available. Please install requirements.txt")
    
    return app


if __name__ == "__main__":
    if DEPENDENCIES_AVAILABLE:
        uvicorn.run(
            "api.main:app",
            host=settings.api.api_host,
            port=settings.api.api_port,
            workers=settings.api.api_workers,
            reload=settings.debug,
            log_level=settings.data.log_level.lower()
        )
    else:
        print("Cannot start server: required dependencies not available")
        print("Please install requirements.txt") 