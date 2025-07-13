"""
RAG检索链模块
实现基于检索增强生成的问答系统
"""
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# 处理依赖包可能未安装的情况
try:
    from langchain.chains import ConversationalRetrievalChain, RetrievalQA
    from langchain.chains.query_constructor.base import AttributeInfo
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.schema import BaseRetriever, Document
    from langchain.chains.base import Chain
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: LangChain dependencies not available. Please install requirements.txt")

from config.settings import get_settings
from config.prompts import PromptTemplates
from models.llm_config import get_llm, get_embeddings
from vectorstores.knowledge_base import KnowledgeBaseBuilder


class ECommerceRAGChain:
    """电商RAG检索链"""
    
    def __init__(self, 
                 llm: Optional[Any] = None,
                 vectorstore: Optional[Any] = None,
                 embeddings: Optional[Any] = None):
        self.settings = get_settings()
        self.llm = llm or get_llm()
        self.vectorstore = vectorstore
        self.embeddings = embeddings or get_embeddings()
        self.prompt_templates = PromptTemplates()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化向量存储
        if self.vectorstore is None:
            self._init_vectorstore()
        
        # 初始化链
        self._init_chains()
    
    def _init_vectorstore(self):
        """初始化向量存储"""
        if not DEPENDENCIES_AVAILABLE:
            return
        
        try:
            kb_builder = KnowledgeBaseBuilder(self.embeddings)
            self.vectorstore = kb_builder.build_knowledge_base(
                data_path=self.settings.data.knowledge_base_path,
                persist_directory=self.settings.vector_store.chroma_persist_directory,
                force_rebuild=False
            )
            
            if self.vectorstore is None:
                self.logger.error("Failed to initialize vectorstore")
            else:
                self.logger.info("Vectorstore initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing vectorstore: {e}")
    
    def _init_chains(self):
        """初始化检索链"""
        if not DEPENDENCIES_AVAILABLE or self.vectorstore is None or self.llm is None:
            return
        
        try:
            # 创建检索器
            self.retriever = self._create_retriever()
            
            # 创建RAG链
            self.rag_chain = self._create_rag_chain()
            
            # 创建对话链
            self.conversation_chain = self._create_conversation_chain()
            
            self.logger.info("RAG chains initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing chains: {e}")
            self.retriever = None
            self.rag_chain = None
            self.conversation_chain = None
    
    def _create_retriever(self) -> Optional[Any]:
        """创建检索器"""
        if not DEPENDENCIES_AVAILABLE or self.vectorstore is None:
            return None
        
        try:
            # 基础检索器
            base_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.settings.vector_store.retrieval_top_k,
                    "score_threshold": self.settings.vector_store.retrieval_score_threshold
                }
            )
            
            # 创建自查询检索器（如果支持）
            if hasattr(self.llm, 'predict') and self.llm is not None:
                try:
                    metadata_field_info = [
                        AttributeInfo(
                            name="source",
                            description="文档来源：FAQ、产品手册、规则文档等",
                            type="string",
                        ),
                        AttributeInfo(
                            name="category",
                            description="文档分类：FAQ、产品信息、规则政策、使用手册、服务支持等",
                            type="string",
                        ),
                        AttributeInfo(
                            name="priority",
                            description="文档优先级：1-5，数字越大优先级越高",
                            type="integer",
                        ),
                    ]
                    
                    self_query_retriever = SelfQueryRetriever.from_llm(
                        self.llm,
                        self.vectorstore,
                        "电商客服知识库，包含FAQ、产品信息、规则文档等内容",
                        metadata_field_info,
                        verbose=False
                    )
                    
                    return self_query_retriever
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create self-query retriever: {e}")
                    return base_retriever
            
            return base_retriever
            
        except Exception as e:
            self.logger.error(f"Error creating retriever: {e}")
            return None
    
    def _create_rag_chain(self) -> Optional[Any]:
        """创建RAG链"""
        if not DEPENDENCIES_AVAILABLE or self.retriever is None or self.llm is None:
            return None
        
        try:
            # 获取系统提示词
            system_prompt = self.prompt_templates.CUSTOMER_SERVICE_SYSTEM_PROMPT
            
            # 创建提示词模板
            prompt = ChatPromptTemplate.from_template(system_prompt)
            
            # 创建文档处理链
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            
            # 创建检索链
            retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
            
            return retrieval_chain
            
        except Exception as e:
            self.logger.error(f"Error creating RAG chain: {e}")
            return None
    
    def _create_conversation_chain(self) -> Optional[Any]:
        """创建对话链"""
        if not DEPENDENCIES_AVAILABLE or self.retriever is None or self.llm is None:
            return None
        
        try:
            # 创建对话检索链
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                return_source_documents=True,
                verbose=False,
                chain_type="stuff"
            )
            
            return conversation_chain
            
        except Exception as e:
            self.logger.error(f"Error creating conversation chain: {e}")
            return None
    
    def query(self, 
              question: str,
              chat_history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """执行查询"""
        if not DEPENDENCIES_AVAILABLE:
            return {
                "answer": "系统暂时不可用，请稍后再试。",
                "source_documents": [],
                "error": "Dependencies not available"
            }
        
        try:
            # 使用对话链（如果有历史记录）
            if chat_history and self.conversation_chain is not None:
                result = self.conversation_chain.invoke({
                    "question": question,
                    "chat_history": chat_history
                })
                
                return {
                    "answer": result.get("answer", "很抱歉，我无法为您提供相关信息。"),
                    "source_documents": result.get("source_documents", []),
                    "question": question,
                    "chat_history": chat_history
                }
            
            # 使用RAG链
            elif self.rag_chain is not None:
                result = self.rag_chain.invoke({
                    "input": question
                })
                
                return {
                    "answer": result.get("answer", "很抱歉，我无法为您提供相关信息。"),
                    "source_documents": result.get("context", []),
                    "question": question
                }
            
            # 回退到基础查询
            else:
                return self._fallback_query(question)
                
        except Exception as e:
            self.logger.error(f"Error in query: {e}")
            return {
                "answer": "很抱歉，处理您的问题时出现了错误。请稍后再试。",
                "source_documents": [],
                "error": str(e)
            }
    
    def _fallback_query(self, question: str) -> Dict[str, Any]:
        """回退查询方法"""
        try:
            if self.vectorstore is None:
                return {
                    "answer": "知识库暂时不可用。",
                    "source_documents": []
                }
            
            # 直接使用向量存储搜索
            docs = self.vectorstore.similarity_search(
                question,
                k=self.settings.vector_store.retrieval_top_k
            )
            
            if not docs:
                return {
                    "answer": "很抱歉，我在知识库中没有找到相关信息。",
                    "source_documents": []
                }
            
            # 构建上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 使用简单的模板
            if self.llm is not None:
                prompt = f"""
                基于以下信息回答用户问题：
                
                上下文：
                {context}
                
                问题：{question}
                
                请提供专业、准确的回答：
                """
                
                try:
                    answer = self.llm.predict(prompt)
                    return {
                        "answer": answer,
                        "source_documents": docs
                    }
                except Exception as e:
                    self.logger.error(f"Error in LLM prediction: {e}")
            
            # 如果LLM不可用，返回文档片段
            return {
                "answer": f"根据知识库信息：\n\n{context[:500]}...",
                "source_documents": docs
            }
            
        except Exception as e:
            self.logger.error(f"Error in fallback query: {e}")
            return {
                "answer": "很抱歉，查询过程中出现错误。",
                "source_documents": []
            }
    
    def add_documents(self, documents: List[Any]) -> bool:
        """添加新文档到知识库"""
        if not DEPENDENCIES_AVAILABLE or self.vectorstore is None:
            return False
        
        try:
            kb_builder = KnowledgeBaseBuilder(self.embeddings)
            success = kb_builder.add_documents_to_vectorstore(
                self.vectorstore,
                documents
            )
            
            if success:
                self.logger.info(f"Added {len(documents)} documents to knowledge base")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return False
    
    def search_documents(self, 
                        query: str, 
                        top_k: int = None,
                        score_threshold: float = None) -> List[Any]:
        """搜索相关文档"""
        if not DEPENDENCIES_AVAILABLE or self.vectorstore is None:
            return []
        
        try:
            top_k = top_k or self.settings.vector_store.retrieval_top_k
            score_threshold = score_threshold or self.settings.vector_store.retrieval_score_threshold
            
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # 过滤低分结果
            filtered_results = [
                doc for doc, score in results 
                if score >= score_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not DEPENDENCIES_AVAILABLE or self.vectorstore is None:
            return {}
        
        try:
            kb_builder = KnowledgeBaseBuilder(self.embeddings)
            stats = kb_builder.get_vectorstore_stats(self.vectorstore)
            
            stats.update({
                "llm_available": self.llm is not None,
                "retriever_available": self.retriever is not None,
                "rag_chain_available": self.rag_chain is not None,
                "conversation_chain_available": self.conversation_chain is not None
            })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}
    
    def update_settings(self, **kwargs):
        """更新设置"""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        
        # 重新初始化链
        self._init_chains()


class FAQMatcher:
    """FAQ精确匹配器"""
    
    def __init__(self, faq_data: Optional[List[Dict[str, str]]] = None):
        self.faq_data = faq_data or []
        self.logger = logging.getLogger(__name__)
    
    def load_faq_data(self, faq_file: str) -> bool:
        """加载FAQ数据"""
        try:
            import json
            with open(faq_file, 'r', encoding='utf-8') as f:
                self.faq_data = json.load(f)
            
            self.logger.info(f"Loaded {len(self.faq_data)} FAQ entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading FAQ data: {e}")
            return False
    
    def match(self, query: str) -> Optional[str]:
        """匹配FAQ"""
        query_lower = query.lower()
        
        for faq in self.faq_data:
            question = faq.get('question', '').lower()
            keywords = faq.get('keywords', [])
            
            # 精确匹配
            if query_lower == question:
                return faq.get('answer', '')
            
            # 关键词匹配
            if keywords:
                for keyword in keywords:
                    if keyword.lower() in query_lower:
                        return faq.get('answer', '')
        
        return None


def create_rag_chain(llm: Optional[Any] = None,
                    vectorstore: Optional[Any] = None,
                    embeddings: Optional[Any] = None) -> ECommerceRAGChain:
    """便捷函数：创建RAG链"""
    return ECommerceRAGChain(llm=llm, vectorstore=vectorstore, embeddings=embeddings)


# 导出主要类和函数
__all__ = [
    "ECommerceRAGChain",
    "FAQMatcher",
    "create_rag_chain"
] 