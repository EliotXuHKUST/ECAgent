"""
知识库构建和管理模块
支持多种文档格式的加载、分割和向量化存储
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# 处理依赖包可能未安装的情况
try:
    from langchain.document_loaders import (
        TextLoader, 
        PyPDFLoader, 
        CSVLoader,
        JSONLoader,
        DirectoryLoader,
        UnstructuredMarkdownLoader,
        UnstructuredWordDocumentLoader
    )
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        TokenTextSplitter
    )
    from langchain.vectorstores import Chroma
    from langchain.schema import Document
    from langchain.embeddings.base import Embeddings
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: LangChain dependencies not available. Please install requirements.txt")

from config.settings import get_settings
from models.llm_config import get_embeddings


class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(self, embeddings: Optional[Any] = None):
        self.settings = get_settings()
        self.embeddings = embeddings or get_embeddings()
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, self.settings.data.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_text_splitter(self, 
                         splitter_type: str = "recursive",
                         chunk_size: int = None,
                         chunk_overlap: int = None) -> Optional[Any]:
        """获取文本分割器"""
        if not DEPENDENCIES_AVAILABLE:
            return None
        
        chunk_size = chunk_size if chunk_size is not None else self.settings.vector_store.chunk_size
        chunk_overlap = chunk_overlap if chunk_overlap is not None else self.settings.vector_store.chunk_overlap
        
        if splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            )
        elif splitter_type == "character":
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n\n"
            )
        elif splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            self.logger.error(f"Unknown splitter type: {splitter_type}")
            return None
    
    def get_document_loader(self, file_path: str) -> Optional[Any]:
        """根据文件类型获取文档加载器"""
        if not DEPENDENCIES_AVAILABLE:
            return None
        
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.txt':
                return TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                return PyPDFLoader(file_path)
            elif file_extension == '.csv':
                return CSVLoader(file_path)
            elif file_extension == '.json':
                return JSONLoader(
                    file_path=file_path,
                    jq_schema='.[]' if self._is_json_array(file_path) else '.',
                    text_content=False
                )
            elif file_extension == '.md':
                return UnstructuredMarkdownLoader(file_path)
            elif file_extension in ['.doc', '.docx']:
                return UnstructuredWordDocumentLoader(file_path)
            else:
                self.logger.warning(f"Unsupported file type: {file_extension}")
                return None
        except Exception as e:
            self.logger.error(f"Error creating loader for {file_path}: {e}")
            return None
    
    def _is_json_array(self, file_path: str) -> bool:
        """检查JSON文件是否为数组格式"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return isinstance(data, list)
        except Exception:
            return False
    
    def load_documents_from_file(self, file_path: str) -> List[Any]:
        """从单个文件加载文档"""
        if not DEPENDENCIES_AVAILABLE:
            return []
        
        loader = self.get_document_loader(file_path)
        if loader is None:
            return []
        
        try:
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': Path(file_path).suffix.lower(),
                    'category': self._get_category_from_path(file_path),
                    'priority': self._get_priority_from_filename(Path(file_path).name)
                })
            
            self.logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading documents from {file_path}: {e}")
            return []
    
    def load_documents_from_directory(self, directory_path: str) -> List[Any]:
        """从目录加载所有文档"""
        if not DEPENDENCIES_AVAILABLE:
            return []
        
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        # 支持的文件类型
        supported_extensions = {'.txt', '.pdf', '.csv', '.json', '.md', '.doc', '.docx'}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                file_docs = self.load_documents_from_file(str(file_path))
                documents.extend(file_docs)
        
        self.logger.info(f"Loaded {len(documents)} documents from directory {directory_path}")
        return documents
    
    def _get_category_from_path(self, file_path: str) -> str:
        """从文件路径推断分类"""
        path_lower = file_path.lower()
        
        if 'faq' in path_lower:
            return 'FAQ'
        elif any(keyword in path_lower for keyword in ['product', '产品', 'specification', '规格']):
            return '产品信息'
        elif any(keyword in path_lower for keyword in ['rule', '规则', 'policy', '政策']):
            return '规则政策'
        elif any(keyword in path_lower for keyword in ['manual', '手册', 'guide', '指南']):
            return '使用手册'
        elif any(keyword in path_lower for keyword in ['service', '服务', 'support', '支持']):
            return '服务支持'
        else:
            return '其他'
    
    def _get_priority_from_filename(self, filename: str) -> int:
        """从文件名推断优先级"""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['important', '重要', 'urgent', '紧急']):
            return 5
        elif any(keyword in filename_lower for keyword in ['high', '高']):
            return 4
        elif 'faq' in filename_lower:
            return 3
        elif any(keyword in filename_lower for keyword in ['common', '常见']):
            return 3
        else:
            return 2
    
    def split_documents(self, documents: List[Any], splitter_type: str = "recursive") -> List[Any]:
        """分割文档"""
        if not DEPENDENCIES_AVAILABLE or not documents:
            return documents
        
        text_splitter = self.get_text_splitter(splitter_type)
        if text_splitter is None:
            return documents
        
        try:
            splits = text_splitter.split_documents(documents)
            self.logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
            return splits
        except Exception as e:
            self.logger.error(f"Error splitting documents: {e}")
            return documents
    
    def create_vectorstore(self, 
                          documents: List[Any],
                          persist_directory: str = None,
                          collection_name: str = None) -> Optional[Any]:
        """创建向量存储"""
        if not DEPENDENCIES_AVAILABLE or not documents:
            return None
        
        if self.embeddings is None:
            self.logger.error("Embeddings model not available")
            return None
        
        persist_directory = persist_directory if persist_directory is not None else self.settings.vector_store.chroma_persist_directory
        collection_name = collection_name if collection_name is not None else self.settings.vector_store.chroma_collection_name
        
        try:
            # 创建目录
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            # 创建向量存储
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            
            self.logger.info(f"Created vectorstore with {len(documents)} documents")
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Error creating vectorstore: {e}")
            return None
    
    def load_existing_vectorstore(self, 
                                 persist_directory: str = None,
                                 collection_name: str = None) -> Optional[Any]:
        """加载现有的向量存储"""
        if not DEPENDENCIES_AVAILABLE:
            return None
        
        if self.embeddings is None:
            self.logger.error("Embeddings model not available")
            return None
        
        persist_directory = persist_directory if persist_directory is not None else self.settings.vector_store.chroma_persist_directory
        collection_name = collection_name if collection_name is not None else self.settings.vector_store.chroma_collection_name
        
        if not os.path.exists(persist_directory):
            self.logger.warning(f"Vectorstore directory does not exist: {persist_directory}")
            return None
        
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            
            self.logger.info(f"Loaded existing vectorstore from {persist_directory}")
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Error loading vectorstore: {e}")
            return None
    
    def build_knowledge_base(self, 
                           data_path: str,
                           persist_directory: str = None,
                           collection_name: str = None,
                           force_rebuild: bool = False) -> Optional[Any]:
        """构建知识库"""
        persist_directory = persist_directory or self.settings.vector_store.chroma_persist_directory
        
        # 检查是否已存在知识库
        if not force_rebuild and os.path.exists(persist_directory):
            self.logger.info("Loading existing knowledge base...")
            vectorstore = self.load_existing_vectorstore(persist_directory, collection_name)
            if vectorstore is not None:
                return vectorstore
        
        self.logger.info("Building new knowledge base...")
        
        # 加载文档
        if os.path.isfile(data_path):
            documents = self.load_documents_from_file(data_path)
        else:
            documents = self.load_documents_from_directory(data_path)
        
        if not documents:
            self.logger.error("No documents loaded")
            return None
        
        # 分割文档
        splits = self.split_documents(documents)
        
        # 创建向量存储
        vectorstore = self.create_vectorstore(splits, persist_directory, collection_name)
        
        return vectorstore
    
    def add_documents_to_vectorstore(self, 
                                   vectorstore: Any,
                                   documents: List[Any]) -> bool:
        """向现有向量存储添加文档"""
        if not DEPENDENCIES_AVAILABLE or not documents:
            return False
        
        try:
            # 分割文档
            splits = self.split_documents(documents)
            
            # 添加到向量存储
            vectorstore.add_documents(splits)
            
            self.logger.info(f"Added {len(splits)} document chunks to vectorstore")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vectorstore: {e}")
            return False
    
    def search_similar_documents(self, 
                               vectorstore: Any,
                               query: str,
                               top_k: int = None,
                               score_threshold: float = None) -> List[Any]:
        """搜索相似文档"""
        if not DEPENDENCIES_AVAILABLE or vectorstore is None:
            return []
        
        top_k = top_k or self.settings.vector_store.retrieval_top_k
        score_threshold = score_threshold or self.settings.vector_store.retrieval_score_threshold
        
        try:
            # 使用相似度搜索
            results = vectorstore.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # 过滤低分结果
            filtered_results = [
                doc for doc, score in results 
                if score >= score_threshold
            ]
            
            self.logger.info(f"Found {len(filtered_results)} similar documents for query: {query}")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error searching similar documents: {e}")
            return []
    
    def get_vectorstore_stats(self, vectorstore: Any) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        if not DEPENDENCIES_AVAILABLE or vectorstore is None:
            return {}
        
        try:
            collection = vectorstore._collection
            stats = {
                'document_count': collection.count(),
                'collection_name': collection.name,
                'persist_directory': vectorstore.persist_directory
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting vectorstore stats: {e}")
            return {}


def create_knowledge_base(data_path: str, 
                         persist_directory: str = None,
                         force_rebuild: bool = False) -> Optional[Any]:
    """便捷函数：创建知识库"""
    builder = KnowledgeBaseBuilder()
    return builder.build_knowledge_base(
        data_path=data_path,
        persist_directory=persist_directory,
        force_rebuild=force_rebuild
    )


def load_knowledge_base(persist_directory: str = None) -> Optional[Any]:
    """便捷函数：加载知识库"""
    builder = KnowledgeBaseBuilder()
    return builder.load_existing_vectorstore(persist_directory)


# 导出主要类和函数
__all__ = [
    "KnowledgeBaseBuilder",
    "create_knowledge_base", 
    "load_knowledge_base"
] 