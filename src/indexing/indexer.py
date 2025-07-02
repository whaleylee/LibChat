#!/usr/bin/env python3
"""
Faiss Indexer Module

本模块提供基于Faiss和LlamaIndex的向量索引功能，
用于将代码块转换为向量并构建高效的检索索引。
"""

import os
import networkx as nx
import pickle
from typing import List, Any
from pathlib import Path
from loguru import logger
import faiss

# LlamaIndex核心组件
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.settings import Settings

# HuggingFace模型
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# 向量存储
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore

# 导入自定义的CodeChunk类型
try:
    from ..chunker.ast_chunker import CodeChunk
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from src.chunker.ast_chunker import CodeChunk


class FaissIndexer:
    """
    基于Faiss的向量索引器
    
    使用HuggingFace嵌入模型和Faiss向量存储构建高效的代码检索索引。
    支持将代码块转换为向量表示并持久化存储。
    """
    
    def __init__(self, embed_model_name: str = 'BAAI/bge-large-en-v1.5') -> None:
        """
        初始化Faiss索引器
        
        Args:
            embed_model_name (str): HuggingFace嵌入模型名称，
                                   默认使用'BAAI/bge-large-en-v1.5'
        """
        self.embed_model_name = embed_model_name
        logger.info(f"初始化FaissIndexer，嵌入模型: {embed_model_name}")
        
        try:
            # 从环境变量获取HuggingFace token
            import os
            from dotenv import load_dotenv
            
            # 加载.env文件
            load_dotenv()
            
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                logger.info("使用环境变量中的HuggingFace token")
                # Remove quotes if present
                hf_token = hf_token.strip('"')
                # 设置HuggingFace token
                os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
                os.environ['HF_TOKEN'] = hf_token
            else:
                logger.warning("未找到HuggingFace token，可能会导致模型下载失败")
            
            # 初始化HuggingFace嵌入模型
            self.embed_model = HuggingFaceEmbedding(
                model_name=embed_model_name,
                trust_remote_code=True
            )
            
            # 设置全局嵌入模型
            Settings.embed_model = self.embed_model
            # 注意：LLM 将在查询阶段单独初始化，这里只需要嵌入模型
            
            logger.info(f"成功初始化嵌入模型: {embed_model_name}")
            
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {e}")
            raise
    
    def _convert_chunks_to_documents(self, chunks: List[CodeChunk]) -> List[Document]:
        """
        将CodeChunk列表转换为LlamaIndex Document对象列表
        
        Args:
            chunks (List[CodeChunk]): 代码块列表
            
        Returns:
            List[Document]: LlamaIndex Document对象列表
        """
        logger.debug(f"开始转换 {len(chunks)} 个代码块为Document对象")
        
        documents = []
        
        for i, chunk in enumerate(chunks):
            try:
                # 创建Document对象
                doc = Document(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    doc_id=f"chunk_{i}_{chunk.metadata.get('node_type', 'unknown')}"
                )
                
                documents.append(doc)
                
                logger.debug(
                    f"转换代码块 {i+1}: {chunk.metadata.get('node_type', 'unknown')} "
                    f"from {chunk.metadata.get('file_path', 'unknown')}"
                )
                
            except Exception as e:
                logger.warning(f"转换代码块 {i+1} 时发生错误: {e}，跳过该块")
                continue
        
        logger.info(f"成功转换 {len(documents)} 个Document对象")
        return documents
    
    def create_index(self, chunks: List[CodeChunk], save_path: str) -> VectorStoreIndex:
        """
        创建并持久化向量索引
        
        将代码块转换为向量表示，构建Faiss索引并保存到磁盘。
        
        Args:
            chunks (List[CodeChunk]): 要索引的代码块列表
            save_path (str): 索引保存路径
            
        Returns:
            VectorStoreIndex: 构建完成的向量索引
            
        Raises:
            Exception: 当索引创建或保存失败时抛出异常
        """
        logger.info(f"开始创建向量索引，代码块数量: {len(chunks)}")
        logger.info(f"索引保存路径: {save_path}")
        
        try:
            # 确保保存目录存在
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 将CodeChunk转换为Document对象
            documents = self._convert_chunks_to_documents(chunks)
            
            if not documents:
                logger.warning("没有有效的文档可以索引")
                raise ValueError("没有有效的文档可以索引")
            
            # 设置全局嵌入模型，确保在构建过程中生效
            Settings.embed_model = self.embed_model

            # 获取嵌入维度
            try:
                embed_dim = len(self.embed_model.get_text_embedding("test"))
            except Exception as e:
                logger.warning(f"无法动态获取嵌入模型维度，将使用默认值 1024。错误: {e}")
                embed_dim = 1024

            # 创建Faiss索引核心
            faiss_index = faiss.IndexFlatL2(embed_dim)

            # 创建Faiss向量存储
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            # 创建存储上下文，并将Faiss向量存储设为默认
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 直接使用VectorStoreIndex.from_documents构建索引
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context,
                show_progress=True
            )
            
            logger.info("向量索引构建完成")
            
            # 持久化索引到磁盘
            logger.debug(f"开始持久化索引到: {save_path}")
            index.storage_context.persist(persist_dir=save_path)





            # 使用LlamaIndex的统一持久化方法，让框架自动处理不同存储类型
            logger.info(f"索引成功保存到: {save_path}")
            
            # 保存元数据
            from datetime import datetime
            import json
            from llama_index.core import __version__
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "num_documents": len(documents),
                "num_chunks": len(chunks),
                "embedding_model": self.embed_model_name,
                "llama_index_version": __version__
            }
            metadata_path = os.path.join(save_path, 'metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            logger.info(f"元数据持久化到: {metadata_path}")
            
            # 记录索引统计信息
            self._log_index_stats(chunks, documents)
            
            return index
            
        except Exception as e:
            logger.error(f"创建索引时发生错误: {e}")
            raise
    
    def load_index(self, load_path: str) -> VectorStoreIndex:
        """
        从磁盘加载已保存的索引
        
        Args:
            load_path (str): 索引加载路径
            
        Returns:
            VectorStoreIndex: 加载的向量索引
            
        Raises:
            Exception: 当索引加载失败时抛出异常
        """
        logger.info(f"开始从磁盘加载索引: {load_path}")
        
        try:
            if not os.path.exists(load_path):
                logger.error(f"索引路径不存在: {load_path}")
                raise FileNotFoundError(f"索引路径不存在: {load_path}")

            self._fix_encoding_issues(load_path)
            
            # 设置全局嵌入模型
            Settings.embed_model = self.embed_model

            # 尝试标准方法加载存储上下文
            try:
                storage_context = StorageContext.from_defaults(persist_dir=load_path)
                index = load_index_from_storage(storage_context)
                logger.info("使用标准方法成功加载索引")
            except Exception as e:
                logger.warning(f"标准方法加载失败: {e}，尝试显式Faiss方法")
                # 如果标准方法失败，尝试显式加载Faiss向量存储
                vector_store = FaissVectorStore.from_persist_dir(load_path)
                storage_context = StorageContext.from_defaults(
                    persist_dir=load_path, vector_store=vector_store
                )
                index = load_index_from_storage(storage_context)
                logger.info("使用显式Faiss向量存储方式成功加载索引")
            
            logger.info(f"成功加载索引: {load_path}")
            return index

        except Exception as e:
            logger.error(f"加载索引时发生错误: {e}")
            raise
    
    def _log_index_stats(self, chunks: List[CodeChunk], documents: List[Document]) -> None:
        """
        记录索引统计信息
        
        Args:
            chunks (List[CodeChunk]): 原始代码块列表
            documents (List[Document]): 转换后的文档列表
        """
        # 统计代码块类型
        chunk_types = {}
        for chunk in chunks:
            node_type = chunk.metadata.get('node_type', 'unknown')
            chunk_types[node_type] = chunk_types.get(node_type, 0) + 1
        
        # 统计文件来源
        file_sources = set()
        for chunk in chunks:
            file_path = chunk.metadata.get('file_path')
            if file_path:
                file_sources.add(file_path)
        
        logger.info("=== 索引统计信息 ===")
        logger.info(f"总代码块数量: {len(chunks)}")
        logger.info(f"成功索引文档数量: {len(documents)}")
        logger.info(f"代码块类型分布: {chunk_types}")
        logger.info(f"涉及文件数量: {len(file_sources)}")
        logger.info(f"使用嵌入模型: {self.embed_model_name}")
        logger.info("===================")
    
    def _fix_encoding_issues(self, index_path: str) -> None:
        """
        修复索引文件的编码问题
        注意：只处理真正的JSON文件，跳过向量存储的二进制文件
        
        Args:
            index_path (str): 索引目录路径
        """
        import json
        
        # 只处理真正的JSON文件，不包括向量存储文件（它们是二进制格式）
        json_files = ['docstore.json', 'index_store.json', 'metadata.json', 'graph_store.json']
        
        for json_file in json_files:
            file_path = Path(index_path) / json_file
            if file_path.exists():
                try:
                    # 首先尝试以UTF-8编码读取文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.stat().st_size == 0:
                            logger.warning(f"{json_file} 是空文件，跳过处理")
                            continue
                        json.load(f)
                    logger.debug(f"{json_file} 编码正常")
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    logger.warning(f"{json_file} 存在编码或JSON格式问题: {e}，正在尝试修复...")
                    try:
                        # 以二进制模式读取文件
                        with open(file_path, 'rb') as f:
                            raw_content = f.read()
                        
                        # 在Windows上，llama_index可能使用系统默认编码保存
                        # 依次尝试utf-8、cp1252、latin-1编码
                        content = None
                        for encoding in ['utf-8', 'cp1252', 'latin-1']:
                            try:
                                content = raw_content.decode(encoding)
                                logger.info(f"成功使用 {encoding} 编码解码 {json_file}")
                                break
                            except UnicodeDecodeError:
                                logger.debug(f"尝试使用 {encoding} 解码失败")
                                continue
                        
                        if content is None:
                            logger.error(f"无法使用任何编码解码 {json_file}")
                            continue
                        
                        # 解析JSON内容
                        data = json.loads(content)
                        
                        # 重新以UTF-8编码写入
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        
                        logger.info(f"成功修复 {json_file} 的编码问题")
                    except Exception as fix_e:
                        logger.error(f"修复 {json_file} 编码失败: {fix_e}。保留原文件。")


def save_knowledge_graph(graph: nx.DiGraph, path: str) -> None:
    """
    将知识图谱保存到磁盘
    
    使用pickle序列化NetworkX图对象并保存到指定路径。
    
    Args:
        graph (nx.DiGraph): 要保存的知识图谱
        path (str): 保存路径
        
    Raises:
        Exception: 当保存失败时抛出异常
    """
    logger.info(f"开始保存知识图谱到: {path}")
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 使用pickle序列化并保存图对象
        with open(path, 'wb') as f:
            pickle.dump(graph, f)
            
        logger.info(f"知识图谱成功保存到: {path}")
        logger.info(f"图谱包含 {len(graph.nodes())} 个节点和 {len(graph.edges())} 条边")
        
    except Exception as e:
        logger.error(f"保存知识图谱失败: {e}")
        raise


def load_knowledge_graph(path: str) -> nx.DiGraph:
    """
    从磁盘加载知识图谱
    
    使用pickle反序列化从指定路径加载NetworkX图对象。
    
    Args:
        path (str): 图谱文件路径
        
    Returns:
        nx.DiGraph: 加载的知识图谱
        
    Raises:
        FileNotFoundError: 当文件不存在时抛出异常
        Exception: 当加载失败时抛出异常
    """
    logger.info(f"开始从磁盘加载知识图谱: {path}")
    
    try:
        if not os.path.exists(path):
            logger.error(f"知识图谱文件不存在: {path}")
            raise FileNotFoundError(f"知识图谱文件不存在: {path}")
            
        # 使用pickle反序列化加载图对象
        with open(path, 'rb') as f:
            graph = pickle.load(f)
            
        logger.info(f"知识图谱成功加载: {path}")
        logger.info(f"图谱包含 {len(graph.nodes())} 个节点和 {len(graph.edges())} 条边")
        
        return graph
        
    except Exception as e:
        logger.error(f"加载知识图谱失败: {e}")
        raise