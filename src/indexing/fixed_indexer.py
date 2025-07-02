#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的索引器模块

本模块解决了原有索引器中的关键问题：
1. FaissVectorStore保存的是二进制文件而非JSON的问题
2. 扩展支持多种文件类型而不仅限于Python文件
3. 改进错误处理和编码问题
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from llama_index.core import (
        VectorStoreIndex, 
        Document, 
        StorageContext,
        Settings
    )
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import faiss
    LLAMA_INDEX_AVAILABLE = True
except ImportError as e:
    logger.error(f"LlamaIndex导入失败: {e}")
    LLAMA_INDEX_AVAILABLE = False

from ..chunker.multi_language_chunker import MultiLanguageChunker, CodeChunk
from ..source_inspector.inspector import PackageInspector


class FixedIndexer:
    """
    修复后的索引器
    
    主要改进：
    1. 正确处理FaissVectorStore的二进制存储格式
    2. 支持多种编程语言和文件类型
    3. 改进的错误处理和恢复机制
    4. 更好的元数据管理
    """
    
    def __init__(self, index_dir: str = "indexes"):
        """初始化修复后的索引器"""
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("LlamaIndex不可用，无法创建索引器")
        
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # 使用多语言分块器
        self.chunker = MultiLanguageChunker()
        
        # 索引文件路径

        
        logger.info(f"FixedIndexer初始化完成，索引目录: {self.index_dir}")

    def _get_index_paths(self, index_name: str) -> Dict[str, Path]:
        """根据索引名称获取所有相关的索引文件路径。"""
        index_path_dir = self.index_dir / index_name
        index_path_dir.mkdir(parents=True, exist_ok=True)
        return {
            "vector_store": index_path_dir / "vector_store.faiss",
            "docstore": index_path_dir / "docstore.json",
            "index_store": index_path_dir / "index_store.json",
            "metadata": index_path_dir / "metadata.json",
            "index_dir": index_path_dir
        }
    
    def create_and_persist_vector_index(self, package_path: str, 
                                       index_name: str,
                                       embedding_model: str = "text-embedding-ada-002",
                                       force_rebuild: bool = False) -> bool:
        """
        创建并持久化向量索引
        
        Args:
            package_path: 要索引的包路径
            embedding_model: 嵌入模型名称
            force_rebuild: 是否强制重建索引
            
        Returns:
            bool: 是否成功创建索引
        """
        logger.info(f"开始创建向量索引: {package_path}")
        
        try:
            # 检查是否需要重建
            if not force_rebuild and self._index_exists(index_name):
                logger.info("索引已存在且未要求强制重建")
                return True
            
            # 清理旧索引
            self._cleanup_old_index(index_name)
            
            # 设置嵌入模型
            logger.info(f"使用嵌入模型: {embedding_model}")
            if embedding_model and "bge-" in embedding_model:
                Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
            else:
                # 默认或OpenAI模型
                Settings.embed_model = OpenAIEmbedding(model=embedding_model)
            
            # 收集并分块所有支持的文件
            all_chunks = self._collect_and_chunk_files(package_path)
            
            if not all_chunks:
                logger.warning("没有找到可分块的文件")
                return False
            
            # 转换为Document对象
            documents = self._chunks_to_documents(all_chunks)
            
            # 根据模型确定维度
            if "m3" in Settings.embed_model.model_name:
                dimension = 1024
            elif "large" in Settings.embed_model.model_name:
                dimension = 1024
            elif "base" in Settings.embed_model.model_name:
                dimension = 768
            elif "small" in Settings.embed_model.model_name:
                dimension = 384
            else:
                # 默认为OpenAI的维度
                dimension = 1536
            logger.info(f"模型 {Settings.embed_model.model_name} 的维度设置为: {dimension}")

            # 创建Faiss向量存储
            faiss_index = faiss.IndexFlatL2(dimension)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            # 创建存储上下文
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 创建索引
            logger.info(f"开始创建向量索引，文档数量: {len(documents)}")
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context
            )
            
            # 持久化索引
            success = self._persist_index_properly(index, storage_context, all_chunks, index_name)
            
            if success:
                logger.info("向量索引创建并持久化成功")
            else:
                logger.error("向量索引持久化失败")
            
            return success
            
        except Exception as e:
            logger.error(f"创建向量索引失败: {e}", exc_info=True)
            return False
    
    def _collect_and_chunk_files(self, package_path: Optional[str]) -> List[CodeChunk]:
        """收集并分块所有支持的文件"""
        logger.info(f"开始收集和分块文件: {package_path}")
        
        if not package_path:
            logger.error("无效的包路径: None")
            return []

        all_chunks = []
        package_path_obj = Path(package_path)
        
        # 检查是本地路径还是已安装的包
        if package_path_obj.exists():
            if package_path_obj.is_file():
                if self.chunker.is_supported_file(str(package_path_obj)):
                    chunks = self.chunker.chunk_file(str(package_path_obj))
                    all_chunks.extend(chunks)
                else:
                    logger.warning(f"不支持的文件类型: {package_path_obj}")
            else:
                chunks = self.chunker.chunk_directory(str(package_path_obj))
                all_chunks.extend(chunks)
        else:
            # 尝试作为已安装的包处理
            logger.info(f"路径 '{package_path}' 不存在，尝试作为已安装的包处理")
            inspector = PackageInspector(repo_path, package_path=package_path)
            source_files = inspector.get_source_files()
            if not source_files:
                logger.error(f"无法找到包 '{package_path}' 的源文件")
                return []
            
            for file_path in source_files:
                if self.chunker.is_supported_file(str(file_path)):
                    chunks = self.chunker.chunk_file(str(file_path))
                    all_chunks.extend(chunks)
                else:
                    logger.warning(f"不支持的文件类型: {file_path}")
        
        logger.info(f"文件收集和分块完成，共生成 {len(all_chunks)} 个代码块")
        
        # 保存分块统计信息
        summary = self.chunker.get_chunk_summary(all_chunks)
        self._save_chunk_summary(summary)
        
        return all_chunks
    
    def _chunks_to_documents(self, chunks: List[CodeChunk]) -> List[Document]:
        """将代码块转换为Document对象"""
        documents = []
        
        for i, chunk in enumerate(chunks):
            # 创建文档内容
            content = chunk.text
            
            # 添加元数据信息到内容中
            metadata = chunk.metadata
            file_path = metadata.get('file_path', 'unknown')
            # 确保file_path是字符串，而不是Path对象
            if isinstance(file_path, Path):
                file_path = str(file_path)
            language = metadata.get('language', 'unknown')
            node_type = metadata.get('node_type', 'unknown')
            start_line = metadata.get('start_line', 0)
            end_line = metadata.get('end_line', 0)
            
            # 构建富文本内容
            enriched_content = f"""
文件: {file_path}
语言: {language}
类型: {node_type}
行数: {start_line}-{end_line}

{content}
"""
            
            # 创建Document
            doc = Document(
                text=enriched_content,
                metadata={
                    'chunk_id': i,
                    'file_path': str(file_path),  # 确保是字符串
                    'language': language,
                    'node_type': node_type,
                    'start_line': start_line,
                    'end_line': end_line,
                    'chunk_method': metadata.get('chunk_method', 'unknown')
                }
            )
            
            documents.append(doc)
        
        return documents
    
    def _persist_index_properly(self, index: VectorStoreIndex, storage_context: StorageContext, all_chunks: List[CodeChunk], index_name: str) -> bool:
        """
        正确持久化索引，包括向量存储、文档存储、索引存储和元数据。
        """
        logger.info("开始持久化索引...")
        index_paths = self._get_index_paths(index_name)
        try:
            # 使用LlamaIndex的统一持久化方法
            index.storage_context.persist(persist_dir=str(index_paths["index_dir"]))
            logger.info(f"索引统一持久化到: {index_paths['index_dir']}")

            # 保存元数据
            from datetime import datetime
            from llama_index.core import __version__
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "num_documents": len(index.docstore.docs),
                "num_chunks": len(all_chunks),
                "embedding_model": Settings.embed_model.model_name,
                "llama_index_version": __version__
            }
            with open(index_paths["metadata"], 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            logger.info(f"元数据持久化到: {index_paths['metadata']}")

            return True
        except Exception as e:
            logger.error(f"持久化索引失败: {e}")
            return False
    
    def _fix_encoding_issues(self, index_path: Path) -> None:
        """
        修复索引文件的编码问题
        
        Args:
            index_path (Path): 索引目录路径
        """
        # 只处理真正的JSON文件，跳过向量存储文件（它们是二进制格式）
        json_files = ['docstore.json', 'index_store.json', 'metadata.json', 'graph_store.json']
        
        # 跳过向量存储文件，因为它们实际上是二进制格式
        vector_store_files = ['default__vector_store.json', 'image__vector_store.json']
        
        logger.info(f"开始修复索引目录 {index_path} 中的编码问题")
        logger.info(f"将跳过向量存储文件: {vector_store_files}")
        
        for json_file in json_files:
            file_path = index_path / json_file
            if file_path.exists():
                try:
                    # First, try to open with utf-8, which is the desired encoding
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.stat().st_size == 0:
                            logger.warning(f"{json_file} is empty, which is invalid for JSON. Skipping.")
                            continue
                        json.load(f)
                    logger.debug(f"{json_file} 编码正常")
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    logger.warning(f"{json_file} 存在编码或JSON格式问题: {e}，正在尝试修复...")
                    try:
                        # Read as bytes
                        with open(file_path, 'rb') as f:
                            raw_content = f.read()
                        
                        # On Windows, llama_index might save with default system encoding
                        # We try utf-8 first, then common windows codepage, then latin-1 as a fallback.
                        content = None
                        for encoding in ['utf-8', 'cp1252', 'latin-1']:
                            try:
                                content = raw_content.decode(encoding)
                                logger.info(f"Successfully decoded {json_file} with {encoding}")
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if content is None:
                            logger.error(f"Could not decode {json_file} with any of the attempted encodings.")
                            continue # Skip to next file

                        # Now that we have the content as a string, parse it
                        data = json.loads(content)
                        
                        # Write it back as utf-8
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        
                        logger.info(f"成功修复 {json_file} 的编码问题")
                    except Exception as fix_e:
                        logger.error(f"修复 {json_file} 编码失败: {fix_e}。保留原文件。")

    def load_index(self, index_name: str, embedding_model: str = "text-embedding-ada-002") -> Optional[VectorStoreIndex]:
        """加载索引"""
        logger.info("开始加载索引")
        
        try:
            index_paths = self._get_index_paths(index_name)
            self.vector_store_path = index_paths["vector_store"]
            self.docstore_path = index_paths["docstore"]
            self.index_store_path = index_paths["index_store"]
            self.metadata_path = index_paths["metadata"]

            if not self._index_exists(index_name):
                logger.warning(f"索引文件不存在: {index_name}")
                return None

            # 尝试修复潜在的编码问题
            self._fix_encoding_issues(index_paths["index_dir"])

            # 设置嵌入模型，这对于加载带有特定嵌入的索引至关重要
            logger.info(f"加载索引时设置嵌入模型: {embedding_model}")
            if embedding_model and "bge-" in embedding_model:
                Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
            else:
                Settings.embed_model = OpenAIEmbedding(model=embedding_model)

            # 加载FaissVectorStore
            logger.info("正在加载FaissVectorStore...")
            vector_store = FaissVectorStore.from_persist_dir(str(index_paths["index_dir"]))
            
            # 创建存储上下文
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(index_paths["index_dir"])
            )
            
            # 从存储上下文加载索引
            from llama_index.core import load_index_from_storage
            index = load_index_from_storage(storage_context)
            logger.info("索引加载成功")
            return index

            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return None
    
    def _cleanup_old_index(self, index_name: str):
        """清理旧的索引文件"""
        logger.info(f"清理旧的索引文件: {index_name}...")
        index_paths = self._get_index_paths(index_name)
        try:
            # 删除向量存储目录
            if index_paths["vector_store"].exists():
                os.remove(index_paths["vector_store"])
            if index_paths["docstore"].exists():
                os.remove(index_paths["docstore"])
            if index_paths["index_store"].exists():
                os.remove(index_paths["index_store"])
            if index_paths["metadata"].exists():
                os.remove(index_paths["metadata"])
            logger.info(f"旧索引文件 {index_name} 清理完成")
        except Exception as e:
            logger.error(f"清理旧索引文件 {index_name} 失败: {e}")

    def _index_exists(self, index_name: str) -> bool:
        """检查索引文件是否存在"""
        index_paths = self._get_index_paths(index_name)
        # LlamaIndex的persist方法会创建多个文件，我们只检查核心的docstore.json
        return index_paths["docstore"].exists()



    
    def _save_chunk_summary(self, summary: Dict[str, Any]):
        """保存分块统计信息"""
        summary_path = self.index_dir / "chunk_summary.json"
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.debug(f"分块统计信息已保存到: {summary_path}")
        except Exception as e:
            logger.warning(f"保存分块统计信息失败: {e}")
    
    def get_index_info(self) -> Dict[str, Any]:
        """获取索引信息"""
        info = {
            'index_exists': self._index_exists(),
            'index_dir': str(self.index_dir),
            'supported_extensions': self.chunker.get_supported_extensions()
        }
        
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                info['metadata'] = metadata
            except Exception as e:
                logger.warning(f"读取元数据失败: {e}")
        
        return info
    
    def build_index_from_chunks(self, chunks: List[CodeChunk], index_name: str, 
                                   embedding_model: str = "BAAI/bge-small-en-v1.5",
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        从代码块直接构建索引
        
        Args:
            chunks: 代码块列表
            index_name: 索引名称
            embedding_model: 嵌入模型名称
            metadata: 额外的元数据
            
        Returns:
            bool: 是否成功创建索引
        """
        logger.info(f"开始从代码块构建索引: {index_name}")
        
        try:
            if not chunks:
                logger.warning("没有提供代码块")
                return False
            
            # 清理旧索引
            self._cleanup_old_index(index_name)
            
            # 设置嵌入模型
            logger.info(f"使用嵌入模型: {embedding_model}")
            if embedding_model and "bge-" in embedding_model:
                Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
            else:
                Settings.embed_model = OpenAIEmbedding(model=embedding_model)
            
            # 转换为Document对象
            documents = self._chunks_to_documents(chunks)
            
            # 根据模型确定维度
            if "m3" in Settings.embed_model.model_name:
                dimension = 1024
            elif "large" in Settings.embed_model.model_name:
                dimension = 1024
            elif "base" in Settings.embed_model.model_name:
                dimension = 768
            elif "small" in Settings.embed_model.model_name:
                dimension = 384
            else:
                dimension = 1536
            logger.info(f"模型 {Settings.embed_model.model_name} 的维度设置为: {dimension}")

            # 创建Faiss向量存储
            faiss_index = faiss.IndexFlatL2(dimension)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            # 创建存储上下文
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 创建索引
            logger.info(f"开始创建向量索引，文档数量: {len(documents)}")
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context
            )
            
            # 持久化索引
            success = self._persist_index_from_chunks(index, storage_context, chunks, index_name, metadata)
            
            if success:
                logger.info("从代码块构建的向量索引创建并持久化成功")
            else:
                logger.error("从代码块构建的向量索引持久化失败")
            
            return success
            
        except Exception as e:
            logger.error(f"从代码块构建向量索引失败: {e}", exc_info=True)
            return False
    
    def _persist_index_from_chunks(self, index: VectorStoreIndex, storage_context: StorageContext, 
                                  chunks: List[CodeChunk], index_name: str, 
                                  extra_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        持久化从代码块构建的索引
        """
        logger.info("开始持久化从代码块构建的索引...")
        index_paths = self._get_index_paths(index_name)
        try:
            # 使用LlamaIndex的统一持久化方法
            index.storage_context.persist(persist_dir=str(index_paths["index_dir"]))
            logger.info(f"索引统一持久化到: {index_paths['index_dir']}")

            # 保存元数据
            from datetime import datetime
            from llama_index.core import __version__
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "num_documents": len(index.docstore.docs),
                "num_chunks": len(chunks),
                "embedding_model": Settings.embed_model.model_name,
                "llama_index_version": __version__
            }
            
            # 合并额外元数据，并确保所有路径都是字符串
            if extra_metadata:
                for key, value in extra_metadata.items():
                    if isinstance(value, Path):
                        metadata[key] = str(value)
                    else:
                        metadata[key] = value
            
            with open(index_paths["metadata"], 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            logger.info(f"元数据持久化到: {index_paths['metadata']}")

            return True
        except Exception as e:
            logger.error(f"持久化从代码块构建的索引失败: {e}")
            return False

    def rebuild_index(self, repo_path: str, index_name: str = "default", embedding_model: str = "BAAI/bge-small-en-v1.5", package_path: Optional[str] = None) -> bool:
        """
        重建索引
        """
        logger.info(f"开始重建索引: {repo_path}, 索引名称: {index_name}")

        # 如果没有提供物理路径，则尝试通过包名查找
        if not package_path:
            inspector = PackageInspector(repo_path)
            path_to_index = inspector._get_package_path()
            if not path_to_index:
                logger.error(f"无法找到包 '{repo_path}' 的路径，索引构建中止。")
                return False
            package_path = str(path_to_index)
        
        return self.create_and_persist_vector_index(package_path, index_name, embedding_model, force_rebuild=True)
    
    def query_index(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """查询索引"""
        index = self.load_index()
        if not index:
            logger.error("无法加载索引")
            return []
        
        try:
            query_engine = index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query)
            
            results = []
            for node in response.source_nodes:
                results.append({
                    'text': node.text,
                    'metadata': node.metadata,
                    'score': node.score if hasattr(node, 'score') else None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"查询索引失败: {e}")
            return []