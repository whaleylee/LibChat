#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG检索器模块

本模块实现了基于知识图谱的增强检索器，结合向量检索和图遍历技术，
提供更丰富、更准确的上下文信息检索功能。
"""

import networkx as nx
from typing import List, Optional, Set
from loguru import logger

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from llama_index.core.retrievers import VectorIndexRetriever
    LLAMA_INDEX_AVAILABLE = True
except ImportError as e:
    logger.error(f"LlamaIndex导入失败: {e}")
    LLAMA_INDEX_AVAILABLE = False

from ..reranking.reranker import SentenceTransformerReranker


class GraphRAGRetriever(BaseRetriever):
    """
    基于知识图谱的增强检索器
    
    GraphRAGRetriever结合了向量检索和知识图谱遍历技术，实现了一个两阶段的检索策略：
    1. 入口点定位：使用向量检索找到语义最相关的初始节点
    2. 上下文扩展：通过图遍历扩展相关节点，获取更丰富的上下文信息
    
    这种方法能够发现传统向量检索可能遗漏的相关信息，特别是那些在语义上不直接相关
    但在代码结构上紧密关联的内容（如函数调用关系、类继承关系等）。
    
    Attributes:
        vector_retriever (VectorIndexRetriever): 基础向量检索器，用于找到语义相关的入口点
        knowledge_graph (nx.DiGraph): 代码知识图谱，包含代码块之间的结构关系
        reranker (SentenceTransformerReranker): 重排序器，用于对扩展后的结果进行精确排序
        expansion_depth (int): 图遍历的最大深度，控制上下文扩展的范围
    """
    
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        knowledge_graph: nx.DiGraph,
        reranker: SentenceTransformerReranker,
        expansion_depth: int = 1
    ):
        """
        初始化GraphRAG检索器
        
        Args:
            vector_retriever (VectorIndexRetriever): 基础向量检索器实例，用于语义检索
            knowledge_graph (nx.DiGraph): 已加载的知识图谱，包含代码块间的关系
            reranker (SentenceTransformerReranker): 重排序器实例，用于最终结果排序
            expansion_depth (int, optional): 图遍历深度，默认为1。
                - 1: 只扩展直接邻居
                - 2: 扩展到二度邻居
                - 更大值: 扩展更远的关系，但可能引入噪声
        
        Raises:
            ValueError: 当expansion_depth小于0时抛出
            TypeError: 当参数类型不正确时抛出
        """
        super().__init__()
        
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("LlamaIndex不可用，无法创建GraphRAG检索器")
        
        if expansion_depth < 0:
            raise ValueError("expansion_depth必须为非负整数")
        
        self.vector_retriever = vector_retriever
        self.knowledge_graph = knowledge_graph
        self.reranker = reranker
        self.expansion_depth = expansion_depth
        
        logger.info(f"GraphRAG检索器初始化完成")
        logger.info(f"知识图谱节点数: {len(self.knowledge_graph.nodes())}")
        logger.info(f"知识图谱边数: {len(self.knowledge_graph.edges())}")
        logger.info(f"扩展深度: {self.expansion_depth}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        执行GraphRAG检索的核心方法
        
        实现两阶段检索策略：
        1. 入口点定位：使用向量检索找到最相关的初始节点
        2. 上下文扩展：通过知识图谱遍历扩展相关节点
        3. 重新排序：对扩展后的结果进行精确重排
        
        Args:
            query_bundle (QueryBundle): 包含查询文本和相关元数据的查询包
        
        Returns:
            List[NodeWithScore]: 经过图扩展和重排序的检索结果列表，
                                按相关性从高到低排序
        
        Raises:
            Exception: 当检索过程中发生错误时抛出
        """
        try:
            logger.info(f"开始GraphRAG检索，查询: {query_bundle.query_str}")
            
            # 阶段一：入口点定位
            logger.debug("阶段一：使用向量检索定位入口点")
            entry_nodes = self.vector_retriever.retrieve(query_bundle)
            
            if not entry_nodes:
                logger.warning("向量检索未返回任何结果")
                return []
            
            logger.info(f"向量检索找到 {len(entry_nodes)} 个入口点")
            
            # 阶段二：上下文扩展
            logger.debug("阶段二：通过知识图谱扩展上下文")
            expanded_node_ids = self._expand_context_via_graph(entry_nodes)
            
            logger.info(f"图扩展后共有 {len(expanded_node_ids)} 个相关节点")
            
            # 重新获取节点对象
            logger.debug("重新构建扩展后的节点列表")
            expanded_nodes = self._rebuild_nodes_from_graph(expanded_node_ids, query_bundle)
            
            if not expanded_nodes:
                logger.warning("图扩展后未能重建任何节点")
                return entry_nodes  # 回退到原始结果
            
            # 阶段三：最终重排
            logger.debug("阶段三：对扩展结果进行重排序")
            final_results = self._rerank_expanded_results(expanded_nodes, query_bundle)
            
            logger.info(f"GraphRAG检索完成，返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"GraphRAG检索失败: {e}", exc_info=True)
            # 发生错误时回退到基础向量检索结果
            logger.info("回退到基础向量检索结果")
            return self.vector_retriever.retrieve(query_bundle)
    
    def _expand_context_via_graph(self, entry_nodes: List[NodeWithScore]) -> Set[str]:
        """
        通过知识图谱扩展上下文节点
        
        从入口点节点开始，在知识图谱中进行广度优先遍历，
        收集指定深度内的所有相关节点ID。
        
        Args:
            entry_nodes (List[NodeWithScore]): 向量检索得到的入口点节点列表
        
        Returns:
            Set[str]: 扩展后的所有相关节点ID集合
        """
        expanded_node_ids = set()
        
        # 首先添加所有入口点节点的ID
        for node in entry_nodes:
            node_id = self._extract_node_id(node)
            if node_id:
                expanded_node_ids.add(node_id)
        
        logger.debug(f"入口点节点数: {len(expanded_node_ids)}")
        
        # 如果扩展深度为0，直接返回入口点
        if self.expansion_depth == 0:
            return expanded_node_ids
        
        # 对每个入口点进行图遍历扩展
        for node in entry_nodes:
            node_id = self._extract_node_id(node)
            if not node_id or node_id not in self.knowledge_graph:
                continue
            
            # 使用BFS遍历指定深度的邻居
            try:
                # 获取所有在指定深度内的边
                bfs_edges = nx.bfs_edges(
                    self.knowledge_graph, 
                    node_id, 
                    depth_limit=self.expansion_depth
                )
                
                # 收集所有相关节点
                for source, target in bfs_edges:
                    expanded_node_ids.add(source)
                    expanded_node_ids.add(target)
                    
            except nx.NetworkXError as e:
                logger.warning(f"图遍历节点 {node_id} 时出错: {e}")
                continue
        
        logger.debug(f"图扩展完成，总节点数: {len(expanded_node_ids)}")
        return expanded_node_ids
    
    def _extract_node_id(self, node: NodeWithScore) -> Optional[str]:
        """
        从NodeWithScore对象中提取节点ID
        
        Args:
            node (NodeWithScore): 包含评分的节点对象
        
        Returns:
            Optional[str]: 节点ID，如果提取失败则返回None
        """
        try:
            # 尝试从节点元数据中获取node_id
            if hasattr(node.node, 'metadata') and 'node_id' in node.node.metadata:
                return node.node.metadata['node_id']
            
            # 尝试从节点ID字段获取
            if hasattr(node.node, 'node_id'):
                return node.node.node_id
            
            # 尝试从节点的id_字段获取
            if hasattr(node.node, 'id_'):
                return node.node.id_
            
            logger.warning(f"无法从节点中提取node_id: {type(node.node)}")
            return None
            
        except Exception as e:
            logger.warning(f"提取节点ID时出错: {e}")
            return None
    
    def _rebuild_nodes_from_graph(self, node_ids: Set[str], query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        根据节点ID从知识图谱中重建NodeWithScore对象列表
        
        Args:
            node_ids (Set[str]): 需要重建的节点ID集合
            query_bundle (QueryBundle): 原始查询包，用于计算相关性评分
        
        Returns:
            List[NodeWithScore]: 重建的节点列表
        """
        rebuilt_nodes = []
        
        for node_id in node_ids:
            if node_id not in self.knowledge_graph:
                continue
            
            try:
                # 从图中获取节点属性
                node_data = self.knowledge_graph.nodes[node_id]
                
                # 检查节点是否包含必要的代码块信息
                if 'chunk' not in node_data:
                    logger.debug(f"节点 {node_id} 缺少chunk信息")
                    continue
                
                chunk = node_data['chunk']
                
                # 创建Document对象
                from llama_index.core import Document
                doc = Document(
                    text=chunk.text,
                    metadata=chunk.to_dict()
                )
                
                # 创建NodeWithScore对象（初始评分为0，后续会重新计算）
                node_with_score = NodeWithScore(
                    node=doc,
                    score=0.0
                )
                
                rebuilt_nodes.append(node_with_score)
                
            except Exception as e:
                logger.warning(f"重建节点 {node_id} 时出错: {e}")
                continue
        
        logger.debug(f"成功重建 {len(rebuilt_nodes)} 个节点")
        return rebuilt_nodes
    
    def _rerank_expanded_results(self, expanded_nodes: List[NodeWithScore], query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        对扩展后的结果进行重新排序
        
        使用重排序器对图扩展后的节点列表进行精确排序，
        确保最相关的内容排在前面。
        
        Args:
            expanded_nodes (List[NodeWithScore]): 扩展后的节点列表
            query_bundle (QueryBundle): 原始查询包
        
        Returns:
            List[NodeWithScore]: 重新排序后的节点列表
        """
        try:
            if not expanded_nodes:
                return []
            
            # 使用重排序器进行最终排序
            reranked_results = self.reranker.rerank(
                query=query_bundle.query_str,
                nodes=expanded_nodes
            )
            
            logger.debug(f"重排序完成，结果数量: {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 如果重排序失败，返回原始扩展结果
            return expanded_nodes
    
    def get_retriever_info(self) -> dict:
        """
        获取检索器的配置信息
        
        Returns:
            dict: 包含检索器配置的字典
        """
        return {
            "type": "GraphRAGRetriever",
            "expansion_depth": self.expansion_depth,
            "graph_nodes": len(self.knowledge_graph.nodes()),
            "graph_edges": len(self.knowledge_graph.edges()),
            "vector_retriever_type": type(self.vector_retriever).__name__,
            "reranker_type": type(self.reranker).__name__
        }