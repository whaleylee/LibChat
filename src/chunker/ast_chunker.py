#!/usr/bin/env python3
"""
AST Chunker Module

本模块提供基于tree-sitter的AST结构化分块功能，
用于将Python源代码按照语法结构进行智能分块。
"""

import tree_sitter
import tree_sitter_languages
import networkx as nx
from dataclasses import dataclass, asdict
from typing import List, Generator, Dict, Any
from loguru import logger


@dataclass
class CodeChunk:
    """
    代码块数据类
    
    用于存储分块后的代码片段及其元数据信息。
    
    Attributes:
        text (str): 代码块的文本内容
        metadata (dict): 元数据信息，包含文件名和起始行号等
        node_id (str): 在知识图谱中唯一标识节点的ID
    """
    text: str
    metadata: Dict[str, Any]
    node_id: str

    def to_dict(self) -> Dict[str, Any]:
        """将CodeChunk对象转换为字典。"""
        return asdict(self)


class ASTChunker:
    """
    基于AST的代码分块器
    
    使用tree-sitter解析Python源代码，按照语法结构（如函数、类定义）
    进行智能分块，便于后续的索引和检索。
    """
    
    def __init__(self) -> None:
        """
        初始化AST分块器
        
        设置tree-sitter解析器并加载Python语言库。
        """
        logger.info("初始化ASTChunker")
        
        # 初始化tree-sitter解析器
        self.parser = tree_sitter.Parser()
        
        # 加载Python语言库
        try:
            python_language = tree_sitter_languages.get_language('python')
            self.parser.set_language(python_language)
            logger.debug("成功加载Python语言库")
        except Exception as e:
            logger.error(f"加载Python语言库失败: {e}")
            raise
        
        # 定义感兴趣的代码节点类型
        self.CODE_NODE_TYPES = {'function_definition', 'class_definition'}
        logger.debug(f"设置目标节点类型: {self.CODE_NODE_TYPES}")
    
    def _find_target_nodes(self, node: tree_sitter.Node) -> Generator[tree_sitter.Node, None, None]:
        """
        递归查找目标节点
        
        使用深度优先遍历递归地查找所有类型在CODE_NODE_TYPES中的节点。
        
        Args:
            node (tree_sitter.Node): 要遍历的AST节点
            
        Yields:
            tree_sitter.Node: 匹配目标类型的AST节点
        """
        # 检查当前节点是否为目标类型
        if node.type in self.CODE_NODE_TYPES:
            logger.debug(f"找到目标节点: {node.type} at line {node.start_point[0] + 1}")
            yield node
        
        # 递归遍历子节点
        for child in node.children:
            yield from self._find_target_nodes(child)
    
    def chunk_source_code(self, file_path: str, code: str) -> List[CodeChunk]:
        """
        对源代码进行分块
        
        使用tree-sitter解析源代码，按照AST结构进行分块，
        为每个函数和类定义创建独立的代码块。
        
        Args:
            file_path (str): 源文件路径
            code (str): 源代码字符串
            
        Returns:
            List[CodeChunk]: 包含所有代码块的列表
        """
        logger.info(f"开始对文件进行AST分块: {file_path}")
        
        chunks = []
        
        try:
            # 将代码字符串转换为字节
            code_bytes = code.encode('utf-8')
            
            # 使用tree-sitter解析代码
            tree = self.parser.parse(code_bytes)
            root_node = tree.root_node
            
            logger.debug(f"成功解析AST，根节点类型: {root_node.type}")
            
            # 将代码按行分割，用于提取具体的代码块文本
            code_lines = code.split('\n')
            
            # 遍历所有目标节点
            for node in self._find_target_nodes(root_node):
                try:
                    # 获取节点的起始和结束位置
                    start_row = node.start_point[0]
                    end_row = node.end_point[0]
                    start_col = node.start_point[1]
                    end_col = node.end_point[1]
                    
                    # 提取代码块文本
                    if start_row == end_row:
                        # 单行代码块
                        chunk_text = code_lines[start_row][start_col:end_col]
                    else:
                        # 多行代码块
                        chunk_lines = []
                        
                        # 第一行（从start_col开始）
                        chunk_lines.append(code_lines[start_row][start_col:])
                        
                        # 中间的完整行
                        for i in range(start_row + 1, end_row):
                            if i < len(code_lines):
                                chunk_lines.append(code_lines[i])
                        
                        # 最后一行（到end_col结束）
                        if end_row < len(code_lines):
                            chunk_lines.append(code_lines[end_row][:end_col])
                        
                        chunk_text = '\n'.join(chunk_lines)
                    
                    # 创建元数据
                    metadata = {
                        'file_path': file_path,
                        'start_line': start_row + 1,  # 转换为1基索引
                        'end_line': end_row + 1,
                        'node_type': node.type,
                        'start_col': start_col,
                        'end_col': end_col
                    }
                    
                    # 创建代码块对象
                    # 临时node_id，在create_knowledge_graph中会被重新赋值
                    temp_node_id = f"{file_path}::{start_row + 1}"
                    chunk = CodeChunk(text=chunk_text, metadata=metadata, node_id=temp_node_id)
                    chunks.append(chunk)
                    
                    logger.debug(f"创建代码块: {node.type} at lines {start_row + 1}-{end_row + 1}")
                    
                except Exception as e:
                    logger.warning(f"处理节点时发生错误: {e}，跳过该节点")
                    continue
            
            logger.info(f"成功创建 {len(chunks)} 个代码块")
            return chunks
            
        except Exception as e:
            logger.error(f"解析代码时发生错误: {e}")
            return []

    def create_knowledge_graph(self, source_files: Dict[str, str]) -> nx.DiGraph:
        """
        构建代码知识图谱

        遍历所有源文件，创建代码块节点，并建立它们之间的关系（如调用、继承、导入）。

        Args:
            source_files (Dict[str, str]): 包含文件路径和代码内容的字典

        Returns:
            nx.DiGraph: 构建完成的知识图谱
        """
        logger.info("开始构建知识图谱...")
        graph = nx.DiGraph()
        all_chunks = {}

        # 第一遍：创建所有节点
        logger.info("第一遍：创建所有代码块节点")
        for file_path, code in source_files.items():
            chunks = self.chunk_source_code(file_path, code)
            for chunk in chunks:
                # 更新node_id的计算方式
                chunk.node_id = f"{chunk.metadata['file_path']}::{chunk.text.splitlines()[0]}"
                node_attributes = chunk.to_dict()
                raw_node_type = chunk.metadata.get('node_type', 'unknown')
                # 映射到更简洁的类型
                if raw_node_type == 'function_definition':
                    node_type = 'function'
                elif raw_node_type == 'class_definition':
                    node_type = 'class'
                else:
                    node_type = raw_node_type
                node_attributes['type'] = node_type
                graph.add_node(chunk.node_id, **node_attributes)
                all_chunks[chunk.node_id] = chunk
        logger.info(f"图谱中总共添加了 {len(graph.nodes)} 个节点")

        # 第二遍：创建边（关系）
        logger.info("第二遍：分析代码关系并创建边")
        for file_path, code in source_files.items():
            try:
                tree = self.parser.parse(code.encode('utf-8'))
                root_node = tree.root_node
                self._find_relationships(root_node, file_path, graph, all_chunks)
            except Exception as e:
                logger.error(f"在文件 {file_path} 中创建边时出错: {e}")

        logger.info(f"知识图谱构建完成，包含 {len(graph.nodes())} 个节点和 {len(graph.edges())} 条边。")
        return graph

    def _find_relationships(self, node: tree_sitter.Node, file_path: str, graph: nx.DiGraph, all_chunks: Dict[str, CodeChunk]):
        """递归遍历AST，寻找并创建代码关系。"""
        # 可以在这里添加更复杂的逻辑来处理不同类型的关系
        # 例如，处理继承关系
        if node.type == 'class_definition':
            class_name_node = node.child_by_field_name('name')
            if class_name_node:
                class_name = class_name_node.text.decode('utf8')
                caller_node_id = self._find_node_id_by_name(class_name, file_path, all_chunks)
                if caller_node_id:
                    # 查找父类
                    superclass_node = node.child_by_field_name('superclass')
                    if superclass_node:
                        parent_class_name = superclass_node.text.decode('utf8')
                        callee_node_id = self._find_node_id_by_name(parent_class_name, file_path, all_chunks, search_globally=True)
                        if callee_node_id:
                            graph.add_edge(caller_node_id, callee_node_id, label='inherits_from')
                            logger.debug(f"添加继承边: {caller_node_id} -> {callee_node_id}")

        # 处理调用关系
        if node.type == 'call':
            # 找到调用者（所在的函数或类）
            caller_context_node = self._get_context_node(node)
            if caller_context_node:
                caller_name_node = caller_context_node.child_by_field_name('name')
                if caller_name_node:
                    caller_name = caller_name_node.text.decode('utf8')
                    caller_node_id = self._find_node_id_by_name(caller_name, file_path, all_chunks)
                    
                    # 找到被调用者
                    callee_name_node = node.child_by_field_name('function')
                    if callee_name_node and caller_node_id:
                        callee_name = callee_name_node.text.decode('utf8').split('.')[-1] # 处理 a.b() 的情况
                        callee_node_id = self._find_node_id_by_name(callee_name, file_path, all_chunks, search_globally=True)
                        if callee_node_id and caller_node_id != callee_node_id:
                            graph.add_edge(caller_node_id, callee_node_id, label='calls')
                            logger.debug(f"添加调用边: {caller_node_id} -> {callee_node_id}")

        # 递归遍历子节点
        for child in node.children:
            self._find_relationships(child, file_path, graph, all_chunks)

    def _get_context_node(self, node: tree_sitter.Node) -> tree_sitter.Node | None:
        """向上遍历AST，找到包含当前节点的函数或类定义。"""
        current = node.parent
        while current:
            if current.type in self.CODE_NODE_TYPES:
                return current
            current = current.parent
        return None

    def _find_node_id_by_name(self, name: str, current_file: str, all_chunks: Dict[str, CodeChunk], search_globally: bool = False) -> str | None:
        """根据名称查找节点的ID。"""
        # 优先在当前文件中查找
        for node_id, chunk in all_chunks.items():
            if chunk.metadata['file_path'] == current_file:
                # 简单的名称匹配逻辑
                if f"def {name}(" in chunk.text or f"class {name}" in chunk.text:
                    return node_id
        
        if search_globally:
            # 如果需要，则在所有文件中查找
            for node_id, chunk in all_chunks.items():
                if f"def {name}(" in chunk.text or f"class {name}" in chunk.text:
                    return node_id
        return None

    
    def get_chunk_summary(self, chunks: List[CodeChunk]) -> Dict[str, int]:
        """
        获取分块统计信息
        
        统计不同类型代码块的数量。
        
        Args:
            chunks (List[CodeChunk]): 代码块列表
            
        Returns:
            Dict[str, int]: 包含各类型代码块数量的字典
        """
        summary = {}
        for chunk in chunks:
            node_type = chunk.metadata.get('node_type', 'unknown')
            summary[node_type] = summary.get(node_type, 0) + 1
        
        logger.info(f"分块统计: {summary}")
        return summary