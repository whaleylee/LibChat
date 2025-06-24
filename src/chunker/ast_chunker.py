#!/usr/bin/env python3
"""
AST Chunker Module

本模块提供基于tree-sitter的AST结构化分块功能，
用于将Python源代码按照语法结构进行智能分块。
"""

import tree_sitter
import tree_sitter_languages
from dataclasses import dataclass
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
    """
    text: str
    metadata: Dict[str, Any]


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
                    chunk = CodeChunk(text=chunk_text, metadata=metadata)
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