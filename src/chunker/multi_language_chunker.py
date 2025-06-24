#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语言代码分块器模块

本模块提供支持多种编程语言和文件类型的代码分块功能，
扩展了原有的AST分块器，支持更多文件类型的处理。
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

try:
    import tree_sitter
    import tree_sitter_languages
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter不可用，将使用基于正则表达式的分块方法")

from .ast_chunker import CodeChunk


class MultiLanguageChunker:
    """
    多语言代码分块器
    
    支持多种编程语言和文件类型的代码分块，包括：
    - Python (.py)
    - JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
    - Java (.java)
    - C/C++ (.c, .cpp, .h, .hpp)
    - Go (.go)
    - Rust (.rs)
    - 文档文件 (.md, .txt, .rst)
    - 配置文件 (.json, .yaml, .yml, .toml, .ini)
    """
    
    # 支持的文件扩展名映射到语言
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript', 
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript', 
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.go': 'go',
        '.rs': 'rust',
        '.md': 'markdown',
        '.txt': 'text',
        '.rst': 'rst',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.sql': 'sql',
        '.sh': 'bash',
        '.bat': 'batch',
        '.ps1': 'powershell'
    }
    
    # Tree-sitter支持的语言
    TREE_SITTER_LANGUAGES = {
        'python', 'javascript', 'typescript', 'java', 
        'c', 'cpp', 'go', 'rust'
    }
    
    def __init__(self):
        """初始化多语言分块器"""
        logger.info("初始化MultiLanguageChunker")
        self.parsers = {}
        
        if TREE_SITTER_AVAILABLE:
            self._init_tree_sitter_parsers()
        
        # 初始化基于正则表达式的分块模式
        self._init_regex_patterns()
    
    def _init_tree_sitter_parsers(self):
        """初始化tree-sitter解析器"""
        for lang in self.TREE_SITTER_LANGUAGES:
            try:
                parser = tree_sitter.Parser()
                language = tree_sitter_languages.get_language(lang)
                parser.set_language(language)
                self.parsers[lang] = parser
                logger.debug(f"成功初始化{lang}解析器")
            except Exception as e:
                logger.warning(f"初始化{lang}解析器失败: {e}")
    
    def _init_regex_patterns(self):
        """初始化正则表达式模式"""
        self.regex_patterns = {
            'python': {
                'function': r'^\s*(def\s+\w+\s*\([^)]*\)\s*:)',
                'class': r'^\s*(class\s+\w+\s*(?:\([^)]*\))?\s*:)'
            },
            'javascript': {
                'function': r'^\s*(function\s+\w+\s*\([^)]*\)|\w+\s*:\s*function\s*\([^)]*\)|\w+\s*=\s*\([^)]*\)\s*=>)',
                'class': r'^\s*(class\s+\w+\s*(?:extends\s+\w+)?\s*\{)'
            },
            'java': {
                'method': r'^\s*((?:public|private|protected|static|final|abstract|synchronized|native|strictfp)\s+)*\w+\s+\w+\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{',
                'class': r'^\s*((?:public|private|protected|static|final|abstract)\s+)*class\s+\w+\s*(?:extends\s+\w+)?\s*(?:implements\s+[^{]+)?\s*\{'
            },
            'go': {
                'function': r'^\s*(func\s+(?:\([^)]*\)\s*)?\w+\s*\([^)]*\)(?:\s*[^{]+)?\s*\{)'
            }
        }
    
    def is_supported_file(self, file_path: str) -> bool:
        """检查文件是否支持分块"""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def get_file_language(self, file_path: str) -> Optional[str]:
        """获取文件对应的编程语言"""
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext)
    
    def chunk_file(self, file_path: str) -> List[CodeChunk]:
        """对文件进行分块"""
        if not self.is_supported_file(file_path):
            logger.warning(f"不支持的文件类型: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return []
        
        language = self.get_file_language(file_path)
        
        # 尝试使用tree-sitter分块
        if (TREE_SITTER_AVAILABLE and 
            language in self.TREE_SITTER_LANGUAGES and 
            language in self.parsers):
            return self._chunk_with_tree_sitter(file_path, content, language)
        
        # 回退到正则表达式分块
        return self._chunk_with_regex(file_path, content, language)
    
    def _chunk_with_tree_sitter(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """使用tree-sitter进行分块"""
        logger.debug(f"使用tree-sitter分块: {file_path} ({language})")
        
        chunks = []
        parser = self.parsers[language]
        
        try:
            tree = parser.parse(content.encode('utf-8'))
            root_node = tree.root_node
            
            # 定义不同语言的目标节点类型
            target_types = {
                'python': {'function_definition', 'class_definition'},
                'javascript': {'function_declaration', 'method_definition', 'class_declaration'},
                'typescript': {'function_declaration', 'method_definition', 'class_declaration'},
                'java': {'method_declaration', 'class_declaration'},
                'c': {'function_definition'},
                'cpp': {'function_definition', 'class_specifier'},
                'go': {'function_declaration', 'method_declaration'},
                'rust': {'function_item', 'impl_item'}
            }
            
            node_types = target_types.get(language, set())
            content_lines = content.split('\n')
            
            for node in self._find_nodes_by_type(root_node, node_types):
                chunk_text = self._extract_node_text(node, content_lines)
                if chunk_text.strip():
                    metadata = {
                        'file_path': file_path,
                        'language': language,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'node_type': node.type,
                        'chunk_method': 'tree_sitter'
                    }
                    chunks.append(CodeChunk(text=chunk_text, metadata=metadata))
            
        except Exception as e:
            logger.error(f"tree-sitter分块失败 {file_path}: {e}")
            return self._chunk_with_regex(file_path, content, language)
        
        return chunks
    
    def _chunk_with_regex(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """使用正则表达式进行分块"""
        logger.debug(f"使用正则表达式分块: {file_path} ({language})")
        
        chunks = []
        
        # 对于文档和配置文件，按段落或节分块
        if language in ['markdown', 'text', 'rst']:
            return self._chunk_document(file_path, content, language)
        
        if language in ['json', 'yaml', 'toml', 'ini', 'xml']:
            return self._chunk_config_file(file_path, content, language)
        
        # 对于代码文件，使用正则表达式查找函数和类
        patterns = self.regex_patterns.get(language, {})
        if not patterns:
            # 如果没有特定模式，按固定行数分块
            return self._chunk_by_lines(file_path, content, language)
        
        lines = content.split('\n')
        current_chunk_start = 0
        
        for i, line in enumerate(lines):
            for pattern_type, pattern in patterns.items():
                if re.match(pattern, line, re.MULTILINE):
                    # 如果找到新的函数/类定义，保存之前的块
                    if current_chunk_start < i:
                        chunk_text = '\n'.join(lines[current_chunk_start:i]).strip()
                        if chunk_text:
                            metadata = {
                                'file_path': file_path,
                                'language': language,
                                'start_line': current_chunk_start + 1,
                                'end_line': i,
                                'node_type': 'code_block',
                                'chunk_method': 'regex'
                            }
                            chunks.append(CodeChunk(text=chunk_text, metadata=metadata))
                    
                    current_chunk_start = i
                    break
        
        # 添加最后一个块
        if current_chunk_start < len(lines):
            chunk_text = '\n'.join(lines[current_chunk_start:]).strip()
            if chunk_text:
                metadata = {
                    'file_path': file_path,
                    'language': language,
                    'start_line': current_chunk_start + 1,
                    'end_line': len(lines),
                    'node_type': 'code_block',
                    'chunk_method': 'regex'
                }
                chunks.append(CodeChunk(text=chunk_text, metadata=metadata))
        
        return chunks
    
    def _chunk_document(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """分块文档文件"""
        chunks = []
        
        if language == 'markdown':
            # 按标题分块
            sections = re.split(r'^#+\s+', content, flags=re.MULTILINE)
            current_line = 1
            
            for i, section in enumerate(sections):
                if section.strip():
                    lines_count = section.count('\n') + 1
                    metadata = {
                        'file_path': file_path,
                        'language': language,
                        'start_line': current_line,
                        'end_line': current_line + lines_count - 1,
                        'node_type': 'markdown_section',
                        'chunk_method': 'regex'
                    }
                    chunks.append(CodeChunk(text=section.strip(), metadata=metadata))
                    current_line += lines_count
        else:
            # 按段落分块
            paragraphs = re.split(r'\n\s*\n', content)
            current_line = 1
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    lines_count = paragraph.count('\n') + 1
                    metadata = {
                        'file_path': file_path,
                        'language': language,
                        'start_line': current_line,
                        'end_line': current_line + lines_count - 1,
                        'node_type': 'paragraph',
                        'chunk_method': 'regex'
                    }
                    chunks.append(CodeChunk(text=paragraph.strip(), metadata=metadata))
                    current_line += lines_count + 1  # +1 for the empty line
        
        return chunks
    
    def _chunk_config_file(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """分块配置文件"""
        # 对于配置文件，通常作为整体处理
        metadata = {
            'file_path': file_path,
            'language': language,
            'start_line': 1,
            'end_line': content.count('\n') + 1,
            'node_type': 'config_file',
            'chunk_method': 'whole_file'
        }
        return [CodeChunk(text=content, metadata=metadata)]
    
    def _chunk_by_lines(self, file_path: str, content: str, language: str, chunk_size: int = 50) -> List[CodeChunk]:
        """按行数分块"""
        chunks = []
        lines = content.split('\n')
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            
            metadata = {
                'file_path': file_path,
                'language': language,
                'start_line': i + 1,
                'end_line': min(i + chunk_size, len(lines)),
                'node_type': 'line_chunk',
                'chunk_method': 'line_based'
            }
            chunks.append(CodeChunk(text=chunk_text, metadata=metadata))
        
        return chunks
    
    def _find_nodes_by_type(self, node, target_types):
        """递归查找指定类型的节点"""
        if node.type in target_types:
            yield node
        
        for child in node.children:
            yield from self._find_nodes_by_type(child, target_types)
    
    def _extract_node_text(self, node, content_lines):
        """从节点提取文本"""
        start_row = node.start_point[0]
        end_row = node.end_point[0]
        start_col = node.start_point[1]
        end_col = node.end_point[1]
        
        if start_row == end_row:
            return content_lines[start_row][start_col:end_col]
        else:
            chunk_lines = []
            chunk_lines.append(content_lines[start_row][start_col:])
            
            for i in range(start_row + 1, end_row):
                if i < len(content_lines):
                    chunk_lines.append(content_lines[i])
            
            if end_row < len(content_lines):
                chunk_lines.append(content_lines[end_row][:end_col])
            
            return '\n'.join(chunk_lines)
    
    def chunk_directory(self, directory_path: str) -> List[CodeChunk]:
        """对目录中的所有支持文件进行分块"""
        logger.info(f"开始分块目录: {directory_path}")
        
        all_chunks = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"目录不存在: {directory_path}")
            return []
        
        # 递归查找所有支持的文件
        for file_path in directory.rglob('*'):
            if file_path.is_file() and self.is_supported_file(str(file_path)):
                logger.debug(f"处理文件: {file_path}")
                chunks = self.chunk_file(str(file_path))
                all_chunks.extend(chunks)
        
        logger.info(f"目录分块完成，共生成 {len(all_chunks)} 个代码块")
        return all_chunks
    
    def get_supported_extensions(self) -> List[str]:
        """获取支持的文件扩展名列表"""
        return list(self.SUPPORTED_EXTENSIONS.keys())
    
    def get_chunk_summary(self, chunks: List[CodeChunk]) -> Dict[str, Any]:
        """获取分块统计信息"""
        summary = {
            'total_chunks': len(chunks),
            'by_language': {},
            'by_node_type': {},
            'by_chunk_method': {},
            'files_processed': set()
        }
        
        for chunk in chunks:
            metadata = chunk.metadata
            
            # 按语言统计
            language = metadata.get('language', 'unknown')
            summary['by_language'][language] = summary['by_language'].get(language, 0) + 1
            
            # 按节点类型统计
            node_type = metadata.get('node_type', 'unknown')
            summary['by_node_type'][node_type] = summary['by_node_type'].get(node_type, 0) + 1
            
            # 按分块方法统计
            chunk_method = metadata.get('chunk_method', 'unknown')
            summary['by_chunk_method'][chunk_method] = summary['by_chunk_method'].get(chunk_method, 0) + 1
            
            # 记录处理的文件
            file_path = metadata.get('file_path')
            if file_path:
                summary['files_processed'].add(file_path)
        
        summary['files_processed'] = len(summary['files_processed'])
        
        logger.info(f"分块统计: {summary}")
        return summary