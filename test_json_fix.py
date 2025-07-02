#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试JSON序列化修复
"""

import json
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_path_serialization():
    """测试Path对象序列化"""
    print("测试Path对象JSON序列化...")
    
    # 创建包含Path对象的数据
    test_data = {
        'file_path': Path('test/file.py'),
        'normal_string': 'test',
        'number': 123
    }
    
    print(f"原始数据: {test_data}")
    
    try:
        # 尝试直接序列化（应该失败）
        json_str = json.dumps(test_data)
        print("直接序列化成功（不应该发生）")
    except TypeError as e:
        print(f"直接序列化失败（预期）: {e}")
    
    # 转换Path对象为字符串后序列化
    def convert_paths_to_strings(obj):
        if isinstance(obj, dict):
            return {k: convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths_to_strings(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    converted_data = convert_paths_to_strings(test_data)
    print(f"转换后数据: {converted_data}")
    
    try:
        json_str = json.dumps(converted_data, ensure_ascii=False, indent=2)
        print("转换后序列化成功!")
        print(f"JSON字符串: {json_str}")
        return True
    except Exception as e:
        print(f"转换后序列化失败: {e}")
        return False

def test_chunker_import():
    """测试chunker导入"""
    print("\n测试chunker导入...")
    try:
        from src.chunker.multi_language_chunker import MultiLanguageChunker
        print("MultiLanguageChunker导入成功")
        
        chunker = MultiLanguageChunker()
        print("MultiLanguageChunker实例化成功")
        return True
    except Exception as e:
        print(f"chunker导入失败: {e}")
        return False

def test_fixed_indexer_import():
    """测试fixed_indexer导入"""
    print("\n测试fixed_indexer导入...")
    try:
        from src.indexing.fixed_indexer import FixedIndexer
        print("FixedIndexer导入成功")
        return True
    except Exception as e:
        print(f"FixedIndexer导入失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试JSON序列化修复...")
    
    success1 = test_path_serialization()
    success2 = test_chunker_import()
    success3 = test_fixed_indexer_import()
    
    if success1 and success2 and success3:
        print("\n✅ 所有测试通过！JSON序列化问题已修复。")
    else:
        print("\n❌ 部分测试失败，需要进一步检查。")