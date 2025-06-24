#!/usr/bin/env python3

import sys
sys.path.append('.')

from llama_index.core.storage.storage_context import DEFAULT_VECTOR_STORE
print(f"DEFAULT_VECTOR_STORE 常量值: '{DEFAULT_VECTOR_STORE}'")

# 检查索引目录中的实际文件
from pathlib import Path
index_path = Path('./indexes/requests')
print(f"\n索引目录文件:")
for file in index_path.iterdir():
    print(f"  {file.name}")

# 检查是否有以 DEFAULT_VECTOR_STORE 命名的文件
expected_file = f"{DEFAULT_VECTOR_STORE}__vector_store.json"
expected_path = index_path / expected_file
print(f"\n期望的向量存储文件: {expected_file}")
print(f"文件是否存在: {expected_path.exists()}")

# 检查所有向量存储文件
print(f"\n所有向量存储文件:")
for file in index_path.glob("*__vector_store.json"):
    print(f"  {file.name}")
    # 读取文件内容
    import json
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'embedding_dict' in data:
            print(f"    嵌入字典大小: {len(data['embedding_dict'])}")
    except Exception as e:
        print(f"    读取失败: {e}")