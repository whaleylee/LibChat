#!/usr/bin/env python3

import sys
sys.path.append('.')

from src.indexing.fixed_indexer import FixedIndexer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
import numpy as np

def debug_embedding_and_retrieval():
    import sys
    log_file_path = "debug_retrieval_output.log"
    sys.stdout = open(log_file_path, "w", encoding="utf-8")
    sys.stderr = open(log_file_path, "a", encoding="utf-8") # Append stderr to the same file
    print("开始调试嵌入和检索功能...")
    
    # 设置嵌入模型
    embed_model = HuggingFaceEmbedding(
        model_name='BAAI/bge-small-en-v1.5',
        trust_remote_code=True
    )
    Settings.embed_model = embed_model
    print("嵌入模型设置完成")
    
    # 测试嵌入模型
    test_text = "如何发送GET请求？"
    try:
        embedding = embed_model.get_text_embedding(test_text)
        print(f"\n测试嵌入成功:")
        print(f"  文本: {test_text}")
        print(f"  嵌入维度: {len(embedding)}")
        print(f"  嵌入前5个值: {embedding[:5]}")
    except Exception as e:
        print(f"\n嵌入测试失败: {e}")
        return
    
    index_path = Path('./indexes/requests')
    print(f"\n索引路径: {index_path}")
    print(f"索引路径存在: {index_path.exists()}")
    
    try:
        # 使用FaissIndexer加载索引
        indexer = FixedIndexer('./indexes')
        indexer.rebuild_index("../LibChat", index_name="requests", embedding_model="text-embedding-ada-002")
        index = indexer.load_index(index_name="requests")
        print(f"\n索引加载成功，类型: {type(index)}")
        
        # 检查向量存储
        if hasattr(index, '_vector_store'):
            vector_store = index._vector_store
            print(f"向量存储类型: {type(vector_store)}")
            
            # 检查向量存储的详细信息
            if hasattr(vector_store, '_data'):
                data = vector_store._data
                print(f"向量存储数据类型: {type(data)}")
                if hasattr(data, 'embedding_dict'):
                    embedding_dict = data.embedding_dict
                    print(f"嵌入字典大小: {len(embedding_dict)}")
                    if len(embedding_dict) > 0:
                        first_key = list(embedding_dict.keys())[0]
                        first_embedding = embedding_dict[first_key]
                        print(f"第一个嵌入维度: {len(first_embedding)}")
                        print(f"第一个嵌入前5个值: {first_embedding[:5]}")
                else:
                    print("向量存储数据中没有embedding_dict")
            else:
                print("向量存储中没有_data属性")
        
        # 检查docstore
        if hasattr(index, '_storage_context') and hasattr(index._storage_context, 'docstore'):
            docstore = index._storage_context.docstore
            print(f"\n文档存储中的文档数量: {len(docstore.docs)}")
        
        # 尝试直接查询向量存储
        print("\n=== 直接查询向量存储 ===")
        if hasattr(index, '_vector_store'):
            vector_store = index._vector_store
            try:
                # 获取查询嵌入
                query_embedding = embed_model.get_text_embedding("GET request")
                print(f"查询嵌入维度: {len(query_embedding)}")
                
                # 尝试直接查询向量存储
                from llama_index.core.vector_stores.types import VectorStoreQuery
                query_obj = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=5
                )
                
                result = vector_store.query(query_obj)
                print(f"向量存储查询结果类型: {type(result)}")
                if result is not None:
                    print(f"查询结果节点数: {len(result.nodes) if result.nodes else 0}")
                    print(f"相似度分数: {result.similarities}")
                else:
                    print("查询结果为None")
                
                if result is not None and result.nodes:
                     for i, node in enumerate(result.nodes[:2]):
                         print(f"\n节点 {i+1}:")
                         print(f"  文本: {node.text[:100]}...")
                         print(f"  元数据: {node.metadata}")
                    
            except Exception as e:
                print(f"直接查询向量存储失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 尝试检索
        print("\n=== 使用VectorIndexRetriever检索 ===")
        retriever = VectorIndexRetriever(index=index, similarity_top_k=20)
        
        test_queries = ["GET", "request", "http", "requests", "get", "如何发送GET请求？"]
        for query in test_queries:
            try:
                nodes = retriever.retrieve(query)
                print(f"查询 '{query}': {len(nodes)} 个节点")
                if len(nodes) > 0:
                    print(f"  第一个结果: {nodes[0].text[:100]}...")
                    if hasattr(nodes[0], 'score'):
                        print(f"  相似度分数: {nodes[0].score}")
            except Exception as e:
                print(f"查询 '{query}' 失败: {e}")
                
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_embedding_and_retrieval()