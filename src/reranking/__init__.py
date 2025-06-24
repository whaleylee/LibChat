"""重排序模块。

该模块提供基于机器学习模型的文档重排序功能，用于提高检索系统的精度。
主要包含SentenceTransformerReranker类，使用CrossEncoder模型对检索结果进行重新排序。
"""

from .reranker import SentenceTransformerReranker

__all__ = ['SentenceTransformerReranker']