from typing import List, Union, Any
from loguru import logger
import torch
from sentence_transformers import CrossEncoder


class SentenceTransformerReranker:
    """
    基于SentenceTransformers CrossEncoder的文档重排序器。
    
    该类使用预训练的CrossEncoder模型对检索到的文档进行重新排序，
    通过计算查询与文档之间的相关性分数来提高检索质量。
    
    Attributes:
        model_name (str): 使用的CrossEncoder模型名称
        top_n (int): 返回的顶部文档数量
        model (CrossEncoder): 加载的CrossEncoder模型
        device (str): 计算设备（cpu或cuda）
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-reranker-large', top_n: int = 5) -> None:
        """
        初始化SentenceTransformerReranker。
        
        Args:
            model_name (str): CrossEncoder模型名称，默认为'BAAI/bge-reranker-large'
            top_n (int): 返回的顶部文档数量，默认为5
        """
        self.model_name = model_name
        self.top_n = top_n
        
        # 设置计算设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {self.device}")
        
        # 加载CrossEncoder模型
        try:
            logger.info(f"正在加载CrossEncoder模型: {model_name}")
            self.model = CrossEncoder(model_name, device=self.device)
            logger.info(f"成功加载模型: {model_name}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def rerank(self, query: str, documents: List[Union[str, Any]]) -> List[Union[str, Any]]:
        """
        对文档列表进行重新排序。
        
        该方法接收查询字符串和文档列表，使用CrossEncoder模型计算
        查询与每个文档的相关性分数，然后按分数从高到低排序。
        
        Args:
            query (str): 查询字符串
            documents (List[Union[str, Any]]): 待排序的文档列表，
                可以是字符串列表或包含text属性的对象列表（如NodeWithScore）
        
        Returns:
            List[Union[str, Any]]: 重新排序后的顶部文档列表
        
        Raises:
            ValueError: 当文档列表为空时
            Exception: 当模型预测失败时
        """
        if not documents:
            logger.warning("文档列表为空，返回空列表")
            return []
        
        if not query.strip():
            logger.warning("查询字符串为空，返回原始文档列表")
            return documents[:self.top_n]
        
        logger.info(f"开始重排序 {len(documents)} 个文档")
        
        try:
            # 准备查询-文档对
            query_doc_pairs = []
            for doc in documents:
                # 处理不同类型的文档对象
                if isinstance(doc, str):
                    doc_text = doc
                elif hasattr(doc, 'text'):
                    # 处理NodeWithScore等对象
                    doc_text = doc.text
                elif hasattr(doc, 'node') and hasattr(doc.node, 'text'):
                    # 处理嵌套的node对象
                    doc_text = doc.node.text
                elif hasattr(doc, 'content'):
                    # 处理content属性
                    doc_text = doc.content
                else:
                    # 尝试转换为字符串
                    doc_text = str(doc)
                    logger.warning(f"未知文档类型，转换为字符串: {type(doc)}")
                
                query_doc_pairs.append([query, doc_text])
            
            # 使用模型预测相关性分数
            logger.debug(f"正在计算 {len(query_doc_pairs)} 个查询-文档对的相关性分数")
            scores = self.model.predict(query_doc_pairs)
            
            # 将文档与分数配对并排序
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回顶部文档
            top_documents = [doc for doc, score in doc_score_pairs[:self.top_n]]
            
            logger.info(f"重排序完成，返回前 {len(top_documents)} 个文档")
            logger.debug(f"顶部文档分数: {[score for _, score in doc_score_pairs[:self.top_n]]}")
            
            return top_documents
            
        except Exception as e:
            logger.error(f"重排序过程中发生错误: {e}")
            # 发生错误时返回原始文档的前top_n个
            logger.warning("返回原始文档列表作为备选")
            return documents[:self.top_n]
    
    def get_scores(self, query: str, documents: List[Union[str, Any]]) -> List[float]:
        """
        获取查询与文档的相关性分数，不进行排序。
        
        Args:
            query (str): 查询字符串
            documents (List[Union[str, Any]]): 文档列表
        
        Returns:
            List[float]: 相关性分数列表，与输入文档顺序对应
        """
        if not documents:
            return []
        
        try:
            # 准备查询-文档对
            query_doc_pairs = []
            for doc in documents:
                if isinstance(doc, str):
                    doc_text = doc
                elif hasattr(doc, 'text'):
                    doc_text = doc.text
                elif hasattr(doc, 'node') and hasattr(doc.node, 'text'):
                    doc_text = doc.node.text
                elif hasattr(doc, 'content'):
                    doc_text = doc.content
                else:
                    doc_text = str(doc)
                
                query_doc_pairs.append([query, doc_text])
            
            # 计算分数
            scores = self.model.predict(query_doc_pairs)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            
        except Exception as e:
            logger.error(f"计算分数时发生错误: {e}")
            return [0.0] * len(documents)
    
    def __repr__(self) -> str:
        """返回对象的字符串表示。"""
        return f"SentenceTransformerReranker(model_name='{self.model_name}', top_n={self.top_n}, device='{self.device}')"