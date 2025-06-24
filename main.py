#!/usr/bin/env python3
"""
LibChat - 本地Python库智能问答系统

这是一个基于RAG（检索增强生成）技术的命令行应用，能够对本地安装的Python库进行智能问答。
系统通过AST解析、向量索引、重排序和大语言模型生成，提供准确的代码相关问答服务。

主要功能：
1. build命令：构建指定Python库的知识库索引
2. ask命令：基于构建的知识库回答用户问题

作者：LibChat Team
版本：1.0.0
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv

# 导入llama_index相关模块
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever

# 导入OpenAI相关模块
import openai
from openai import OpenAI

# 导入我们自定义的模块
from src.source_inspector.inspector import PackageInspector
from src.chunker.ast_chunker import ASTChunker
from src.indexing.fixed_indexer import FixedIndexer
from src.reranking.reranker import SentenceTransformerReranker

# 加载环境变量
load_dotenv()

# 创建CLI应用和控制台
app = typer.Typer(
name="libchat",
    help="本地Python库智能问答系统 - 基于RAG技术的代码问答助手",
    add_completion=False
)
console = Console()

# 配置常量
CONFIG = {
    "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    "reranker_model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large"), 
    "qwen_model": "qwen-turbo",
    "qwen_api_key": os.getenv("QWEN_API_KEY"),
    "qwen_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "index_dir": "./indexes",
    "top_k_retrieval": 20,
    "top_k_rerank": 5,
    "max_tokens": 1024,
    "temperature": 0.1
}

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/libchat.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB"
)


def ensure_directories() -> None:
    """确保必要的目录存在。"""
    Path(CONFIG["index_dir"]).mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)


def process_query(query: str, index_path: str) -> str:
    """
    处理用户查询，执行完整的RAG流程。
    
    Args:
        query: 用户查询
        index_path: 索引路径
        
    Returns:
        生成的答案
    """
    try:
        # 1. 加载索引
        logger.info(f"正在加载索引: {index_path}")
        indexer = FixedIndexer(index_dir=CONFIG["index_dir"])
        index_name = Path(index_path).name
        index = indexer.load_index(index_name, embedding_model=CONFIG["embedding_model"])
        
        if index is None:
            logger.error(f"无法加载索引: {index_name}")
            return f"抱歉，无法加载索引 '{index_name}'。请确保索引已正确构建。"
        
        # 2. 检索相关文档
        logger.info(f"正在检索查询: {query}")
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=CONFIG["top_k_retrieval"]
        )
        retrieved_nodes = retriever.retrieve(query)
        
        if not retrieved_nodes:
            return "抱歉，我没有找到与您的问题相关的信息。"
        
        logger.info(f"检索到 {len(retrieved_nodes)} 个相关文档")
        
        # 3. 重排序（如果有足够的文档）
        if len(retrieved_nodes) > CONFIG["top_k_rerank"]:
            logger.info("正在进行重排序...")
            reranker = SentenceTransformerReranker(
                 model_name=CONFIG["reranker_model"],
                 top_n=CONFIG["top_k_rerank"]
             )
            
            # 准备重排序的文档
            documents = [node.text for node in retrieved_nodes]
            reranked_docs = reranker.rerank(query, documents)
            
            # 使用重排序后的文档（reranked_docs直接是字符串列表）
            context_texts = reranked_docs
            logger.info(f"重排序后保留 {len(context_texts)} 个最相关文档")
        else:
            # 如果文档数量不多，直接使用检索结果
            context_texts = [node.text for node in retrieved_nodes]
            logger.info(f"直接使用 {len(context_texts)} 个检索文档")
        
        # 4. 构建上下文
        context = "\n\n".join(context_texts)
        
        # 5. 生成答案
        logger.info("正在生成答案...")
        answer = generate_answer_with_llm(query, context)
        
        return answer
        
    except Exception as e:
        logger.error(f"处理查询失败: {e}", exc_info=True)
        return f"处理查询时发生错误: {str(e)}"


def generate_answer_with_llm(query: str, context: str) -> str:
    """
    使用Qwen API基于上下文生成答案。
    
    Args:
        query: 用户查询
        context: 检索到的上下文
        
    Returns:
        生成的答案
    """
    try:
        # 初始化Qwen客户端（使用OpenAI兼容接口）
        client = OpenAI(
            api_key=CONFIG["qwen_api_key"],
            base_url=CONFIG["qwen_base_url"]
        )
        
        # 构建消息
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的Python代码助手。请基于提供的代码上下文回答用户的问题。提供准确、详细的答案，如果可能的话，包含相关的代码示例。如果上下文中没有足够的信息来回答问题，请诚实地说明。"
            },
            {
                "role": "user",
                "content": f"""上下文信息：
                {context}

                用户问题：{query}

                请基于上述上下文回答问题。"""
            }
        ]
        
        # 打印完整的提示词到日志
        logger.info(f"构建的提示词（Prompt）: {json.dumps(messages, indent=2, ensure_ascii=False)}")

        # 调用Qwen API
        logger.info(f"正在调用Qwen API ({CONFIG['qwen_model']})...")
        response = client.chat.completions.create(
            model=CONFIG["qwen_model"],
            messages=messages,
            max_tokens=CONFIG["max_tokens"],
            temperature=CONFIG["temperature"]
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info("Qwen API调用成功")
        
        return answer if answer else "抱歉，我无法基于提供的上下文生成合适的答案。"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Qwen API调用失败: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        logger.error(f"错误详情: {error_msg}")
        
        # 检查是否是配额不足的错误
        if "429" in error_msg or "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            logger.warning("Qwen API配额不足，请检查您的计费详情")
            quota_msg = "\n⚠️  Qwen API配额不足提示：\n" + \
                       "您的Qwen API配额已用完。请访问阿里云控制台检查您的计费详情。\n" + \
                       "现在将使用简单的基于检索的答案生成。\n"
            simple_answer = generate_simple_answer(query, context)
            return quota_msg + simple_answer
        
        logger.info("回退到简单答案生成...")
        return generate_simple_answer(query, context)


def generate_simple_answer(query: str, context: str) -> str:
    """
    当LLM不可用时的简单答案生成备用方案。
    
    Args:
        query: 用户查询
        context: 检索到的上下文
        
    Returns:
        基于检索结果的简单答案
    """
    # 简单的基于关键词匹配的答案生成
    lines = context.split('\n')
    relevant_lines = []
    
    query_words = query.lower().split()
    
    for line in lines:
        if any(word in line.lower() for word in query_words):
            relevant_lines.append(line.strip())
    
    if relevant_lines:
        answer = "基于检索到的代码，以下是相关信息：\n\n"
        answer += "\n".join(relevant_lines[:5])  # 限制显示前5行
        if len(relevant_lines) > 5:
            answer += "\n\n...（还有更多相关内容）"
    else:
        answer = "抱歉，我在检索到的代码中没有找到与您问题直接相关的信息。\n\n检索到的部分内容：\n\n"
        answer += context[:500] + "..." if len(context) > 500 else context
    
    return answer


@app.command("test", help="运行完整的端到端测试")
def run_test(
    package_name: str = typer.Option("typer", "--package", "-p", help="用于测试的包名"),
    query: str = typer.Option("如何创建一个CLI应用程序？", "--query", "-q", help="用于测试的查询")
) -> None:
    """执行完整的端到端测试。"""
    console.print(Panel(f"[bold green]开始端到端测试[/bold green]", title="测试流程", expand=False))
    
    # 1. 强制重建索引
    test_index_name = "test_index"
    console.print(f"[cyan]步骤 1: 强制重建索引 '{test_index_name}' (使用包: {package_name})[/cyan]")
    build_index(package_name, test_index_name, force_rebuild=True)
    
    # 2. 使用测试查询提问
    console.print(f"[cyan]步骤 2: 使用测试查询提问[/cyan]")
    console.print(f"[yellow]查询:[/yellow] {query}")
    ask_question(query, test_index_name)
    
    console.print(Panel(f"[bold green]端到端测试完成[/bold green]", title="测试流程", expand=False))

@app.command("build", help="构建指定Python库的知识库索引")
def build_index(
    package_name: str = typer.Argument(..., help="要索引的Python包的名称"),
    index_name: str = typer.Option("default", "--index", "-i", help="要创建的索引的名称"),
    force_rebuild: bool = typer.Option(False, "--force", "-f", help="强制重新构建索引"),
    package_path: Optional[str] = typer.Option(None, "--path", "-p", help="手动指定包的物理路径")
) -> None:
    """
    构建指定Python库的知识库索引。
    
    该命令会：
    1. 检查并获取指定包的所有源文件
    2. 使用AST解析器对源代码进行结构化分块
    3. 创建向量索引并保存到本地
    
    Args:
        package_name: Python包名称（如 'numpy', 'pandas' 等）
        force: 是否强制重新构建索引
    """
    ensure_directories()

    # 使用包名作为代码库路径
    repo_path = package_name
    index_dir = Path(CONFIG["index_dir"])

    console.print(Panel.fit(
        f"[bold blue]开始为 '{repo_path}' 构建知识库索引 (名称: {index_name})[/bold blue]",
        border_style="blue"
    ))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task_build_index = progress.add_task("正在创建并持久化向量索引...", total=None)
            logger.info(f"开始为代码库 '{repo_path}' 构建索引 '{index_name}'")

            indexer = FixedIndexer(index_dir=str(index_dir))

            # 调用 FixedIndexer 的 rebuild_index 方法
            success = indexer.rebuild_index(
                repo_path, 
                index_name=index_name, 
                embedding_model=CONFIG["embedding_model"],
                package_path=package_path
            )

            if not success:
                console.print("[red]错误：构建索引失败。[/red]")
                raise typer.Exit(code=1)

            progress.update(task_build_index, description="索引创建完成")
            final_index_path = index_dir / index_name
            logger.info(f"索引已保存到 {final_index_path}")

        console.print(Panel.fit(
            f"[bold green]✅ 成功为 '{repo_path}' 构建知识库索引！[/bold green]\n"
            f"📁 索引名称: {index_name}\n"
            f"📍 存储位置: {final_index_path}",
            border_style="green",
            title="构建完成"
        ))
        
    except Exception as e:
        console.print(f"[red]构建过程中发生错误: {e}[/red]")
        logger.error(f"构建索引失败: {e}", exc_info=True)
        raise e


@app.command("ask", help="基于构建的知识库回答用户问题")
def ask_question(
    query: str = typer.Argument(..., help="您要提出的问题"),
    index_name: str = typer.Option("default", "--index", "-i", help="要使用的索引名称")
) -> None:
    """处理用户查询并打印答案。"""
    ensure_directories()
    index_path = Path(CONFIG["index_dir"]) / index_name

    if not index_path.exists():
        console.print(f"[red]错误：索引 '{index_name}' 不存在。请先使用 'build' 命令构建它。[/red]")
        raise typer.Exit(code=1)

    console.print(Panel.fit(
        f"[bold blue]正在使用索引 '{index_name}' 回答问题[/bold blue]",
        border_style="blue"
    ))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("正在生成答案...", total=None)
            answer = process_query(query, str(index_path))
            progress.update(task, description="答案生成完成")
        
        console.print(Markdown(f"**你问**：{query}\n\n**回答**：{answer}"))
    except Exception as e:
        console.print(f"[red]处理查询时发生错误: {e}[/red]")
        logger.error(f"查询处理失败: {e}", exc_info=True)

if __name__ == "__main__":
    app()