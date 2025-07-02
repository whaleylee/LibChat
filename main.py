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

# 解决Windows上rich库的UnicodeEncodeError
# 在Windows上，通过设置环境变量强制使用UTF-8编码
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
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
from src.indexing.indexer import save_knowledge_graph, load_knowledge_graph
from src.reranking.reranker import SentenceTransformerReranker
from src.retrieval.graph_retriever import GraphRAGRetriever
from src.github_crawler.github_crawler import GitHubCrawler

# 加载环境变量
load_dotenv()

# 创建CLI应用和控制台
app = typer.Typer(
name="libchat",
    help="本地Python库智能问答系统 - 基于RAG技术的代码问答助手",
    add_completion=False
)


# 配置常量
CONFIG = {
    "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    "reranker_model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large"), 
    "qwen_model": "qwen-turbo",
    "qwen_api_key": os.getenv("QWEN_API_KEY"),
    "qwen_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "index_dir": "./indexes",
     "log_dir": "./logs",
     "temp_dir": "./temp",
     "upload_dir": "./uploads",
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
    Path(CONFIG["index_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["temp_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["upload_dir"]).mkdir(parents=True, exist_ok=True)


def process_query(query: str, index_path: str) -> str:
    """
    处理用户查询，执行完整的GraphRAG流程。
    
    Args:
        query: 用户查询
        index_path: 索引路径
        
    Returns:
        生成的答案
    """
    try:
        # 1. 加载向量索引
        logger.info(f"正在加载向量索引: {index_path}")
        indexer = FixedIndexer(index_dir=CONFIG["index_dir"])
        index_name = Path(index_path).name
        index = indexer.load_index(index_name, embedding_model=CONFIG["embedding_model"])
        
        if index is None:
            logger.error(f"无法加载索引: {index_name}")
            return f"抱歉，无法加载索引 '{index_name}'。请确保索引已正确构建。"
        
        # 2. 加载知识图谱
        graph_path = str(Path(index_path) / f"knowledge_graph_{index_name}.gpickle")
        logger.info(f"正在加载知识图谱: {graph_path}")
        
        try:
            knowledge_graph = load_knowledge_graph(graph_path)
        except Exception as e:
            logger.warning(f"加载知识图谱失败: {e}，回退到基础向量检索")
            # 回退到原始的向量检索流程
            return _fallback_vector_retrieval(query, index)
        
        # 3. 初始化所有组件
        logger.info("正在初始化检索组件...")
        
        # 基础向量检索器
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=CONFIG["top_k_retrieval"]
        )
        
        # 重排序器
        reranker = SentenceTransformerReranker(
            model_name=CONFIG["reranker_model"],
            top_n=CONFIG["top_k_rerank"]
        )
        
        # GraphRAG检索器
        expansion_depth = CONFIG.get("expansion_depth", 1)
        graph_retriever = GraphRAGRetriever(
            vector_retriever=vector_retriever,
            knowledge_graph=knowledge_graph,
            reranker=reranker,
            expansion_depth=expansion_depth
        )
        
        # 4. 执行GraphRAG检索
        logger.info(f"正在执行GraphRAG检索: {query}")
        retrieved_nodes = graph_retriever.retrieve(query)
        
        if not retrieved_nodes:
            logger.warning("GraphRAG检索未返回结果，回退到基础向量检索")
            return _fallback_vector_retrieval(query, index)
        
        logger.info(f"GraphRAG检索完成，获得 {len(retrieved_nodes)} 个高质量结果")
        
        # 5. 构建上下文
        context_texts = []
        for node in retrieved_nodes:
            if hasattr(node.node, 'text'):
                context_texts.append(node.node.text)
            elif hasattr(node, 'text'):
                context_texts.append(node.text)
        
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
    console = Console()
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

@app.command("build-github", help="从GitHub仓库构建知识库索引")
def build_github_index(
    github_url: str = typer.Argument(..., help="GitHub仓库URL"),
    index_name: str = typer.Option(None, "--index", "-i", help="要创建的索引的名称（默认使用仓库名）"),
    force_rebuild: bool = typer.Option(False, "--force", "-f", help="强制重新构建索引"),
    max_size_mb: int = typer.Option(100, "--max-size", "-s", help="仓库最大大小限制（MB）")
) -> None:
    """
    从GitHub仓库构建知识库索引。
    
    该命令会：
    1. 克隆指定的GitHub仓库
    2. 分析仓库中的所有代码文件
    3. 使用多语言分块器对代码进行结构化分块
    4. 创建向量索引并保存到本地
    
    Args:
        github_url: GitHub仓库URL
        index_name: 索引名称（可选，默认使用仓库名）
        force_rebuild: 是否强制重新构建索引
        max_size_mb: 仓库大小限制（MB）
    """
    console = Console()
    ensure_directories()

    try:
        # 初始化GitHub爬虫
        crawler = GitHubCrawler(max_repo_size=max_size_mb)
        
        console.print(Panel.fit(
            f"[bold blue]开始从GitHub仓库构建知识库索引[/bold blue]\n"
            f"🔗 仓库URL: {github_url}",
            border_style="blue"
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # 克隆仓库
            task_clone = progress.add_task("正在克隆GitHub仓库...", total=None)
            repo_info = crawler.clone_repository(github_url)
            
            if not repo_info:
                console.print("[red]错误：克隆仓库失败[/red]")
                raise typer.Exit(code=1)
            
            # 如果没有指定索引名称，使用仓库名
            if not index_name:
                index_name = repo_info['name']
            
            progress.update(task_clone, description=f"仓库克隆完成 - {repo_info['name']}")
            
            # 分析代码文件
            task_analyze = progress.add_task("正在分析代码文件...", total=None)
            analysis_result = crawler.analyze_repository(repo_info['local_path'])
            
            if not analysis_result or not analysis_result['chunks']:
                console.print("[red]错误：未找到可分析的代码文件[/red]")
                raise typer.Exit(code=1)
            
            progress.update(task_analyze, description=f"代码分析完成 - 找到 {len(analysis_result['chunks'])} 个代码块")
            
            # 构建向量索引
            task_build = progress.add_task("正在构建向量索引...", total=None)
            
            index_dir = Path(CONFIG["index_dir"])
            indexer = FixedIndexer(index_dir=str(index_dir))
            
            # 使用分析结果构建索引
            success = indexer.build_index_from_chunks(
                chunks=analysis_result['chunks'],
                index_name=index_name,
                embedding_model=CONFIG["embedding_model"],
                metadata={
                    'source_type': 'github',
                    'repository_url': github_url,
                    'repository_name': repo_info['name'],
                    'total_files': analysis_result['total_files'],
                    'supported_files': analysis_result['supported_files']
                }
            )
            
            if not success:
                console.print("[red]错误：构建索引失败[/red]")
                raise typer.Exit(code=1)
            
            progress.update(task_build, description="向量索引构建完成")
            
            # 构建知识图谱（如果有Python文件）
            python_files = {f: content for f, content in analysis_result.get('file_contents', {}).items() 
                          if f.endswith('.py')}
            
            if python_files:
                task_kg = progress.add_task("正在构建知识图谱...", total=None)
                
                ast_chunker = ASTChunker()
                knowledge_graph = ast_chunker.create_knowledge_graph(python_files)
                
                # 保存知识图谱
                final_index_path = index_dir / index_name
                final_index_path.mkdir(parents=True, exist_ok=True)
                graph_path = str(final_index_path / f"knowledge_graph_{index_name}.gpickle")
                save_knowledge_graph(knowledge_graph, graph_path)
                
                progress.update(task_kg, description=f"知识图谱构建完成 - 节点: {len(knowledge_graph.nodes())}, 边: {len(knowledge_graph.edges())}")

        console.print(Panel.fit(
            f"[bold green]✅ 成功从GitHub仓库构建知识库索引！[/bold green]\n"
            f"📁 索引名称: {index_name}\n"
            f"🔗 源仓库: {github_url}\n"
            f"📊 分析文件: {analysis_result['supported_files']}/{analysis_result['total_files']}\n"
            f"📍 存储位置: {index_dir / index_name}",
            border_style="green",
            title="构建完成"
        ))
        
    except Exception as e:
        console.print(f"[red]构建过程中发生错误: {e}[/red]")
        logger.error(f"GitHub索引构建失败: {e}", exc_info=True)
        raise e


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
    console = Console()
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
            # 构建知识图谱
            task_kg = progress.add_task("正在构建知识图谱...", total=None)
            logger.info(f"开始为代码库 '{repo_path}' 构建知识图谱")
            
            # 获取源文件
            inspector = PackageInspector(repo_path)
            source_files = inspector.get_source_files()
            
            if not source_files:
                console.print(f"[red]错误：未找到包 '{repo_path}' 的源文件[/red]")
                raise typer.Exit(code=1)
            
            # 构建知识图谱
            ast_chunker = ASTChunker()
            source_files_dict = {}
            
            for file_path in source_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        source_files_dict[str(file_path)] = file_content
                except Exception as e:
                    logger.warning(f"读取文件 {file_path} 失败: {e}")
                    continue
            
            # 创建知识图谱
            knowledge_graph = ast_chunker.create_knowledge_graph(source_files_dict)
            
            # 保存知识图谱
            final_index_path = index_dir / index_name
            final_index_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            graph_path = str(final_index_path / f"knowledge_graph_{index_name}.gpickle")
            save_knowledge_graph(knowledge_graph, graph_path)
            
            progress.update(task_kg, description=f"知识图谱构建完成 - 节点: {len(knowledge_graph.nodes())}, 边: {len(knowledge_graph.edges())}")
            logger.info(f"知识图谱构建完成 - 节点数: {len(knowledge_graph.nodes())}, 边数: {len(knowledge_graph.edges())}")
            
            # 构建向量索引
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


def _fallback_vector_retrieval(query: str, index) -> str:
    """
    当GraphRAG检索失败时的回退方案，使用基础向量检索
    
    Args:
        query: 用户查询
        index: 向量索引
        
    Returns:
        生成的答案
    """
    logger.info("执行回退向量检索流程")
    
    try:
        # 基础向量检索
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=CONFIG["top_k_retrieval"]
        )
        retrieved_nodes = retriever.retrieve(query)
        
        if not retrieved_nodes:
            return "抱歉，我没有找到与您的问题相关的信息。"
        
        # 重排序（如果有足够的文档）
        if len(retrieved_nodes) > CONFIG["top_k_rerank"]:
            reranker = SentenceTransformerReranker(
                model_name=CONFIG["reranker_model"],
                top_n=CONFIG["top_k_rerank"]
            )
            
            documents = [node.text for node in retrieved_nodes]
            reranked_docs = reranker.rerank(query, documents)
            context_texts = reranked_docs
        else:
            context_texts = [node.text for node in retrieved_nodes]
        
        context = "\n\n".join(context_texts)
        
        # 生成答案
        answer = generate_answer_with_llm(query, context)
        return answer
        
    except Exception as e:
        logger.error(f"回退检索也失败: {e}")
        return f"抱歉，检索过程中发生错误: {str(e)}"


@app.command("ask", help="基于构建的知识库回答用户问题")
def ask_question(
    query: str = typer.Argument(..., help="您要提出的问题"),
    index_name: str = typer.Option("default", "--index", "-i", help="要使用的索引名称"),
    expansion_depth: int = typer.Option(1, "--depth", "-d", help="图遍历扩展深度")
) -> None:
    """
    处理用户查询并打印答案，使用GraphRAG增强检索。
    
    Args:
        query: 用户问题
        index_name: 要使用的索引名称
        expansion_depth: 知识图谱遍历的扩展深度
    """
    console = Console()
    ensure_directories()
    index_path = Path(CONFIG["index_dir"]) / index_name
    graph_path = index_path / f"knowledge_graph_{index_name}.gpickle"

    # 检查索引是否存在
    if not index_path.exists():
        console.print(f"[red]错误：索引 '{index_name}' 不存在。请先使用 'build' 命令构建它。[/red]")
        raise typer.Exit(code=1)
    
    # 检查知识图谱是否存在
    if not Path(graph_path).exists():
        console.print(f"[yellow]警告：知识图谱文件不存在，将使用基础向量检索。[/yellow]")
        console.print(f"[yellow]建议重新运行 'build' 命令以生成知识图谱。[/yellow]")

    console.print(Panel.fit(
        f"[bold blue]正在使用GraphRAG增强检索回答问题[/bold blue]\n"
        f"📊 索引: {index_name}\n"
        f"🕸️ 图遍历深度: {expansion_depth}",
        border_style="blue",
        title="GraphRAG检索"
    ))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task_load = progress.add_task("正在加载索引和知识图谱...", total=None)
            
            # 临时更新配置中的扩展深度
            original_depth = CONFIG.get("expansion_depth", 1)
            CONFIG["expansion_depth"] = expansion_depth
            
            try:
                answer = process_query(query, str(index_path))
                progress.update(task_load, description="GraphRAG检索完成")
            finally:
                # 恢复原始配置
                CONFIG["expansion_depth"] = original_depth
        
        # 显示结果
        console.print("\n" + "="*60)
        console.print(Markdown(f"**🤔 你问**：{query}\n\n**🤖 回答**：{answer}"))
        console.print("="*60)
        
    except Exception as e:
        console.print(f"[red]处理查询时发生错误: {e}[/red]")
        logger.error(f"查询处理失败: {e}", exc_info=True)

if __name__ == "__main__":
    app()