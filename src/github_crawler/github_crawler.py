#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub仓库爬虫模块

本模块提供GitHub仓库的自动克隆、代码分析和索引构建功能。
支持公开仓库的自动下载和本地缓存管理。
"""

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
from loguru import logger

try:
    import git
    from git import Repo
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    logger.warning("GitPython不可用，将使用subprocess调用git命令")

import subprocess
import time
import requests
from ..chunker.multi_language_chunker import MultiLanguageChunker, CodeChunk


class GitHubCrawler:
    """
    GitHub仓库爬虫
    
    提供GitHub仓库的自动克隆、代码分析和本地缓存管理功能。
    支持多种GitHub URL格式的解析和处理。
    
    Attributes:
        cache_dir (Path): 本地缓存目录
        chunker (MultiLanguageChunker): 多语言代码分块器
        max_repo_size (int): 最大仓库大小限制（MB）
        timeout (int): 网络请求超时时间（秒）
    """
    
    def __init__(self, cache_dir: str = "./github_cache", max_repo_size: int = 500, timeout: int = 300, max_retries: int = 3, retry_delay: int = 5):
        """
        初始化GitHub爬虫
        
        Args:
            cache_dir (str): 本地缓存目录路径
            max_repo_size (int): 最大仓库大小限制（MB），默认500MB
            timeout (int): 网络请求超时时间（秒），默认300秒
            max_retries (int): 克隆失败时的最大重试次数
            retry_delay (int): 每次重试之间的延迟（秒）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.chunker = MultiLanguageChunker()
        self.max_repo_size = max_repo_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._last_progress_update = 0
        
        logger.info(f"GitHub爬虫初始化完成，缓存目录: {self.cache_dir}")
        
        # 检查git可用性
        if not self._check_git_available():
            raise RuntimeError("Git不可用，请确保已安装Git并添加到PATH环境变量")
    
    def _check_git_available(self) -> bool:
        """检查GitPython是否可用"""
        if not GIT_AVAILABLE:
            logger.error("GitPython未安装，无法执行Git操作。请运行 'pip install GitPython' 进行安装。")
            return False
        return True
    
    def parse_github_url(self, url: str) -> Optional[Dict[str, str]]:
        """
        解析GitHub URL，提取仓库信息
        
        支持的URL格式：
        - https://github.com/owner/repo
        - https://github.com/owner/repo.git
        - git@github.com:owner/repo.git
        - https://github.com/owner/repo/tree/branch
        
        Args:
            url (str): GitHub仓库URL
            
        Returns:
            Optional[Dict[str, str]]: 包含owner、repo、branch等信息的字典，解析失败返回None
        """
        logger.debug(f"解析GitHub URL: {url}")
        
        # 清理URL
        url = url.strip()
        
        # 处理SSH格式
        if url.startswith('git@github.com:'):
            # git@github.com:owner/repo.git
            match = re.match(r'git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/?$', url)
            if match:
                owner, repo = match.groups()
                return {
                    'owner': owner,
                    'repo': repo,
                    'branch': 'main',  # 默认分支
                    'clone_url': url,
                    'https_url': f'https://github.com/{owner}/{repo}'
                }
        
        # 处理HTTPS格式
        elif url.startswith('https://github.com/'):
            # 移除末尾的斜杠和.git
            url = url.rstrip('/').rstrip('.git')
            
            # 匹配不同的URL模式
            patterns = [
                # https://github.com/owner/repo/tree/branch
                r'https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)',
                # https://github.com/owner/repo
                r'https://github\.com/([^/]+)/([^/]+)/?$'
            ]
            
            for pattern in patterns:
                match = re.match(pattern, url)
                if match:
                    groups = match.groups()
                    owner, repo = groups[0], groups[1]
                    branch = groups[2] if len(groups) > 2 else 'main'
                    
                    return {
                        'owner': owner,
                        'repo': repo,
                        'branch': branch,
                        'clone_url': f'https://github.com/{owner}/{repo}.git',
                        'https_url': f'https://github.com/{owner}/{repo}'
                    }
        
        logger.error(f"无法解析GitHub URL: {url}")
        return None
    
    def check_repo_size(self, repo_info: Dict[str, str]) -> bool:
        """
        检查仓库大小是否在限制范围内
        
        Args:
            repo_info (Dict[str, str]): 仓库信息
            
        Returns:
            bool: 仓库大小是否符合要求
        """
        try:
            api_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['repo']}"
            logger.debug(f"检查仓库大小: {api_url}")
            
            response = requests.get(api_url, timeout=30)
            if response.status_code == 200:
                repo_data = response.json()
                size_kb = repo_data.get('size', 0)
                size_mb = size_kb / 1024
                
                logger.info(f"仓库大小: {size_mb:.2f} MB")
                
                if size_mb > self.max_repo_size:
                    logger.warning(f"仓库大小 ({size_mb:.2f} MB) 超过限制 ({self.max_repo_size} MB)")
                    return False
                
                return True
            else:
                logger.warning(f"无法获取仓库信息，状态码: {response.status_code}")
                # 如果无法获取大小信息，允许继续（可能是私有仓库或API限制）
                return True
                
        except Exception as e:
            logger.warning(f"检查仓库大小时出错: {e}")
            # 出错时允许继续
            return True
    
    def clone_repository(self, github_url: str, force_update: bool = False) -> Optional[Path]:
        """
        克隆GitHub仓库到本地
        
        Args:
            github_url (str): GitHub仓库URL
            force_update (bool): 是否强制更新已存在的仓库
            
        Returns:
            Optional[Path]: 本地仓库路径，克隆失败返回None
        """
        repo_info = self.parse_github_url(github_url)
        if not repo_info:
            logger.error(f"无法解析GitHub URL: {github_url}")
            return None

        repo_name = f"{repo_info['owner']}_{repo_info['repo']}"
        local_path = self.cache_dir / repo_name
        
        logger.info(f"准备克隆仓库: {repo_info['https_url']}")
        
        # 检查本地是否已存在
        if local_path.exists():
            if force_update:
                logger.info(f"强制更新，删除现有仓库: {local_path}")
                shutil.rmtree(local_path)
            else:
                logger.info(f"仓库已存在，使用缓存: {local_path}")
                return local_path
        
        if not self._check_git_available():
            return None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"尝试第 {attempt + 1}/{self.max_retries} 次克隆...")
                # 尝试克隆指定分支，如果失败则尝试 'main' 和 'master'
                branches_to_try = [repo_info['branch'], 'main', 'master']
                cloned = False
                for branch in branches_to_try:
                    try:
                        logger.info(f"尝试克隆分支: {branch}")
                        Repo.clone_from(
                            url=repo_info['clone_url'],
                            to_path=local_path,
                            branch=branch,
                            depth=1,
                            progress=self._clone_progress
                        )
                        cloned = True
                        break
                    except git.exc.GitCommandError as branch_e:
                        logger.warning(f"克隆分支 '{branch}' 失败: {branch_e}")
                        if local_path.exists():
                            shutil.rmtree(local_path)
                        # 如果是最后一个分支，则抛出错误
                        if branch == branches_to_try[-1]:
                            raise branch_e
                        # 否则，继续尝试下一个分支
                        time.sleep(1) # 短暂等待，避免连续失败
                
                if not cloned:
                    raise git.exc.GitCommandError("所有尝试的分支克隆均失败", "clone")
                logger.info(f"仓库克隆成功: {local_path}")
                return local_path
            except git.exc.GitCommandError as e:
                logger.error(f"克隆失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if local_path.exists():
                    shutil.rmtree(local_path)
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("已达到最大重试次数，克隆失败。")
                    return None
            except Exception as e:
                logger.error(f"克隆过程中发生未知错误: {e}")
            if local_path.exists():
                shutil.rmtree(local_path)
            return None
        return None
    
    def _clone_progress(self, op_code, cur_count, max_count=None, message=''):
        """克隆进度的回调函数"""
        current_time = time.time()
        if current_time - self._last_progress_update < 1.0 and cur_count != max_count:
            return

        self._last_progress_update = current_time
        
        if max_count:
            progress = cur_count / max_count * 100
            print(f'克隆进度: {progress:.2f}% ({cur_count}/{max_count}) {message}', end='\r')
        else:
            print(f'克隆进度: {cur_count} {message}', end='\r')
        
        if cur_count == max_count:
            print()
    
    def analyze_repository(self, repo_path: Path) -> List[CodeChunk]:
        """
        分析仓库中的所有代码文件
        
        Args:
            repo_path (Path): 本地仓库路径
            
        Returns:
            List[CodeChunk]: 代码块列表
        """
        logger.info(f"开始分析仓库: {repo_path}")
        
        all_chunks = []
        
        # 定义要忽略的目录和文件
        ignore_dirs = {
            '.git', '.github', '__pycache__', '.pytest_cache',
            'node_modules', '.vscode', '.idea', 'venv', 'env',
            '.conda', 'dist', 'build', '.tox', '.coverage',
            'htmlcov', '.mypy_cache', '.ruff_cache'
        }
        
        ignore_files = {
            '.gitignore', '.gitattributes', 'LICENSE', 'CHANGELOG',
            '.DS_Store', 'Thumbs.db'
        }
        
        # 递归遍历仓库目录
        for root, dirs, files in os.walk(repo_path):
            # 过滤要忽略的目录
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                file_path = Path(root) / file
                
                # 跳过要忽略的文件
                if file in ignore_files:
                    continue
                
                # 检查文件是否支持分块
                if self.chunker.is_supported_file(str(file_path)):
                    try:
                        chunks = self.chunker.chunk_file(str(file_path))
                        all_chunks.extend(chunks)
                        logger.debug(f"分析文件: {file_path.relative_to(repo_path)}, 生成 {len(chunks)} 个代码块")
                    except Exception as e:
                        logger.warning(f"分析文件失败 {file_path}: {e}")
                        continue
        
        logger.info(f"仓库分析完成，共生成 {len(all_chunks)} 个代码块")
        return all_chunks
    
    def crawl_and_analyze(self, github_url: str, force_update: bool = False) -> Optional[Dict[str, Any]]:
        """
        完整的GitHub仓库爬取和分析流程
        
        Args:
            github_url (str): GitHub仓库URL
            force_update (bool): 是否强制更新已存在的仓库
            
        Returns:
            Optional[Dict[str, Any]]: 包含仓库信息和代码块的字典，失败返回None
        """
        logger.info(f"开始GitHub仓库爬取和分析: {github_url}")
        
        # 1. 解析URL
        repo_info = self.parse_github_url(github_url)
        if not repo_info:
            logger.error("GitHub URL解析失败")
            return None
        
        logger.info(f"仓库信息: {repo_info['owner']}/{repo_info['repo']} (分支: {repo_info['branch']})")
        
        # 2. 检查仓库大小
        if not self.check_repo_size(repo_info):
            logger.error("仓库大小超过限制")
            return None
        
        # 3. 克隆仓库
        repo_path = self.clone_repository(github_url, force_update)
        if not repo_path:
            logger.error("仓库克隆失败")
            return None
        
        # 4. 分析代码
        chunks = self.analyze_repository(repo_path)
        if not chunks:
            logger.warning("未找到可分析的代码文件")
            return None
        
        # 5. 返回结果
        result = {
            'repo_info': repo_info,
            'local_path': str(repo_path),
            'chunks': chunks,
            'total_chunks': len(chunks),
            'supported_files': len([c for c in chunks if c.metadata.get('chunk_method')]),
            'summary': self.chunker.get_chunk_summary(chunks)
        }
        
        logger.info(f"GitHub仓库爬取和分析完成: {result['total_chunks']} 个代码块")
        return result
    
    def cleanup_cache(self, keep_recent: int = 5) -> None:
        """
        清理缓存目录，保留最近的仓库
        
        Args:
            keep_recent (int): 保留最近的仓库数量
        """
        logger.info(f"开始清理缓存，保留最近 {keep_recent} 个仓库")
        
        if not self.cache_dir.exists():
            return
        
        # 获取所有仓库目录，按修改时间排序
        repo_dirs = []
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                repo_dirs.append((item, item.stat().st_mtime))
        
        # 按修改时间降序排序
        repo_dirs.sort(key=lambda x: x[1], reverse=True)
        
        # 删除多余的仓库
        for repo_dir, _ in repo_dirs[keep_recent:]:
            try:
                shutil.rmtree(repo_dir)
                logger.info(f"删除缓存仓库: {repo_dir.name}")
            except Exception as e:
                logger.warning(f"删除缓存仓库失败 {repo_dir}: {e}")
        
        logger.info("缓存清理完成")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            Dict[str, Any]: 缓存统计信息
        """
        if not self.cache_dir.exists():
            return {'total_repos': 0, 'total_size_mb': 0, 'repos': []}
        
        repos = []
        total_size = 0
        
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                try:
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    total_size += size
                    
                    repos.append({
                        'name': item.name,
                        'path': str(item),
                        'size_mb': size / (1024 * 1024),
                        'modified_time': item.stat().st_mtime
                    })
                except Exception as e:
                    logger.warning(f"获取仓库信息失败 {item}: {e}")
        
        return {
            'total_repos': len(repos),
            'total_size_mb': total_size / (1024 * 1024),
            'repos': sorted(repos, key=lambda x: x['modified_time'], reverse=True)
        }