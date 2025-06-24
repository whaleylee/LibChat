#!/usr/bin/env python3
"""
Package Inspector Module

本模块提供PackageInspector类，用于检查已安装的Python库，
找到其源文件位置并收集所有.py文件路径。
"""

import importlib
import inspect
import logging
import copy
from pathlib import Path
from typing import List, Optional
from loguru import logger


class PackageInspector:
    """
    Python包检查器
    
    用于检查已安装的Python库，找到其源文件位置并收集所有.py文件路径。
    支持单文件模块和包目录两种情况。
    """
    
    def __init__(self, package_name: str, package_path: Optional[str] = None) -> None:
        """
        初始化包检查器
        
        Args:
            package_name (str): 要检查的包名
            package_path (Optional[str]): 包的物理路径（可选）
        """
        self.package_name = package_name
        self.package_path = Path(package_path) if package_path else None
        logger.info(f"初始化PackageInspector，目标包: {package_name}, 指定路径: {self.package_path}")
    
    def _get_package_path(self) -> Optional[Path]:
        """
        获取包的根目录路径
        
        尝试导入指定的包并获取其源文件路径。
        处理单文件模块和包目录两种情况。
        
        Returns:
            Optional[Path]: 包的根目录路径，如果找不到则返回None
        """
        if self.package_path and self.package_path.exists():
            logger.info(f"使用提供的路径: {self.package_path}")
            return self.package_path

        try:
            logger.debug(f"尝试导入包: {self.package_name}")
            package = importlib.import_module(self.package_name)
            
            # 尝试获取包的源文件路径
            source_file = inspect.getsourcefile(package)
            if source_file:
                source_path = Path(source_file)
                logger.debug(f"通过inspect.getsourcefile获取到源文件: {source_path}")
                
                # 如果是__init__.py，我们找到了包目录
                if source_path.name == '__init__.py':
                    package_path = source_path.parent
                # 否则，它可能是一个单文件模块或包目录
                else:
                    package_path = source_path
                    
                logger.info(f"成功获取包路径: {package_path}")
                return package_path
            
            # 如果inspect.getsourcefile失败，尝试使用__path__属性
            if hasattr(package, '__path__'):
                # 对于包，__path__是一个列表
                package_paths = package.__path__
                if package_paths:
                    package_path = Path(package_paths[0])
                    logger.info(f"通过__path__属性获取包路径: {package_path}")
                    return package_path
            
            # 如果都失败了，尝试使用__file__属性
            if hasattr(package, '__file__') and package.__file__:
                file_path = Path(package.__file__)
                if file_path.name == '__init__.py':
                    package_path = file_path.parent
                else:
                    package_path = file_path.parent
                logger.info(f"通过__file__属性获取包路径: {package_path}")
                return package_path
                
            logger.warning(f"无法确定包 {self.package_name} 的源文件路径")
            return None
            
        except ImportError as e:
            logger.warning(f"无法直接导入包 {self.package_name}: {e}。尝试使用find_spec查找。")
            try:
                spec = importlib.util.find_spec(self.package_name)
                if spec and spec.origin:
                    source_path = Path(spec.origin)
                    if source_path.name == '__init__.py':
                        package_path = source_path.parent
                    else:
                        package_path = source_path
                    logger.info(f"通过find_spec成功获取包路径: {package_path}")
                    return package_path
                else:
                    logger.error(f"无法通过find_spec找到包 {self.package_name} 的规范或源")
                    return None
            except Exception as find_e:
                logger.error(f"使用find_spec查找包 {self.package_name} 时出错: {find_e}")
                return None
        except Exception as e:
            logger.error(f"获取包路径时发生未知错误: {e}")
            return None
    
    def get_source_files(self) -> List[Path]:
        """
        获取包中所有的Python源文件
        
        递归查找包目录下所有的.py文件，并返回其绝对路径列表。
        
        Returns:
            List[Path]: 包含所有.py文件绝对路径的列表
        """
        logger.info(f"开始收集包 {self.package_name} 的源文件")
        
        # 获取包的根目录路径
        package_path = self._get_package_path()
        if package_path is None:
            logger.warning(f"无法获取包 {self.package_name} 的路径，返回空列表")
            return []
        
        if not package_path.exists():
            logger.error(f"包路径不存在: {package_path}")
            return []
        
        source_files = []
        
        try:
            if package_path.is_file() and package_path.suffix == '.py':
                # 如果是单个Python文件
                source_files.append(package_path.resolve())
                logger.debug(f"添加单文件模块: {package_path}")
            elif package_path.is_dir():
                # 如果是目录，递归查找所有.py文件
                for py_file in package_path.rglob('*.py'):
                    if py_file.is_file():
                        source_files.append(py_file.resolve())
                        logger.debug(f"找到Python文件: {py_file}")
            
            logger.info(f"成功收集到 {len(source_files)} 个Python源文件")
            return source_files
            
        except Exception as e:
            logger.error(f"收集源文件时发生错误: {e}")
            return []