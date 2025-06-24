#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储文件编码问题诊断和修复工具

问题分析：
1. default__vector_store.json 文件被写入了二进制数据而不是 JSON 文本
2. 错误 'utf-8' codec can't decode byte 0x80 表明文件包含非文本数据
3. 这可能是 llama-index 在保存向量数据时出现的序列化问题
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('vector_store_fix.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VectorStoreEncodingFixer:
    """向量存储编码问题修复器"""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.backup_dir = self.index_path / "backup"
        
    def diagnose_file(self, file_path: Path) -> Dict[str, Any]:
        """诊断文件问题"""
        diagnosis = {
            "file_exists": file_path.exists(),
            "file_size": 0,
            "is_binary": False,
            "is_empty": False,
            "is_valid_json": False,
            "encoding_error": None,
            "json_error": None,
            "first_bytes": None
        }
        
        if not file_path.exists():
            return diagnosis
            
        try:
            diagnosis["file_size"] = file_path.stat().st_size
            diagnosis["is_empty"] = diagnosis["file_size"] == 0
            
            # 读取前20个字节来判断是否为二进制
            with open(file_path, 'rb') as f:
                first_bytes = f.read(20)
                diagnosis["first_bytes"] = first_bytes.hex() if first_bytes else None
                
                # 检查是否包含非文本字符
                try:
                    first_bytes.decode('utf-8')
                    diagnosis["is_binary"] = False
                except UnicodeDecodeError:
                    diagnosis["is_binary"] = True
                    
        except Exception as e:
            logger.error(f"读取文件时出错: {e}")
            return diagnosis
            
        # 尝试作为 JSON 读取
        if not diagnosis["is_binary"] and not diagnosis["is_empty"]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                diagnosis["is_valid_json"] = True
            except UnicodeDecodeError as e:
                diagnosis["encoding_error"] = str(e)
            except json.JSONDecodeError as e:
                diagnosis["json_error"] = str(e)
                
        return diagnosis
        
    def create_backup(self, file_path: Path) -> Optional[Path]:
        """创建文件备份"""
        if not file_path.exists():
            return None
            
        self.backup_dir.mkdir(exist_ok=True)
        backup_path = self.backup_dir / f"{file_path.name}.backup"
        
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"已创建备份: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return None
            
    def fix_binary_file(self, file_path: Path) -> bool:
        """修复二进制文件问题"""
        logger.warning(f"检测到 {file_path.name} 包含二进制数据，这通常表明 llama-index 序列化出现问题")
        
        # 创建备份
        backup_path = self.create_backup(file_path)
        if not backup_path:
            return False
            
        try:
            # 删除损坏的文件
            file_path.unlink()
            logger.info(f"已删除损坏的文件: {file_path}")
            
            # 创建一个空的有效 JSON 文件作为占位符
            empty_vector_store = {
                "__type__": "faiss",
                "__version__": "0.1.0",
                "data": {},
                "metadata": {
                    "created_at": None,
                    "dimension": None,
                    "total_vectors": 0
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(empty_vector_store, f, ensure_ascii=False, indent=2)
                
            logger.info(f"已创建新的空向量存储文件: {file_path}")
            logger.warning("注意: 需要重新创建索引以恢复向量数据")
            return True
            
        except Exception as e:
            logger.error(f"修复文件失败: {e}")
            # 尝试恢复备份
            if backup_path and backup_path.exists():
                try:
                    shutil.copy2(backup_path, file_path)
                    logger.info("已恢复原文件")
                except Exception as restore_e:
                    logger.error(f"恢复文件失败: {restore_e}")
            return False
            
    def fix_encoding_issues(self) -> Dict[str, bool]:
        """修复所有编码问题"""
        results = {}
        
        json_files = [
            'default__vector_store.json',
            'docstore.json', 
            'index_store.json',
            'image__vector_store.json',
            'graph_store.json'
        ]
        
        for json_file in json_files:
            file_path = self.index_path / json_file
            results[json_file] = False
            
            if not file_path.exists():
                logger.info(f"{json_file} 不存在，跳过")
                results[json_file] = True
                continue
                
            # 诊断文件
            diagnosis = self.diagnose_file(file_path)
            logger.info(f"\n{json_file} 诊断结果:")
            for key, value in diagnosis.items():
                logger.info(f"  {key}: {value}")
                
            # 根据诊断结果进行修复
            if diagnosis["is_empty"]:
                logger.warning(f"{json_file} 为空文件，删除")
                file_path.unlink()
                results[json_file] = True
                
            elif diagnosis["is_binary"]:
                results[json_file] = self.fix_binary_file(file_path)
                
            elif diagnosis["encoding_error"] or diagnosis["json_error"]:
                # 尝试原有的编码修复方法
                results[json_file] = self._try_encoding_fix(file_path)
                
            elif diagnosis["is_valid_json"]:
                logger.info(f"{json_file} 正常，无需修复")
                results[json_file] = True
                
        return results
        
    def _try_encoding_fix(self, file_path: Path) -> bool:
        """尝试原有的编码修复方法"""
        backup_path = self.create_backup(file_path)
        if not backup_path:
            return False
            
        try:
            # 尝试用 latin-1 读取然后重新保存为 UTF-8
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            
            # 尝试解析 JSON
            data = json.loads(content)
            
            # 重新保存为 UTF-8
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功修复 {file_path.name} 的编码问题")
            return True
            
        except Exception as e:
            logger.error(f"编码修复失败: {e}")
            # 恢复备份
            if backup_path.exists():
                try:
                    shutil.copy2(backup_path, file_path)
                    logger.info("已恢复原文件")
                except Exception as restore_e:
                    logger.error(f"恢复文件失败: {restore_e}")
            return False
            
    def validate_fix(self) -> bool:
        """验证修复结果"""
        json_files = [
            'default__vector_store.json',
            'docstore.json', 
            'index_store.json'
        ]
        
        all_valid = True
        for json_file in json_files:
            file_path = self.index_path / json_file
            if file_path.exists():
                diagnosis = self.diagnose_file(file_path)
                if not diagnosis["is_valid_json"] and not diagnosis["is_empty"]:
                    logger.error(f"{json_file} 仍然存在问题")
                    all_valid = False
                else:
                    logger.info(f"{json_file} 验证通过")
                    
        return all_valid

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="修复向量存储文件编码问题")
    parser.add_argument("index_path", help="索引目录路径")
    parser.add_argument("--diagnose-only", action="store_true", help="仅诊断，不修复")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.index_path):
        logger.error(f"索引目录不存在: {args.index_path}")
        return 1
        
    fixer = VectorStoreEncodingFixer(args.index_path)
    
    if args.diagnose_only:
        # 仅诊断
        json_files = ['default__vector_store.json', 'docstore.json', 'index_store.json']
        for json_file in json_files:
            file_path = Path(args.index_path) / json_file
            if file_path.exists():
                diagnosis = fixer.diagnose_file(file_path)
                print(f"\n{json_file} 诊断结果:")
                for key, value in diagnosis.items():
                    print(f"  {key}: {value}")
    else:
        # 执行修复
        logger.info(f"开始修复索引目录: {args.index_path}")
        results = fixer.fix_encoding_issues()
        
        logger.info("\n修复结果:")
        for file_name, success in results.items():
            status = "成功" if success else "失败"
            logger.info(f"  {file_name}: {status}")
            
        # 验证修复结果
        if fixer.validate_fix():
            logger.info("\n所有文件修复验证通过")
            return 0
        else:
            logger.error("\n部分文件修复失败")
            return 1

if __name__ == "__main__":
    exit(main())