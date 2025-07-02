#!/usr/bin/env python3
"""
LibChat Web应用启动脚本

这个脚本用于启动LibChat的Web界面，提供以下功能：
1. 智能问答界面
2. 知识图谱可视化
3. 索引管理
4. GraphRAG检索

使用方法:
    python run_web.py

然后在浏览器中访问: http://localhost:5000
"""

import os
import sys
import webbrowser
import time
from threading import Timer
from dotenv import load_dotenv

# 解决Windows下的编码问题
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 加载环境变量
load_dotenv()

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import flask
        print("✅ Flask 已安装")
    except ImportError:
        print("❌ Flask 未安装，请运行: pip install flask==2.3.3")
        return False
    
    try:
        from llama_index.core import VectorStoreIndex
        print("✅ LlamaIndex 已安装")
    except ImportError:
        print("❌ LlamaIndex 未安装，请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_api_key():
    """检查API密钥配置"""
    # 检查Qwen API密钥
    qwen_key = os.getenv('QWEN_API_KEY')
    if qwen_key:
        print(f"✅ Qwen API密钥已配置 (***{qwen_key[-4:]})")
        return True
    
    # 检查OpenAI API密钥（备用）
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key and openai_key != 'your-api-key-here':
        print(f"✅ OpenAI API密钥已配置 (***{openai_key[-4:]})")
        return True
    
    print("⚠️  警告: 未检测到有效的API密钥")
    print("   请设置您的API密钥:")
    print("   Qwen: set QWEN_API_KEY=your_qwen_api_key")
    print("   或 OpenAI: set OPENAI_API_KEY=your_openai_api_key")
    print("")
    return False

def check_indexes():
    """检查可用的索引"""
    indexes_dir = "indexes"
    if not os.path.exists(indexes_dir):
        print("⚠️  未找到indexes目录，请先构建一些索引")
        print("   示例: python main.py build typer --index my_typer_index")
        return []
    
    indexes = [d for d in os.listdir(indexes_dir) if os.path.isdir(os.path.join(indexes_dir, d))]
    if indexes:
        print(f"✅ 找到 {len(indexes)} 个可用索引: {', '.join(indexes)}")
    else:
        print("⚠️  indexes目录为空，请先构建一些索引")
        print("   示例: python main.py build typer --index my_typer_index")
    
    return indexes

def open_browser():
    """延迟打开浏览器"""
    time.sleep(2)  # 等待服务器启动
    webbrowser.open('http://localhost:5000')

from app import create_app

def main():
    print("🚀 启动 LibChat Web应用...")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请安装必要的依赖后重试")
        sys.exit(1)
    
    check_api_key()
    check_indexes()
    
    print("=" * 50)
    print("🌍 Web服务已准备就绪, 请在浏览器中打开 http://localhost:5000")
    


    print("🔧 正在创建Flask应用...")
    try:
        app = create_app()
        if app:
            print("✅ Flask应用创建成功")
        else:
            print("❌ Flask应用创建失败，无法启动服务器。")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 创建Flask应用时发生致命错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 启动浏览器
    Timer(2, open_browser).start()

    print("\n🌐 启动Web服务器...")
    print("📍 访问地址: http://localhost:5000")
    print("⏹️  按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    # 使用 waitress 启动生产环境服务器
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000, threads=8)
    except Exception as e:
        print(f"❌ 启动服务器时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()