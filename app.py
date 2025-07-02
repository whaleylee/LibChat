from flask import Flask, render_template, request, jsonify
import os
import sys
import json
import subprocess
import importlib
import ast
from typing import Dict, List, Any
from pathlib import Path
import zipfile
from src.chunker.multi_language_chunker import MultiLanguageChunker
from src.chunker.ast_chunker import ASTChunker

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(f"当前工作目录: {os.getcwd()}")
print(f"app.py 绝对路径: {os.path.abspath(__file__)}")

# 直接导入main.py中已经测试完成的函数
from main import process_query, CONFIG, ensure_directories
from src.indexing.indexer import load_knowledge_graph
from src.indexing.fixed_indexer import FixedIndexer
from src.github_crawler.github_crawler import GitHubCrawler



# 确保必要的目录存在
ensure_directories()

class WebRAGSystem:
    def __init__(self):
        import time
        print("开始初始化 WebRAGSystem...")
        start_time = time.time()
        self.current_index = None
        self.current_graph = None
        self.indexer = FixedIndexer(index_dir=CONFIG["index_dir"])
        self.available_indexes = self._get_available_indexes()
        end_time = time.time()
        print(f"WebRAGSystem 初始化完成，耗时: {end_time - start_time:.2f} 秒")
    
    def _get_available_indexes(self) -> List[str]:
        """获取可用的索引列表"""
        import time
        print("开始获取可用索引列表...")
        start_time = time.time()
        indexes_dir = Path(CONFIG["index_dir"])
        if not indexes_dir.exists():
            print("索引目录不存在。")
            return []
        
        indexes = [d.name for d in indexes_dir.iterdir() if d.is_dir()]
        end_time = time.time()
        print(f"获取可用索引列表完成，找到 {len(indexes)} 个索引，耗时: {end_time - start_time:.2f} 秒")
        return indexes
    
    def load_index(self, index_name: str) -> bool:
        """检查指定的索引是否存在"""
        index_path = Path(CONFIG["index_dir"]) / index_name
        return index_path.exists()
    
    def query(self, question: str, index_name: str, expansion_depth: int = 2) -> Dict[str, Any]:
        """执行查询并返回结果"""
        try:
            # 检查索引是否存在
            if not self.load_index(index_name):
                return {"error": f"索引 '{index_name}' 不存在"}
            
            # 构建索引路径
            index_path = str(Path(CONFIG["index_dir"]) / index_name)
            
            # 临时更新配置中的扩展深度
            original_depth = CONFIG.get("expansion_depth", 1)
            CONFIG["expansion_depth"] = expansion_depth
            
            try:
                # 直接调用main.py中已经测试完成的process_query函数
                answer = process_query(question, index_path)
                
                # 获取图谱可视化数据
                graph_data = self._get_graph_visualization_data(index_name)
                
                # 确保返回正确的数据结构
                result = {
                    "answer": answer,
                    "graph_data": graph_data,
                    "sources": [],  # 可以后续扩展添加来源信息
                    "used_graph": graph_data is not None  # 添加图谱使用标志
                }
                
                print(f"查询结果: answer={len(answer) if answer else 0}字符, graph_data={graph_data is not None}")
                return result
                
            finally:
                 # 恢复原始配置
                 CONFIG["expansion_depth"] = original_depth
            
        except Exception as e:
            print(f"查询错误: {str(e)}")
            return {"error": f"查询失败: {str(e)}"}
    
    def _get_graph_visualization_data(self, index_name: str) -> Dict[str, Any]:
        """获取知识图谱可视化数据"""
        try:
            # 构建正确的图谱文件路径
            graph_path = Path(CONFIG["index_dir"]) / index_name / f"knowledge_graph_{index_name}.gpickle"

            if not graph_path.exists():
                print(f"未找到索引 {index_name} 的知识图谱文件")
                return None
            
            print(f"加载知识图谱: {graph_path}")
            knowledge_graph = load_knowledge_graph(graph_path)
            if not knowledge_graph:
                print(f"知识图谱加载失败: {graph_path}")
                return None
            
            # 构建简化的可视化数据（显示图谱的一个子集）
            nodes = []
            edges = []
            
            # 限制显示的节点数量以避免界面过于复杂
            node_count = 0
            max_nodes = 50
            
            for node_id, node_data in knowledge_graph.nodes(data=True):
                if node_count >= max_nodes:
                    break
                    
                # 使用os.path.basename来获取更简洁的标签
                label = os.path.basename(str(node_data.get('label', node_id)))
                nodes.append({
                    "id": str(node_id),
                    "label": label[:30],  # 截断以防标签过长
                    "full_label": label, # 添加完整标签用于工具提示
                    "type": node_data.get('type', 'unknown'),
                    "file_path": node_data.get('metadata', {}).get('file_path', ''),
                    "size": 15,
                    "color": self._get_node_color(node_data.get('type', 'unknown'))
                })
                node_count += 1
            
            # 添加对应的边
            node_ids = {node["id"] for node in nodes}
            for source, target, edge_data in knowledge_graph.edges(data=True):
                if str(source) in node_ids and str(target) in node_ids:
                    edges.append({
                        "source": str(source),
                        "target": str(target),
                        "label": edge_data.get('relation', '')
                    })
            
            return {
                "nodes": nodes,
                "edges": edges
            }
        except Exception as e:
            print(f"获取图谱可视化数据失败: {e}")
            return None
    
    def _get_node_color(self, node_type: str) -> str:
        """根据节点类型返回颜色"""
        color_map = {
            'function': '#FF6B6B',
            'class': '#4ECDC4',
            'variable': '#45B7D1',
            'import': '#96CEB4',
            'file': '#FFEAA7',
            'unknown': '#DDA0DD'
        }
        return color_map.get(node_type, '#DDA0DD')
    
    def check_library_installed(self, library_name: str) -> bool:
        """检查Python库是否已安装"""
        try:
            importlib.import_module(library_name)
            return True
        except ImportError:
            return False
    
    def install_library(self, library_name: str) -> Dict[str, Any]:
        """安装Python库"""
        try:
            # 使用pip安装库
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', library_name],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"成功安装 {library_name}",
                    "output": result.stdout
                }
            else:
                return {
                    "success": False,
                    "error": f"安装失败: {result.stderr}",
                    "output": result.stdout
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "安装超时（超过5分钟）"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"安装过程中出错: {str(e)}"
            }
    
    def build_github_index(self, github_url: str, index_name: str = None, max_size_mb: int = 100) -> Dict[str, Any]:
        """从GitHub仓库构建知识库索引"""
        try:
            # 初始化GitHub爬虫
            crawler = GitHubCrawler(max_repo_size=max_size_mb)
            
            # 2. 分析仓库
            analysis_result = crawler.crawl_and_analyze(github_url, force_update=True)
            
            if not analysis_result or not analysis_result.get('chunks'):
                return {
                    "success": False,
                    "error": "分析GitHub仓库失败或未找到可分析的代码文件"
                }

            repo_info = analysis_result['repo_info']

            # 如果没有提供索引名称，使用仓库名称
            if not index_name:
                index_name = f"{repo_info['repo']}_index"
            
            # 构建向量索引
            success = self.indexer.build_index_from_chunks(
                chunks=analysis_result['chunks'],
                index_name=index_name,
                embedding_model=CONFIG["embedding_model"],
                metadata={
                    'source_type': 'github',
                    'repository_url': github_url,
                    'repository_name': repo_info['repo'],
                    'total_chunks': analysis_result['total_chunks'],
                    'summary': json.dumps(analysis_result['summary'])
                }
            )
            
            if not success:
                return {
                    "success": False,
                    "error": "构建索引失败"
                }
            
            # 构建知识图谱（如果有Python文件）
            python_files = {f: content for f, content in analysis_result.get('file_contents', {}).items() 
                          if f.endswith('.py')}
            
            if python_files:
                from src.chunker.ast_chunker import ASTChunker
                from src.indexing.indexer import save_knowledge_graph
                
                ast_chunker = ASTChunker()
                knowledge_graph = ast_chunker.create_knowledge_graph(python_files)
                
                # 保存知识图谱
                index_dir = Path(CONFIG["index_dir"])
                final_index_path = index_dir / index_name
                final_index_path.mkdir(parents=True, exist_ok=True)
                graph_path = str(final_index_path / f"knowledge_graph_{index_name}.gpickle")
                save_knowledge_graph(knowledge_graph, graph_path)
            
            # 更新可用索引列表
            self.available_indexes = self._get_available_indexes()
            
            return {
                "success": True,
                "message": f"成功从GitHub仓库构建索引 '{index_name}'",
                "index_name": index_name,
                "repository_name": repo_info['repo'],
                "total_chunks": analysis_result['total_chunks']
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"构建GitHub索引失败: {str(e)}"
            }

    def build_library_index(self, library_name: str, index_name: str = None) -> Dict[str, Any]:
        """为Python库构建知识库索引"""
        try:
            # 验证库名称格式
            if not library_name or not library_name.replace('_', '').replace('-', '').isalnum():
                return {
                    "success": False,
                    "error": "库名称格式无效，只能包含字母、数字、下划线和连字符"
                }
            
            # 如果没有提供索引名称，使用库名称
            if not index_name:
                index_name = f"{library_name}_index"
            
            # 检查磁盘空间（简单检查）
            import shutil
            free_space = shutil.disk_usage('.').free
            if free_space < 1024 * 1024 * 100:  # 100MB
                return {
                    "success": False,
                    "error": "磁盘空间不足，至少需要100MB可用空间"
                }
            
            # 检查库是否已安装
            if not self.check_library_installed(library_name):
                # 尝试安装库
                install_result = self.install_library(library_name)
                if not install_result["success"]:
                    return install_result
            
            # 使用main.py中的构建功能
            # 这里需要调用类似于 python main.py build library_name --index index_name 的功能
            result = subprocess.run(
                [sys.executable, 'main.py', 'build', library_name, '--index', index_name],
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            if result.returncode == 0:
                # 更新可用索引列表
                self.available_indexes = self._get_available_indexes()
                return {
                    "success": True,
                    "message": f"成功为 {library_name} 构建索引 {index_name}",
                    "index_name": index_name,
                    "output": result.stdout
                }
            else:
                return {
                    "success": False,
                    "error": f"构建索引失败: {result.stderr}",
                    "output": result.stdout
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "构建索引超时（超过10分钟）"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"构建过程中出错: {str(e)}"
            }

    def _find_node_code(self, file_path: str, node_name: str, node_type: str) -> str:
        """使用AST从文件中查找特定节点（函数或类）的源代码"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if node_type == 'function' and isinstance(node, ast.FunctionDef) and node.name == node_name:
                    return ast.get_source_segment(source_code, node)
                elif node_type == 'class' and isinstance(node, ast.ClassDef) and node.name == node_name:
                    return ast.get_source_segment(source_code, node)
            
            return f"# 未在 {file_path} 中找到 {node_type} {node_name}"
        except Exception as e:
            return f"# 解析文件时出错: {e}"

    def get_node_detail(self, index_name: str, node_id: str) -> Dict[str, Any]:
        """获取节点的详细信息，例如函数源码"""
        try:
            # 加载知识图谱
            graph_path = Path(CONFIG["index_dir"]) / index_name / f"knowledge_graph_{index_name}.gpickle"
            if not graph_path.exists():
                return {"error": f"Knowledge graph for index '{index_name}' not found."}
            
            kg = load_knowledge_graph(str(graph_path))

            if node_id not in kg.nodes:
                return {"error": f"Node '{node_id}' not found in the knowledge graph."}

            node_data = kg.nodes[node_id]
            node_type = node_data.get("type", "unknown")

            response = {
                "id": node_id,
                "type": node_type,
                "label": node_data.get("label", node_id),
                "source_code": ""
            }

            if node_type in ['function', 'class']:
                file_path = node_data.get('metadata', {}).get('file_path')
                if file_path and Path(file_path).exists():
                    # 从 node_id 中更准确地提取名称
                    # 例如 '.../file.py::def my_func(...' -> 'my_func'
                    # 或者 '.../file.py::class MyClass:' -> 'MyClass'
                    try:
                        name_part = node_id.split('::')[-1]
                        if node_type == 'function':
                            # 'def my_func(...' -> 'my_func'
                            node_name = name_part.split('(')[0].replace('def ', '').strip()
                        else: # class
                            # 'class MyClass:' -> 'MyClass'
                            node_name = name_part.split(':')[0].replace('class ', '').strip()
                        
                        response["source_code"] = self._find_node_code(file_path, node_name, node_type)
                    except IndexError:
                        response["source_code"] = "# Failed to parse node name from ID."
                else:
                    response["source_code"] = "# Source file not found."
            
            return response
        except Exception as e:
            return {"error": f"Failed to get node details: {e}"}


# 使用应用工厂模式
def create_app():
    """创建并配置Flask应用实例"""
    print("开始创建Flask应用...")
    app = Flask(__name__)

    # 在应用上下文中实例化RAG系统
    with app.app_context():
        print("正在实例化WebRAGSystem...")
        try:
            app.rag_system = WebRAGSystem()
            print("WebRAGSystem实例化成功。")
            print(f"可用的索引: {app.rag_system.available_indexes}")
        except Exception as e:
            print(f"WebRAGSystem实例化失败: {e}")
            app.rag_system = None

    register_routes(app)

    return app

def register_routes(app):
    """将所有API路由注册到应用"""
    print("开始注册API路由...")

    @app.route('/')
    def index():
        """主页"""
        rag_system = app.rag_system
        return render_template('index.html', indexes=rag_system.available_indexes if rag_system else [])

    @app.route('/api/node_detail')
    def node_detail_route():
        rag_system = app.rag_system
        index_name = request.args.get('index_name')
        node_id = request.args.get('node_id')
        if not all([index_name, node_id]):
            return jsonify({"error": "Missing index_name or node_id"}), 400
        
        details = rag_system.get_node_detail(index_name, node_id)
        return jsonify(details)

    @app.route('/api/load_index', methods=['POST'])
    def load_index_route():
        """检查索引是否存在API"""
        rag_system = app.rag_system
        if not rag_system:
            return jsonify({"success": False, "error": "RAG系统未初始化"})
        data = request.get_json()
        index_name = data.get('index_name')
        
        if not index_name:
            return jsonify({"success": False, "error": "请提供索引名称"})
        
        exists = rag_system.load_index(index_name)
        if exists:
            return jsonify({"success": True, "message": f"索引 {index_name} 可用"})
        else:
            return jsonify({"success": False, "error": f"索引 {index_name} 不存在"})

    @app.route('/api/query', methods=['POST'])
    def query_route():
        """查询API"""
        rag_system = app.rag_system
        if not rag_system:
            return jsonify({"error": "RAG系统未初始化"})
        data = request.get_json()
        question = data.get('question', '').strip()
        index_name = data.get('index_name', 'default')
        expansion_depth = data.get('expansion_depth', 2)
        
        if not question:
            return jsonify({"error": "请提供查询问题"})
        
        if not index_name:
            return jsonify({"error": "请选择一个索引"})
        
        result = rag_system.query(question, index_name, expansion_depth)
        return jsonify(result)

    @app.route('/api/indexes')
    def get_indexes_route():
        """获取可用索引列表API"""
        rag_system = app.rag_system
        if not rag_system:
            return jsonify({"indexes": []})
        return jsonify({"indexes": rag_system.available_indexes})

    @app.route('/api/check_library', methods=['POST'])
    def check_library_route():
        """检查Python库是否已安装API"""
        rag_system = app.rag_system
        if not rag_system:
            return jsonify({"error": "RAG系统未初始化"})

    @app.route('/api/upload_folder', methods=['POST'])
    def upload_folder_route():
        """上传文件夹并构建索引API"""
        rag_system = app.rag_system
        if not rag_system:
            return jsonify({"success": False, "error": "RAG系统未初始化"})

        if 'file' not in request.files:
            return jsonify({"success": False, "error": "未找到文件"}), 400

        file = request.files['file']
        index_name = request.form.get('index_name')

        if file.filename == '':
            return jsonify({"success": False, "error": "未选择文件"}), 400

        if not index_name:
            return jsonify({"success": False, "error": "请提供索引名称"}), 400

        if not file.filename.endswith('.zip'):
            return jsonify({"success": False, "error": "只支持.zip文件"}), 400

        try:
            # 保存上传的zip文件
            upload_dir = Path(CONFIG["upload_dir"])
            upload_dir.mkdir(parents=True, exist_ok=True)
            zip_path = upload_dir / file.filename
            file.save(zip_path)

            # 解压文件
            import zipfile
            extract_to_dir = upload_dir / zip_path.stem  # 使用zip文件名作为解压目录
            extract_to_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_dir)
            
            # 删除zip文件
            os.remove(zip_path)

            # 使用MultiLanguageChunker处理解压后的文件
            from src.chunker.multi_language_chunker import MultiLanguageChunker
            chunker = MultiLanguageChunker()
            
            all_chunks = []
            file_contents = {}
            for root, _, files in os.walk(extract_to_dir):
                for f_name in files:
                    file_path = Path(root) / f_name
                    if chunker.is_supported_file(str(file_path)):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            all_chunks.extend(chunker.chunk_file(str(file_path)))
                            file_contents[str(file_path)] = content
                        except Exception as e:
                            print(f"处理文件 {file_path} 失败: {e}")

            if not all_chunks:
                return jsonify({"success": False, "error": "未找到可处理的代码或文档文件"})

            # 构建向量索引
            success = rag_system.indexer.build_index_from_chunks(
                chunks=all_chunks,
                index_name=index_name,
                embedding_model=CONFIG["embedding_model"],
                metadata={
                    'source_type': 'uploaded_folder',
                    'folder_name': zip_path.stem,
                    'total_chunks': len(all_chunks)
                }
            )

            if not success:
                return jsonify({"success": False, "error": "构建索引失败"})
            
            # 构建知识图谱（如果有Python文件）
            python_files = {f: content for f, content in file_contents.items() 
                          if f.endswith('.py')}
            
            if python_files:
                from src.chunker.ast_chunker import ASTChunker
                from src.indexing.indexer import save_knowledge_graph
                
                ast_chunker = ASTChunker()
                knowledge_graph = ast_chunker.create_knowledge_graph(python_files)
                
                # 保存知识图谱
                index_dir = Path(CONFIG["index_dir"])
                final_index_path = index_dir / index_name
                final_index_path.mkdir(parents=True, exist_ok=True)
                graph_path = str(final_index_path / f"knowledge_graph_{index_name}.gpickle")
                save_knowledge_graph(knowledge_graph, graph_path)

            # 更新可用索引列表
            rag_system.available_indexes = rag_system._get_available_indexes()

            return jsonify({
                "success": True,
                "message": f"成功从上传文件夹构建索引 '{index_name}'",
                "index_name": index_name,
                "total_chunks": len(all_chunks)
            })

        except Exception as e:
            return jsonify({"success": False, "error": f"处理上传文件夹失败: {str(e)}"}), 500
        data = request.get_json()
        library_name = data.get('library_name', '').strip()
        
        if not library_name:
            return jsonify({"error": "请提供库名称"})
        
        is_installed = rag_system.check_library_installed(library_name)
        return jsonify({
            "library_name": library_name,
            "installed": is_installed
        })

    @app.route('/api/install_library', methods=['POST'])
    def install_library_route():
        """安装Python库API"""
        rag_system = app.rag_system
        if not rag_system:
            return jsonify({"error": "RAG系统未初始化"})
        data = request.get_json()
        library_name = data.get('library_name', '').strip()
        
        if not library_name:
            return jsonify({"error": "请提供库名称"})
        
        result = rag_system.install_library(library_name)
        return jsonify(result)

    @app.route('/api/build_library_index', methods=['POST'])
    def build_library_index_route():
        """为Python库构建知识库索引API"""
        rag_system = app.rag_system
        if not rag_system:
            return jsonify({"error": "RAG系统未初始化"})
        data = request.get_json()
        library_name = data.get('library_name', '').strip()
        index_name = data.get('index_name', '').strip()
        
        if not library_name:
            return jsonify({"error": "请提供库名称"})
        
        # 如果没有提供索引名称，使用默认名称
        if not index_name:
            index_name = f"{library_name}_index"
        
        result = rag_system.build_library_index(library_name, index_name)
        return jsonify(result)

    @app.route('/api/build_github_index', methods=['POST'])
    def build_github_index_route():
        """从GitHub仓库构建知识库索引API"""
        rag_system = app.rag_system
        if not rag_system:
            return jsonify({"error": "RAG系统未初始化"})
        data = request.get_json()
        github_url = data.get('github_url', '').strip()
        index_name = data.get('index_name', '').strip()
        max_size_mb = data.get('max_size_mb', 100)
        
        if not github_url:
            return jsonify({"error": "请提供GitHub仓库URL"})
        
        # 验证URL格式
        if not (github_url.startswith('https://github.com/') or github_url.startswith('http://github.com/')):
            return jsonify({"error": "请提供有效的GitHub仓库URL"})
        
        result = rag_system.build_github_index(github_url, index_name, max_size_mb)
        return jsonify(result)
    
    print("API路由注册完成。")
    print(f"已注册的路由: {list(app.url_map.iter_rules())}")

if __name__ == '__main__':
    print("启动Web服务器...")
    app = create_app()
    if app:
        print("Flask应用创建成功，准备运行...")
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000)
    else:
        print("Flask应用创建失败，服务器未启动。")