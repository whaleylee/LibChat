# LibChat - 智能代码问答系统
(2025/7/2更新：增加了图增强RAG，增加web界面，增加对git仓库和本地代码包的支持，增加以知识图谱为基础的相关代码查看功能）

## 项目简介

LibChat 是一个基于 GraphRAG 技术的本地代码库智能问答系统。

## 🚀 核心功能

- **代码分析**: 基于AST的结构化代码解析，支持函数、类、变量等代码元素的精确提取
- **多模态检索**: 结合向量检索、BM25关键词匹配和知识图谱的混合检索策略
- **知识图谱构建**: 自动构建代码间的依赖关系图，支持可视化展示
- **多语言支持**: 支持Python、JavaScript、TypeScript、Java、C++等多种编程语言
- **重排序**: 使用CrossEncoder模型对检索结果进行重排序，提高答案质量
- **GitHub集成**: 支持直接输入GitHub仓库URL，自动克隆和分析整个代码库
- **Web界面**: 提供直观的Web界面，支持实时查询和知识图谱可视化
- **命令行工具**: 支持批量处理和自动化集成


## 技术架构

### 核心模块

#### 1. 代码分块器 (Chunker)
- **AST分块器** (`ast_chunker.py`)：基于抽象语法树的Python代码分块
- **多语言分块器** (`multi_language_chunker.py`)：支持多种编程语言的智能分块
- 使用 Tree-sitter 进行精确的语法分析
- 支持函数、类、方法级别的代码块提取

#### 2. 索引器 (Indexer)
- **修复索引器** (`fixed_indexer.py`)：改进的向量索引构建器
- 支持 FAISS 向量存储和持久化
- 集成多种嵌入模型（OpenAI, BGE系列）
- 提供索引管理和重建功能

#### 3. 检索器 (Retriever)
- **GraphRAG检索器** (`graph_retriever.py`)：图增强检索实现
- 结合向量检索和图遍历技术
- 两阶段检索策略：入口点定位 + 上下文扩展
- 支持可配置的扩展深度

#### 4. 重排序器 (Reranker)
- **句子变换器重排序器** (`reranker.py`)：基于CrossEncoder的结果重排
- 使用 BGE-reranker 模型提升检索精度
- 支持批量处理和GPU加速

#### 5. 源码检查器 (Inspector)
- **包检查器** (`inspector.py`)：Python包源码定位和收集
- 支持已安装包和本地项目的源码发现
- 智能处理包结构和模块依赖

## 快速开始

### 环境准备

1. **Python 环境**
   ```bash
   # 确保 Python 3.8+
   python --version
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置环境变量**
   ```bash
   # 设置 OpenAI API Key（用于嵌入和生成）
   export OPENAI_API_KEY="your-api-key-here"
   
   # 或者在 .env 文件中配置
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

### 构建知识库

使用命令行工具构建代码知识库：

```bash
# 为本地项目构建索引
python main.py build /path/to/your/project

# 为已安装的 Python 包构建索引
python main.py build --package numpy

# 使用自定义嵌入模型
python main.py build /path/to/project --embedding-model BAAI/bge-small-en-v1.5
```

### 📚 构建知识库

#### Python库分析
```bash
# 构建指定Python库的知识库
python main.py build numpy

# 指定自定义索引名称
python main.py build requests --index-name my-requests-index
```

#### GitHub仓库分析
```bash
# 从GitHub仓库构建知识库
python main.py build-github https://github.com/user/repo

# 指定自定义索引名称
python main.py build-github https://github.com/fastapi/fastapi --index-name fastapi-analysis

# 设置最大仓库大小限制（MB）
python main.py build-github https://github.com/user/repo --max-size 100
```

### 启动Web界面

```bash
# 启动 Web 应用
python run_web.py

# 或者直接运行
python run.py
```

访问 `http://localhost:5000` 开始使用Web界面。

## 使用指南

### 命令行模式

```bash
# 构建索引
python main.py build /path/to/your/code

# 开始问答
python main.py ask "如何使用这个库的主要功能？"
python main.py ask "这个项目中有哪些重要的类？"
python main.py ask "解释一下数据处理的流程"
```

### Web界面功能

#### 左侧面板 - 智能问答
- **问题输入**：支持自然语言查询
- **历史记录**：保存问答历史
- **索引管理**：查看和管理已构建的索引
- **库安装**：直接安装Python包并构建索引

#### 右侧面板 - 知识图谱可视化
- **交互式图谱**：可缩放、拖拽的知识图谱
- **节点详情**：点击节点查看代码详情
- **关系展示**：显示代码间的调用和依赖关系
- **源码查看**：直接查看相关源代码

## 技术栈

### 核心技术
- **GraphRAG**：图增强检索生成技术
- **LlamaIndex**：文档索引和检索框架
- **FAISS**：高效向量相似度搜索
- **Tree-sitter**：多语言代码解析
- **NetworkX**：图数据结构和算法
- **Sentence-Transformers**：文本嵌入和重排序

### 前端技术
- **Flask**：Web框架
- **D3.js**：数据可视化
- **Bootstrap**：响应式UI框架
- **jQuery**：前端交互

### AI模型支持
- **OpenAI Embeddings**：text-embedding-ada-002
- **BGE系列模型**：bge-small/base/large-en-v1.5
- **BGE重排序模型**：bge-reranker-large
- **多语言支持**：bge-m3（多语言嵌入）

## 项目结构

```
LibChat/
├── .gitignore                   # Git忽略文件
├── README.md                    # 项目文档
├── app.py                       # Flask应用
├── github_cache/                # GitHub仓库缓存目录
├── main.py                      # 命令行入口
├── requirements.txt             # 依赖列表
├── run.py                       # 简单启动脚本
├── run_web.py                   # Web应用启动脚本
├── src/                         # 核心源码目录
│   ├── __init__.py              # src模块初始化文件
│   ├── chunker/                 # 代码分块模块
│   │   ├── __init__.py          # chunker模块初始化文件
│   │   ├── ast_chunker.py       # AST分块器
│   │   └── multi_language_chunker.py  # 多语言分块器
│   ├── data_ingestion/          # 数据摄取模块
│   ├── generation/              # 生成模块
│   │   └── __init__.py          # generation模块初始化文件
│   ├── github_crawler/          # GitHub爬虫模块
│   │   ├── __init__.py          # github_crawler模块初始化文件
│   │   └── github_crawler.py    # GitHub爬虫实现
│   ├── indexing/                # 索引构建模块
│   │   ├── __init__.py          # indexing模块初始化文件
│   │   ├── fixed_indexer.py     # 修复索引器
│   │   └── indexer.py           # 索引器实现
│   ├── pipeline/                # 管道模块
│   │   └── __init__.py          # pipeline模块初始化文件
│   ├── reranking/               # 重排序模块
│   │   ├── __init__.py          # reranking模块初始化文件
│   │   └── reranker.py          # 重排序器
│   ├── retrieval/               # 检索模块
│   │   ├── __init__.py          # retrieval模块初始化文件
│   │   └── graph_retriever.py   # GraphRAG检索器
│   ├── source_inspector/        # 源码检查模块
│   │   ├── __init__.py          # source_inspector模块初始化文件
│   │   └── inspector.py         # 包检查器
│   └── utils/                   # 工具模块
│       └── __init__.py          # utils模块初始化文件
├── static/                      # 静态资源
│   └── css/                     # 样式文件
├── temp/                        # 临时文件目录
├── templates/                   # Web模板
│   └── index.html               # 主页面模板
├── test_encoding_index/         # 测试编码索引目录
└── uploads/                     # 上传文件目录
```




## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。
