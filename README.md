# LibChat - 本地Python库智能问答系统

LibChat 是一个基于 RAG (Retrieval-Augmented Generation) 技术的命令行应用，能够对您本地安装的 Python 库进行智能问答。通过结合 AST 解析、向量索引、重排序和大型语言模型，LibChat 能够为您提供精准、高效的代码相关问答服务。

## ✨ 主要功能

- **构建知识库**: 使用 `build` 命令，为指定的 Python 库创建本地知识库索引。
- **智能问答**: 使用 `ask` 命令，基于已构建的知识库，回答关于该库的各种问题。
- **精准检索**: 采用先进的检索和重排序技术，确保答案的相关性和准确性。
- **易于使用**: 简洁的命令行界面，让您能够轻松上手。

## 🚀 快速开始

### 1. 环境准备

首先，请确保您已经安装了 Python 3.8 或更高版本。然后，克隆本仓库并安装所需的依赖：

```bash
git clone https://github.com/whaleylee/LibChat.git
cd LibChat
pip install -r requirements.txt
```

### 2. 配置

项目中使用了大型语言模型（LLM）API，您需要在项目根目录下创建一个 `.env` 文件，并填入您的 API 密钥等信息。例如，您的 `.env` 文件可能包含以下内容：

```
HF_TOKEN=hf_YOUR_HUGGINGFACE_TOKEN
QWEN_API_KEY=sk-your-qwen-api-key
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-large
```

请将 `hf_YOUR_HUGGINGFACE_TOKEN` 和 `sk-your-qwen-api-key` 替换为您的实际 Hugging Face 和 Qwen API 密钥。`EMBEDDING_MODEL` 和 `RERANKER_MODEL` 变量可以根据您的需求进行修改，以使用不同的嵌入和重排序模型。可以参考 `.env.example` 文件。

### 3. 构建知识库

在使用问答功能之前，您需要先为您想要查询的库构建一个知识库。例如，要为 `requests` 库构建知识库，请运行：

```bash
python main.py build requests
```

### 4. 开始提问

知识库构建完成后，您就可以开始提问了。例如，向 `requests` 库提问：

```bash
python main.py ask requests "如何使用requests发送一个带header的POST请求？"
```

同时，为了方便您的使用，也可以开启可视化界面，运行以下指令：

```bash
streamlit run app.py
```

在浏览器访问http://localhost:8501/，即可开启前端界面。

## 🛠️ 技术栈

- **命令行界面**: Typer, Rich
- **代码解析**: Tree-sitter
- **RAG 框架**: LlamaIndex
- **向量存储**: Faiss
- **模型**: Sentence-Transformers, Qwen (通过 OpenAI 兼容接口)

## 🤝 贡献

我们欢迎任何形式的贡献！如果您有任何建议或问题，请随时提交 Issue 或 Pull Request。

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证。
