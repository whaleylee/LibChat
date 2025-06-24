import streamlit as st
import queue
import threading
from loguru import logger
from main import ask_question
import time

st.set_page_config(page_title="LibChat", page_icon="📚")
st.title("📚 LibChat 本地问答系统")

# 输入向量索引路径
index_path = st.text_input("📂 输入你想查询的库：", value="requests")

# 输入问题
query = st.text_area("❓ 输入你的问题：", height=100)


# 配置日志队列
log_queue = queue.Queue()
logger.remove()
logger.add(lambda msg: log_queue.put(msg.record["message"]), level="INFO")

if st.button("🚀 提交"):
    if not query.strip():
        st.warning("请输入问题")
    else:
        st.subheader("🛠️ 进度信息")
        log_display = st.empty()
        logs = []

        result_state = {"answer": None, "error": None, "done": False}

        def task():
            try:
                result_state["answer"] = ask_question(index_name=index_path, query=query)
            except Exception as e:
                result_state["error"] = str(e)
            finally:
                result_state["done"] = True

        # 启动后台问答线程
        thread = threading.Thread(target=task)
        thread.start()

        with st.spinner("正在生成回答..."):
            # 日志展示区域
            log_display_latest = st.empty()  # 显示最新一条
            with st.expander("🔽 查看更多日志进度", expanded=True):  # 默认展开
                log_display_expanded = st.empty()  # 显示最近10条
            while not result_state["done"]:
                while not log_queue.empty():
                    logs.append(log_queue.get())
                if logs:
                    # 显示最新一条日志
                    log_display_latest.markdown(f"🟢 当前进度：`{logs[-1]}`")
                    # 显示最近10条（在展开框中）
                    log_display_expanded.markdown("```\n🧭" + "\n🧭".join(logs[-5:]) + "\n```")
                time.sleep(0.2)

        # 展示结果或错误
        if result_state["error"]:
            st.error(f"❌ 出错：{result_state['error']}")
        else:
            st.subheader("💡 回答结果：")
            st.write(result_state["answer"])