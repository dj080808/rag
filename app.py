import os
import uuid
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional

# ==========================================
# 1. 环境配置与组件初始化
# ==========================================
# 🚨 请在这里替换为你自己在阿里云百炼平台申请的真实 API Key
# os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxx"  
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "devops_knowledge_base"
LOG_FILE_PATH = "error.log"

@st.cache_resource
def init_core_services():
    """初始化底层的数据库客户端、嵌入模型和大语言模型"""
    client = QdrantClient(url=QDRANT_URL)
    embed_model = DashScopeEmbeddings(model="text-embedding-v3")
    chat_model = ChatTongyi(model="qwen-plus", temperature=0.2)
    return client, embed_model, chat_model

qdrant_client, embeddings, llm = init_core_services()

# 初始化 Qdrant 集合：显式对齐通义千问嵌入模型的 1024 维度
if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

def init_mock_log():
    """如果本地不存在日志文件，自动初始化一份用于测试 Tool 的模拟实时日志"""
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
            f.write("2026-05-31 15:30:01 [INFO] 网元管理系统初始化成功。\n")
            f.write("2026-05-31 15:31:45 [WARN] CPRI 通信链路高并发预警。\n")
            f.write("2026-05-31 15:32:12 [ERROR] org.apache.catalina.connector.ClientAbortException: java.io.IOException: Broken pipe\n")
            f.write("2026-05-31 15:33:00 [ERROR] io.lettuce.core.RedisChannelHandler - Connection timeout during cluster topology refresh.\n")

init_mock_log()

# ==========================================
# 2. 数据结构定义 (Pydantic)
# ==========================================
class LLMGeneratedTags(BaseModel):
    """约束大模型只输出我们需要的标签数组，屏蔽多余口水话"""
    tags: List[str] = Field(
        description="根据提供的问题标题、描述、原因和方案，抽取并打上 2-4 个技术栈或模块标签。例如: ['Redis', 'LTE', 'Java']"
    )

# ==========================================
# 3. 后端核心逻辑层
# ==========================================
def search_knowledge_base_logic(query: str) -> str:
    """利用最新的 query_points 接口接收生成好的向量，实现语义检索"""
    query_vector = embeddings.embed_query(query)
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=2
    )
    if not response or not response.points:
        return "知识库中未找到相关排障记录。"
    
    formatted_results = []
    for hit in response.points:
        p = hit.payload
        # 使用 .get() 安全防御取值，防止库内脏数据或旧数据字段不匹配导致崩溃
        v_title = p.get('title', '无标题')
        v_pronto = p.get('pronto_id', '无单号')
        v_desc = p.get('description', '无描述')
        v_error = p.get('error_code', '无')
        v_cause = p.get('root_cause', '无原因')
        v_sol = p.get('solution', '无方案')
        v_tags = p.get('tags', [])
        
        formatted_results.append(
            f"### 📋 [{v_pronto}] {v_title}\n\n"
            f"- **现象描述:** {v_desc}\n"
            f"- **关键报错:** `{v_error}`\n"
            f"- **根本原因:** {v_cause}\n"
            f"- **解决方案:** {v_sol}\n"
            f"- **自动标签:** `{', '.join(v_tags) if v_tags else '无'}`"
        )
    return "\n\n---\n\n".join(formatted_results)

def smart_record_logic(title: str, pronto_id: str, description: str, root_cause: str, error_code: str, solution: str):
    """
    半自动化录入（带向量文本脱水截断版）：
    人工控制核心内容 -> 超长日志/文本自动脱水 -> 干净文本算向量 -> 完整文本存Payload
    """
    st.caption("🔍 正在启动数据预处理流...")
    
    # ─── 🚀 【核心优化：文本与日志脱水】 ───
    # 如果现象描述太长，大模型介入提炼，否则保持原样
    if len(description) > 600:
        st.caption("🔄 检测到【现象描述】字段过长，正在提取核心特征...")
        desc_summary_prompt = "你是一个 DevOps 专家。请将下面这段冗长的故障现象描述或日志，压缩提炼为 200 字以内的精简摘要，务必保留核心报错和受影响的方法名。"
        clean_desc = llm.invoke([("system", desc_summary_prompt), ("human", description)]).content
    else:
        clean_desc = description

    # 如果根本原因里贴了大段日志，大模型介入脱水
    if len(root_cause) > 600:
        st.caption("🔄 检测到【根本原因】字段包含长日志，正在进行脱水清洗...")
        cause_summary_prompt = "你是一个日志分析专家。下面是一段极长的故障原因或堆栈。请在保留核心异常类名、关键参数和直接诱因的前提下，将其精简压缩到 200 字以内，剔除无用时戳和重复流水。"
        clean_cause = llm.invoke([("system", cause_summary_prompt), ("human", root_cause)]).content
    else:
        clean_cause = root_cause

    # 1. 组合干净的上下文让大模型生成标签（不仅速度更快，大模型也不会被日志噪音误导）
    context_for_tags = f"标题: {title}\n描述: {clean_desc}\n原因: {clean_cause}\n报错: {error_code}\n方案: {solution}"
    
    structured_llm = llm.with_structured_output(LLMGeneratedTags)
    system_prompt = "你是一个优秀的资深运维专家。请仔细阅读用户提供的排障单内容，为其提取出最贴切的 2-4 个标签。"
    
    try:
        llm_output = structured_llm.invoke([
            ("system", system_prompt),
            ("human", context_for_tags)
        ])
        auto_tags = llm_output.tags
    except Exception:
        auto_tags = ["Auto-Tagged-Error"]
    
    # 2. 组装最终 Payload 结构：
    # 【关键点】这里存入的 description 和 root_cause 依然是用户输入的【全量原始文本】，确保看详情时不漏掉蛛丝马迹
    final_payload = {
        "title": title,
        "pronto_id": pronto_id,
        "description": description, 
        "root_cause": root_cause,  
        "error_code": error_code,
        "solution": solution,
        "tags": auto_tags
    }
    
    # 3. 生成高维特征向量：
    # 【关键点】我们【只拿脱水后的干净文本】去计算向量，彻底避免向量均值化和失焦
    text_to_vector = f"标题: {title} 现象摘要: {clean_desc} 原因摘要: {clean_cause} 核心报错: {error_code} 解决方案: {solution}"
    vector = embeddings.embed_query(text_to_vector)
    
    # 4. 持久化存入 Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=final_payload
            )
        ]
    )
    return final_payload

# ==========================================
# 4. LangChain 外部工具调用扩展 (Tool)
# ==========================================
@tool
def fetch_live_error_log(keyword: str) -> str:
    """当用户要求查看当前实时日志、查查最新报错，或者问题中带有'刚刚'、'当前'等时间词汇时调用此工具。参数 keyword 为过滤日志的核心词。"""
    if not os.path.exists(LOG_FILE_PATH):
        return "错误：未找到系统实时日志文件 error.log"
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()[-50:] # 获取最后 50 行
        matched_lines = [line.strip() for line in lines if keyword.lower() in line.lower()]
        if not matched_lines:
            return f"在最新的实时日志中，未捕捉到包含关键字 '{keyword}' 的报错。"
        return "【实时日志捕获结果】\n" + "\n".join(matched_lines)
    except Exception as e:
        return f"读取实时日志失败: {e}"

# ==========================================
# 5. Streamlit 现代化前端 UI 渲染
# ==========================================
st.set_page_config(page_title="DevOps 智能排障看板", page_icon="🛠️", layout="wide")

st.title("🛠️ DevOps 项目维护智能排障 Agent")
st.caption("基于 Qwen-Plus 与 Qdrant 向量数据库，由你掌控核心资产，大模型作为助手高效赋能。")

# 侧边栏导航控制
page = st.sidebar.radio("功能导航", ["🔍 故障智能检索", "✍️ 维护日志智能录入"])
# ---- 功能展示区：故障智能检索（流式输出版） ----
if page == "🔍 故障智能检索":
    st.header("智能故障诊断与检索")
    st.markdown("支持**模糊现象提问、查历史故障库**。你也可以直接对它说：*“帮我查查刚刚日志里有什么报错，顺便看看以前有人解决过没。”*")
    
    user_input = st.text_input("输入当前遇到的问题或操作指令:", 
                               placeholder="例如：帮我看看刚才日志里的超时错，顺便看看怎么搞")
    
    if st.button("开始诊断分析", type="primary"):
        if user_input.strip() == "":
            st.warning("请先输入一些内容吧！")
        else:
            # 1. 静态前置处理（查日志、查数据库都需要时间，我们用 st.spinner 包裹）
            with st.spinner("Agent 正在搜集线索并检索向量数据库..."):
                # 注册日志查看工具，并先让大模型进行意图路由判断
                llm_with_tools = llm.bind_tools([fetch_live_error_log])
                intent_check = llm_with_tools.invoke([
                    ("system", "你是一个运维 Agent。请判断用户是否需要查看当下的实时日志。如果是，请立即调用 fetch_live_error_log 工具。"),
                    ("human", user_input)
                ])
                
                live_log_context = ""
                if intent_check.tool_calls:
                    for tool_call in intent_check.tool_calls:
                        if tool_call["name"] == "fetch_live_error_log":
                            st.caption("🤖 **Agent 动作**：检测到需要查看实时日志，正在读取本地 `error.log`...")
                            live_log_context = fetch_live_error_log.invoke(tool_call["args"])
                            with st.chat_message("assistant"):
                                st.text(live_log_context)
                
                # 根据获取的上下文去查向量数据库
                search_query = live_log_context if live_log_context else user_input
                st.caption("🤖 **Agent 动作**：正在对线索进行高维特征转换，检索 Qdrant 知识库...")
                tool_output = search_knowledge_base_logic(search_query)
            
            # 2. ─── 🚀 【核心优化：严格截断幻觉，查不到直接说不知道】 ───
            st.subheader("💡 Agent 最终诊断报告：")
            
            # 检查知识库返回结果是否包含有效数据（通过判断关键字）
            if "知识库中未找到相关排障记录" in tool_output:
                with st.chat_message("assistant"):
                    st.error("❌ 抱歉，当前内部知识库中未检索到任何与该故障相关的相似历史案例。")
                    st.markdown("**💡 建议行动：** 本系统为严格防御模式，已拦截大模型的公开通识幻觉。请前往相关平台手动排查，或在解决后将该案例录入知识库。")
            else:
                # 只有当知识库真正捞出东西时，才允许大模型组织语言
                with st.chat_message("assistant"):
                    
                    # 设定最高优先级的 System 约束，不允许有任何发散
                    strict_prompt = (
                        "你是一个极度严谨的内部运维专家助手。你的回答必须【完全基于】下方提供的【内部故障库参考资料】。\n"
                        "⚠️ 铁律：\n"
                        "1. 严格禁止使用你自身对公开网络、开源软件的通用常识来脑补、扩充或捏造解决方案。\n"
                        "2. 如果参考资料里的方案不完整，请直接基于资料原样陈述，绝对不允许发明任何内部系统的参数、命令或流程。\n"
                        "3. 必须保持回答的真实性，拒绝任何幻觉。\n\n"
                        f"用户当前请求: {user_input}\n"
                        f"当前实时日志线索: {live_log_context if live_log_context else '未提取到实时日志'}\n"
                        f"【唯一合法的内部故障库参考资料】:\n{tool_output}\n\n"
                        "请根据上述合法资料，为用户梳理并总结出标准修复建议："
                    )
                    
                    text_placeholder = st.empty()
                    full_response = ""
                    
                    # 流式渲染
                    for chunk in llm.stream(strict_prompt):
                        full_response += chunk.content
                        text_placeholder.markdown(full_response + "┃")
                    
                    text_placeholder.markdown(full_response)

# ---- 功能展示区：全新可控结构化录入表单 ----
elif page == "✍️ 维护日志智能录入":
    st.header("📋 结构化故障单知识录入")
    st.markdown("自主维护核心 Payload 结构。其中 **Tags** 会由通义千问在提交时依据你填写的细节自动解析生成。")
    
    # 采用标准 Form 机制，确保只有用户点击按钮后才触发一整套流水线
    with st.form("issue_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("📌 问题标题 (Title) *", placeholder="例：A网元因DNS配置错误导致通信中断")
            pronto_id = st.text_input("🆔 Pronto ID / 故障单号 *", placeholder="例：PRONTO-2026-0522")
        with col2:
            error_code = st.text_input("🚨 错误码 / 核心报错 (Error Code)", placeholder="例：io.lettuce.core.RedisChannelHandler")
            
        description = st.text_area("📝 现象描述 (Description) *", placeholder="请输入详细的故障发生现象、用户反馈或可疑的错误堆栈段落...")
        root_cause = st.text_area("🔍 根本原因 (Root Cause) *", placeholder="请写明导致该故障的深层技术本质和排查结论...")
        solution = st.text_area("💡 解决方案 (Solution) *", placeholder="请写明具体的修复步骤、修改的代码行、配置项文件或相关的重启命令...")
        
        # 表单提交
        submit_button = st.form_submit_button("智能归档并生成标签", type="primary")
        
        if submit_button:
            if not (title and pronto_id and description and root_cause and solution):
                st.error("❌ 带有 * 的字段属于核心必填项，请填写完整后再提交！")
            else:
                with st.spinner("小秘书正在帮你定制智能标签，并准备将数据落盘到 Qdrant..."):
                    try:
                        result = smart_record_logic(
                            title=title,
                            pronto_id=pronto_id,
                            description=description,
                            root_cause=root_cause,
                            error_code=error_code,
                            solution=solution
                        )
                        st.success(f"🎉 故障单 [{result['pronto_id']}] 已成功归档入向量库！")
                        st.markdown("### 💾 实际写入向量数据库的完整 Payload 数据：")
                        st.json(result)
                    except Exception as e:
                        st.error(f"知识录入过程中发生异常: {e}")

# 左下角数据库实例运行监控组件
st.sidebar.markdown("---")
try:
    collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    st.sidebar.metric(label="📊 本地故障库已存案例数", value=f"{collection_info.points_count} 条")
except Exception:
    st.sidebar.error("📊 Qdrant 数据库连接异常")