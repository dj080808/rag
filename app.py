import os
import re
import uuid
import hashlib
import json
from datetime import datetime
from typing import List, Optional

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, ScoredPoint,
    Filter, FieldCondition, MatchAny, MatchText, PayloadSchemaType,
)
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

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

# 为 tags 字段创建 payload 索引，使标签过滤从 O(n) 降为 O(log n)
try:
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="tags",
        field_schema=PayloadSchemaType.KEYWORD,
    )
except Exception:
    pass  # 索引已存在则跳过

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
# 3. 检索优化辅助层（缓存、标签提取、关键词提取、查询扩展、重排序）
# ==========================================

# --- 嵌入缓存：避免重复查询反复调用 embedding API ---
_embedding_cache: dict = {}
EMBEDDING_CACHE_MAX = 500

def cached_embed_query(text: str) -> List[float]:
    """对 embed_query 做 LRU 缓存，减少 DashScope API 调用次数与延迟"""
    cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    result = embeddings.embed_query(text)
    # 超过上限时淘汰最旧的 100 条
    if len(_embedding_cache) > EMBEDDING_CACHE_MAX:
        for old_key in list(_embedding_cache.keys())[:100]:
            del _embedding_cache[old_key]
    _embedding_cache[cache_key] = result
    return result

# --- 标签缓存 ---
_tag_cache: dict = {}
TAG_CACHE_MAX = 300

def extract_query_tags(query_text: str) -> List[str]:
    """用 LLM 从用户查询中提取 2-4 个技术标签，匹配入库时的标签体系"""
    cache_key = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
    if cache_key in _tag_cache:
        return _tag_cache[cache_key]

    structured_llm = llm.with_structured_output(LLMGeneratedTags)
    try:
        output = structured_llm.invoke([
            ("system", "从用户的排障查询中提取2-4个最相关的技术栈或模块标签。仅输出标签列表，不要额外解释。"),
            ("human", query_text)
        ])
        tags = output.tags
    except Exception:
        tags = []

    if len(_tag_cache) > TAG_CACHE_MAX:
        for old_key in list(_tag_cache.keys())[:50]:
            del _tag_cache[old_key]
    _tag_cache[cache_key] = tags
    return tags

# --- 错误码 / 关键词提取：弥补向量搜索丢失精确匹配 ---
def extract_error_patterns(query: str) -> List[str]:
    """从查询文本中提取可能的错误码、类名、异常名"""
    patterns = []
    # Java 风格全限定类名: com.foo.BarException
    java_class = re.findall(r'([a-zA-Z][\w]*\.[\w.]*[A-Z]\w*)', query)
    patterns.extend(java_class)
    # 错误码: ERR-1234, ECONNREFUSED, HTTP_500
    error_codes = re.findall(r'([A-Z]{2,}[-_]\d+|[A-Z_]{4,})', query)
    patterns.extend(error_codes)
    # 常见关键词: timeout, broken pipe, OOM 等（长度 > 3）
    short_keywords = re.findall(r'\b(timeout|broken.?pipe|OOM|deadlock|race.?condition)\b', query, re.IGNORECASE)
    patterns.extend([k.upper() for k in short_keywords])
    return list(set(patterns))

# --- 查询扩展：生成语义等价变体，弥补口语化与技术文档的词汇鸿沟 ---
def expand_query(query: str, n_variants: int = 2) -> List[str]:
    """用 LLM 生成查询的语义等价变体，提升召回率"""
    if not query or len(query) < 10:
        return [query]
    expansion_prompt = (
        f"将以下运维排障查询改写为{n_variants}个语义等价但用词不同的变体。"
        "每个变体保留核心技术细节，但可以改变措辞、侧重不同角度。"
        "直接输出变体，每行一个，不要编号。\n\n"
        f"原始查询: {query}"
    )
    try:
        result = llm.invoke(expansion_prompt)
        variants = [
            q.strip() for q in result.content.strip().split("\n")
            if q.strip() and q.strip() != query
        ]
        return [query] + variants[:n_variants]
    except Exception:
        return [query]

# --- LLM 重排序：向量粗筛 → LLM 精排，提升精确率 ---
def rerank_with_llm(
    query: str, candidates: List[ScoredPoint], top_n: int = 3
) -> List[ScoredPoint]:
    """用 LLM 对向量检索候选进行相关性重排序"""
    if len(candidates) <= top_n:
        return candidates

    candidate_texts = []
    for i, point in enumerate(candidates):
        p = point.payload
        if p is None:
            continue
        summary = (
            f"[{i}] 标题: {p.get('title', '')} | "
            f"现象: {str(p.get('description', ''))[:120]} | "
            f"标签: {', '.join(p.get('tags', []))}"
        )
        candidate_texts.append(summary)

    ranking_prompt = (
        f"用户问题: {query}\n\n"
        "以下是检索到的候选排障案例。请按与用户问题的相关性从高到低排序，"
        "仅返回排序后的索引号（逗号分隔），例如: 2, 0, 5, 1, 3, 4\n\n"
        + "\n".join(candidate_texts)
    )

    try:
        result = llm.invoke(ranking_prompt)
        indices = [int(x.strip()) for x in result.content.split(",")]
        reranked = [candidates[i] for i in indices if i < len(candidates)]
        return reranked[:top_n]
    except Exception:
        return candidates[:top_n]

# ==========================================
# 4. 后端核心逻辑层
# ==========================================
def search_knowledge_base_logic(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.45,
    use_tag_filter: bool = True,
    use_expansion: bool = False,
    use_rerank: bool = True,
) -> tuple[str, dict]:
    """
    增强版语义检索：
    - 分数阈值过滤低质量结果
    - 标签感知过滤（先过滤 → 不够则回退全局）
    - 错误码关键词加权
    - 可选查询扩展 + LLM 重排序
    返回 (formatted_text, diagnostics_dict)
    """
    diagnostics = {
        "query": query,
        "query_tags": [],
        "variants_used": 1,
        "candidates_raw": 0,
        "candidates_after_rerank": 0,
        "top_score": 0.0,
        "min_score": 0.0,
        "filter_applied": False,
        "fell_back": False,
        "rerank_applied": False,
        "expansion_applied": False,
    }

    # 1. 查询扩展（可选）
    if use_expansion:
        query_variants = expand_query(query)
        diagnostics["variants_used"] = len(query_variants)
        diagnostics["expansion_applied"] = len(query_variants) > 1
    else:
        query_variants = [query]

    # 2. 标签提取（用于过滤）
    query_tags: List[str] = []
    if use_tag_filter:
        query_tags = extract_query_tags(query)
        diagnostics["query_tags"] = query_tags

    # 3. 错误码 / 关键词提取（用于 should 加权）
    error_keywords = extract_error_patterns(query)

    # 4. 多查询搜索 + 去重融合
    all_points: dict[str, ScoredPoint] = {}

    for variant in query_variants:
        query_vector = cached_embed_query(variant)

        # 构建 query_filter
        query_filter = None
        should_conditions = []

        # should: 关键词加权
        for kw in error_keywords:
            should_conditions.append(
                FieldCondition(key="error_code", match=MatchText(text=kw))
            )
            should_conditions.append(
                FieldCondition(key="description", match=MatchText(text=kw))
            )

        if query_tags and use_tag_filter:
            try:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="tags",
                            match=MatchAny(any=query_tags),
                        )
                    ],
                    should=should_conditions if should_conditions else None,
                )
                diagnostics["filter_applied"] = True
            except Exception:
                query_filter = (
                    Filter(should=should_conditions) if should_conditions else None
                )
        elif should_conditions:
            query_filter = Filter(should=should_conditions)

        # 执行向量检索
        try:
            response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k * 3 if use_rerank else top_k,
                score_threshold=score_threshold,
            )
        except Exception:
            # 过滤语法不兼容时回退到无过滤搜索
            response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k * 3 if use_rerank else top_k,
                score_threshold=score_threshold,
            )

        if response and response.points:
            for point in response.points:
                pid = point.id
                if pid not in all_points or point.score > all_points[pid].score:
                    all_points[pid] = point

    # 5. 标签过滤回退：结果太少则去掉过滤再搜一次
    if diagnostics["filter_applied"] and len(all_points) < 2:
        diagnostics["fell_back"] = True
        query_vector = cached_embed_query(query)
        try:
            response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k * 3 if use_rerank else top_k,
                score_threshold=score_threshold,
            )
            if response and response.points:
                for point in response.points:
                    pid = point.id
                    if pid not in all_points or point.score > all_points[pid].score:
                        all_points[pid] = point
        except Exception:
            pass

    candidates = sorted(
        all_points.values(), key=lambda p: p.score, reverse=True
    )
    diagnostics["candidates_raw"] = len(candidates)

    if not candidates:
        return "知识库中未找到相关排障记录。", diagnostics

    # 6. LLM 重排序（可选）
    if use_rerank and len(candidates) > top_k:
        diagnostics["rerank_applied"] = True
        candidates = rerank_with_llm(query, candidates, top_n=top_k)

    diagnostics["candidates_after_rerank"] = len(candidates)
    diagnostics["top_score"] = round(candidates[0].score, 4) if candidates else 0.0
    diagnostics["min_score"] = round(candidates[-1].score, 4) if candidates else 0.0

    # 7. 格式化结果（含相似度分数）
    formatted_results = []
    for hit in candidates:
        p = hit.payload
        if p is None:
            continue
        v_title = p.get("title", "无标题")
        v_pronto = p.get("pronto_id", "无单号")
        v_desc = p.get("description", "无描述")
        v_error = p.get("error_code", "无")
        v_cause = p.get("root_cause", "无原因")
        v_sol = p.get("solution", "无方案")
        v_tags = p.get("tags", [])
        v_score = getattr(hit, "score", 0.0)

        formatted_results.append(
            f"### 📋 [{v_pronto}] {v_title}  (匹配度: {v_score:.1%})\n\n"
            f"- **现象描述:** {v_desc}\n"
            f"- **关键报错:** `{v_error}`\n"
            f"- **根本原因:** {v_cause}\n"
            f"- **解决方案:** {v_sol}\n"
            f"- **自动标签:** `{', '.join(v_tags) if v_tags else '无'}`"
        )
    return "\n\n---\n\n".join(formatted_results), diagnostics

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
    vector = cached_embed_query(text_to_vector)
    
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
# 5. LangChain 外部工具调用扩展 (Tool)
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
# 6. Streamlit 现代化前端 UI 渲染
# ==========================================
st.set_page_config(page_title="DevOps 智能排障看板", page_icon="🛠️", layout="wide")

st.title("🛠️ DevOps 项目维护智能排障 Agent")
st.caption("基于 Qwen-Plus 与 Qdrant 向量数据库，由你掌控核心资产，大模型作为助手高效赋能。")

# 侧边栏导航控制
page = st.sidebar.radio("功能导航", ["🔍 故障智能检索", "✍️ 维护日志智能录入"])

# 侧边栏检索参数
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ 检索参数")
retrieval_top_k = st.sidebar.slider(
    "检索深度 (Top-K)", min_value=1, max_value=10, value=5,
    help="从知识库中检索的匹配案例数量"
)
retrieval_score_threshold = st.sidebar.slider(
    "最低匹配度阈值", min_value=0.0, max_value=1.0, value=0.45, step=0.05,
    help="低于此阈值的结果将被过滤，0 表示不过滤"
)

with st.sidebar.expander("🔧 高级检索选项"):
    use_tag_filter = st.checkbox(
        "启用标签过滤", value=True,
        help="从查询中提取技术标签，仅搜索匹配标签的案例（结果太少时自动回退）"
    )
    use_query_expansion = st.checkbox(
        "启用查询扩展", value=False,
        help="生成查询的语义等价变体，提升召回率（增加 LLM 调用）"
    )
    use_llm_rerank = st.checkbox(
        "启用 LLM 重排序", value=True,
        help="向量粗筛后用 LLM 精排，提升精确率"
    )
    show_diagnostics = st.checkbox(
        "显示检索诊断", value=False,
        help="展示检索过程的详细诊断信息"
    )

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

                # 标签过滤提示
                if use_tag_filter:
                    query_tags = extract_query_tags(search_query)
                    if query_tags:
                        st.caption(f"🏷️ 提取到查询标签: `{', '.join(query_tags)}`")

                tool_output, diagnostics = search_knowledge_base_logic(
                    query=search_query,
                    top_k=retrieval_top_k,
                    score_threshold=retrieval_score_threshold,
                    use_tag_filter=use_tag_filter,
                    use_expansion=use_query_expansion,
                    use_rerank=use_llm_rerank,
                )

            # 诊断信息展示
            if show_diagnostics:
                with st.expander("🔬 检索质量诊断", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("候选结果数", diagnostics["candidates_raw"])
                        st.metric("标签过滤", "✓" if diagnostics["filter_applied"] else "✗")
                    with col2:
                        st.metric("重排序后", diagnostics["candidates_after_rerank"])
                        st.metric("已回退", "✓" if diagnostics["fell_back"] else "✗")
                    with col3:
                        st.metric("最高匹配度", f"{diagnostics['top_score']:.2%}")
                        st.metric("最低匹配度", f"{diagnostics['min_score']:.2%}")
                    st.caption(f"查询标签: `{', '.join(diagnostics['query_tags']) if diagnostics['query_tags'] else '无'}`")
                    st.caption(f"查询变体数: {diagnostics['variants_used']} | "
                              f"重排序: {'✓' if diagnostics['rerank_applied'] else '✗'} | "
                              f"扩展: {'✓' if diagnostics['expansion_applied'] else '✗'}")
                    st.caption(f"原始查询: {diagnostics['query']}")
                    st.json(diagnostics)

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