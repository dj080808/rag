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
    SparseVector, SparseVectorParams,
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

# 初始化 Qdrant 集合：命名向量 (dense 语义 + sparse 词法) 支持 RRF 混合检索
COLLECTION_EXISTS = qdrant_client.collection_exists(collection_name=COLLECTION_NAME)
NEEDS_MIGRATION = False

if not COLLECTION_EXISTS:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
else:
    # 检查旧集合是否需要迁移（无 named vectors）
    try:
        info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        has_sparse = hasattr(info.config.params, "sparse_vectors") and info.config.params.sparse_vectors
        if not has_sparse:
            NEEDS_MIGRATION = True
    except Exception:
        pass

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

# --- 稀疏向量生成器（n-gram，纯 Python 无额外依赖） ---
# 用于 RRF 混合检索中的关键词/词法匹配通道
NGRAM_VOCAB_SIZE = 10000  # 哈希桶数量

def _ngrams(text: str, n: int) -> List[str]:
    """生成文本的字符级 n-gram（适配中英文混合）"""
    # 在文本前后加边界标记，让首尾字符也能形成 n-gram
    text = "^" + text + "$"
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def text_to_sparse_vector(text: str) -> SparseVector:
    """
    将文本转为稀疏向量（TF-IDF 风格的 n-gram 加权）。
    同时捕获中文字符 n-gram 和英文单词片段，支持混合检索。
    """
    # 生成 1-gram / 2-gram / 3-gram
    all_ngrams = _ngrams(text, 1) + _ngrams(text, 2) + _ngrams(text, 3)

    # 统计频率
    freq: dict[int, float] = {}
    for ng in all_ngrams:
        idx = abs(hash(ng)) % NGRAM_VOCAB_SIZE
        freq[idx] = freq.get(idx, 0.0) + 1.0

    # 简单 IDF 抑制：高频 n-gram（如纯空格、常见字符）降权
    max_freq = max(freq.values()) if freq else 1.0
    indices = []
    values = []
    for idx, count in freq.items():
        # sublinear TF scaling: 1 + log(tf)，抑制高频噪音
        tf = 1.0 + (count / max_freq) ** 0.5
        indices.append(idx)
        values.append(round(tf, 4))

    return SparseVector(indices=indices, values=values)

def text_to_sparse_vector_dict(text: str) -> dict:
    """同上，但返回 dict 格式（用于 Qdrant PointStruct 的 named vector）"""
    sv = text_to_sparse_vector(text)
    return {"indices": sv.indices, "values": sv.values}

# --- Python 端 RRF (Reciprocal Rank Fusion) ---
# Qdrant 1.18 server 不支持 native Fusion.RRF，在客户端实现 RRF 合并
RRF_K = 60  # RRF 平滑常数

def rrf_fuse(
    dense_results: List[ScoredPoint],
    sparse_results: List[ScoredPoint],
    top_k: int = 10,
) -> List[ScoredPoint]:
    """
    将稠密向量和稀疏向量的搜索结果用 RRF 公式融合：
    RRF_score(d) = sum_over_channels( 1 / (k + rank_i(d)) )
    """
    rrf_scores: dict[str, float] = {}
    point_map: dict[str, ScoredPoint] = {}

    # 稠密通道
    for rank, point in enumerate(dense_results):
        pid = point.id
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if pid not in point_map:
            point_map[pid] = point

    # 稀疏通道
    for rank, point in enumerate(sparse_results):
        pid = point.id
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if pid not in point_map:
            point_map[pid] = point

    # 按 RRF 分数排序
    sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)
    fused = []
    for pid in sorted_ids[:top_k]:
        p = point_map[pid]
        # 用 RRF 分数替换原始分数
        p.score = rrf_scores[pid]
        fused.append(p)
    return fused

# --- HyDE: 假设文档嵌入，弥合口语查询与技术文档的语义鸿沟 ---
def generate_hypothetical_doc(query: str) -> str:
    """
    让 LLM"脑补"一段假设的排障记录，用技术语言描述用户问题。
    然后用这段假设文档的 embedding 去检索，比直接用口语查询更精准。
    """
    if not query or len(query) < 5:
        return query

    hyde_prompt = (
        "你是一个资深运维专家。请根据用户的问题，撰写一段假设的故障排障记录摘要，"
        "包含：标题、故障现象、关键报错类名/错误码、根本原因。"
        "使用专业术语，150 字以内，不要编号或列表格式。"
        f"\n\n用户问题: {query}\n\n假设的排障记录:"
    )

    try:
        result = llm.invoke(hyde_prompt)
        hypothetical = result.content.strip()
        if not hypothetical or len(hypothetical) < 10:
            return query
        return hypothetical
    except Exception:
        return query

# ==========================================
# 4. 后端核心逻辑层
# ==========================================

def rewrite_query_with_history(chat_history: List[dict], current_query: str) -> str:
    """
    上下文感知查询改写：将对话历史中的关键信息融入当前查询，
    使向量检索能利用上下文消歧/补全。
    例如: 历史 "Redis 超时" + 当前 "ChannelHandler 那个" → "Redis ChannelHandler 超时"
    """
    if not chat_history or len(chat_history) < 2:
        return current_query

    # 取最近 6 轮对话作为上下文
    recent = chat_history[-6:]
    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:300]}"
        for m in recent
    )

    rewrite_prompt = (
        "你是一个查询改写专家。根据对话历史，将用户的当前问题改写为一个独立、完整、"
        "包含上下文中关键技术细节的检索查询。保留所有错误码、类名、关键报错信息。"
        "直接输出改写后的查询，不要加任何前缀或解释。\n\n"
        f"对话历史:\n{history_text}\n\n"
        f"当前用户问题: {current_query}\n\n"
        "改写后的检索查询:"
    )

    try:
        result = llm.invoke(rewrite_prompt)
        rewritten = result.content.strip()
        # 防御：如果 LLM 返回空或过长，用原文
        if not rewritten or len(rewritten) > 500:
            return current_query
        return rewritten
    except Exception:
        return current_query

def search_knowledge_base_logic(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.45,
    use_tag_filter: bool = True,
    use_expansion: bool = False,
    use_rerank: bool = True,
    use_rrf: bool = True,
) -> tuple[str, dict]:
    """
    RRF 混合检索：
    - 稠密向量 (语义) + 稀疏向量 (词法/n-gram) → RRF 融合
    - 分数阈值 + 标签感知过滤 + 查询扩展 + LLM 重排序
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
        "rrf_enabled": use_rrf and not NEEDS_MIGRATION,
        "needs_migration": NEEDS_MIGRATION,
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

    # 4. 构建 query_filter
    def _is_full_class_name(kw: str) -> bool:
        """判断是否完整类名/异常名（如 io.lettuce.core.RedisChannelHandler）"""
        return bool(re.match(r'^[a-zA-Z][\w]*(\.[\w]+)+[A-Z]\w*$', kw))

    def build_filter(tags: List[str], keywords: List[str]):
        must_conditions = []
        should_conditions = []

        for kw in keywords:
            if _is_full_class_name(kw):
                # 完整类名 → must: 精确匹配错误码字段
                must_conditions.append(
                    FieldCondition(key="error_code", match=MatchText(text=kw))
                )
            else:
                # 普通关键词 → should: 加权匹配
                should_conditions.append(
                    FieldCondition(key="error_code", match=MatchText(text=kw))
                )
                should_conditions.append(
                    FieldCondition(key="description", match=MatchText(text=kw))
                )

        # 合并标签条件
        if tags:
            must_conditions.append(
                FieldCondition(key="tags", match=MatchAny(any=tags))
            )

        if must_conditions or should_conditions:
            try:
                return Filter(
                    must=must_conditions if must_conditions else None,
                    should=should_conditions if should_conditions else None,
                )
            except Exception:
                return Filter(should=should_conditions) if should_conditions else None
        return None

    query_filter = build_filter(query_tags, error_keywords)
    if query_filter and query_tags:
        diagnostics["filter_applied"] = True

    # 5. RRF 混合检索：稠密 + 稀疏 → Python 端 RRF 融合
    all_points: dict[str, ScoredPoint] = {}
    rrf_enabled = diagnostics["rrf_enabled"]

    for variant in query_variants:
        dense_vector = cached_embed_query(variant)
        sparse_vector = text_to_sparse_vector(variant)

        try:
            if rrf_enabled:
                # ─── RRF: 分别做 dense + sparse 搜索，再客户端融合 ───
                dense_resp = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=dense_vector,
                    using="dense",
                    query_filter=query_filter,
                    limit=top_k * 3 if use_rerank else top_k,
                    score_threshold=score_threshold,
                )
                dense_points = dense_resp.points if dense_resp else []

                sparse_resp = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=sparse_vector,
                    using="sparse",
                    query_filter=query_filter,
                    limit=top_k * 3 if use_rerank else top_k,
                    score_threshold=0.0,  # 稀疏分数范围不同，阈值在 RRF 后统一处理
                )
                sparse_points = sparse_resp.points if sparse_resp else []

                fused = rrf_fuse(dense_points, sparse_points, top_k=top_k * 3 if use_rerank else top_k)
                for point in fused:
                    pid = point.id
                    if pid not in all_points or point.score > all_points[pid].score:
                        all_points[pid] = point
            else:
                # 回退：仅稠密向量
                response = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=dense_vector,
                    using="dense",
                    query_filter=query_filter,
                    limit=top_k * 3 if use_rerank else top_k,
                    score_threshold=score_threshold,
                )
                if response and response.points:
                    for point in response.points:
                        pid = point.id
                        if pid not in all_points or point.score > all_points[pid].score:
                            all_points[pid] = point
        except Exception:
            # 过滤语法不兼容 / sparse 不可用 → 回退 dense-only
            try:
                response = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=dense_vector,
                    using="dense",
                    limit=top_k * 3 if use_rerank else top_k,
                    score_threshold=score_threshold,
                )
                if response and response.points:
                    for point in response.points:
                        pid = point.id
                        if pid not in all_points or point.score > all_points[pid].score:
                            all_points[pid] = point
            except Exception:
                continue

    # 6. 标签过滤回退：结果太少则去掉过滤再搜一次
    if diagnostics["filter_applied"] and len(all_points) < 2:
        diagnostics["fell_back"] = True
        unfiltered_filter = build_filter([], error_keywords)
        dense_vector = cached_embed_query(query)
        sparse_vector = text_to_sparse_vector(query)

        try:
            if rrf_enabled:
                dense_resp = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=dense_vector, using="dense", query_filter=unfiltered_filter,
                    limit=top_k * 3 if use_rerank else top_k,
                    score_threshold=score_threshold,
                )
                sparse_resp = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=sparse_vector, using="sparse", query_filter=unfiltered_filter,
                    limit=top_k * 3 if use_rerank else top_k,
                    score_threshold=0.0,
                )
                fallback_fused = rrf_fuse(
                    dense_resp.points if dense_resp else [],
                    sparse_resp.points if sparse_resp else [],
                    top_k=top_k * 3 if use_rerank else top_k,
                )
                for point in fallback_fused:
                    pid = point.id
                    if pid not in all_points or point.score > all_points[pid].score:
                        all_points[pid] = point
            else:
                response = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=dense_vector, using="dense", query_filter=unfiltered_filter,
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

    candidates = sorted(all_points.values(), key=lambda p: p.score, reverse=True)
    diagnostics["candidates_raw"] = len(candidates)

    if not candidates:
        return "知识库中未找到相关排障记录。", diagnostics

    # 7. LLM 重排序（可选）
    if use_rerank and len(candidates) > top_k:
        diagnostics["rerank_applied"] = True
        candidates = rerank_with_llm(query, candidates, top_n=top_k)

    diagnostics["candidates_after_rerank"] = len(candidates)
    diagnostics["top_score"] = round(candidates[0].score, 4) if candidates else 0.0
    diagnostics["min_score"] = round(candidates[-1].score, 4) if candidates else 0.0

    # 8. 格式化结果（含相似度分数）
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
    # 【关键点】只拿脱水后的干净文本计算向量，dense + sparse 双通道
    text_to_vector = f"标题: {title} 现象摘要: {clean_desc} 原因摘要: {clean_cause} 核心报错: {error_code} 解决方案: {solution}"
    dense_vector = cached_embed_query(text_to_vector)

    # 4. 持久化存入 Qdrant
    new_id = str(uuid.uuid4())
    if NEEDS_MIGRATION:
        # 旧集合不支持命名向量，使用旧格式（仅 dense）
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=new_id,
                    vector=dense_vector,
                    payload=final_payload,
                )
            ]
        )
    else:
        # 新集合：命名向量 (dense 语义 + sparse 词法) — 支持 RRF 混合检索
        sparse_vector = text_to_sparse_vector_dict(text_to_vector)
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=new_id,
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_vector,
                    },
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

# 侧边栏：检索模式
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 检索模式")

# 会话状态初始化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # {role, content, diagnostics, msg_id}
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "上下文连续检索"
if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # msg_id → "relevant" | "irrelevant"
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0  # 用于生成唯一消息 ID

search_mode = st.sidebar.radio(
    "检索模式",
    ["上下文连续检索", "全新独立检索"],
    help="**上下文连续检索**：将历史对话代入，每次检索叠加上下文，逐步逼近目标。\n\n"
         "**全新独立检索**：每次检索独立进行，不携带历史上下文。"
)

# 模式切换时清空历史
if search_mode != st.session_state.search_mode:
    st.session_state.search_mode = search_mode
    st.session_state.chat_history = []

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
with col2:
    history_len = len(st.session_state.chat_history)
    st.caption(f"已积累 {history_len} 条消息")

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
    use_rrf = st.checkbox(
        "启用 RRF 混合检索", value=True,
        help="稠密向量 (语义) + 稀疏向量 (词法) → RRF 融合，显著提升精确率"
    )
    use_tag_filter = st.checkbox(
        "启用标签过滤", value=True,
        help="从查询中提取技术标签，仅搜索匹配标签的案例（结果太少时自动回退）"
    )
    use_hyde = st.checkbox(
        "启用 HyDE 假设文档", value=False,
        help="让 LLM 先生成一段假设排障记录，再用它去检索（+1 LLM 调用，显著提升口语查询精确率）"
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

# 集合迁移提示与一键升级
if NEEDS_MIGRATION:
    st.sidebar.warning(
        "⚠️ 当前知识库格式较旧，不支持 RRF 混合检索。"
        "新录入的案例将自动使用旧格式（仅 dense）。"
    )
    if st.sidebar.button("🔄 一键迁移到 RRF 双向量格式", type="secondary",
                         help="删除旧集合并创建支持 dense+sparse 的新集合（已有数据将丢失，请先备份）"):
        try:
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(size=1024, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )
            # 重建 tags 索引
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tags",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            NEEDS_MIGRATION = False
            st.sidebar.success("✅ 迁移完成！现支持 RRF 混合检索（dense + sparse）。")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"迁移失败: {e}")

# ---- 功能展示区：故障智能检索（对话式） ----
if page == "🔍 故障智能检索":
    st.header("智能故障诊断与检索")

    is_contextual = (search_mode == "上下文连续检索")

    if is_contextual:
        st.markdown(
            "**🔄 上下文连续检索模式** — 每次检索携带历史对话，逐步逼近目标。"
            "输入追加信息即可细化查询。"
        )
    else:
        st.markdown(
            "**🆕 全新独立检索模式** — 每次检索彼此独立，不携带任何历史上下文。"
        )

    # 渲染历史消息
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 反馈按钮（仅对助手消息）
            if msg["role"] == "assistant" and msg.get("msg_id"):
                mid = msg["msg_id"]
                current_feedback = st.session_state.feedback.get(mid)
                fc1, fc2, fc_space = st.columns([1, 1, 15])
                with fc1:
                    if st.button("👍", key=f"rel_{mid}",
                                 help="结果相关",
                                 type="primary" if current_feedback == "relevant" else "secondary"):
                        st.session_state.feedback[mid] = "relevant"
                        st.rerun()
                with fc2:
                    if st.button("👎", key=f"irrel_{mid}",
                                 help="结果不相关",
                                 type="primary" if current_feedback == "irrelevant" else "secondary"):
                        st.session_state.feedback[mid] = "irrelevant"
                        st.rerun()
                if current_feedback:
                    with fc_space:
                        st.caption(
                            "✅ 感谢反馈" if current_feedback == "relevant"
                            else "📝 已记录，将用于优化"
                        )

            if msg.get("diagnostics") and show_diagnostics:
                with st.expander("🔬 本次检索诊断", expanded=False):
                    d = msg["diagnostics"]
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("候选数", d.get("candidates_raw", 0))
                        st.metric("标签过滤", "✓" if d.get("filter_applied") else "✗")
                    with c2:
                        st.metric("重排序后", d.get("candidates_after_rerank", 0))
                        st.metric("已回退", "✓" if d.get("fell_back") else "✗")
                    with c3:
                        st.metric("最高匹配度", f"{d.get('top_score', 0):.2%}")
                        st.metric("最低匹配度", f"{d.get('min_score', 0):.2%}")
                    st.caption(
                        f"查询标签: `{', '.join(d.get('query_tags', [])) or '无'}` | "
                        f"RRF: {'✓' if d.get('rrf_enabled') else '✗'} | "
                        f"HyDE: {'✓' if d.get('hyde_applied') else '✗'} | "
                        f"重排序: {'✓' if d.get('rerank_applied') else '✗'}"
                    )
                    if d.get("rewritten_query"):
                        st.caption(f"改写后查询: {d['rewritten_query']}")

    # 聊天输入
    user_input = st.chat_input("输入当前遇到的问题或操作指令...",
                               key="chat_input_key")

    if user_input and user_input.strip():
        # 新增用户消息（带唯一 ID）
        st.session_state.msg_counter += 1
        user_msg_id = f"u{st.session_state.msg_counter}"
        st.session_state.chat_history.append({
            "role": "user", "content": user_input, "msg_id": user_msg_id,
        })

        with st.spinner("Agent 正在搜集线索并检索向量数据库..."):
            # 1. 上下文查询改写（仅上下文模式）
            if is_contextual and len(st.session_state.chat_history) > 1:
                search_query = rewrite_query_with_history(
                    st.session_state.chat_history[:-1],
                    user_input
                )
            else:
                search_query = user_input

            # 1.5 HyDE 假设文档生成（可选）
            hyde_applied = False
            if use_hyde:
                st.caption("🤖 **Agent 动作**：正在生成假设排障文档以提升检索精度...")
                hyde_doc = generate_hypothetical_doc(search_query)
                if hyde_doc != search_query:
                    search_query = hyde_doc
                    hyde_applied = True

            # 2. 日志意图检测
            llm_with_tools = llm.bind_tools([fetch_live_error_log])
            intent_check = llm_with_tools.invoke([
                ("system", "你是一个运维 Agent。请判断用户是否需要查看当下的实时日志。如果是，请立即调用 fetch_live_error_log 工具。"),
                ("human", user_input)
            ])

            live_log_context = ""
            if intent_check.tool_calls:
                for tool_call in intent_check.tool_calls:
                    if tool_call["name"] == "fetch_live_error_log":
                        live_log_context = fetch_live_error_log.invoke(tool_call["args"])
                        st.session_state.msg_counter += 1
                        log_msg = "📋 **实时日志捕获:**\n```\n" + live_log_context + "\n```"
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": log_msg,
                            "diagnostics": None,
                            "msg_id": f"a{st.session_state.msg_counter}",
                        })

            final_search_query = live_log_context if live_log_context else search_query

            if use_tag_filter:
                query_tags = extract_query_tags(final_search_query)
                if query_tags:
                    st.caption(f"🏷️ 提取到查询标签: `{', '.join(query_tags)}`")

            # 3. 向量检索
            tool_output, diagnostics = search_knowledge_base_logic(
                query=final_search_query,
                top_k=retrieval_top_k,
                score_threshold=retrieval_score_threshold,
                use_tag_filter=use_tag_filter,
                use_expansion=use_query_expansion,
                use_rerank=use_llm_rerank,
                use_rrf=use_rrf,
            )
            diagnostics["rewritten_query"] = search_query if search_query != user_input else ""
            diagnostics["hyde_applied"] = hyde_applied

        # 4. LLM 回答
        if "知识库中未找到相关排障记录" in tool_output:
            answer = (
                "❌ 抱歉，当前内部知识库中未检索到任何与该故障相关的相似历史案例。\n\n"
                "**💡 建议行动：** 本系统为严格防御模式，已拦截大模型的公开通识幻觉。"
                "请前往相关平台手动排查，或在解决后将该案例录入知识库。"
            )
        else:
            history_context = ""
            if is_contextual and len(st.session_state.chat_history) > 1:
                recent = st.session_state.chat_history[-6:]
                history_context = "【近期对话历史】\n" + "\n".join(
                    f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:200]}"
                    for m in recent[:-1]
                ) + "\n\n"

            strict_prompt = (
                "你是一个极度严谨的内部运维专家助手。你的回答必须【完全基于】下方提供的【内部故障库参考资料】。\n"
                "⚠️ 铁律：\n"
                "1. 严格禁止使用你自身对公开网络、开源软件的通用常识来脑补、扩充或捏造解决方案。\n"
                "2. 如果参考资料里的方案不完整，请直接基于资料原样陈述，绝对不允许发明任何内部系统的参数、命令或流程。\n"
                "3. 必须保持回答的真实性，拒绝任何幻觉。\n\n"
                f"{history_context}"
                f"用户当前请求: {user_input}\n"
                f"当前实时日志线索: {live_log_context if live_log_context else '未提取到实时日志'}\n"
                f"【唯一合法的内部故障库参考资料】:\n{tool_output}\n\n"
                "请根据上述合法资料，为用户梳理并总结出标准修复建议："
            )

            answer = ""
            for chunk in llm.stream(strict_prompt):
                answer += chunk.content

        # 5. 记录助手回答
        st.session_state.msg_counter += 1
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "diagnostics": diagnostics,
            "msg_id": f"a{st.session_state.msg_counter}",
        })

        # 6. 全新模式下，只保留最后一轮
        if not is_contextual:
            st.session_state.chat_history = st.session_state.chat_history[-2:]

        st.rerun()

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