# app.py

import os
import re
import json
import openai
import tiktoken
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from typing import List, Dict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
from zipfile import ZipFile

# ─────────────────────────────
# 1. 설정
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
EMBED_MODEL = "text-embedding-3-small"
GPT_ANALYZER_MODEL = "gpt-5-nano"
GPT_STRUCTURER_MODEL = "gpt-3.5-turbo"
CHUNK_TOKEN_SIZE = 500
MIN_CLUSTER_SIZE = 2
RECOMMEND_TOP_N = 3
OUTPUT_ZIP_PATH = "outputs/summaries.zip"

openai.api_key = OPENAI_API_KEY
for path in ["data", "outputs"]:
    os.makedirs(path, exist_ok=True)

# ─────────────────────────────
# 2. 유틸
def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def load_file(file) -> str:
    ext = os.path.splitext(file.name)[-1].lower()
    if ext == ".pdf":
        return "\n".join([page.extract_text() or "" for page in PdfReader(file).pages])
    return file.read().decode("utf-8")

def split_chunks(text: str, max_tokens: int = CHUNK_TOKEN_SIZE) -> List[str]:
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks, current = [], ""
    for sent in sentences:
        tentative = f"{current} {sent}".strip() if current else sent
        if count_tokens(tentative) <= max_tokens:
            current = tentative
        else:
            if current: chunks.append(current)
            current = sent
    if current: chunks.append(current)
    return chunks

# ─────────────────────────────
# 3. 임베딩
def get_embedding(text: str) -> list[float]:
    return openai.Embedding.create(model=EMBED_MODEL, input=text)['data'][0]['embedding']

def process_and_store_embeddings(chunks: List[str], doc_id: str):
    vectors = [get_embedding(c) for c in chunks]
    avg_vector = [sum(x)/len(x) for x in zip(*vectors)]
    return avg_vector

# ─────────────────────────────
# 4. 클러스터링
def determine_best_k(vectors, k_range=(2, 8)) -> int:
    best_k, best_score = k_range[0], -1
    for k in range(*k_range):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(vectors)
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

def cluster_embeddings(vectors: List[List[float]], doc_ids: List[str], auto_k=True, fixed_k=4) -> Dict[str, int]:
    X = np.array(vectors)
    k = determine_best_k(X) if auto_k else fixed_k
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    return {doc_id: int(cid) for doc_id, cid in zip(doc_ids, labels)}

def merge_small_clusters(cluster_map: Dict[str, int], min_size=MIN_CLUSTER_SIZE) -> Dict[str, int]:
    counter = Counter(cluster_map.values())
    small = [cid for cid, cnt in counter.items() if cnt < min_size]
    major = counter.most_common(1)[0][0]
    return {doc_id: (major if cid in small else cid) for doc_id, cid in cluster_map.items()}

# ─────────────────────────────
# 5. GPT 해석
def summarize_cluster(document_texts: list[str]) -> dict:
    content = "\n\n".join(document_texts)
    sys = """당신은 여러 문서를 분석하여 공통된 의미를 정리하는 정보 분석가입니다.
규칙:
- 사고 과정이나 분석 이유를 절대 설명하지 마세요.
- 개별 문서를 직접 언급하지 마세요.
- 여러 문서에 공통적으로 나타나는 핵심 의미만 추출하세요.
- 간결하고 명확하게 작성하세요.
- 출력은 반드시 JSON 형식만 사용하세요."""
    user = f"""
아래는 동일한 의미적 클러스터에 속한 여러 문서의 내용입니다.

문서 내용:
{content}

작업 지시:
1. 이 문서 묶음을 대표하는 클러스터 주제를 하나 생성하세요. (최대 12단어)
2. 클러스터 전체를 요약하는 문장을 3~5문장으로 작성하세요.
3. 클러스터를 가장 잘 설명하는 핵심 키워드 5~8개를 추출하세요.

출력 형식 (JSON만):
{{
  "cluster_topic": "",
  "cluster_summary": "",
  "keywords": []
}}
"""
    res = openai.ChatCompletion.create(
        model=GPT_ANALYZER_MODEL,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.4
    )
    return json.loads(res.choices[0].message.content)

# ─────────────────────────────
# 6. 추천
def recommend_by_cosine(new_vec, existing_vecs, top_n=RECOMMEND_TOP_N):
    sims = cosine_similarity([new_vec], existing_vecs)[0]
    top_idxs = np.argsort(sims)[::-1][:top_n]
    return top_idxs, sims[top_idxs]

def explain_document_similarity(target_doc: str, related_docs: list[tuple[str, str]]) -> dict:
    rel_txt = "\n".join([f"- {doc_id}: {text}" for doc_id, text in related_docs])
    user_prompt = f"""기준 문서:
{target_doc}

연관 문서 목록:
{rel_txt}

작업 지시:
각 연관 문서가 기준 문서와 왜 함께 읽으면 좋은지 한 문장으로 설명하세요.

출력 형식 (JSON만):
{{
  "recommendations": [
    {{
      "document_id": "",
      "reason": ""
    }}
  ]
}}"""
    sys = """당신은 문서 간의 공통 주제를 간단히 설명하는 분석가입니다.
규칙:
- 문서 내용을 요약하거나 재작성하지 마세요.
- 공통된 주제 또는 관점만 한 문장으로 설명하세요.
- 출력은 반드시 JSON 형식만 사용하세요."""
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user_prompt}],
        temperature=0.3
    )
    return json.loads(res.choices[0].message.content)

# ─────────────────────────────
# 7. Streamlit UI
st.set_page_config(page_title="📄 문서 분석", layout="wide")
st.title("📄 문서 의미 분석 및 추천 플랫폼")

if "doc_texts" not in st.session_state:
    st.session_state.doc_texts = {}
    st.session_state.doc_vectors = {}

uploaded_files = st.file_uploader("문서를 업로드하세요 (.pdf, .txt, .md)", type=["pdf", "txt", "md"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("문서 업로드 및 임베딩 처리 중..."):
        progress = st.progress(0)
        for i, file in enumerate(uploaded_files):
            doc_id = file.name
            st.markdown(f"📄 `{doc_id}` 처리 중...")
            text = load_file(file)
            chunks = split_chunks(text)
            avg_vec = process_and_store_embeddings(chunks, doc_id)
            st.session_state.doc_texts[doc_id] = text
            st.session_state.doc_vectors[doc_id] = avg_vec
            progress.progress((i + 1) / len(uploaded_files))
        st.success("✅ 문서 처리 완료!")

if st.button("🚀 의미 기반 분석 실행"):
    with st.spinner("클러스터링 및 GPT 요약 중..."):
        doc_ids = list(st.session_state.doc_vectors.keys())
        vectors = list(st.session_state.doc_vectors.values())

        cluster_map = cluster_embeddings(vectors, doc_ids)
        cluster_map = merge_small_clusters(cluster_map)

        clusters = defaultdict(list)
        for doc_id, cid in cluster_map.items():
            clusters[cid].append(doc_id)

        summaries = {}
        for cid, doc_list in clusters.items():
            texts = [st.session_state.doc_texts[d] for d in doc_list]
            summary = summarize_cluster(texts)
            summaries[cid] = summary
            with st.expander(f"📁 클러스터 {cid}"):
                st.write(f"📌 주제: **{summary['cluster_topic']}**")
                st.info(summary['cluster_summary'])
                st.write("🔑 키워드: " + ", ".join([f"`{kw}`" for kw in summary["keywords"]]))
                st.write("📄 문서:")
                for doc in doc_list:
                    st.markdown(f"- {doc}")

        st.subheader("📚 유사 문서 추천")
        if len(doc_ids) > 1:
            target = doc_ids[0]
            target_vec = st.session_state.doc_vectors[target]
            others = [v for i, v in enumerate(vectors) if doc_ids[i] != target]
            other_ids = [d for d in doc_ids if d != target]
            top_idxs, _ = recommend_by_cosine(target_vec, others)
            top_docs = [other_ids[i] for i in top_idxs]
            related = [(d, st.session_state.doc_texts[d]) for d in top_docs]
            reasons = explain_document_similarity(st.session_state.doc_texts[target], related)
            for r in reasons["recommendations"]:
                st.markdown(f"🔗 **{r['document_id']}**: {r['reason']}")

        with ZipFile(OUTPUT_ZIP_PATH, "w") as zipf:
            for cid, data in summaries.items():
                md = f"# 클러스터 {cid}\n\n"
                md += f"**주제:** {data['cluster_topic']}\n\n"
                md += f"**요약:**\n{data['cluster_summary']}\n\n"
                md += f"**키워드:** {', '.join(data['keywords'])}\n"
                zipf.writestr(f"cluster_{cid}.md", md)

        with open(OUTPUT_ZIP_PATH, "rb") as f:
            st.download_button("📦 요약 결과 ZIP 다운로드", f, file_name="summaries.zip")
