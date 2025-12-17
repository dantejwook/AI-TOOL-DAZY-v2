import streamlit as st
import time
import zipfile
import os
from pathlib import Path
import openai

# ----------------------------
# 🌈 기본 페이지 설정
# ----------------------------
st.set_page_config(page_title="AI dazy document sorter (Fast Edition)", page_icon="🗂️", layout="wide")

# ----------------------------
# 🔐 OpenAI API Key 자동 감지 및 캐싱
# ----------------------------
@st.cache_data(show_spinner=False)
def get_openai_key():
    return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

openai.api_key = get_openai_key()

if not openai.api_key:
    st.sidebar.error("🚨 OpenAI API Key가 없습니다. .streamlit/secrets.toml 또는 환경변수를 확인하세요.")
else:
    st.sidebar.success("✅ OpenAI API Key 로드 완료")

# ----------------------------
# 🎨 스타일 커스터마이징
# ----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fc;
        font-family: 'Pretendard', sans-serif;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #4a6cf7;
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #3451c1;
    }
    .status-bar {
        background-color: #e9ecef;
        border-radius: 6px;
        padding: 0.5em;
        margin-top: 20px;
        font-size: 0.9em;
    }
    .log-box {
        background-color: #fff;
        border-radius: 6px;
        padding: 0.8em;
        margin-top: 10px;
        height: 120px;
        overflow-y: auto;
        font-size: 0.85em;
        border: 1px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# 🧭 사이드바 설정
# ----------------------------
st.sidebar.title("⚙️ 설정")
if st.sidebar.button("🔁 다시 시작"):
    st.rerun()

lang = st.sidebar.selectbox("🌐 언어 선택", ["한국어", "English"])

# ----------------------------
# 📁 메인 UI 구성
# ----------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📤 파일 업로드")
    uploaded_files = st.file_uploader(
        "문서를 업로드하세요 (.md, .pdf, .txt)",
        accept_multiple_files=True,
        type=["md", "pdf", "txt"],
    )

with right_col:
    st.subheader("📦 ZIP 다운로드")
    zip_placeholder = st.empty()

# ----------------------------
# ⚙️ 상태 표시 + 로그 관리
# ----------------------------
status_placeholder = st.empty()
log_box = st.empty()
log_messages = []

def log(msg):
    log_messages.append(msg)
    log_html = "<div class='log-box'>" + "<br>".join(log_messages[-10:]) + "</div>"
    log_box.markdown(log_html, unsafe_allow_html=True)

# ----------------------------
# 💾 ZIP 생성 (캐시 적용)
# ----------------------------
@st.cache_resource
def create_zip(files, output_dir):
    zip_filename = "result_documents.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for file in files:
            file_path = output_dir / file.name
            zipf.write(file_path, arcname=file_path.name)
    return zip_filename

# ----------------------------
# 🚀 메인 로직
# ----------------------------
if uploaded_files:
    log("파일 업로드 완료 ✅")
    total = len(uploaded_files)
    output_dir = Path("output_docs")
    output_dir.mkdir(exist_ok=True)

    with st.spinner("⚙️ 문서를 정리 중입니다... 잠시만 기다려주세요."):
        for i, file in enumerate(uploaded_files, start=1):
            file_path = output_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())
            progress = int((i / total) * 100)
            status_placeholder.markdown(f"<div class='status-bar'>[{progress}% processing ({i}/{total} complete)]</div>", unsafe_allow_html=True)
            log(f"📄 문서 처리 중: {file.name}")

        # ZIP 파일 생성
        zip_filename = create_zip(uploaded_files, output_dir)

    with open(zip_filename, "rb") as f:
        zip_placeholder.download_button(
            label="📥 정리된 ZIP 파일 다운로드",
            data=f,
            file_name=zip_filename,
            mime="application/zip",
        )

    log("✅ 모든 파일이 성공적으로 정리되었습니다.")
    status_placeholder.markdown(
        f"<div class='status-bar'>[100% complete – 모든 문서 정리 완료]</div>",
        unsafe_allow_html=True,
    )

else:
    status_placeholder.markdown(
        "<div class='status-bar'>[0% processing (0/0 complete)]</div>",
        unsafe_allow_html=True,
    )
    log_box.markdown("<div class='log-box'>대기 중...</div>", unsafe_allow_html=True)
