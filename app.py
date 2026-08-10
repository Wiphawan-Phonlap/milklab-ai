"""MilkLab RAG Chatbot (S3).

Run locally: streamlit run app.py
Deploy: push to GitHub then Actions deploys to HuggingFace Space

นักศึกษาต้องเติม TODO 5 จุด ใน Session 3 Lab 2.2
"""

import json
import os
import uuid
from datetime import datetime, timezone

import faiss
import numpy as np
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHAT_MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TRACE_FILE_NAME = "traces.jsonl"


def _trace_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), TRACE_FILE_NAME)


def _new_trace_id() -> str:
    return uuid.uuid4().hex


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_trace_record(record: dict) -> None:
    trace_path = _trace_file_path()
    try:
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _start_span(trace_id: str, span_name: str, attributes: dict | None = None) -> dict:
    return {
        "trace_id": trace_id,
        "span_id": uuid.uuid4().hex,
        "span_name": span_name,
        "parent_span_id": None,
        "started_at": _now_iso(),
        "ended_at": None,
        "duration_ms": None,
        "status": "ok",
        "attributes": attributes or {},
    }


def _finish_span(span_record: dict) -> None:
    ended_at = _now_iso()
    span_record["ended_at"] = ended_at
    started_at = datetime.fromisoformat(span_record["started_at"])
    ended = datetime.fromisoformat(ended_at)
    span_record["duration_ms"] = round(
        (ended - started_at).total_seconds() * 1000, 2)
    _append_trace_record(span_record)


def _split_text_to_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""
    normalized = "\n".join(line.strip() for line in text.splitlines())
    normalized = "\n".join(line for line in normalized.splitlines() if line)

    chunks: list[str] = []
    start = 0
    text_len = len(normalized)
    if text_len == 0:
        return chunks

    step = max(1, chunk_size - overlap)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


@st.cache_resource
def load_index():
    """TODO 1+2+3: โหลด menu_kb.md, split เป็น chunk, encode ด้วย sentence-transformers,
    สร้าง faiss index. Cache เพราะโหลด model ครั้งแรกใช้เวลา 30 วินาที

    Returns: (model, index, chunks_list)
    """
    kb_path = os.path.join(os.path.dirname(__file__), "menu_kb.md")
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_text = f.read()

    chunks = _split_text_to_chunks(kb_text)
    if not chunks:
        raise RuntimeError("menu_kb.md is empty or could not be chunked")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return model, index, chunks


def retrieve_top_k(
    query: str, model, index, chunks: list[str], k: int = 3, trace_id: str | None = None
) -> tuple[list[str], dict]:
    """TODO 4: encode query, search index, return top-k chunks"""
    span_record = _start_span(
        trace_id or _new_trace_id(),
        "retrieve_top_k",
        {"query": query, "k": k, "chunk_count": len(chunks)},
    )
    try:
        query_vector = model.encode([query], convert_to_numpy=True)
        query_vector = np.asarray(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_vector)

        top_k = min(k, len(chunks))
        _, indices = index.search(query_vector, top_k)
        result = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
        span_record["attributes"]["result_count"] = len(result)
        return result, span_record
    except Exception as exc:
        span_record["status"] = "error"
        span_record["attributes"]["error"] = str(exc)
        raise
    finally:
        _finish_span(span_record)


def generate_answer(
    query: str, context_chunks: list[str], trace_id: str | None = None
) -> tuple[str, dict]:
    """TODO 5: ส่ง query + context ไป Gemini, return answer

    Hint: build prompt that says "ตอบจากข้อมูลต่อไปนี้เท่านั้น ถ้าไม่มีใน context ให้บอกว่าไม่รู้"
    """
    span_record = _start_span(
        trace_id or _new_trace_id(),
        "generate_answer",
        {"query": query, "context_count": len(context_chunks)},
    )
    answer = ""

    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            span_record["status"] = "error"
            span_record["attributes"]["error"] = "missing GOOGLE_API_KEY"
            answer = "ไม่พบ GOOGLE_API_KEY ใน environment"
            span_record["attributes"]["answer_preview"] = answer
            return answer, span_record

        if not context_chunks:
            answer = "ไม่พบข้อมูลที่เกี่ยวข้องในคลังความรู้"
            span_record["attributes"]["answer_preview"] = answer
            return answer, span_record

        context_text = "\n\n".join(
            f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks, 1))
        prompt = f"""คุณคือผู้ช่วยของร้าน MilkLab°
ตอบคำถามโดยอ้างอิงจากข้อมูลใน CONTEXT เท่านั้น
ถ้าข้อมูลไม่พอหรือไม่มีใน CONTEXT ให้ตอบว่า "ไม่ทราบจากข้อมูลที่มี"

CONTEXT:
{context_text}

QUESTION:
{query}

คำตอบภาษาไทยแบบกระชับ:"""

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=CHAT_MODEL_NAME, contents=prompt)
        answer = (response.text or "ไม่ทราบจากข้อมูลที่มี").strip()
        span_record["attributes"]["answer_preview"] = answer[:240]
        return answer, span_record
    except Exception as exc:
        span_record["status"] = "error"
        span_record["attributes"]["error"] = str(exc)
        raise
    finally:
        _finish_span(span_record)


def main():
    st.set_page_config(page_title="MilkLab° RAG", page_icon="🥛")
    st.title("MilkLab° RAG Chatbot")
    st.caption("ถามอะไรเกี่ยวกับ MilkLab ได้ ตอบจาก menu_kb.md")

    try:
        model, index, chunks = load_index()
    except NotImplementedError as exc:
        st.error(f"TODO not implemented: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("ถามอะไรเกี่ยวกับ MilkLab"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นข้อมูล..."):
                trace_id = _new_trace_id()
                context, retrieve_span = retrieve_top_k(
                    prompt, model, index, chunks, trace_id=trace_id)
                answer, generate_span = generate_answer(
                    prompt, context, trace_id=trace_id)
            st.write(answer)
            with st.expander("Trace"):
                st.json({
                    "trace_id": trace_id,
                    "spans": [retrieve_span, generate_span],
                })
            with st.expander("Source chunks"):
                for i, c in enumerate(context, 1):
                    st.markdown(f"**[{i}]** {c}")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
