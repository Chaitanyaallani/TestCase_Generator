import streamlit as st
from groq import Groq
import pytesseract
from PIL import Image
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import chromadb
from sentence_transformers import SentenceTransformer
import io
import os

st.set_page_config(page_title="QA Test Case Generator", layout="wide")

# ---------------- SESSION ----------------
defaults = {
    "extracted_text": None,
    "parsed_req": None,
    "similar_cases": None,
    "test_cases": None,
    "excel_bytes": None,
    "stage": 0,
    "tc_count": 0,
    "rag_loaded": False,
    "rag_count": 0,
    "rag_docs": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- CACHE ----------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("all_test_cases")
    return client, collection

# ---------------- HELPERS ----------------
def call_groq(prompt, api_key):
    try:
        client = Groq(api_key=api_key)
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content
    except Exception as e:
        st.error(f"Groq Error: {e}")
        return ""

def run_ocr(img):
    return pytesseract.image_to_string(img).strip()

def load_excel_to_rag(excel_bytes):
    embed_model = load_embed_model()
    _, col = load_chroma()
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))

    total = 0
    docs = []

    for s in wb.sheetnames:
        ws = wb[s]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not any(row):
                continue
            txt = " ".join([str(v) for v in row if v])
            docs.append(txt)
            emb = embed_model.encode(txt).tolist()
            try:
                col.add(documents=[txt], embeddings=[emb], ids=[f"id_{total}"])
                total += 1
            except:
                pass

    st.session_state.rag_docs = docs
    st.session_state.rag_count = total
    st.session_state.rag_loaded = True
    return total

def rag_retrieve(query):
    embed_model = load_embed_model()
    _, col = load_chroma()

    if col.count() == 0:
        return "No past test cases."

    emb = embed_model.encode(query).tolist()
    res = col.query(query_embeddings=[emb], n_results=5)
    docs = res["documents"][0] if res["documents"] else []
    return "\n".join(docs)

# ---------------- EXCEL ----------------
def build_excel(test_cases):
    wb = openpyxl.Workbook()
    ws = wb.active

    headers = ["TC ID", "Title", "Preconditions", "Steps", "Expected", "Priority", "Status"]

    for i, h in enumerate(headers, 1):
        ws.cell(row=1, column=i, value=h)

    row_num = 2

    for line in test_cases.split("\n"):
        if "|" not in line:
            continue

        parts = [p.strip() for p in line.strip("|").split("|")]

        # FIX: ensure 6 cols + add Status
        while len(parts) < 6:
            parts.append("")
        parts = parts[:6]
        parts.append("Not Run")

        for col_num, val in enumerate(parts, 1):
            ws.cell(row=row_num, column=col_num, value=val)

        row_num += 1

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

# ---------------- UI ----------------
st.title("QA Test Case Generator")

api_key = st.text_input("Groq API Key", type="password")

uploaded_excel = st.file_uploader("Upload past test cases")

if uploaded_excel and not st.session_state.rag_loaded:
    load_excel_to_rag(uploaded_excel.read())

text = st.text_area("Enter feature description")

num_cases = st.slider("Test cases", 5, 30, 10)

if st.button("Generate"):

    parsed = call_groq(f"Summarize requirements:\n{text}", api_key)

    similar = rag_retrieve(parsed)

    pos = num_cases

    # -------- PROMPT 1 --------
    prompt1 = f"""You are a senior QA engineer. Generate exactly {pos} POSITIVE test cases.

Requirements:
{parsed}

Reference style:
{similar}

CRITICAL RULES — STRICTLY FOLLOW:
- EXACTLY 6 columns
- Format strictly pipe separated
- 5 steps mandatory

| TC001 | Title | Preconditions | 1. Step;2;3;4;5 | Expected | High |

Generate exactly {pos} rows.
"""

    batch1 = call_groq(prompt1, api_key)

    # -------- CLEAN OUTPUT --------
    lines = []
    for l in batch1.split("\n"):
        if "|" in l:
            parts = l.strip("|").split("|")
            if len(parts) >= 6:
                lines.append(l)

    final_text = "\n".join(lines)

    excel = build_excel(final_text)

    st.session_state.test_cases = final_text
    st.session_state.excel_bytes = excel

# ---------------- OUTPUT ----------------
if st.session_state.test_cases:
    st.text_area("Generated", st.session_state.test_cases, height=300)

    st.download_button(
        "Download Excel",
        st.session_state.excel_bytes,
        file_name="test_cases.xlsx"
    )
