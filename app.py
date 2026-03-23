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

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QA Test Case Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
*, html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #F7F6F3; }
.app-header {
    background: #1A1A2E; border-radius: 0 0 24px 24px;
    padding: 2.5rem 2rem 2rem; margin: -1rem -1rem 2rem -1rem;
    position: relative; overflow: hidden;
}
.app-header::after {
    content: '🧪'; position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%); font-size: 5rem; opacity: 0.08;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 2rem;
    font-weight: 700; color: #E8F4FD; margin: 0; letter-spacing: -0.5px;
}
.app-sub { color: #7B8FA1; font-size: 0.9rem; margin-top: 0.4rem; font-weight: 300; }
.tag {
    display: inline-block; background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12); color: #A8C8E8;
    padding: 0.15rem 0.65rem; border-radius: 4px; font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace; margin: 0.3rem 0.2rem 0 0;
}
.card-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    color: #9B8EA0; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;
}
.result-preview {
    background: #1A1A2E; border-radius: 8px; padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #A8C8E8;
    white-space: pre-wrap; max-height: 220px; overflow-y: auto; margin-top: 0.5rem;
}
.metric-strip { display: flex; gap: 1rem; margin: 1rem 0; }
.metric {
    flex: 1; background: #FFFFFF; border: 1px solid #E8E5DF;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #1A1A2E; }
.metric-lbl { font-size: 0.7rem; color: #9B8EA0; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }
.success-box {
    background: #E8F5E9; border: 1px solid #A5D6A7; border-left: 4px solid #2E7D32;
    border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1rem 0;
    color: #1B5E20; font-weight: 500;
}
.rag-status-loaded {
    background: #E8F5E9; border: 1px solid #A5D6A7; border-left: 3px solid #2E7D32;
    border-radius: 6px; padding: 0.6rem 0.8rem; font-size: 0.8rem;
    color: #1B5E20; margin-top: 0.5rem;
}
.rag-status-empty {
    background: #FFF3CD; border: 1px solid #FFD580; border-left: 3px solid #F59E0B;
    border-radius: 6px; padding: 0.6rem 0.8rem; font-size: 0.8rem;
    color: #92400E; margin-top: 0.5rem;
}
.stButton > button {
    background: #1A1A2E !important; color: #E8F4FD !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important;
    font-weight: 600 !important; letter-spacing: 1px !important;
    padding: 0.65rem 1.5rem !important; width: 100% !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #2D2D4E !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(26,26,46,0.25) !important;
}
.stButton > button:disabled { background: #C8C5BF !important; color: #8A8580 !important; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E8E5DF; }
.sidebar-section {
    background: #F7F6F3; border: 1px solid #E8E5DF;
    border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
}
.sidebar-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    text-transform: uppercase; letter-spacing: 2px; color: #9B8EA0; margin-bottom: 0.6rem;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# KEY FIX: rag_loaded flag ensures Excel is never re-processed on rerun
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "extracted_text"  : None,
    "parsed_req"      : None,
    "similar_cases"   : None,
    "test_cases"      : None,
    "excel_bytes"     : None,
    "stage"           : 0,
    "tc_count"        : 0,
    "rag_loaded"      : False,   # True once Excel is loaded into ChromaDB
    "rag_count"       : 0,       # How many test cases are in ChromaDB
    "rag_docs"        : [],      # Stores all docs in memory as backup
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES — loaded once, never reloaded
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI model...")
def load_embed_model():
    """Load embedding model once and cache it permanently."""
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

@st.cache_resource(show_spinner="Setting up database...")
def load_chroma():
    """Create ChromaDB client once and cache it permanently."""
    client     = chromadb.Client()
    collection = client.get_or_create_collection("all_test_cases")
    return client, collection


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def call_groq(prompt: str, api_key: str, max_tokens: int = 2000) -> str:
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model      = "llama-3.1-8b-instant",
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = max_tokens
    )
    return response.choices[0].message.content


def run_ocr(image: Image.Image) -> str:
    return pytesseract.image_to_string(
        image.convert("RGB"), config="--psm 6"
    ).strip()


def load_excel_to_rag(excel_bytes: bytes) -> int:
    """
    Load Excel into ChromaDB AND save docs to session_state as backup.
    This ensures RAG data survives Streamlit reruns.
    """
    embed_model         = load_embed_model()
    _, collection       = load_chroma()
    all_docs            = []
    total               = 0

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        if ws.max_row < 3:
            continue

        headers = [
            str(c.value).strip().lower() if c.value else f"col{i}"
            for i, c in enumerate(ws[1])
        ]

        for row in ws.iter_rows(min_row=2, values_only=True):
            if not any(row):
                continue
            row_text = "\n".join(
                f"{headers[i]}: {v}"
                for i, v in enumerate(row)
                if v is not None and i < len(headers)
            )
            if not row_text.strip():
                continue

            all_docs.append(row_text)
            emb = embed_model.encode(row_text).tolist()
            try:
                collection.add(
                    documents  = [row_text],
                    embeddings = [emb],
                    ids        = [f"tc_{sheet_name}_{total}"]
                )
                total += 1
            except Exception:
                pass

    # Save docs to session state as backup
    st.session_state.rag_docs  = all_docs
    st.session_state.rag_count = total
    st.session_state.rag_loaded = True
    return total


def rag_retrieve(query: str) -> str:
    """
    Retrieve similar test cases.
    First tries ChromaDB, falls back to session_state docs if ChromaDB is empty.
    """
    embed_model   = load_embed_model()
    _, collection = load_chroma()

    # If ChromaDB got reset but we have docs in session_state, reload them
    if collection.count() == 0 and st.session_state.rag_docs:
        for i, doc in enumerate(st.session_state.rag_docs):
            emb = embed_model.encode(doc).tolist()
            try:
                collection.add(
                    documents  = [doc],
                    embeddings = [emb],
                    ids        = [f"tc_restored_{i}"]
                )
            except Exception:
                pass

    if collection.count() == 0:
        return "No past test cases loaded."

    emb     = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings = [emb],
        n_results        = min(5, collection.count())
    )
    docs = results["documents"][0] if results["documents"] else []
    return "\n\n---\n\n".join(docs) if docs else "No similar cases found."


def build_excel(test_cases_text: str, feature_name: str) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Test Cases"

    h_fill  = PatternFill("solid", fgColor="1A1A2E")
    a_fill  = PatternFill("solid", fgColor="EBF3FB")
    h_font  = Font(bold=True, color="E8F4FD", name="Calibri", size=11)
    n_font  = Font(name="Calibri", size=10)
    c_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    l_align = Alignment(horizontal="left",   vertical="top",    wrap_text=True)
    thin    = Side(style="thin", color="D0CCC8")
    bdr     = Border(left=thin, right=thin, top=thin, bottom=thin)

    headers = ["TC ID", "Title", "Preconditions", "Test Steps", "Expected Result", "Priority", "Status"]
    widths  = [10, 28, 25, 40, 32, 12, 12]

    for c, (h, w) in enumerate(zip(headers, widths), 1):
        cell           = ws.cell(row=1, column=c, value=h)
        cell.fill      = h_fill
        cell.font      = h_font
        cell.alignment = c_align
        cell.border    = bdr
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[1].height = 30

    lines = [
        l.strip() for l in test_cases_text.split("\n")
        if "|" in l and "---" not in l and l.strip() != "|"
    ]
    for ri, line in enumerate(lines, 2):
        parts = [p.strip() for p in line.strip("|").split("|")]
        while len(parts) < 7:
            parts.append("")
        if not parts[6]:
            parts[6] = "Not Run"
        for ci, val in enumerate(parts[:7], 1):
            cell           = ws.cell(row=ri, column=ci, value=val)
            cell.font      = n_font
            cell.border    = bdr
            cell.alignment = c_align if ci in [1, 6, 7] else l_align
            if ri % 2 == 0:
                cell.fill = a_fill
        ws.row_dimensions[ri].height = 45

    ws2 = wb.create_sheet("Summary")
    for i, (k, v) in enumerate([
        ("Feature",      feature_name),
        ("Total TCs",    f"=COUNTA('Test Cases'!A2:A1000)"),
        ("Generated By", "AI Test Case Generator"),
        ("Model",        "Groq LLaMA 3.1"),
    ], 1):
        ws2.cell(row=i, column=1, value=k).font = Font(bold=True, name="Calibri")
        ws2.cell(row=i, column=2, value=v)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
    <div class="app-title">QA Test Case Generator</div>
    <div class="app-sub">Upload feature images or paste description → Get professional test cases in Excel</div>
    <div style="margin-top:0.8rem">
        <span class="tag">GROQ AI</span>
        <span class="tag">TESSERACT OCR</span>
        <span class="tag">RAG</span>
        <span class="tag">CHROMADB</span>
        <span class="tag">EXCEL EXPORT</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:

    # API Key
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🔑 Groq API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Paste your gsk_... key",
        label_visibility="hidden"
    )
    st.caption("Free key → [console.groq.com](https://console.groq.com)")
    st.markdown('</div>', unsafe_allow_html=True)

    # RAG Upload — KEY FIX: only loads if not already loaded
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Past Test Cases (RAG)</div>', unsafe_allow_html=True)
    st.caption("Upload your master Excel file — works for ALL features")

    past_excel = st.file_uploader(
        "Upload past test cases Excel",
        type=["xlsx", "xls"],
        label_visibility="hidden"
    )

    if past_excel and not st.session_state.rag_loaded:
        # Only loads ONCE — never reloads on rerun
        with st.spinner("Loading test cases into RAG..."):
            count = load_excel_to_rag(past_excel.read())
        st.success(f"✅ {count} test cases loaded")

    elif past_excel and st.session_state.rag_loaded:
        # Already loaded — show status without reloading
        st.markdown(
            f'<div class="rag-status-loaded">'
            f'✅ {st.session_state.rag_count} test cases already in RAG — ready to use'
            f'</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="rag-status-empty">'
            '⚠️ No past test cases uploaded yet. Upload Excel for better results.'
            '</div>',
            unsafe_allow_html=True
        )

    # Clear RAG button
    if st.session_state.rag_loaded:
        if st.button("🗑️ Clear RAG & Upload New File"):
            st.session_state.rag_loaded = False
            st.session_state.rag_count  = 0
            st.session_state.rag_docs   = []
            _, col = load_chroma()
            try:
                col.delete(where={"id": {"$ne": ""}})
            except Exception:
                pass
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🎛️ Settings</div>', unsafe_allow_html=True)
    num_cases    = st.slider("Number of test cases", 5, 45, 15)
    include_neg  = st.toggle("Include negative tests",  value=True)
    include_edge = st.toggle("Include edge cases",       value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Pipeline Status
    st.markdown("---")
    st.markdown('<div class="sidebar-title">📋 Pipeline Status</div>', unsafe_allow_html=True)
    stage = st.session_state.stage or 0
    steps = [
        "🔍 OCR — Read Image",
        "🧠 Parse Requirements",
        "📚 RAG — Find Similar",
        "⚙️ Generate Test Cases",
        "📊 Export to Excel",
    ]
    for i, label in enumerate(steps):
        if i < stage:
            st.markdown(f'<div style="color:#2E7D32;font-size:0.85rem;padding:0.2rem 0;">✅ {label}</div>', unsafe_allow_html=True)
        elif i == stage and stage > 0:
            st.markdown(f'<div style="color:#1565C0;font-size:0.85rem;padding:0.2rem 0;">⏳ {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:#9B8EA0;font-size:0.85rem;padding:0.2rem 0;">○ {label}</div>', unsafe_allow_html=True)


# ── Main Layout ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-label">📤 Feature Input</div>', unsafe_allow_html=True)

    tab_img, tab_txt = st.tabs(["🖼️ Upload Images", "📝 Type / Paste Text"])

    with tab_img:
        images = st.file_uploader(
            "Upload feature images",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            label_visibility="hidden"
        )
        if images:
            for img in images:
                st.image(img, caption=img.name, use_container_width=True)

    with tab_txt:
        manual_text = st.text_area(
            "Feature description",
            height=200,
            placeholder="""Example:
Feature: Add Issue Pattern
User selects a Rule Pattern from dropdown.
Options: Contains, Equals, Starts With, Ends With
User clicks Save. Validation: field cannot be empty.

Templates: incident report, email, PDF, CSV, dashboard.
Include: positive, negative, edge cases.""",
            label_visibility="hidden"
        )

    st.markdown("")

    has_image = bool(images)
    has_text  = bool(manual_text and manual_text.strip())
    has_input = has_image or has_text
    can_run   = has_input and bool(api_key)

    if not api_key:
        st.caption("🔑 Add your Groq API key in the sidebar to enable")
    elif not has_input:
        st.caption("⬆️ Upload an image or type a description above")

    if st.button("⚡ GENERATE TEST CASES", disabled=not can_run):

        # ── Stage 1: OCR ──────────────────────────────────────
        st.session_state.stage = 1
        combined_text = ""

        if has_image:
            with st.spinner("🔍 Extracting text from images..."):
                for img_file in images:
                    img  = Image.open(img_file)
                    text = run_ocr(img)
                    if text:
                        combined_text += f"\n\n[From: {img_file.name}]\n{text}"

        if has_text:
            combined_text += f"\n\n[Manual Description]\n{manual_text.strip()}"

        combined_text = combined_text.strip()
        if not combined_text:
            st.error("❌ Could not extract text. Please add a description in the text tab.")
            st.stop()

        st.session_state.extracted_text = combined_text

        # ── Stage 2: Parse Requirements ───────────────────────
        st.session_state.stage = 2
        with st.spinner("🧠 Parsing requirements with Groq AI..."):
            parse_prompt = f"""You are a senior business analyst.
Read the feature story and extract structured requirements.

Return EXACTLY in this format:
Feature Name: [name]
Description: [brief description]
UI Elements: [list all buttons, dropdowns, fields]
User Actions: [list what user can do]
Expected Behaviors: [list expected outcomes]
Validations: [list all validation rules]
Edge Cases: [list potential edge cases]

Feature Story:
{combined_text}
"""
            parsed = call_groq(parse_prompt, api_key, max_tokens=800)
            st.session_state.parsed_req = parsed

        # ── Stage 3: RAG ──────────────────────────────────────
        st.session_state.stage = 3
        with st.spinner("📚 Searching past test cases..."):
            similar = rag_retrieve(parsed)
            st.session_state.similar_cases = similar

        # ── Stage 4: Generate — 2 fast batches ────────────────
        st.session_state.stage = 4

        if include_neg and include_edge:
            pos  = num_cases // 3 + (num_cases % 3)
            neg  = num_cases // 3
            edge = num_cases - pos - neg
        else:
            pos, neg, edge = num_cases, 0, 0

        with st.spinner(f"⚙️ Generating positive test cases..."):
            prompt1 = f"""You are a senior QA engineer. Generate exactly {pos} POSITIVE test cases.

Requirements:
{parsed}

Reference style — follow this EXACT format from past test cases:
{similar}

Rules:
- Each test MUST have exactly 5 numbered steps
- Format: 1. Navigate to...; 2. Open/Select...; 3. Enter/Click...; 4. Submit...; 5. Verify...
- Return ONLY pipe-delimited rows. No headings. No extra text.

| TC001 | Title | Preconditions | 1. Step; 2. Step; 3. Step; 4. Step; 5. Step | Expected result | High |

Generate exactly {pos} rows starting TC001.
"""
            batch1 = call_groq(prompt1, api_key, max_tokens=2000)

        if neg > 0 or edge > 0:
            with st.spinner(f"⚙️ Generating negative and edge test cases..."):
                start_num = pos + 1
                prompt2   = f"""You are a senior QA engineer.
Generate {neg} NEGATIVE and {edge} EDGE test cases.

Requirements:
{parsed}

NEGATIVE: invalid inputs, unauthorized access, service unavailable, corrupted data.
EDGE: empty fields, boundary values, special chars, leap year, timezone, large datasets.

Rules:
- Each test MUST have exactly 5 numbered steps
- Return ONLY pipe-delimited rows. No headings. No extra text.

| TC{start_num:03d} | Title | Preconditions | 1. Step; 2. Step; 3. Step; 4. Step; 5. Step | Expected | Medium |

Generate exactly {neg + edge} rows starting TC{start_num:03d}.
"""
                batch2     = call_groq(prompt2, api_key, max_tokens=2000)
                test_cases = batch1 + "\n" + batch2
        else:
            test_cases = batch1

        st.session_state.test_cases = test_cases
        lines = [l for l in test_cases.split("\n") if "|" in l and "---" not in l]
        st.session_state.tc_count   = len(lines)

        # ── Stage 5: Excel Export ─────────────────────────────
        st.session_state.stage = 5
        with st.spinner("📊 Building Excel file..."):
            feature_name = "feature"
            if images:
                feature_name = os.path.splitext(images[0].name)[0]
            elif manual_text:
                first_line   = manual_text.strip().split("\n")[0]
                feature_name = first_line[:30].replace("Feature:", "").strip() or "feature"

            excel = build_excel(test_cases, feature_name)
            st.session_state.excel_bytes = excel

        st.rerun()


# ── Results ────────────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card-label">📋 Results</div>', unsafe_allow_html=True)

    if not st.session_state.test_cases:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E8E5DF;border-radius:12px;
                    padding:4rem 2rem;text-align:center;">
            <div style="font-size:3rem">⏳</div>
            <div style="margin-top:0.8rem;font-size:0.9rem;color:#9B8EA0;">
                Results will appear here after generation
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        tc        = st.session_state.tc_count or 0
        chars     = len(st.session_state.extracted_text or "")
        rag_total = st.session_state.rag_count or 0

        st.markdown(f"""
        <div class="success-box">
            ✅ &nbsp; <strong>{tc} test cases generated successfully!</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-strip">
            <div class="metric">
                <div class="metric-val">{tc}</div>
                <div class="metric-lbl">Test Cases</div>
            </div>
            <div class="metric">
                <div class="metric-val">{chars}</div>
                <div class="metric-lbl">Chars Read</div>
            </div>
            <div class="metric">
                <div class="metric-val">{rag_total}</div>
                <div class="metric-lbl">RAG Cases</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["📊 Test Cases", "🔍 OCR Output", "🧠 Requirements"])

        with t1:
            st.markdown(
                f'<div class="result-preview">{st.session_state.test_cases}</div>',
                unsafe_allow_html=True
            )
        with t2:
            st.markdown(
                f'<div class="result-preview">{st.session_state.extracted_text or "—"}</div>',
                unsafe_allow_html=True
            )
        with t3:
            st.markdown(
                f'<div class="result-preview">{st.session_state.parsed_req or "—"}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        fname = "feature"
        if images:
            fname = os.path.splitext(images[0].name)[0]
        elif manual_text:
            fname = manual_text.strip().split("\n")[0][:30].replace("Feature:", "").strip() or "feature"

        st.download_button(
            label     = "⬇️ DOWNLOAD EXCEL TEST CASES",
            data      = st.session_state.excel_bytes,
            file_name = f"{fname}_test_cases.xlsx",
            mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if st.button("🔄 Generate New"):
            for k in ["extracted_text", "parsed_req", "similar_cases",
                      "test_cases", "excel_bytes", "tc_count"]:
                st.session_state[k] = None
            st.session_state.stage = 0
            st.rerun()
