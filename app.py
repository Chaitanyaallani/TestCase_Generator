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
    white-space: pre-wrap; max-height: 280px; overflow-y: auto; margin-top: 0.5rem;
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
.step-box {
    background: #EBF3FB; border: 1px solid #90CAF9; border-left: 4px solid #1565C0;
    border-radius: 8px; padding: 0.8rem 1.2rem; margin: 0.4rem 0;
    color: #0D47A1; font-size: 0.85rem;
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
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "extracted_text"    : None,
    "feature_understand": None,
    "coverage_index"    : None,
    "mapping_report"    : None,
    "test_cases_raw"    : None,
    "excel_bytes"       : None,
    "stage"             : 0,
    "tc_count"          : 0,
    "pos_count"         : 0,
    "neg_count"         : 0,
    "edge_count"        : 0,
    "rag_loaded"        : False,
    "rag_count"         : 0,
    "rag_docs"          : [],
    "last_tc_id"        : 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI model...")
def load_embed_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

@st.cache_resource(show_spinner="Setting up database...")
def load_chroma():
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


def load_excel_to_rag(excel_bytes: bytes) -> tuple:
    """
    Load Excel into ChromaDB AND extract last TC ID for auto-increment.
    Returns (total_loaded, last_tc_id)
    """
    embed_model   = load_embed_model()
    _, collection = load_chroma()
    all_docs      = []
    total         = 0
    last_tc_id    = 3000  # default start

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

            # Extract last TC ID for auto-increment
            tc_id_val = str(row[0]) if row[0] else ""
            nums = [int(s) for s in tc_id_val.split("-") if s.isdigit()]
            if nums:
                last_tc_id = max(last_tc_id, max(nums))

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

    st.session_state.rag_docs   = all_docs
    st.session_state.rag_count  = total
    st.session_state.rag_loaded = True
    st.session_state.last_tc_id = last_tc_id
    return total, last_tc_id


def rag_retrieve(query: str) -> str:
    embed_model   = load_embed_model()
    _, collection = load_chroma()

    if collection.count() == 0 and st.session_state.rag_docs:
        for i, doc in enumerate(st.session_state.rag_docs):
            emb = embed_model.encode(doc).tolist()
            try:
                collection.add(documents=[doc], embeddings=[emb], ids=[f"tc_restored_{i}"])
            except Exception:
                pass

    if collection.count() == 0:
        return "No past test cases loaded."

    emb     = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[emb], n_results=min(5, collection.count()))
    docs    = results["documents"][0] if results["documents"] else []
    return "\n\n---\n\n".join(docs) if docs else "No similar cases found."


def build_excel(test_cases_text: str, feature_name: str,
                pos_count: int, neg_count: int, edge_count: int) -> bytes:
    """
    Build Excel with Copilot-style formatting:
    - Summary sheet with counts
    - Test case sheet with color coding per type
    - Green = Positive, Red/Orange = Negative, Yellow = Edge
    """
    wb = openpyxl.Workbook()

    # ── SUMMARY SHEET ─────────────────────────────────────────
    ws_sum            = wb.active
    ws_sum.title      = "SUMMARY"
    sum_headers       = ["Feature / Module Name", "Total Test Cases",
                         "Positive Count", "Negative Count", "Edge Case Count",
                         "Acceptance Criteria Covered", "Gaps Identified"]
    h_fill = PatternFill("solid", fgColor="1A1A2E")
    h_font = Font(bold=True, color="E8F4FD", name="Calibri", size=11)

    for c, h in enumerate(sum_headers, 1):
        cell           = ws_sum.cell(row=1, column=c, value=h)
        cell.fill      = h_fill
        cell.font      = h_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        ws_sum.column_dimensions[get_column_letter(c)].width = 22

    total = pos_count + neg_count + edge_count
    ws_sum.cell(row=2, column=1, value=feature_name)
    ws_sum.cell(row=2, column=2, value=total)
    ws_sum.cell(row=2, column=3, value=pos_count)
    ws_sum.cell(row=2, column=4, value=neg_count)
    ws_sum.cell(row=2, column=5, value=edge_count)
    ws_sum.cell(row=2, column=6, value="Extracted from feature input")
    ws_sum.cell(row=2, column=7, value="Legacy system integration, Performance testing")
    ws_sum.freeze_panes = "A2"

    # ── TEST CASE SHEET ────────────────────────────────────────
    ws = wb.create_sheet(title=feature_name[:31])

    tc_headers = ["Test Case ID", "Test Case Title", "Prerequisites",
                  "Test Steps", "Expected Result", "Test Type", "Priority", "Related TC ID"]
    widths     = [14, 35, 30, 50, 40, 12, 10, 14]

    # Color fills per type
    pos_fill  = PatternFill("solid", fgColor="E2EFDA")  # light green
    neg_fill  = PatternFill("solid", fgColor="FCE4D6")  # light orange/red
    edge_fill = PatternFill("solid", fgColor="FFF2CC")  # light yellow
    alt_pos   = PatternFill("solid", fgColor="D5E8C8")
    alt_neg   = PatternFill("solid", fgColor="F4D0BC")
    alt_edge  = PatternFill("solid", fgColor="F5E6AA")

    n_font  = Font(name="Calibri", size=10)
    c_align = Alignment(horizontal="center", vertical="top", wrap_text=True)
    l_align = Alignment(horizontal="left",   vertical="top", wrap_text=True)
    thin    = Side(style="thin", color="D0CCC8")
    bdr     = Border(left=thin, right=thin, top=thin, bottom=thin)

    for c, (h, w) in enumerate(zip(tc_headers, widths), 1):
        cell           = ws.cell(row=1, column=c, value=h)
        cell.fill      = h_fill
        cell.font      = h_font
        cell.alignment = c_align
        cell.border    = bdr
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"

    # Parse pipe-delimited rows
    lines = [
        l.strip() for l in test_cases_text.split("\n")
        if "|" in l and "---" not in l and l.strip() != "|"
    ]

    for ri, line in enumerate(lines, 2):
        parts = [p.strip() for p in line.strip("|").split("|")]
        while len(parts) < 8:
            parts.append("")
        if not parts[7]:
            parts[7] = "None"

        # Determine fill color based on Test Type (column 6, index 5)
        tc_type = parts[5].strip().lower() if len(parts) > 5 else ""
        if "negative" in tc_type or "neg" in tc_type:
            fill = neg_fill if ri % 2 == 0 else alt_neg
        elif "edge" in tc_type:
            fill = edge_fill if ri % 2 == 0 else alt_edge
        else:
            fill = pos_fill if ri % 2 == 0 else alt_pos

        for ci, val in enumerate(parts[:8], 1):
            cell           = ws.cell(row=ri, column=ci, value=val)
            cell.font      = n_font
            cell.border    = bdr
            cell.fill      = fill
            cell.alignment = c_align if ci in [1, 6, 7, 8] else l_align
        ws.row_dimensions[ri].height = 60

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS — following Copilot's 8-step process
# ══════════════════════════════════════════════════════════════════════════════

def step1_parse_input(combined_text: str, api_key: str) -> str:
    """Step 1: Parse input and build Feature Understanding."""
    prompt = f"""You are a Senior QA Architect.
Parse the feature input below and extract a complete FEATURE UNDERSTANDING.

Return EXACTLY in this format:
Feature Name        : [name]
User Roles          : [list all roles: Admin, User, Guest etc]
Screens / Pages     : [list all screens mentioned]
UI Components       : [list all buttons, fields, dropdowns, tabs]
User Flows          : [numbered list of all user flows]
Acceptance Criteria : [numbered list of all acceptance criteria]
Business Rules      : [constraints, validations, permissions]
Error / Edge States : [empty states, error messages, boundary conditions]
Out of Scope        : [anything explicitly excluded]

Feature Input:
{combined_text}
"""
    return call_groq(prompt, api_key, max_tokens=1000)


def step2_build_coverage_index(similar_cases: str, api_key: str) -> str:
    """Step 2: Build coverage index from existing test cases."""
    if not similar_cases or similar_cases == "No past test cases loaded.":
        return "No existing coverage found. All test cases will be new."

    prompt = f"""You are a Senior QA Architect.
Analyse these existing test cases and build a COVERAGE INDEX.

List what is already covered and identify GAP AREAS that need new test cases.

Format:
COVERAGE INDEX:
[List existing TCs with ID and title]

GAP AREAS:
[List flows, screens, or scenarios NOT yet covered]

Existing Test Cases:
{similar_cases}
"""
    return call_groq(prompt, api_key, max_tokens=800)


def step3_generate_test_cases(feature_understanding: str, coverage_index: str,
                               similar_cases: str, num_cases: int,
                               last_tc_id: int, api_key: str,
                               include_neg: bool, include_edge: bool) -> str:
    """Steps 3-5: Generate test cases following Copilot's exact rules."""

    # Calculate split
    if include_neg and include_edge:
        pos  = num_cases // 3 + (num_cases % 3)
        neg  = num_cases // 3
        edge = num_cases - pos - neg
    elif include_neg:
        pos  = num_cases // 2 + (num_cases % 2)
        neg  = num_cases - pos
        edge = 0
    elif include_edge:
        pos  = num_cases // 2 + (num_cases % 2)
        neg  = 0
        edge = num_cases - pos
    else:
        pos, neg, edge = num_cases, 0, 0

    start_id = last_tc_id + 1

    prompt = f"""You are a Senior QA Architect generating enterprise-level UI test cases.

FEATURE UNDERSTANDING:
{feature_understanding}

EXISTING COVERAGE (do NOT duplicate these):
{coverage_index}

SIMILAR PAST TEST CASES (match this exact style and quality):
{similar_cases}

GENERATION RULES — STRICTLY FOLLOW:
1. Generate {pos} Positive + {neg} Negative + {edge} Edge = {num_cases} total test cases
2. TC IDs must start from TC-{start_id:04d} and auto-increment
3. Prerequisites MUST be specific: "User: [role] logged in, [Module] accessible, [Test data] available"
4. Test Steps MUST have exactly 5 numbered steps — each step is ONE UI action:
   1. Navigate to [specific module/URL]
   2. [Click/Enter/Select] [exact element name] 
   3. [Click/Enter/Select] [exact element name]
   4. [Click/Submit/Confirm] [exact button name]
   5. Verify [exact visible UI outcome]
5. Expected Result MUST describe EXACTLY what the user sees — specific screen name, message text, element state
6. Test Type MUST be exactly: Positive / Negative / Edge
7. Priority: High = core flows, Medium = validations, Low = edge/boundary cases
8. Related TC ID: reference existing TC if extending, else "None"
9. DO NOT generate vague steps like "check if it works" or "verify it saves"
10. DO NOT duplicate test cases from the existing coverage index

FORBIDDEN phrases in Expected Result:
- "works correctly"
- "saves successfully" 
- "system behaves as expected"

MANDATORY in Expected Result:
- Exact screen name shown to user
- Exact message text displayed
- Exact element state change (button enabled/disabled, field highlighted etc)

Return ONLY pipe-delimited rows. NO headers. NO extra text. NO markdown.
EXACT FORMAT — 8 columns per row:
| TC-{start_id:04d} | [Clear action-oriented title] | [Specific prerequisites] | 1. [Step]\n2. [Step]\n3. [Step]\n4. [Step]\n5. [Step] | [Exact visible UI outcome] | Positive | High | None |

Generate exactly {num_cases} rows.
"""
    return call_groq(prompt, api_key, max_tokens=4000)


# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
    <div class="app-title">QA Test Case Generator</div>
    <div class="app-sub">Enterprise-grade test cases — powered by the same logic as GitHub Copilot</div>
    <div style="margin-top:0.8rem">
        <span class="tag">GROQ AI</span>
        <span class="tag">OCR</span>
        <span class="tag">RAG</span>
        <span class="tag">8-STEP PIPELINE</span>
        <span class="tag">EXCEL EXPORT</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🔑 Groq API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Groq API Key", type="password",
        placeholder="Paste your gsk_... key",
        label_visibility="hidden"
    )
    st.caption("Free key → [console.groq.com](https://console.groq.com)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Existing Test Cases (RAG)</div>', unsafe_allow_html=True)
    st.caption("Upload your master Excel — app learns your style and avoids duplicates")
    past_excel = st.file_uploader(
        "Upload existing test cases Excel",
        type=["xlsx", "xls"],
        label_visibility="hidden"
    )

    if past_excel and not st.session_state.rag_loaded:
        with st.spinner("Loading all test cases into RAG..."):
            count, last_id = load_excel_to_rag(past_excel.read())
        st.success(f"✅ {count} test cases loaded | Last TC ID: {last_id}")

    elif past_excel and st.session_state.rag_loaded:
        st.markdown(
            f'<div class="rag-status-loaded">'
            f'✅ {st.session_state.rag_count} test cases in RAG<br>'
            f'Last TC ID: {st.session_state.last_tc_id} — new TCs will continue from here'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="rag-status-empty">'
            '⚠️ No existing test cases uploaded. Upload Excel for duplicate prevention and style matching.'
            '</div>',
            unsafe_allow_html=True
        )

    if st.session_state.rag_loaded:
        if st.button("🗑️ Clear & Upload New File"):
            for k in ["rag_loaded", "rag_count", "rag_docs", "last_tc_id"]:
                st.session_state[k] = defaults[k]
            _, col = load_chroma()
            try:
                col.delete(where={"id": {"$ne": ""}})
            except Exception:
                pass
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🎛️ Settings</div>', unsafe_allow_html=True)
    num_cases    = st.slider("Number of test cases", 5, 45, 15)
    include_neg  = st.toggle("Include negative tests",  value=True)
    include_edge = st.toggle("Include edge cases",       value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">📋 Pipeline Status</div>', unsafe_allow_html=True)
    stage = st.session_state.stage or 0
    steps = [
        "🔍 OCR — Read Input",
        "🧠 Parse Feature Understanding",
        "📚 Build Coverage Index",
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
            height=220,
            placeholder="""Paste your Jira story, acceptance criteria, or feature description here.

Acceptance Criteria:
User Roles: admin, regular user
Modules: incident report, dashboard, email, CSV export, Excel export, PDF export""",
            label_visibility="hidden"
        )

    st.markdown("")

    has_image = bool(images)
    has_text  = bool(manual_text and manual_text.strip())
    has_input = has_image or has_text
    can_run   = has_input and bool(api_key)

    if not api_key:
        st.caption("🔑 Add your Groq API key in the sidebar")
    elif not has_input:
        st.caption("⬆️ Upload an image or paste a description above")

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
            combined_text += f"\n\n[Feature Description]\n{manual_text.strip()}"

        combined_text = combined_text.strip()
        if not combined_text:
            st.error("❌ Could not extract text. Please add a description.")
            st.stop()

        st.session_state.extracted_text = combined_text

        # ── Stage 2: Parse Feature Understanding ──────────────
        st.session_state.stage = 2
        with st.spinner("🧠 Step 1 — Parsing feature and building Feature Understanding..."):
            feature_understanding = step1_parse_input(combined_text, api_key)
            st.session_state.feature_understand = feature_understanding

        # ── Stage 3: Build Coverage Index ─────────────────────
        st.session_state.stage = 3
        with st.spinner("📚 Step 2 — Retrieving similar past test cases and building Coverage Index..."):
            similar = rag_retrieve(feature_understanding)
            coverage_index = step2_build_coverage_index(similar, api_key)
            st.session_state.coverage_index = coverage_index

        # ── Stage 4: Generate Test Cases ──────────────────────
        st.session_state.stage = 4
        last_tc_id = st.session_state.last_tc_id or 3000

        with st.spinner(f"⚙️ Steps 3-5 — Generating {num_cases} enterprise-grade test cases..."):
            test_cases_raw = step3_generate_test_cases(
                feature_understanding = feature_understanding,
                coverage_index        = coverage_index,
                similar_cases         = similar,
                num_cases             = num_cases,
                last_tc_id            = last_tc_id,
                api_key               = api_key,
                include_neg           = include_neg,
                include_edge          = include_edge,
            )
            st.session_state.test_cases_raw = test_cases_raw

            # Count by type
            lines     = [l for l in test_cases_raw.split("\n") if "|" in l and "---" not in l]
            pos_count = sum(1 for l in lines if "positive" in l.lower())
            neg_count = sum(1 for l in lines if "negative" in l.lower())
            edge_count= sum(1 for l in lines if "edge" in l.lower())

            st.session_state.tc_count   = len(lines)
            st.session_state.pos_count  = pos_count
            st.session_state.neg_count  = neg_count
            st.session_state.edge_count = edge_count

        # ── Stage 5: Excel Export ─────────────────────────────
        st.session_state.stage = 5
        with st.spinner("📊 Step 6 — Building formatted Excel with color coding..."):
            feature_name = "Feature"
            if images:
                feature_name = os.path.splitext(images[0].name)[0]
            elif manual_text:
                first_line   = manual_text.strip().split("\n")[0]
                feature_name = first_line[:30].replace("Feature:", "").strip() or "Feature"

            excel = build_excel(
                test_cases_text = test_cases_raw,
                feature_name    = feature_name,
                pos_count       = pos_count,
                neg_count       = neg_count,
                edge_count      = edge_count,
            )
            st.session_state.excel_bytes = excel

        st.rerun()


# ── Results ────────────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card-label">📋 Results</div>', unsafe_allow_html=True)

    if not st.session_state.test_cases_raw:
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
        tc        = st.session_state.tc_count   or 0
        pos       = st.session_state.pos_count  or 0
        neg       = st.session_state.neg_count  or 0
        edge      = st.session_state.edge_count or 0
        rag_total = st.session_state.rag_count  or 0

        st.markdown(f"""
        <div class="success-box">
            ✅ &nbsp; <strong>{tc} enterprise-grade test cases generated!</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-strip">
            <div class="metric">
                <div class="metric-val" style="color:#2E7D32">{pos}</div>
                <div class="metric-lbl">✅ Positive</div>
            </div>
            <div class="metric">
                <div class="metric-val" style="color:#C00000">{neg}</div>
                <div class="metric-lbl">❌ Negative</div>
            </div>
            <div class="metric">
                <div class="metric-val" style="color:#F59E0B">{edge}</div>
                <div class="metric-lbl">⚠️ Edge</div>
            </div>
            <div class="metric">
                <div class="metric-val">{rag_total}</div>
                <div class="metric-lbl">📚 RAG Cases</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs([
            "📊 Test Cases",
            "🧠 Feature Understanding",
            "📚 Coverage Index",
            "🔍 OCR Output"
        ])

        with t1:
            st.markdown(
                f'<div class="result-preview">{st.session_state.test_cases_raw}</div>',
                unsafe_allow_html=True
            )
        with t2:
            st.markdown(
                f'<div class="result-preview">{st.session_state.feature_understand or "—"}</div>',
                unsafe_allow_html=True
            )
        with t3:
            st.markdown(
                f'<div class="result-preview">{st.session_state.coverage_index or "—"}</div>',
                unsafe_allow_html=True
            )
        with t4:
            st.markdown(
                f'<div class="result-preview">{st.session_state.extracted_text or "—"}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        fname = "Feature"
        if images:
            fname = os.path.splitext(images[0].name)[0]
        elif manual_text:
            fname = manual_text.strip().split("\n")[0][:30].replace("Feature:", "").strip() or "Feature"

        st.download_button(
            label     = "⬇️ DOWNLOAD EXCEL TEST CASES",
            data      = st.session_state.excel_bytes,
            file_name = f"{fname}_TestCases.xlsx",
            mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if st.button("🔄 Generate New"):
            for k in ["extracted_text", "feature_understand", "coverage_index",
                      "test_cases_raw", "excel_bytes", "tc_count",
                      "pos_count", "neg_count", "edge_count"]:
                st.session_state[k] = None
            st.session_state.stage = 0
            st.rerun()
