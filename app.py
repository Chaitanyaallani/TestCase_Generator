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

*, html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.stApp { background: #F7F6F3; }

.app-header {
    background: #1A1A2E;
    border-radius: 0 0 24px 24px;
    padding: 2.5rem 2rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
}
.app-header::after {
    content: '🧪';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem; opacity: 0.08;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem; font-weight: 700;
    color: #E8F4FD; margin: 0; letter-spacing: -0.5px;
}
.app-sub {
    color: #7B8FA1; font-size: 0.9rem;
    margin-top: 0.4rem; font-weight: 300;
}
.tag {
    display: inline-block;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: #A8C8E8; padding: 0.15rem 0.65rem;
    border-radius: 4px; font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    margin: 0.3rem 0.2rem 0 0; letter-spacing: 0.5px;
}
.card {
    background: #FFFFFF; border: 1px solid #E8E5DF;
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
}
.card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; color: #9B8EA0;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;
}
.result-preview {
    background: #1A1A2E; border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; color: #A8C8E8;
    white-space: pre-wrap; max-height: 220px;
    overflow-y: auto; margin-top: 0.5rem;
}
.metric-strip { display: flex; gap: 1rem; margin: 1rem 0; }
.metric {
    flex: 1; background: #FFFFFF;
    border: 1px solid #E8E5DF;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem; font-weight: 700; color: #1A1A2E;
}
.metric-lbl {
    font-size: 0.7rem; color: #9B8EA0;
    text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem;
}
.success-box {
    background: #E8F5E9; border: 1px solid #A5D6A7;
    border-left: 4px solid #2E7D32; border-radius: 8px;
    padding: 1.2rem 1.5rem; margin: 1rem 0;
    color: #1B5E20; font-weight: 500;
}
.stButton > button {
    background: #1A1A2E !important; color: #E8F4FD !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    letter-spacing: 1px !important; padding: 0.65rem 1.5rem !important;
    width: 100% !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #2D2D4E !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(26,26,46,0.25) !important;
}
.stButton > button:disabled {
    background: #C8C5BF !important;
    color: #8A8580 !important; cursor: not-allowed !important;
}
.stTextInput > div > input,
.stTextArea > div > textarea {
    background: #FFFFFF !important;
    border: 1px solid #D8D5CF !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #1A1A2E !important;
}
[data-testid="stSidebar"] {
    background: #FFFFFF; border-right: 1px solid #E8E5DF;
}
.sidebar-section {
    background: #F7F6F3; border: 1px solid #E8E5DF;
    border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
}
.sidebar-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 2px; color: #9B8EA0; margin-bottom: 0.6rem;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
for k in ["extracted_text", "parsed_req", "similar_cases",
          "test_cases", "excel_bytes", "stage", "tc_count", "rag_count"]:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.stage is None:
    st.session_state.stage = 0


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def call_groq(prompt: str, api_key: str, max_tokens: int = 3000) -> str:
    """Call Groq AI with a prompt and return the response."""
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = max_tokens
    )
    return response.choices[0].message.content


def run_ocr(image: Image.Image) -> str:
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(
        image.convert("RGB"), config="--psm 6"
    ).strip()


@st.cache_resource
def get_rag():
    """Initialize and cache RAG components (runs once)."""
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    db    = chromadb.Client()
    col   = db.get_or_create_collection("all_test_cases")
    return embed, col


def load_excel_to_rag(excel_bytes: bytes) -> int:
    """
    Load ALL sheets from an Excel file into ChromaDB.
    Works for both single-feature and multi-feature master files.
    """
    embed, col = get_rag()
    wb         = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    total      = 0

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        # Skip sheets with very few rows (summary/readme sheets)
        if ws.max_row < 3:
            continue

        # Read headers from row 1
        headers = [
            str(c.value).strip().lower() if c.value else f"col{i}"
            for i, c in enumerate(ws[1])
        ]

        # Load each row as one test case
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not any(row):
                continue

            # Combine all columns into one readable text block
            row_text = "\n".join(
                f"{headers[i]}: {v}"
                for i, v in enumerate(row)
                if v is not None and i < len(headers)
            )

            if not row_text.strip():
                continue

            emb = embed.encode(row_text).tolist()
            try:
                col.add(
                    documents  = [row_text],
                    embeddings = [emb],
                    ids        = [f"tc_{sheet_name}_{total}"]
                )
                total += 1
            except Exception:
                pass  # Skip duplicates

    return total


def rag_retrieve(query: str) -> str:
    """Find top 5 most similar past test cases from ChromaDB."""
    embed, col = get_rag()

    if col.count() == 0:
        return "No past test cases loaded."

    emb     = embed.encode(query).tolist()
    results = col.query(
        query_embeddings = [emb],
        n_results        = min(5, col.count())
    )
    docs = results["documents"][0] if results["documents"] else []
    return "\n\n---\n\n".join(docs) if docs else "No similar cases found."


def build_excel(test_cases_text: str, feature_name: str) -> bytes:
    """Build a formatted Excel file from generated test cases."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Test Cases"

    # Styles
    h_fill  = PatternFill("solid", fgColor="1A1A2E")
    a_fill  = PatternFill("solid", fgColor="EBF3FB")
    h_font  = Font(bold=True, color="E8F4FD", name="Calibri", size=11)
    n_font  = Font(name="Calibri", size=10)
    c_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    l_align = Alignment(horizontal="left",   vertical="top",    wrap_text=True)
    thin    = Side(style="thin", color="D0CCC8")
    bdr     = Border(left=thin, right=thin, top=thin, bottom=thin)

    headers = ["TC ID", "Title", "Preconditions",
               "Test Steps", "Expected Result", "Priority", "Status"]
    widths  = [10, 28, 25, 40, 32, 12, 12]

    # Write headers
    for c, (h, w) in enumerate(zip(headers, widths), 1):
        cell           = ws.cell(row=1, column=c, value=h)
        cell.fill      = h_fill
        cell.font      = h_font
        cell.alignment = c_align
        cell.border    = bdr
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[1].height = 30

    # Parse pipe-delimited rows
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

    # Summary sheet
    ws2 = wb.create_sheet("Summary")
    summary_data = [
        ("Feature",         feature_name),
        ("Total TCs",       f"=COUNTA('Test Cases'!A2:A1000)"),
        ("Generated By",    "AI Test Case Generator"),
        ("Model",           "Groq LLaMA 3.1"),
    ]
    for i, (k, v) in enumerate(summary_data, 1):
        ws2.cell(row=i, column=1, value=k).font = Font(bold=True, name="Calibri")
        ws2.cell(row=i, column=2, value=v)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">QA Test Case Generator</div>
    <div class="app-sub">
        Upload feature images or paste description → Get professional test cases in Excel
    </div>
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

    # ── API Key ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🔑 Groq API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "", type="password",
        placeholder="Paste your gsk_... key",
        label_visibility="collapsed"
    )
    st.caption("Free key → [console.groq.com](https://console.groq.com)")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── RAG Upload ─────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Past Test Cases (RAG)</div>', unsafe_allow_html=True)
    st.caption("Upload your master Excel file — works for ALL features")
    past_excel = st.file_uploader(
        "", type=["xlsx", "xls"],
        label_visibility="collapsed"
    )
    if past_excel:
        with st.spinner("Loading all test cases into RAG..."):
            count = load_excel_to_rag(past_excel.read())
            st.session_state.rag_count = count
        st.success(f"✅ {count} test cases loaded from all features")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Settings ───────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🎛️ Settings</div>', unsafe_allow_html=True)
    num_cases    = st.slider("Number of test cases", 5, 45, 30)
    include_neg  = st.toggle("Include negative tests",  value=True)
    include_edge = st.toggle("Include edge cases",       value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Pipeline Status ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sidebar-title">📋 Pipeline Status</div>', unsafe_allow_html=True)
    stage = st.session_state.stage or 0
    steps = [
        ("🔍", "OCR — Read Image"),
        ("🧠", "Parse Requirements"),
        ("📚", "RAG — Find Similar"),
        ("⚙️", "Generate Test Cases"),
        ("📊", "Export to Excel"),
    ]
    for i, (icon, label) in enumerate(steps):
        if i < stage:
            st.markdown(f'<div style="color:#2E7D32;font-size:0.85rem;padding:0.2rem 0;">✅ {label}</div>', unsafe_allow_html=True)
        elif i == stage and stage > 0:
            st.markdown(f'<div style="color:#1565C0;font-size:0.85rem;padding:0.2rem 0;">⏳ {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:#9B8EA0;font-size:0.85rem;padding:0.2rem 0;">○ {label}</div>', unsafe_allow_html=True)


# ── Main Layout ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — INPUT
# ══════════════════════════════════════════════════════════════════════════════
with col_left:
    st.markdown('<div class="card-label">📤 Feature Input</div>', unsafe_allow_html=True)

    tab_img, tab_txt = st.tabs(["🖼️ Upload Images", "📝 Type / Paste Text"])

    with tab_img:
        images = st.file_uploader(
            "Upload one or more feature images",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if images:
            for img in images:
                st.image(img, caption=img.name, use_container_width=True)

    with tab_txt:
        manual_text = st.text_area(
            "Paste your feature description",
            height=200,
            placeholder="""Example:
Feature: Add Issue Pattern
User selects a Rule Pattern from dropdown.
Dropdown options: Contains, Equals, Starts With, Ends With
User clicks Save to save. Cancel to discard.
Validation: Pattern field cannot be empty.

Templates affected: incident report, email, PDF, CSV...
Include: positive, negative, edge cases.""",
            label_visibility="collapsed"
        )

    st.markdown("")

    # Validation
    has_image = bool(images)
    has_text  = bool(manual_text and manual_text.strip())
    has_input = has_image or has_text
    can_run   = has_input and bool(api_key)

    if not api_key:
        st.caption("🔑 Add Groq API key in the sidebar to enable")
    elif not has_input:
        st.caption("⬆️ Upload an image or type a description above")

    # ── GENERATE BUTTON ────────────────────────────────────────
    if st.button("⚡ GENERATE TEST CASES", disabled=not can_run):

        # ── Stage 1: OCR ──────────────────────────────────────
        st.session_state.stage = 1
        combined_text = ""

        if has_image:
            with st.spinner("🔍 Reading images with OCR..."):
                for img_file in images:
                    img  = Image.open(img_file)
                    text = run_ocr(img)
                    if text:
                        combined_text += f"\n\n[From: {img_file.name}]\n{text}"

        if has_text:
            combined_text += f"\n\n[Manual Description]\n{manual_text.strip()}"

        combined_text = combined_text.strip()

        if not combined_text:
            st.error("❌ Could not extract any text. Please add a description in the text tab.")
            st.stop()

        st.session_state.extracted_text = combined_text

        # ── Stage 2: Parse Requirements ───────────────────────
        st.session_state.stage = 2
        with st.spinner("🧠 Parsing requirements with Groq AI..."):
            parse_prompt = f"""You are a senior business analyst.
Read the feature story below and extract structured requirements.

Return in exactly this format:
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
            parsed = call_groq(parse_prompt, api_key, max_tokens=1000)
            st.session_state.parsed_req = parsed

        # ── Stage 3: RAG ──────────────────────────────────────
        st.session_state.stage = 3
        with st.spinner("📚 Searching past test cases..."):
            similar = rag_retrieve(parsed)
            st.session_state.similar_cases = similar

        # ── Stage 4: Generate Test Cases ──────────────────────
        st.session_state.stage = 4

        # Build type instructions
        neg_inst  = "Include negative test cases (invalid inputs, errors, unauthorized access)." if include_neg  else ""
        edge_inst = "Include edge cases (empty fields, boundary values, special characters, leap year, timezone, large datasets)." if include_edge else ""

        # Calculate split
        if include_neg and include_edge:
            pos  = num_cases // 3 + (num_cases % 3)
            neg  = num_cases // 3
            edge = num_cases - pos - neg
            split_instruction = f"Generate {pos} Positive + {neg} Negative + {edge} Edge cases = {num_cases} total."
        else:
            split_instruction = f"Generate exactly {num_cases} test cases."

        with st.spinner(f"⚙️ Generating {num_cases} test cases..."):
            gen_prompt = f"""You are a senior QA engineer with 10 years experience.
{split_instruction}

Requirements:
{parsed}

Reference style — follow this EXACT style from past test cases:
{similar}

STRICT RULES:
- Each test MUST have exactly 5 numbered steps:
  1. Navigate to...
  2. Open / Select...
  3. Enter / Click...
  4. Submit / Confirm...
  5. Verify result...
- Label each row type: Positive / Negative / Edge
{neg_inst}
{edge_inst}

CRITICAL: Return ONLY pipe-delimited rows. No headings. No extra text. No markdown.
Each row EXACT format:
| TC001 | Title here | Preconditions here | 1. Step one; 2. Step two; 3. Step three; 4. Step four; 5. Step five | Expected result | High |

Columns: TC ID | Title | Preconditions | Test Steps | Expected Result | Priority
Priority: High / Medium / Low
Generate exactly {num_cases} rows.
"""
            test_cases = call_groq(gen_prompt, api_key, max_tokens=4000)
            st.session_state.test_cases = test_cases

            # Count rows
            lines = [
                l for l in test_cases.split("\n")
                if "|" in l and "---" not in l and l.strip() != "|"
            ]
            st.session_state.tc_count = len(lines)

        # ── Stage 5: Excel Export ─────────────────────────────
        st.session_state.stage = 5
        with st.spinner("📊 Building Excel file..."):
            feature_name = "feature"
            if images:
                feature_name = os.path.splitext(images[0].name)[0]
            elif manual_text:
                # Extract feature name from first line
                first_line   = manual_text.strip().split("\n")[0]
                feature_name = first_line[:30].replace("Feature:", "").strip() or "feature"

            excel = build_excel(test_cases, feature_name)
            st.session_state.excel_bytes = excel

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with col_right:
    st.markdown('<div class="card-label">📋 Results</div>', unsafe_allow_html=True)

    if not st.session_state.test_cases:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E8E5DF;
                    border-radius:12px;padding:4rem 2rem;
                    text-align:center;color:#C8C5BF;">
            <div style="font-size:3rem">⏳</div>
            <div style="margin-top:0.8rem;font-size:0.9rem;color:#9B8EA0;">
                Results will appear here after generation
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Success banner
        tc       = st.session_state.tc_count or 0
        chars    = len(st.session_state.extracted_text or "")
        _, col_db = get_rag()
        rag_total = col_db.count()

        st.markdown(f"""
        <div class="success-box">
            ✅ &nbsp; <strong>{tc} test cases generated successfully!</strong>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
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

        # Result tabs
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

        # Download button
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

        # Reset button
        if st.button("🔄 Generate New"):
            for k in ["extracted_text", "parsed_req", "similar_cases",
                      "test_cases", "excel_bytes", "tc_count"]:
                st.session_state[k] = None
            st.session_state.stage = 0
            st.rerun()
