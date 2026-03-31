import streamlit as st
from groq import Groq
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import chromadb
from sentence_transformers import SentenceTransformer
import io, os, re, json
import urllib.request

st.set_page_config(
    page_title="QA Test Case Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
*, html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #F7F6F3; }
.app-header { background:#1A1A2E; border-radius:0 0 24px 24px; padding:2.5rem 2rem 2rem; margin:-1rem -1rem 2rem -1rem; position:relative; overflow:hidden; }
.app-header::after { content:'🧪'; position:absolute; right:2rem; top:50%; transform:translateY(-50%); font-size:5rem; opacity:0.08; }
.app-title { font-family:'IBM Plex Mono',monospace; font-size:2rem; font-weight:700; color:#E8F4FD; margin:0; }
.app-sub { color:#7B8FA1; font-size:0.9rem; margin-top:0.4rem; }
.tag { display:inline-block; background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); color:#A8C8E8; padding:0.15rem 0.65rem; border-radius:4px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:0.3rem 0.2rem 0 0; }
.card-label { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#9B8EA0; text-transform:uppercase; letter-spacing:2px; margin-bottom:0.5rem; }
.result-preview { background:#1A1A2E; border-radius:8px; padding:1rem 1.2rem; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#A8C8E8; white-space:pre-wrap; max-height:280px; overflow-y:auto; margin-top:0.5rem; }
.metric-strip { display:flex; gap:1rem; margin:1rem 0; }
.metric { flex:1; background:#FFFFFF; border:1px solid #E8E5DF; border-radius:10px; padding:1rem; text-align:center; }
.metric-val { font-family:'IBM Plex Mono',monospace; font-size:1.8rem; font-weight:700; color:#1A1A2E; }
.metric-lbl { font-size:0.7rem; color:#9B8EA0; text-transform:uppercase; letter-spacing:1px; margin-top:0.2rem; }
.success-box { background:#E8F5E9; border:1px solid #A5D6A7; border-left:4px solid #2E7D32; border-radius:8px; padding:1.2rem 1.5rem; margin:1rem 0; color:#1B5E20; font-weight:500; }
.warn-box { background:#FFF3CD; border:1px solid #FFD580; border-left:4px solid #F59E0B; border-radius:8px; padding:1rem 1.2rem; margin:0.5rem 0; color:#92400E; font-size:0.88rem; }
.info-box { background:#E3F2FD; border:1px solid #90CAF9; border-left:4px solid #1565C0; border-radius:8px; padding:1rem 1.2rem; margin:0.5rem 0; color:#0D47A1; font-size:0.88rem; }
.rag-ok { background:#E8F5E9; border:1px solid #A5D6A7; border-left:3px solid #2E7D32; border-radius:6px; padding:0.6rem 0.8rem; font-size:0.8rem; color:#1B5E20; margin-top:0.5rem; }
.rag-warn { background:#FFF3CD; border:1px solid #FFD580; border-left:3px solid #F59E0B; border-radius:6px; padding:0.6rem 0.8rem; font-size:0.8rem; color:#92400E; margin-top:0.5rem; }
.stButton > button { background:#1A1A2E !important; color:#E8F4FD !important; border:none !important; border-radius:8px !important; font-family:'IBM Plex Mono',monospace !important; font-size:0.82rem !important; font-weight:600 !important; letter-spacing:1px !important; padding:0.65rem 1.5rem !important; width:100% !important; transition:all 0.2s !important; }
.stButton > button:hover { background:#2D2D4E !important; transform:translateY(-1px) !important; box-shadow:0 4px 16px rgba(26,26,46,0.25) !important; }
.stButton > button:disabled { background:#C8C5BF !important; color:#8A8580 !important; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E8E5DF; }
.sidebar-section { background:#F7F6F3; border:1px solid #E8E5DF; border-radius:10px; padding:1rem; margin-bottom:1rem; }
.sidebar-title { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; text-transform:uppercase; letter-spacing:2px; color:#9B8EA0; margin-bottom:0.6rem; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
defaults = {
    "extracted_text": None, "feature_understand": None,
    "test_cases_parsed": [], "excel_bytes": None,
    "stage": 0, "tc_count": 0, "pos_count": 0, "neg_count": 0, "edge_count": 0,
    "rag_loaded": False, "rag_count": 0, "rag_docs": [], "last_tc_id": 3000,
    "ocr_warnings": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Cached Resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_embed_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

@st.cache_resource(show_spinner="Setting up database...")
def load_chroma():
    client = chromadb.Client()
    col    = client.get_or_create_collection("all_test_cases")
    return client, col


# ── Helper: Groq ───────────────────────────────────────────────────────────────
def call_groq(prompt: str, api_key: str, max_tokens: int = 3000) -> str:
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model      = "llama-3.1-8b-instant",
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = max_tokens
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# OCR — WITH IMAGE PREPROCESSING FOR BETTER RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_image(image: Image.Image) -> Image.Image:
    """Enhance image quality before OCR for better text extraction."""
    img = image.convert("RGB")
    # Increase size for better OCR
    w, h = img.size
    if w < 1000:
        scale = 1000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Enhance contrast and sharpness
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    # Convert to grayscale for better OCR
    img = img.convert("L")
    return img


def run_ocr(image: Image.Image) -> str:
    """Run OCR with preprocessing and multiple config attempts."""
    warnings = []

    # Try with preprocessing first
    try:
        enhanced = preprocess_image(image)
        text = pytesseract.image_to_string(enhanced, config="--psm 6 --oem 3")
        if len(text.strip()) > 30:
            return text.strip(), []
    except Exception:
        pass

    # Try raw RGB
    try:
        text = pytesseract.image_to_string(image.convert("RGB"), config="--psm 6")
        if len(text.strip()) > 30:
            return text.strip(), []
    except Exception:
        pass

    # Try different PSM modes
    for psm in [11, 3, 4]:
        try:
            text = pytesseract.image_to_string(image.convert("L"), config=f"--psm {psm}")
            if len(text.strip()) > 30:
                return text.strip(), [f"Used alternate OCR mode (psm={psm}) for this image"]
        except Exception:
            pass

    # OCR failed — return empty with warning
    return "", ["⚠️ OCR could not extract text from this image. It may contain embedded images, diagrams, or low-resolution content. Please paste the feature description manually in the text tab."]


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from .docx files."""
    try:
        import zipfile
        from xml.etree import ElementTree as ET

        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            if "word/document.xml" not in z.namelist():
                return ""
            xml_content = z.read("word/document.xml")

        tree = ET.fromstring(xml_content)
        ns   = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        texts = []
        for para in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
            para_text = "".join(
                node.text or ""
                for node in para.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
            )
            if para_text.strip():
                texts.append(para_text.strip())
        return "\n".join(texts)
    except Exception as e:
        return ""


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from .txt files."""
    try:
        return file_bytes.decode("utf-8").strip()
    except Exception:
        try:
            return file_bytes.decode("latin-1").strip()
        except Exception:
            return ""


# ══════════════════════════════════════════════════════════════════════════════
# RAG
# ══════════════════════════════════════════════════════════════════════════════
def load_excel_to_rag(excel_bytes: bytes) -> tuple:
    embed = load_embed_model()
    _, col = load_chroma()
    docs, total, last_id = [], 0, 3000
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    for sname in wb.sheetnames:
        ws = wb[sname]
        if ws.max_row < 3: continue
        headers = [str(c.value).strip().lower() if c.value else f"col{i}" for i, c in enumerate(ws[1])]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not any(row): continue
            tc_val = str(row[0]) if row[0] else ""
            for s in re.findall(r'\d+', tc_val):
                last_id = max(last_id, int(s))
            row_text = "\n".join(
                f"{headers[i]}: {v}" for i, v in enumerate(row)
                if v is not None and i < len(headers)
            )
            if not row_text.strip(): continue
            docs.append(row_text)
            emb = embed.encode(row_text).tolist()
            try:
                col.add(documents=[row_text], embeddings=[emb], ids=[f"tc_{sname}_{total}"])
                total += 1
            except Exception:
                pass
    st.session_state.rag_docs   = docs
    st.session_state.rag_count  = total
    st.session_state.rag_loaded = True
    st.session_state.last_tc_id = last_id
    return total, last_id


def rag_retrieve(query: str) -> str:
    embed = load_embed_model()
    _, col = load_chroma()
    if col.count() == 0 and st.session_state.rag_docs:
        for i, doc in enumerate(st.session_state.rag_docs):
            emb = embed.encode(doc).tolist()
            try: col.add(documents=[doc], embeddings=[emb], ids=[f"tc_r_{i}"])
            except: pass
    if col.count() == 0: return "No past test cases loaded."
    emb     = embed.encode(query).tolist()
    results = col.query(query_embeddings=[emb], n_results=min(3, col.count()))
    docs    = results["documents"][0] if results["documents"] else []
    return "\n\n---\n\n".join(docs) if docs else "No similar cases found."


# ══════════════════════════════════════════════════════════════════════════════
# JSON GENERATION — ROBUST WITH RETRY
# ══════════════════════════════════════════════════════════════════════════════
def parse_json_response(raw: str) -> list:
    """Try multiple strategies to extract JSON array from AI response."""

    # Strategy 1: Direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
    except Exception:
        pass

    # Strategy 2: Extract [...] block
    try:
        match = re.search(r'\[[\s\S]*\]', raw)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
    except Exception:
        pass

    # Strategy 3: Strip markdown fences
    try:
        cleaned = re.sub(r'```json|```', '', raw).strip()
        parsed  = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
    except Exception:
        pass

    # Strategy 4: Fix common JSON issues (trailing commas, single quotes)
    try:
        cleaned = re.sub(r',\s*}', '}', raw)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        cleaned = re.sub(r"'", '"', cleaned)
        match   = re.search(r'\[[\s\S]*\]', cleaned)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
    except Exception:
        pass

    return []


def generate_test_cases_as_json(
    combined_text: str,
    feature_understanding: str,
    similar_cases: str,
    num_cases: int,
    last_tc_id: int,
    api_key: str,
    include_neg: bool,
    include_edge: bool
) -> list:
    """Generate test cases as JSON with retry on failure."""

    if include_neg and include_edge:
        pos = num_cases // 3 + (num_cases % 3)
        neg = num_cases // 3
        edge = num_cases - pos - neg
    elif include_neg:
        pos = num_cases // 2 + (num_cases % 2)
        neg = num_cases - pos
        edge = 0
    elif include_edge:
        pos = num_cases // 2 + (num_cases % 2)
        neg = 0
        edge = num_cases - pos
    else:
        pos, neg, edge = num_cases, 0, 0

    start_id = last_tc_id + 1

    # Build the prompt
    prompt = f"""You are a Senior QA Architect. Generate exactly {num_cases} UI test cases.

FEATURE UNDERSTANDING:
{feature_understanding}

PAST TEST CASES STYLE REFERENCE (match this quality):
{similar_cases}

RULES:
- {pos} Positive + {neg} Negative + {edge} Edge cases
- TC IDs start from TC-{start_id:04d}
- Prerequisites: specific role, module, test data
- Steps: exactly 5, each a single specific UI action on a named element
- Expected: exact visible UI outcome — screen name, message text, element state
- NEVER use: "works correctly", "saves successfully", "behaves as expected"

RETURN ONLY a valid JSON array. Nothing else before or after.

[
  {{
    "id": "TC-{start_id:04d}",
    "title": "Verify year format in incident report template displays correctly",
    "prerequisites": "User: ignio admin logged in, Incident report template accessible, Test data with date Jan 23 2026 available",
    "steps": "1. Navigate to incident management system\\n2. Open existing incident with date Jan 23 2026\\n3. Generate incident report using standard template\\n4. Review date format in generated report\\n5. Verify year displays in new required format",
    "expected": "Incident report displays date in correct year format. Template adheres to new standard. No formatting errors visible.",
    "type": "Positive",
    "priority": "High",
    "related_tc": "None"
  }},
  {{
    "id": "TC-{start_id+1:04d}",
    "title": "Verify system rejects invalid year format with error message",
    "prerequisites": "User: ignio admin logged in, Template configuration screen accessible",
    "steps": "1. Navigate to template configuration screen\\n2. Locate the year format input field\\n3. Enter invalid year format e.g. 3-digit year '202'\\n4. Click Save button\\n5. Verify error message appears on screen",
    "expected": "System displays error message 'Invalid year format. Please use the required format.' Save is blocked. Field is highlighted in red.",
    "type": "Negative",
    "priority": "High",
    "related_tc": "None"
  }}
]

Now generate all {num_cases} test cases for this feature:
{combined_text[:800]}

Return ONLY the JSON array starting with [ and ending with ].
"""

    # First attempt
    raw    = call_groq(prompt, api_key, max_tokens=4000)
    result = parse_json_response(raw)

    if result:
        return result

    # Retry with simpler prompt
    retry_prompt = f"""Generate {num_cases} test cases as a JSON array.

Feature: {combined_text[:400]}

Return ONLY this JSON format, no other text:
[
  {{
    "id": "TC-{start_id:04d}",
    "title": "test case title",
    "prerequisites": "User: role logged in, module accessible, test data available",
    "steps": "1. Navigate to page\\n2. Click element\\n3. Enter value\\n4. Submit\\n5. Verify result",
    "expected": "Exact visible UI outcome with screen name and message text",
    "type": "Positive",
    "priority": "High",
    "related_tc": "None"
  }}
]
Generate {num_cases} objects. Types: {pos} Positive, {neg} Negative, {edge} Edge.
Start IDs from TC-{start_id:04d}.
"""
    raw    = call_groq(retry_prompt, api_key, max_tokens=4000)
    result = parse_json_response(raw)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_excel(test_cases: list, feature_name: str) -> bytes:
    wb     = openpyxl.Workbook()
    h_fill = PatternFill("solid", fgColor="1A1A2E")
    h_font = Font(bold=True, color="E8F4FD", name="Calibri", size=11)
    c_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    pos_count  = sum(1 for tc in test_cases if "positive" in str(tc.get("type","")).lower())
    neg_count  = sum(1 for tc in test_cases if "negative" in str(tc.get("type","")).lower())
    edge_count = sum(1 for tc in test_cases if "edge"     in str(tc.get("type","")).lower())

    # Summary sheet
    ws_sum = wb.active
    ws_sum.title = "SUMMARY"
    sum_hdrs = ["Feature / Module Name","Total Test Cases","Positive Count",
                "Negative Count","Edge Case Count","Acceptance Criteria Covered","Gaps Identified"]
    for c, h in enumerate(sum_hdrs, 1):
        cell = ws_sum.cell(row=1, column=c, value=h)
        cell.fill = h_fill; cell.font = h_font; cell.alignment = c_align
        ws_sum.column_dimensions[get_column_letter(c)].width = 22
    ws_sum.cell(row=2, column=1, value=feature_name)
    ws_sum.cell(row=2, column=2, value=len(test_cases))
    ws_sum.cell(row=2, column=3, value=pos_count)
    ws_sum.cell(row=2, column=4, value=neg_count)
    ws_sum.cell(row=2, column=5, value=edge_count)
    ws_sum.cell(row=2, column=6, value="Extracted from feature input")
    ws_sum.cell(row=2, column=7, value="Legacy system integration, Performance testing")
    ws_sum.freeze_panes = "A2"

    # Test case sheet
    ws = wb.create_sheet(title=feature_name[:31])
    tc_hdrs = ["Test Case ID","Test Case Title","Prerequisites",
               "Test Steps","Expected Result","Test Type","Priority","Related TC ID"]
    widths  = [14, 35, 32, 55, 42, 12, 10, 14]

    pos_fill  = PatternFill("solid", fgColor="E2EFDA")
    neg_fill  = PatternFill("solid", fgColor="FCE4D6")
    edge_fill = PatternFill("solid", fgColor="FFF2CC")
    pos_alt   = PatternFill("solid", fgColor="D0E8C5")
    neg_alt   = PatternFill("solid", fgColor="F0D0B8")
    edge_alt  = PatternFill("solid", fgColor="EFE8AA")

    n_font   = Font(name="Calibri", size=10)
    l_align  = Alignment(horizontal="left",   vertical="top", wrap_text=True)
    m_align  = Alignment(horizontal="center", vertical="top", wrap_text=True)
    thin     = Side(style="thin", color="CCCCCC")
    bdr      = Border(left=thin, right=thin, top=thin, bottom=thin)

    for c, (h, w) in enumerate(zip(tc_hdrs, widths), 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = h_fill; cell.font = h_font
        cell.alignment = c_align; cell.border = bdr
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"

    for ri, tc in enumerate(test_cases, 2):
        tc_type = str(tc.get("type", "")).lower()
        is_alt  = ri % 2 == 0
        fill    = (neg_alt if is_alt else neg_fill) if "negative" in tc_type else \
                  (edge_alt if is_alt else edge_fill) if "edge" in tc_type else \
                  (pos_alt  if is_alt else pos_fill)

        values = [
            tc.get("id",           f"TC-{3000+ri:04d}"),
            tc.get("title",        ""),
            tc.get("prerequisites",""),
            tc.get("steps",        ""),
            tc.get("expected",     ""),
            tc.get("type",         ""),
            tc.get("priority",     ""),
            tc.get("related_tc",   "None"),
        ]
        for ci, val in enumerate(values, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.font = n_font; cell.border = bdr; cell.fill = fill
            cell.alignment = m_align if ci in [1,6,7,8] else l_align
        ws.row_dimensions[ri].height = 80

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="app-title">QA Test Case Generator</div>
    <div class="app-sub">Copilot-quality enterprise test cases — Groq AI + RAG + JSON pipeline</div>
    <div style="margin-top:0.8rem">
        <span class="tag">GROQ AI</span><span class="tag">OCR</span>
        <span class="tag">RAG</span><span class="tag">JSON PIPELINE</span><span class="tag">EXCEL EXPORT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🔑 Groq API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input("Groq API Key", type="password",
                            placeholder="Paste your gsk_... key", label_visibility="hidden")
    st.caption("Free key → [console.groq.com](https://console.groq.com)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Existing Test Cases (RAG)</div>', unsafe_allow_html=True)
    st.caption("Upload master Excel → avoids duplicates, matches your style")
    past_excel = st.file_uploader("Upload existing test cases",
                                  type=["xlsx","xls"], label_visibility="hidden")
    if past_excel and not st.session_state.rag_loaded:
        with st.spinner("Loading into RAG..."):
            count, last_id = load_excel_to_rag(past_excel.read())
        st.success(f"✅ {count} test cases loaded | Next ID: TC-{last_id+1:04d}")
    elif past_excel and st.session_state.rag_loaded:
        st.markdown(f'<div class="rag-ok">✅ {st.session_state.rag_count} test cases in RAG<br>Next TC ID: TC-{st.session_state.last_tc_id+1:04d}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rag-warn">⚠️ Upload your master Excel for better quality</div>', unsafe_allow_html=True)
    if st.session_state.rag_loaded:
        if st.button("🗑️ Clear RAG"):
            for k in ["rag_loaded","rag_count","rag_docs","last_tc_id"]:
                st.session_state[k] = defaults[k]
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
    steps = ["🔍 Extract Text","🧠 Parse Feature","📚 RAG Search","⚙️ Generate JSON","📊 Build Excel"]
    for i, label in enumerate(steps):
        color = "#2E7D32" if i < stage else ("#1565C0" if i == stage and stage > 0 else "#9B8EA0")
        icon  = "✅" if i < stage else ("⏳" if i == stage and stage > 0 else "○")
        st.markdown(f'<div style="color:{color};font-size:0.85rem;padding:0.2rem 0;">{icon} {label}</div>', unsafe_allow_html=True)


# ── Main Layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-label">📤 Feature Input</div>', unsafe_allow_html=True)

    # ── Input type selector ────────────────────────────────────────────────────
    tab_img, tab_doc, tab_txt = st.tabs(["🖼️ Images", "📄 Documents (.docx/.txt)", "📝 Text / Jira Story"])

    with tab_img:
        st.markdown("""
        <div class="info-box">
            💡 <strong>Tip:</strong> If your image has embedded diagrams or very small text,
            OCR may not extract it well. In that case, also paste the description in the Text tab.
        </div>
        """, unsafe_allow_html=True)
        images = st.file_uploader(
            "Upload feature images",
            type=["jpg","jpeg","png","webp","bmp"],
            accept_multiple_files=True,
            label_visibility="hidden"
        )
        if images:
            for img in images:
                st.image(img, caption=img.name, use_container_width=True)

    with tab_doc:
        st.markdown("""
        <div class="info-box">
            💡 <strong>Supported:</strong> .docx (Word) and .txt files.<br>
            For VD (Visual Design) links or Figma — copy the text content and paste it in the Text tab instead.
        </div>
        """, unsafe_allow_html=True)
        doc_files = st.file_uploader(
            "Upload .docx or .txt files",
            type=["docx","txt"],
            accept_multiple_files=True,
            label_visibility="hidden"
        )

    with tab_txt:
        manual_text = st.text_area(
            "Feature description",
            height=240,
            placeholder="""Paste your Jira story, acceptance criteria, VD link description, or feature description here.

Example:
Feature: Year Format Changes in Templates
As ignio admin, I want all templates to display dates in new year format.

Acceptance Criteria:
1. Incident report template shows date in new format
2. Email notification template uses new format
3. CSV and Excel exports maintain new format
4. Dashboard filters display new format
5. System rejects invalid year formats with error

Modules: incident report, dashboard, email, CSV export,
Excel export, PDF export, alert template, rule mining

User Roles: ignio admin, regular user""",
            label_visibility="hidden"
        )

    # VD Link input
    st.markdown("**🔗 Or paste a VD / Figma / Confluence link:**")
    vd_link = st.text_input(
        "VD Link",
        placeholder="https://your-vd-link.com/page",
        label_visibility="hidden"
    )
    if vd_link:
        st.markdown("""
        <div class="warn-box">
            ⚠️ <strong>Note:</strong> VD links (Figma, Confluence, Jira) require login — the app cannot access them directly.
            Please open the link, copy the text content, and paste it in the Text tab above.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Determine what input we have
    has_image   = bool(images)
    has_doc     = bool(doc_files)
    has_text    = bool(manual_text and manual_text.strip())
    has_input   = has_image or has_doc or has_text
    can_run     = has_input and bool(api_key)

    if not api_key:     st.caption("🔑 Add your Groq API key in the sidebar")
    elif not has_input: st.caption("⬆️ Upload images, documents, or paste text above")

    # ── GENERATE BUTTON ────────────────────────────────────────────────────────
    if st.button("⚡ GENERATE TEST CASES", disabled=not can_run):

        combined_text = ""
        ocr_warnings  = []

        # ── Stage 1: Extract text from all inputs ──────────────────────────────
        st.session_state.stage = 1

        # From images
        if has_image:
            with st.spinner("🔍 Extracting text from images..."):
                for img_file in images:
                    img          = Image.open(img_file)
                    text, warns  = run_ocr(img)
                    ocr_warnings.extend(warns)
                    if text:
                        combined_text += f"\n\n[From image: {img_file.name}]\n{text}"
                    else:
                        ocr_warnings.append(
                            f"⚠️ No text extracted from '{img_file.name}'. "
                            f"Please paste the content manually in the Text tab."
                        )

        # From documents
        if has_doc:
            with st.spinner("📄 Reading document files..."):
                for doc_file in doc_files:
                    file_bytes = doc_file.read()
                    if doc_file.name.endswith(".docx"):
                        text = extract_text_from_docx(file_bytes)
                        if text:
                            combined_text += f"\n\n[From document: {doc_file.name}]\n{text}"
                        else:
                            ocr_warnings.append(
                                f"⚠️ Could not extract text from '{doc_file.name}'. "
                                f"If it contains only images, please extract the text manually."
                            )
                    elif doc_file.name.endswith(".txt"):
                        text = extract_text_from_txt(file_bytes)
                        if text:
                            combined_text += f"\n\n[From file: {doc_file.name}]\n{text}"

        # From manual text
        if has_text:
            combined_text += f"\n\n[Feature Description]\n{manual_text.strip()}"

        combined_text = combined_text.strip()
        st.session_state.ocr_warnings = ocr_warnings

        # If no text extracted from any source
        if not combined_text:
            st.error("""
❌ Could not extract any text from your inputs.

**What to do:**
1. Open your VD link / Figma / document in browser
2. Copy the text content (Ctrl+A → Ctrl+C)
3. Paste it in the **📝 Text / Jira Story** tab
4. Click Generate again
""")
            st.stop()

        # If text is too short — warn but continue
        if len(combined_text) < 50:
            st.warning(
                "⚠️ Very little text was extracted. Results may be generic. "
                "Consider adding more detail in the Text tab."
            )

        st.session_state.extracted_text = combined_text

        # ── Stage 2: Feature Understanding ────────────────────────────────────
        st.session_state.stage = 2
        with st.spinner("🧠 Parsing feature and extracting acceptance criteria..."):
            fu_prompt = f"""You are a Senior QA Architect.
Parse this feature input and produce a FEATURE UNDERSTANDING.

Return EXACTLY in this format:
Feature Name        : [name]
User Roles          : [all user roles mentioned]
Screens / Pages     : [all screens/pages]
UI Components       : [all buttons, fields, dropdowns, tabs]
User Flows          : [numbered list of user flows]
Acceptance Criteria : [numbered list — one per line]
Business Rules      : [constraints, validations, permissions]
Error / Edge States : [error messages, boundary conditions]
Modules Affected    : [list all specific modules/templates]

Feature Input:
{combined_text}
"""
            feature_understanding = call_groq(fu_prompt, api_key, max_tokens=1000)
            st.session_state.feature_understand = feature_understanding

        # ── Stage 3: RAG ──────────────────────────────────────────────────────
        st.session_state.stage = 3
        with st.spinner("📚 Retrieving similar past test cases..."):
            similar = rag_retrieve(feature_understanding)

        # ── Stage 4: Generate JSON ─────────────────────────────────────────────
        st.session_state.stage = 4
        last_tc_id = st.session_state.last_tc_id or 3000

        with st.spinner(f"⚙️ Generating {num_cases} Copilot-quality test cases..."):
            test_cases = generate_test_cases_as_json(
                combined_text         = combined_text,
                feature_understanding = feature_understanding,
                similar_cases         = similar,
                num_cases             = num_cases,
                last_tc_id            = last_tc_id,
                api_key               = api_key,
                include_neg           = include_neg,
                include_edge          = include_edge,
            )

            if not test_cases:
                st.error("""
❌ AI returned invalid JSON after 2 attempts.

**Possible causes:**
- Feature description is too short or unclear
- Groq API rate limit hit — wait 30 seconds and try again
- Try reducing the number of test cases in the slider

**What to do:** Add more detail to your feature description and try again.
""")
                st.stop()

            st.session_state.test_cases_parsed = test_cases
            st.session_state.tc_count   = len(test_cases)
            st.session_state.pos_count  = sum(1 for t in test_cases if "positive" in t.get("type","").lower())
            st.session_state.neg_count  = sum(1 for t in test_cases if "negative" in t.get("type","").lower())
            st.session_state.edge_count = sum(1 for t in test_cases if "edge"     in t.get("type","").lower())

        # ── Stage 5: Excel ─────────────────────────────────────────────────────
        st.session_state.stage = 5
        with st.spinner("📊 Building formatted Excel with color coding..."):
            feature_name = "Feature"
            if images:        feature_name = os.path.splitext(images[0].name)[0]
            elif doc_files:   feature_name = os.path.splitext(doc_files[0].name)[0]
            elif manual_text:
                first = manual_text.strip().split("\n")[0]
                feature_name = first[:30].replace("Feature:","").strip() or "Feature"
            excel = build_excel(test_cases, feature_name)
            st.session_state.excel_bytes = excel

        st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card-label">📋 Results</div>', unsafe_allow_html=True)

    # Show OCR warnings if any
    if st.session_state.ocr_warnings:
        for w in st.session_state.ocr_warnings:
            st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

    if not st.session_state.test_cases_parsed:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E8E5DF;border-radius:12px;
                    padding:4rem 2rem;text-align:center;">
            <div style="font-size:3rem">⏳</div>
            <div style="margin-top:0.8rem;font-size:0.9rem;color:#9B8EA0;">
                Results will appear here after generation
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        tc   = st.session_state.tc_count   or 0
        pos  = st.session_state.pos_count  or 0
        neg  = st.session_state.neg_count  or 0
        edge = st.session_state.edge_count or 0
        rag  = st.session_state.rag_count  or 0

        st.markdown(f'<div class="success-box">✅ &nbsp; <strong>{tc} enterprise test cases generated!</strong></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-strip">
            <div class="metric"><div class="metric-val" style="color:#2E7D32">{pos}</div><div class="metric-lbl">✅ Positive</div></div>
            <div class="metric"><div class="metric-val" style="color:#C00000">{neg}</div><div class="metric-lbl">❌ Negative</div></div>
            <div class="metric"><div class="metric-val" style="color:#F59E0B">{edge}</div><div class="metric-lbl">⚠️ Edge</div></div>
            <div class="metric"><div class="metric-val">{rag}</div><div class="metric-lbl">📚 RAG</div></div>
        </div>""", unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["📊 Test Cases Preview", "🧠 Feature Understanding", "🔍 Extracted Text"])

        with t1:
            tcs     = st.session_state.test_cases_parsed
            preview = "\n\n".join(
                f"{tc.get('id','')} | {tc.get('type','')} | {tc.get('priority','')}\n"
                f"Title: {tc.get('title','')}\n"
                f"Prerequisites: {tc.get('prerequisites','')}\n"
                f"Steps:\n{tc.get('steps','')}\n"
                f"Expected: {tc.get('expected','')}"
                for tc in tcs[:5]
            ) + (f"\n\n... and {len(tcs)-5} more test cases in Excel" if len(tcs) > 5 else "")
            st.markdown(f'<div class="result-preview">{preview}</div>', unsafe_allow_html=True)

        with t2:
            st.markdown(f'<div class="result-preview">{st.session_state.feature_understand or "—"}</div>', unsafe_allow_html=True)

        with t3:
            st.markdown(f'<div class="result-preview">{st.session_state.extracted_text or "—"}</div>', unsafe_allow_html=True)

        st.markdown("---")

        fname = "Feature"
        if images:        fname = os.path.splitext(images[0].name)[0]
        elif st.session_state.test_cases_parsed:
            fname = st.session_state.test_cases_parsed[0].get("id","Feature").split("-")[0] + "_Feature"

        st.download_button(
            label     = "⬇️ DOWNLOAD EXCEL TEST CASES",
            data      = st.session_state.excel_bytes,
            file_name = f"{fname}_TestCases.xlsx",
            mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if st.button("🔄 Generate New"):
            for k in ["extracted_text","feature_understand","test_cases_parsed",
                      "excel_bytes","tc_count","pos_count","neg_count","edge_count","ocr_warnings"]:
                st.session_state[k] = [] if k in ["test_cases_parsed","ocr_warnings"] else None
            st.session_state.stage = 0
            st.rerun()
