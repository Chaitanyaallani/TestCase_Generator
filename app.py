import streamlit as st
from groq import Groq
import pytesseract
from PIL import Image
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import chromadb
from sentence_transformers import SentenceTransformer
import io, os, re, json

st.set_page_config(page_title="QA Test Case Generator", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

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


# ── Helper: OCR ────────────────────────────────────────────────────────────────
def run_ocr(image: Image.Image) -> str:
    return pytesseract.image_to_string(image.convert("RGB"), config="--psm 6").strip()


# ── Helper: RAG ────────────────────────────────────────────────────────────────
def load_excel_to_rag(excel_bytes: bytes) -> tuple:
    embed, _, col = load_embed_model(), None, load_chroma()[1]
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
            row_text = "\n".join(f"{headers[i]}: {v}" for i, v in enumerate(row) if v is not None and i < len(headers))
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
    embed, _, col = load_embed_model(), None, load_chroma()[1]
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
# CORE FIX: Generate test cases as JSON — no pipe parsing issues ever
# ══════════════════════════════════════════════════════════════════════════════
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
    """
    Generate test cases as JSON array — completely solves the
    multiline step parsing problem that was breaking pipe-delimited output.
    """
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

    # Study examples from Copilot style
    copilot_example = """
EXAMPLE OF REQUIRED QUALITY (match this exactly):

TC-3651 | Verify year format change in incident report template displays correctly
Prerequisites: User: ignio admin logged in, Incident report template accessible, Test incident data available
Steps:
  1. Navigate to incident management system
  2. Open an existing incident with date "Jan 23, 2026"
  3. Generate incident report using standard template
  4. Review the date format in generated report
  5. Verify year is displayed in new required format
Expected: Incident report displays date with correct year format. Template adheres to new format standard. No formatting errors in report.
Type: Positive | Priority: High

TC-3669 | Verify system handles invalid year format input gracefully
Prerequisites: User: ignio admin logged in, Template configuration accessible
Steps:
  1. Navigate to template configuration
  2. Attempt to manually set invalid year format (e.g., 3-digit year)
  3. Try to save configuration
  4. Observe system response
  5. Verify error handling
Expected: System displays appropriate error message. Invalid format is rejected with clear explanation. Original format is preserved.
Type: Negative | Priority: High

TC-3684 | Verify template handles year boundary transition (Dec 31 to Jan 1)
Prerequisites: User: ignio admin logged in, Data spanning year boundary available
Steps:
  1. Generate template with data from Dec 31, 2025 to Jan 1, 2026
  2. Review date formatting across year boundary
  3. Check chronological ordering
  4. Verify no date format inconsistencies
  5. Test with multiple year boundaries
Expected: Template correctly handles year boundary transitions. Date formatting remains consistent across boundaries. Chronological ordering preserved correctly.
Type: Edge | Priority: High
"""

    prompt = f"""You are a Senior QA Architect generating enterprise UI test cases.

FEATURE INPUT:
{combined_text}

FEATURE UNDERSTANDING:
{feature_understanding}

SIMILAR PAST TEST CASES (study their style):
{similar_cases}

{copilot_example}

TASK: Generate {num_cases} test cases ({pos} Positive + {neg} Negative + {edge} Edge).

STRICT QUALITY RULES:
1. Prerequisites: MUST include "User: [specific role] logged in, [specific module] accessible, [specific test data] available"
2. Steps: MUST be EXACTLY 5 steps — each step is ONE specific UI action on a named UI element
3. Expected Result: MUST describe the EXACT visible UI outcome — screen name, message text, element state — NEVER say "works correctly" or "saves successfully"
4. Test Type: MUST be exactly Positive / Negative / Edge
5. Priority: High = core flows and AC coverage, Medium = validations, Low = edge/boundary
6. Title: MUST be clear and action-oriented — start with "Verify..."
7. IDs start from TC-{start_id:04d} and increment

FORBIDDEN in Expected Result: "works correctly", "saves successfully", "behaves as expected", "is displayed correctly" alone

RETURN ONLY valid JSON array. No explanation. No markdown. No extra text.
Format:
[
  {{
    "id": "TC-{start_id:04d}",
    "title": "Verify year format change in incident report template displays correctly",
    "prerequisites": "User: ignio admin logged in, Incident report template accessible, Test incident data available",
    "steps": "1. Navigate to incident management system\\n2. Open existing incident with date Jan 23 2026\\n3. Generate incident report using standard template\\n4. Review date format in generated report\\n5. Verify year displays in new required format",
    "expected": "Incident report displays date with correct year format. Template adheres to new format standard. No formatting errors in report.",
    "type": "Positive",
    "priority": "High",
    "related_tc": "None"
  }}
]

Generate exactly {num_cases} objects in the JSON array.
"""

    raw = call_groq(prompt, api_key, max_tokens=4000)

    # Robust JSON extraction
    try:
        # Try direct parse
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    try:
        # Extract JSON array from response
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
    except Exception:
        pass

    try:
        # Strip markdown fences
        cleaned = re.sub(r'```json|```', '', raw).strip()
        parsed  = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # If all parsing fails, return empty list
    return []


# ── Helper: Build Excel ────────────────────────────────────────────────────────
def build_excel(test_cases: list, feature_name: str) -> bytes:
    """
    Build Excel from parsed list of dicts.
    Color coding: Green=Positive, Orange=Negative, Yellow=Edge
    """
    wb = openpyxl.Workbook()

    # Counts
    pos_count  = sum(1 for tc in test_cases if "positive" in str(tc.get("type","")).lower())
    neg_count  = sum(1 for tc in test_cases if "negative" in str(tc.get("type","")).lower())
    edge_count = sum(1 for tc in test_cases if "edge"     in str(tc.get("type","")).lower())

    # ── SUMMARY SHEET ─────────────────────────────────────────────────────────
    ws_sum       = wb.active
    ws_sum.title = "SUMMARY"
    h_fill = PatternFill("solid", fgColor="1A1A2E")
    h_font = Font(bold=True, color="E8F4FD", name="Calibri", size=11)
    c_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    sum_headers = ["Feature / Module Name","Total Test Cases","Positive Count",
                   "Negative Count","Edge Case Count","Acceptance Criteria Covered","Gaps Identified"]
    for c, h in enumerate(sum_headers, 1):
        cell = ws_sum.cell(row=1, column=c, value=h)
        cell.fill = h_fill; cell.font = h_font; cell.alignment = c_align
        ws_sum.column_dimensions[get_column_letter(c)].width = 22

    ws_sum.cell(row=2, column=1, value=feature_name)
    ws_sum.cell(row=2, column=2, value=len(test_cases))
    ws_sum.cell(row=2, column=3, value=pos_count)
    ws_sum.cell(row=2, column=4, value=neg_count)
    ws_sum.cell(row=2, column=5, value=edge_count)
    ws_sum.cell(row=2, column=6, value="Extracted from feature input")
    ws_sum.cell(row=2, column=7, value="Legacy system integration, Performance impact testing")
    ws_sum.freeze_panes = "A2"

    # ── TEST CASE SHEET ────────────────────────────────────────────────────────
    ws       = wb.create_sheet(title=feature_name[:31])
    tc_hdrs  = ["Test Case ID","Test Case Title","Prerequisites",
                "Test Steps","Expected Result","Test Type","Priority","Related TC ID"]
    widths   = [14, 35, 32, 55, 42, 12, 10, 14]

    pos_fill  = PatternFill("solid", fgColor="E2EFDA")  # green
    neg_fill  = PatternFill("solid", fgColor="FCE4D6")  # orange/red
    edge_fill = PatternFill("solid", fgColor="FFF2CC")  # yellow
    pos_alt   = PatternFill("solid", fgColor="D0E8C5")
    neg_alt   = PatternFill("solid", fgColor="F0D0B8")
    edge_alt  = PatternFill("solid", fgColor="EFE8AA")

    n_font  = Font(name="Calibri", size=10)
    l_align = Alignment(horizontal="left",   vertical="top", wrap_text=True)
    mid_align = Alignment(horizontal="center", vertical="top", wrap_text=True)
    thin    = Side(style="thin", color="CCCCCC")
    bdr     = Border(left=thin, right=thin, top=thin, bottom=thin)

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
        if "negative" in tc_type:
            fill = neg_alt  if is_alt else neg_fill
        elif "edge" in tc_type:
            fill = edge_alt if is_alt else edge_fill
        else:
            fill = pos_alt  if is_alt else pos_fill

        values = [
            tc.get("id",          f"TC-{ri+3000:04d}"),
            tc.get("title",       ""),
            tc.get("prerequisites",""),
            tc.get("steps",       ""),
            tc.get("expected",    ""),
            tc.get("type",        ""),
            tc.get("priority",    ""),
            tc.get("related_tc",  "None"),
        ]

        for ci, val in enumerate(values, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.font   = n_font
            cell.border = bdr
            cell.fill   = fill
            cell.alignment = mid_align if ci in [1,6,7,8] else l_align
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
    <div class="app-sub">Copilot-quality enterprise test cases — powered by Groq AI + RAG</div>
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
    api_key = st.text_input("Groq API Key", type="password", placeholder="Paste your gsk_... key", label_visibility="hidden")
    st.caption("Free key → [console.groq.com](https://console.groq.com)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Existing Test Cases (RAG)</div>', unsafe_allow_html=True)
    st.caption("Upload master Excel → app avoids duplicates and matches your style")
    past_excel = st.file_uploader("Upload existing test cases", type=["xlsx","xls"], label_visibility="hidden")

    if past_excel and not st.session_state.rag_loaded:
        with st.spinner("Loading into RAG..."):
            count, last_id = load_excel_to_rag(past_excel.read())
        st.success(f"✅ {count} test cases loaded | Next ID: TC-{last_id+1:04d}")
    elif past_excel and st.session_state.rag_loaded:
        st.markdown(f'<div class="rag-ok">✅ {st.session_state.rag_count} test cases in RAG<br>Next TC ID: TC-{st.session_state.last_tc_id+1:04d}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rag-warn">⚠️ Upload your master Excel for better quality and duplicate prevention</div>', unsafe_allow_html=True)

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
    steps = ["🔍 OCR — Read Input","🧠 Parse Feature Understanding",
             "📚 RAG — Find Similar Cases","⚙️ Generate Test Cases (JSON)",
             "📊 Build Excel with Color Coding"]
    for i, label in enumerate(steps):
        color = "#2E7D32" if i < stage else ("#1565C0" if i == stage and stage > 0 else "#9B8EA0")
        icon  = "✅" if i < stage else ("⏳" if i == stage and stage > 0 else "○")
        st.markdown(f'<div style="color:{color};font-size:0.85rem;padding:0.2rem 0;">{icon} {label}</div>', unsafe_allow_html=True)


# ── Main Layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-label">📤 Feature Input</div>', unsafe_allow_html=True)
    tab_img, tab_txt = st.tabs(["🖼️ Upload Images", "📝 Paste Feature / Jira Story"])

    with tab_img:
        images = st.file_uploader("Upload feature images", type=["jpg","jpeg","png","webp","bmp"], accept_multiple_files=True, label_visibility="hidden")
        if images:
            for img in images:
                st.image(img, caption=img.name, use_container_width=True)

    with tab_txt:
        manual_text = st.text_area("Feature description", height=240,
            placeholder="""Paste your Jira story, acceptance criteria, or feature description.

Example:
Feature: Year Format Changes in Templates
User: As ignio admin, I want all system templates to use the new year format.

Acceptance Criteria:
1. Incident report template shows date in new format
2. Email notification template uses new year format
3. CSV and Excel exports maintain new format
4. Dashboard date filters display new format
5. System rejects invalid year formats with error

Modules affected: incident report, dashboard, email notifications,
CSV export, Excel export, PDF export, maintenance window,
alert template, rule mining, workitem evidence tab

User Roles: ignio admin, regular user (read-only)""",
            label_visibility="hidden")

    st.markdown("")
    has_image = bool(images)
    has_text  = bool(manual_text and manual_text.strip())
    has_input = has_image or has_text
    can_run   = has_input and bool(api_key)

    if not api_key:       st.caption("🔑 Add your Groq API key in the sidebar")
    elif not has_input:   st.caption("⬆️ Upload an image or paste a feature description")

    if st.button("⚡ GENERATE TEST CASES", disabled=not can_run):

        # Stage 1: OCR
        st.session_state.stage = 1
        combined_text = ""
        if has_image:
            with st.spinner("🔍 Extracting text from images..."):
                for img_file in images:
                    text = run_ocr(Image.open(img_file))
                    if text: combined_text += f"\n\n[From: {img_file.name}]\n{text}"
        if has_text:
            combined_text += f"\n\n[Feature Description]\n{manual_text.strip()}"
        combined_text = combined_text.strip()
        if not combined_text:
            st.error("❌ Could not extract text. Please add a description."); st.stop()
        st.session_state.extracted_text = combined_text

        # Stage 2: Feature Understanding
        st.session_state.stage = 2
        with st.spinner("🧠 Parsing feature and extracting acceptance criteria..."):
            fu_prompt = f"""You are a Senior QA Architect. Parse this feature input and produce a FEATURE UNDERSTANDING.

Return EXACTLY in this format:
Feature Name        : [name]
User Roles          : [all roles]
Screens / Pages     : [all screens]
UI Components       : [all buttons, fields, dropdowns]
User Flows          : [numbered list of flows]
Acceptance Criteria : [numbered list — one per line]
Business Rules      : [constraints, validations, permissions]
Error / Edge States : [error messages, boundary conditions]

Feature Input:
{combined_text}
"""
            feature_understanding = call_groq(fu_prompt, api_key, max_tokens=1000)
            st.session_state.feature_understand = feature_understanding

        # Stage 3: RAG
        st.session_state.stage = 3
        with st.spinner("📚 Retrieving similar past test cases..."):
            similar = rag_retrieve(feature_understanding)

        # Stage 4: Generate as JSON
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
                st.error("❌ AI returned invalid JSON. Please try again.")
                st.stop()

            st.session_state.test_cases_parsed = test_cases
            st.session_state.tc_count   = len(test_cases)
            st.session_state.pos_count  = sum(1 for t in test_cases if "positive" in t.get("type","").lower())
            st.session_state.neg_count  = sum(1 for t in test_cases if "negative" in t.get("type","").lower())
            st.session_state.edge_count = sum(1 for t in test_cases if "edge"     in t.get("type","").lower())

        # Stage 5: Excel
        st.session_state.stage = 5
        with st.spinner("📊 Building formatted Excel with color coding..."):
            feature_name = "Feature"
            if images:   feature_name = os.path.splitext(images[0].name)[0]
            elif manual_text:
                first = manual_text.strip().split("\n")[0]
                feature_name = first[:30].replace("Feature:","").strip() or "Feature"
            excel = build_excel(test_cases, feature_name)
            st.session_state.excel_bytes = excel

        st.rerun()


with col_right:
    st.markdown('<div class="card-label">📋 Results</div>', unsafe_allow_html=True)

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

        st.markdown(f'<div class="success-box">✅ &nbsp; <strong>{tc} Copilot-quality test cases generated!</strong></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-strip">
            <div class="metric"><div class="metric-val" style="color:#2E7D32">{pos}</div><div class="metric-lbl">✅ Positive</div></div>
            <div class="metric"><div class="metric-val" style="color:#C00000">{neg}</div><div class="metric-lbl">❌ Negative</div></div>
            <div class="metric"><div class="metric-val" style="color:#F59E0B">{edge}</div><div class="metric-lbl">⚠️ Edge</div></div>
            <div class="metric"><div class="metric-val">{rag}</div><div class="metric-lbl">📚 RAG</div></div>
        </div>""", unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["📊 Test Cases Preview", "🧠 Feature Understanding", "🔍 OCR Output"])

        with t1:
            tcs = st.session_state.test_cases_parsed
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
        if images:   fname = os.path.splitext(images[0].name)[0]
        elif manual_text:
            fname = manual_text.strip().split("\n")[0][:30].replace("Feature:","").strip() or "Feature"

        st.download_button(
            label     = "⬇️ DOWNLOAD EXCEL TEST CASES",
            data      = st.session_state.excel_bytes,
            file_name = f"{fname}_TestCases.xlsx",
            mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if st.button("🔄 Generate New"):
            for k in ["extracted_text","feature_understand","test_cases_parsed",
                      "excel_bytes","tc_count","pos_count","neg_count","edge_count"]:
                st.session_state[k] = None if k != "test_cases_parsed" else []
            st.session_state.stage = 0
            st.rerun()
