import streamlit as st
import anthropic
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import chromadb
from sentence_transformers import SentenceTransformer
import io, os, re, json, base64

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


# ── Helper: Claude API ─────────────────────────────────────────────────────────
def call_claude(system_prompt: str, user_content, api_key: str, max_tokens: int = 8000) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    if isinstance(user_content, str):
        user_content = [{"type": "text", "text": user_content}]
    response = client.messages.create(
        model      = "claude-sonnet-4-5-20251001",
        max_tokens = max_tokens,
        system     = system_prompt,
        messages   = [{"role": "user", "content": user_content}]
    )
    return response.content[0].text


# ══════════════════════════════════════════════════════════════════════════════
# OCR
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_image(image: Image.Image) -> Image.Image:
    img = image.convert("RGB")
    w, h = img.size
    if w < 1000:
        scale = 1000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.convert("L")
    return img


def run_ocr(image: Image.Image) -> tuple:
    try:
        enhanced = preprocess_image(image)
        text = pytesseract.image_to_string(enhanced, config="--psm 6 --oem 3")
        if len(text.strip()) > 30:
            return text.strip(), []
    except Exception:
        pass
    try:
        text = pytesseract.image_to_string(image.convert("RGB"), config="--psm 6")
        if len(text.strip()) > 30:
            return text.strip(), []
    except Exception:
        pass
    for psm in [11, 3, 4]:
        try:
            text = pytesseract.image_to_string(image.convert("L"), config=f"--psm {psm}")
            if len(text.strip()) > 30:
                return text.strip(), [f"Used alternate OCR mode (psm={psm}) for this image"]
        except Exception:
            pass
    return "", ["⚠️ OCR could not extract text from this image. It may contain embedded images, diagrams, or low-resolution content. Please paste the feature description manually in the text tab."]


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        import zipfile
        from xml.etree import ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            if "word/document.xml" not in z.namelist():
                return ""
            xml_content = z.read("word/document.xml")
        tree = ET.fromstring(xml_content)
        texts = []
        for para in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
            para_text = "".join(
                node.text or ""
                for node in para.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
            )
            if para_text.strip():
                texts.append(para_text.strip())
        return "\n".join(texts)
    except Exception:
        return ""


def extract_text_from_txt(file_bytes: bytes) -> str:
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
    results = col.query(query_embeddings=[emb], n_results=min(10, col.count()))
    docs    = results["documents"][0] if results["documents"] else []
    return "\n\n---\n\n".join(docs) if docs else "No similar cases found."


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — MATCHES GITHUB COPILOT TCprompt.txt QUALITY EXACTLY
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a Senior QA Architect working on an EXISTING enterprise application.

Your job is to generate comprehensive, precise, and non-duplicate UI test cases.

========================================================
STEP 1 — PARSE THE FEATURE INPUT
========================================================

Read all provided input (text + any images provided).

Extract a FEATURE UNDERSTANDING:
Feature Name        : [name from input]
User Roles          : [all roles — Admin, SRE, CIO, Viewer, Capacity Manager, etc.]
Screens / Pages     : [all screens or views]
UI Components       : [all buttons, tabs, dropdowns, tables, modals, fields visible]
User Flows          : [numbered list of every user flow]
Acceptance Criteria : [numbered AC-1, AC-2, etc. — one per line]
Business Rules      : [constraints, validations, permissions]
Error / Edge States : [error messages, empty states, boundary conditions]

========================================================
STEP 2 — DUPLICATE CHECK AGAINST EXISTING TEST CASES
========================================================

Review the existing test cases provided in the user message.
Do NOT generate any test case that duplicates an existing one.
Only generate test cases for NEW gaps.

========================================================
STEP 3 — MANDATORY COVERAGE RULES
========================================================

Positive (min 40%): Happy path for every AC, every user flow, every role.

Negative (min 20%): Missing required fields, invalid data, unauthorized access, wrong role.

Edge Cases (MANDATORY min 30% — DO NOT SKIP THESE):
- Empty state when no data exists
- Maximum character limit on inputs
- Special characters in input fields
- Page refresh mid-flow
- Browser back button during multi-step flow
- Session expiry during active flow
- Boundary value inputs
- Simultaneous actions if applicable

========================================================
CRITICAL FIELD WRITING RULES — READ CAREFULLY
========================================================

PREREQUISITES — MUST be a numbered list with all three of:
1. User role and login state (exact role name)
2. Screen name or URL that must be accessible
3. Specific test data conditions required

CORRECT prerequisites:
1. User is logged in with Capacity Manager role
2. Multi-Cloud Capacity Dashboard is accessible
3. Test data exists for OnPremise environment with at least 8 CI Types

WRONG prerequisites (NEVER write like this):
"User: ignio SRE logged in, Dashboard accessible, Multiple systems configured"

─────────────────────────────────────────────────────

TEST STEPS — MUST be numbered, one UI action per step:
- Every step = exactly ONE UI action (click, type, select, hover, scroll, navigate)
- Reference the EXACT UI element label as seen on screen
- Always begin from navigation to the feature screen
- Never combine two actions in one step

CORRECT steps:
1. Navigate to ignio application URL
2. Click on "Capacity Management" from the main menu
3. Verify Executive Dashboard loads successfully
4. Click on "OnPremise" tab/section
5. Wait for data to load

WRONG steps (NEVER write like this):
1. Log in to Ignio
2. Navigate to dashboard
3. Verify tiles displayed
4. Confirm layout

─────────────────────────────────────────────────────

EXPECTED RESULT — MUST be multi-line with specific detail:
- Line 1: The primary outcome (what happens)
- Line 2+: Specific UI elements visible — exact column names, button labels, message text, counts, data values
- For negative cases: include the EXACT error message text

CORRECT expected result:
OnPremise view loads successfully.
CI Type table displays with columns: CI TYPE, NO OF SYSTEMS, RISK, POSSIBLE RISK, OPTIMIZATION, HEALTHY.
OnPremise specific CI types are shown (Windows, Linux, CiscoRouter, etc.).
Consolidation Duration shows current selection (PPM-RC3-701).

WRONG expected result (NEVER write like this):
"Ten system tiles displayed, each with system name, CPU usage, and memory usage"

─────────────────────────────────────────────────────

PRIORITY:
High   → core user flows, AC-mapped scenarios, role-based access
Medium → form validations, navigation flows, error messages
Low    → edge cases, cosmetic states, boundary value inputs

ABSOLUTELY FORBIDDEN:
- Vague titles like "Test button" or "Check page loads"
- Prerequisites as a single run-on sentence
- Single-line vague expected results
- Phrases like "works correctly", "saves successfully", "behaves as expected"
- Missing edge cases (must be at least 30% of total)
- Duplicate test cases

========================================================
OUTPUT FORMAT — JSON ONLY
========================================================

Return ONLY a valid JSON array. No markdown, no preamble, no text outside the array.

[
  {
    "id": "TC-XXXX",
    "title": "Verify [specific action] [specific condition]",
    "prerequisites": "1. User is logged in with [Role] role\\n2. [Screen name] is accessible\\n3. [Specific test data condition]",
    "steps": "1. Navigate to [URL/screen]\\n2. Click [exact element name]\\n3. [Action] [exact element]\\n4. [Action] [exact element]\\n5. Verify [specific outcome]",
    "expected": "Primary outcome description.\\nSpecific UI element or column visible: [names].\\nExact message shown: '[message text]'.",
    "type": "Positive",
    "priority": "High",
    "related_tc": "None"
  }
]

type must be exactly one of: Positive, Negative, Edge
priority must be exactly one of: High, Medium, Low
"""


# ══════════════════════════════════════════════════════════════════════════════
# JSON PARSING — ROBUST MULTI-STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
def parse_json_response(raw: str) -> list:
    for attempt in [
        lambda r: json.loads(r),
        lambda r: json.loads(re.search(r'\[[\s\S]*\]', r).group()),
        lambda r: json.loads(re.sub(r'```json|```', '', r).strip()),
        lambda r: json.loads(re.search(r'\[[\s\S]*\]',
                             re.sub(r',\s*}', '}', re.sub(r',\s*\]', ']', r))).group()),
    ]:
        try:
            parsed = attempt(raw)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except Exception:
            pass
    return []


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE UNDERSTANDING — CLAUDE
# ══════════════════════════════════════════════════════════════════════════════
def extract_feature_understanding(combined_text: str, api_key: str, image_blocks: list = None) -> str:
    system = "You are a Senior QA Architect. Parse the feature input and return a structured FEATURE UNDERSTANDING."
    user_text = f"""Parse this feature input and return EXACTLY in this format:

Feature Name        : [name]
User Roles          : [all user roles mentioned]
Screens / Pages     : [all screens/pages]
UI Components       : [all buttons, fields, dropdowns, tabs, tables]
User Flows          : [numbered list of user flows]
Acceptance Criteria : [numbered list — AC-1, AC-2, etc. — one per line]
Business Rules      : [constraints, validations, permissions]
Error / Edge States : [error messages, empty states, boundary conditions]
Modules Affected    : [list all specific modules/templates]

Feature Input:
{combined_text}
"""
    content = [{"type": "text", "text": user_text}]
    if image_blocks:
        content.extend(image_blocks)
    return call_claude(system, content, api_key, max_tokens=1500)


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASE GENERATION — CLAUDE WITH FULL CONTEXT + VISION
# ══════════════════════════════════════════════════════════════════════════════
def generate_test_cases_as_json(
    combined_text: str,
    feature_understanding: str,
    similar_cases: str,
    num_cases: int,
    last_tc_id: int,
    api_key: str,
    include_neg: bool,
    include_edge: bool,
    image_blocks: list = None
) -> list:

    if include_neg and include_edge:
        pos  = max(1, int(num_cases * 0.40))
        edge = max(1, int(num_cases * 0.30))
        neg  = num_cases - pos - edge
    elif include_neg:
        pos = num_cases // 2 + (num_cases % 2)
        neg = num_cases - pos
        edge = 0
    elif include_edge:
        pos  = max(1, int(num_cases * 0.60))
        edge = num_cases - pos
        neg  = 0
    else:
        pos, neg, edge = num_cases, 0, 0

    start_id  = last_tc_id + 1
    auto_mode = (num_cases == 999)

    if auto_mode:
        count_instruction = (
            f"Generate as many test cases as needed for COMPLETE coverage of every AC, "
            f"every user flow, every module, every error state, and ALL edge conditions. "
            f"Minimum 30%% must be Edge type. TC IDs start from TC-{start_id:04d}."
        )
    else:
        count_instruction = (
            f"Generate exactly {num_cases} test cases: "
            f"{pos} Positive + {neg} Negative + {edge} Edge. "
            f"TC IDs start from TC-{start_id:04d}."
        )

    user_text = f"""{count_instruction}

========================================================
FEATURE UNDERSTANDING:
========================================================
{feature_understanding}

========================================================
FULL FEATURE INPUT (use ALL of this, not just part of it):
========================================================
{combined_text}

========================================================
EXISTING TEST CASES — DO NOT DUPLICATE ANY OF THESE:
========================================================
{similar_cases}

========================================================
GENERATION INSTRUCTIONS:
========================================================
- Auto-increment TC IDs from TC-{start_id:04d}
- Do NOT duplicate any existing test case listed above
- Prerequisites MUST be a numbered list:
    1. Exact user role and login state
    2. Screen name or URL
    3. Specific test data conditions
- Test Steps MUST be numbered, atomic, one UI action each, referencing exact UI element names
- Expected Result MUST be multi-line with specific UI outcomes, column names, message text
- Edge cases MUST be at least 30%% of total — include empty state, boundary values, session expiry, page refresh, browser back
- Return ONLY the JSON array — no text before [ or after ]
"""

    content = [{"type": "text", "text": user_text}]
    if image_blocks:
        content.extend(image_blocks)

    raw    = call_claude(SYSTEM_PROMPT, content, api_key, max_tokens=8000)
    result = parse_json_response(raw)
    if result:
        return result

    # Retry — simpler prompt, still full text
    retry_text = f"""Generate {num_cases} QA test cases as a JSON array for this feature.

Feature:
{combined_text[:3000]}

Rules:
- TC IDs from TC-{start_id:04d}
- Prerequisites: numbered list — 1. role  2. screen  3. test data
- Steps: numbered, one UI action each, exact element names
- Expected: multi-line, specific UI outcome with element names and message text
- Types: {pos} Positive, {neg} Negative, {edge} Edge
- Edge cases must cover: empty state, boundary value, session expiry, page refresh

Return ONLY the JSON array starting with [ and ending with ].
"""
    raw    = call_claude(SYSTEM_PROMPT, retry_text, api_key, max_tokens=8000)
    return parse_json_response(raw)


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL BUILDER — MATCHES GITHUB COPILOT OUTPUT FORMAT
# ══════════════════════════════════════════════════════════════════════════════
def build_excel(test_cases: list, feature_name: str) -> bytes:
    wb      = openpyxl.Workbook()
    h_fill  = PatternFill("solid", fgColor="1A1A2E")
    h_font  = Font(bold=True, color="E8F4FD", name="Calibri", size=11)
    c_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    pos_count  = sum(1 for tc in test_cases if "positive" in str(tc.get("type","")).lower())
    neg_count  = sum(1 for tc in test_cases if "negative" in str(tc.get("type","")).lower())
    edge_count = sum(1 for tc in test_cases if "edge"     in str(tc.get("type","")).lower())

    # ── SUMMARY sheet ──────────────────────────────────────────────────────────
    ws_sum = wb.active
    ws_sum.title = "SUMMARY"
    sum_hdrs = [
        "Feature / Module Name", "Total Test Cases Generated", "Positive Count",
        "Negative Count", "Edge Case Count", "Acceptance Criteria Covered", "Gaps Identified"
    ]
    sum_widths = [30, 24, 14, 14, 14, 50, 45]
    for c, (h, w) in enumerate(zip(sum_hdrs, sum_widths), 1):
        cell = ws_sum.cell(row=1, column=c, value=h)
        cell.fill = h_fill; cell.font = h_font; cell.alignment = c_align
        ws_sum.column_dimensions[get_column_letter(c)].width = w

    ws_sum.cell(row=2, column=1, value=feature_name)
    ws_sum.cell(row=2, column=2, value=len(test_cases))
    ws_sum.cell(row=2, column=3, value=pos_count)
    ws_sum.cell(row=2, column=4, value=neg_count)
    ws_sum.cell(row=2, column=5, value=edge_count)
    ws_sum.cell(row=2, column=6, value="Extracted from Acceptance Criteria in feature input")
    ws_sum.cell(row=2, column=7, value="Performance testing under load, Cross-browser compatibility")
    for c in range(1, 8):
        ws_sum.cell(row=2, column=c).alignment = Alignment(wrap_text=True, vertical="top")
        ws_sum.cell(row=2, column=c).font = Font(name="Calibri", size=10)
    ws_sum.freeze_panes = "A2"

    # ── Test case sheet ────────────────────────────────────────────────────────
    safe_name = re.sub(r'[\\/*?:\[\]]', '', feature_name)[:31]
    ws      = wb.create_sheet(title=safe_name)
    tc_hdrs = ["Test Case ID", "Test Case Title", "Prerequisites",
               "Test Steps", "Expected Result", "Test Type", "Priority", "Related TC ID"]
    widths  = [14, 38, 40, 55, 52, 12, 10, 14]

    pos_fill  = PatternFill("solid", fgColor="E2EFDA")
    neg_fill  = PatternFill("solid", fgColor="FCE4D6")
    edge_fill = PatternFill("solid", fgColor="FFF2CC")
    pos_alt   = PatternFill("solid", fgColor="D0E8C5")
    neg_alt   = PatternFill("solid", fgColor="F0D0B8")
    edge_alt  = PatternFill("solid", fgColor="EFE8AA")

    n_font  = Font(name="Calibri", size=10)
    l_align = Alignment(horizontal="left",   vertical="top", wrap_text=True)
    m_align = Alignment(horizontal="center", vertical="top", wrap_text=True)
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
        fill    = (neg_alt  if is_alt else neg_fill)  if "negative" in tc_type else \
                  (edge_alt if is_alt else edge_fill) if "edge"     in tc_type else \
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
            cell.alignment = m_align if ci in [1, 6, 7, 8] else l_align
        ws.row_dimensions[ri].height = 90

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# UI — IDENTICAL LAYOUT TO ORIGINAL APP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="app-title">QA Test Case Generator</div>
    <div class="app-sub">Copilot-quality enterprise test cases — Claude AI + RAG + JSON pipeline</div>
    <div style="margin-top:0.8rem">
        <span class="tag">CLAUDE AI</span><span class="tag">OCR</span>
        <span class="tag">RAG</span><span class="tag">JSON PIPELINE</span><span class="tag">EXCEL EXPORT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🔑 Anthropic API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input("Anthropic API Key", type="password",
                            placeholder="Paste your sk-ant-... key", label_visibility="hidden")
    st.caption("Get your key → [console.anthropic.com](https://console.anthropic.com)")
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
    auto_mode    = st.toggle("🤖 Auto — generate ALL scenarios", value=False)
    if auto_mode:
        num_cases = 999
        st.caption("Claude will generate as many test cases as needed to cover all scenarios")
    else:
        num_cases = st.number_input("Number of test cases", min_value=5, max_value=500, value=15, step=5)
        st.caption(f"Will generate {num_cases} test cases")
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

    tab_img, tab_doc, tab_txt = st.tabs(["🖼️ Images", "📄 Documents (.docx/.txt)", "📝 Text / Jira Story"])

    with tab_img:
        st.markdown("""
        <div class="info-box">
            💡 <strong>Tip:</strong> Claude reads images directly — UI screenshots, wireframes, and mockups
            are passed to Claude for visual understanding. No text extraction needed.
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
Feature: Multi-Cloud Capacity Management Dashboard
As a Capacity Manager, I want to view capacity across cloud providers.

Acceptance Criteria:
AC-1: User can switch between OnPremise, AWS, GCP, Azure views
AC-2: CI Type table shows CI TYPE, NO OF SYSTEMS, RISK, POSSIBLE RISK, OPTIMIZATION, HEALTHY
AC-3: Pagination shows correct record count
AC-4: SRE Dashboard shows forecast view with utilization data

User Roles: Capacity Manager, SRE, Viewer""",
            label_visibility="hidden"
        )

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

    has_image = bool(images)
    has_doc   = bool(doc_files)
    has_text  = bool(manual_text and manual_text.strip())
    has_input = has_image or has_doc or has_text
    can_run   = has_input and bool(api_key)

    if not api_key:     st.caption("🔑 Add your Anthropic API key in the sidebar")
    elif not has_input: st.caption("⬆️ Upload images, documents, or paste text above")

    # ── GENERATE BUTTON ────────────────────────────────────────────────────────
    if st.button("⚡ GENERATE TEST CASES", disabled=not can_run):

        combined_text = ""
        ocr_warnings  = []
        image_blocks  = []  # Claude vision content blocks

        # ── Stage 1: Extract text + build image blocks ─────────────────────────
        st.session_state.stage = 1

        if has_image:
            with st.spinner("🔍 Processing images for Claude vision..."):
                for img_file in images:
                    # Pass image directly to Claude (vision) — no OCR dependency
                    img_file.seek(0)
                    img_bytes  = img_file.read()
                    ext        = img_file.name.rsplit(".", 1)[-1].lower()
                    media_map  = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png",
                                  "webp":"image/webp","bmp":"image/png"}
                    media_type = media_map.get(ext, "image/png")
                    b64        = base64.standard_b64encode(img_bytes).decode("utf-8")
                    image_blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64}
                    })
                    # Also run OCR as supplementary text context
                    img_file.seek(0)
                    pil_img    = Image.open(img_file)
                    text, warns = run_ocr(pil_img)
                    ocr_warnings.extend(warns)
                    if text:
                        combined_text += f"\n\n[From image: {img_file.name}]\n{text}"

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

        if has_text:
            combined_text += f"\n\n[Feature Description]\n{manual_text.strip()}"

        combined_text = combined_text.strip()
        st.session_state.ocr_warnings = ocr_warnings

        if not combined_text and not image_blocks:
            st.error("""
❌ Could not extract any text from your inputs.

**What to do:**
1. Open your VD link / Figma / document in browser
2. Copy the text content (Ctrl+A → Ctrl+C)
3. Paste it in the **📝 Text / Jira Story** tab
4. Click Generate again
""")
            st.stop()

        if len(combined_text) < 50 and not image_blocks:
            st.warning(
                "⚠️ Very little text was extracted. Results may be generic. "
                "Consider adding more detail in the Text tab."
            )

        st.session_state.extracted_text = combined_text

        # ── Stage 2: Feature Understanding via Claude ──────────────────────────
        st.session_state.stage = 2
        with st.spinner("🧠 Parsing feature and extracting acceptance criteria..."):
            feature_understanding = extract_feature_understanding(
                combined_text, api_key,
                image_blocks if image_blocks else None
            )
            st.session_state.feature_understand = feature_understanding

        # ── Stage 3: RAG ──────────────────────────────────────────────────────
        st.session_state.stage = 3
        with st.spinner("📚 Retrieving similar past test cases..."):
            similar = rag_retrieve(feature_understanding)

        # ── Stage 4: Generate Test Cases via Claude ────────────────────────────
        st.session_state.stage = 4
        last_tc_id = st.session_state.last_tc_id or 3000

        with st.spinner("⚙️ Generating Copilot-quality test cases (30–60 seconds)..."):
            test_cases = generate_test_cases_as_json(
                combined_text         = combined_text,
                feature_understanding = feature_understanding,
                similar_cases         = similar,
                num_cases             = num_cases,
                last_tc_id            = last_tc_id,
                api_key               = api_key,
                include_neg           = include_neg,
                include_edge          = include_edge,
                image_blocks          = image_blocks if image_blocks else None,
            )

            if not test_cases:
                st.error("""
❌ AI returned invalid JSON after 2 attempts.

**Possible causes:**
- Feature description is too short or unclear
- API rate limit hit — wait 30 seconds and try again

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
            if images:       feature_name = os.path.splitext(images[0].name)[0]
            elif doc_files:  feature_name = os.path.splitext(doc_files[0].name)[0]
            elif manual_text:
                first = manual_text.strip().split("\n")[0]
                feature_name = first[:30].replace("Feature:","").strip() or "Feature"
            excel = build_excel(test_cases, feature_name)
            st.session_state.excel_bytes = excel

        st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card-label">📋 Results</div>', unsafe_allow_html=True)

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
                f"Prerequisites:\n{tc.get('prerequisites','')}\n"
                f"Steps:\n{tc.get('steps','')}\n"
                f"Expected:\n{tc.get('expected','')}"
                for tc in tcs[:5]
            ) + (f"\n\n... and {len(tcs)-5} more test cases in Excel" if len(tcs) > 5 else "")
            st.markdown(f'<div class="result-preview">{preview}</div>', unsafe_allow_html=True)

        with t2:
            st.markdown(f'<div class="result-preview">{st.session_state.feature_understand or "—"}</div>', unsafe_allow_html=True)

        with t3:
            st.markdown(f'<div class="result-preview">{st.session_state.extracted_text or "—"}</div>', unsafe_allow_html=True)

        st.markdown("---")

        fname = "Feature"
        if images:      fname = os.path.splitext(images[0].name)[0]
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
