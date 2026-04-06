import streamlit as st
from groq import Groq
import pytesseract
from PIL import Image, ImageEnhance
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from sentence_transformers import SentenceTransformer
import io, os, re, json, math
import zipfile
from xml.etree import ElementTree as ET

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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
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
.app-title { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 700; color: #E8F4FD; margin: 0; }
.app-sub   { color: #7B8FA1; font-size: 0.9rem; margin-top: 0.4rem; }
.tag { display: inline-block; background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12); color: #A8C8E8; padding: 0.15rem 0.65rem; border-radius: 4px; font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace; margin: 0.3rem 0.2rem 0 0; }
.card-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #9B8EA0; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem; }
.result-preview { background: #1A1A2E; border-radius: 8px; padding: 1rem 1.2rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #A8C8E8; white-space: pre-wrap; max-height: 300px; overflow-y: auto; margin-top: 0.5rem; }
.metric-strip { display: flex; gap: 1rem; margin: 1rem 0; }
.metric { flex: 1; background: #FFFFFF; border: 1px solid #E8E5DF; border-radius: 10px; padding: 1rem; text-align: center; }
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #1A1A2E; }
.metric-lbl { font-size: 0.7rem; color: #9B8EA0; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }
.success-box { background: #E8F5E9; border: 1px solid #A5D6A7; border-left: 4px solid #2E7D32; border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1rem 0; color: #1B5E20; font-weight: 500; }
.warn-box  { background: #FFF3CD; border: 1px solid #FFD580; border-left: 4px solid #F59E0B; border-radius: 8px; padding: 0.8rem 1.2rem; margin: 0.5rem 0; color: #92400E; font-size: 0.88rem; }
.info-box  { background: #E3F2FD; border: 1px solid #90CAF9; border-left: 4px solid #1565C0; border-radius: 8px; padding: 0.8rem 1.2rem; margin: 0.5rem 0; color: #0D47A1; font-size: 0.88rem; }
.rag-ok   { background: #E8F5E9; border: 1px solid #A5D6A7; border-left: 3px solid #2E7D32; border-radius: 6px; padding: 0.6rem 0.8rem; font-size: 0.8rem; color: #1B5E20; margin-top: 0.5rem; }
.rag-warn { background: #FFF3CD; border: 1px solid #FFD580; border-left: 3px solid #F59E0B; border-radius: 6px; padding: 0.6rem 0.8rem; font-size: 0.8rem; color: #92400E; margin-top: 0.5rem; }
.llm-badge { display: inline-block; background: #E3F2FD; border: 1px solid #90CAF9; color: #0D47A1; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; }
.stButton > button { background: #1A1A2E !important; color: #E8F4FD !important; border: none !important; border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important; font-weight: 600 !important; letter-spacing: 1px !important; padding: 0.65rem 1.5rem !important; width: 100% !important; transition: all 0.2s !important; }
.stButton > button:hover { background: #2D2D4E !important; transform: translateY(-1px) !important; box-shadow: 0 4px 16px rgba(26,26,46,0.25) !important; }
.stButton > button:disabled { background: #C8C5BF !important; color: #8A8580 !important; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E8E5DF; }
.sidebar-section { background: #F7F6F3; border: 1px solid #E8E5DF; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
.sidebar-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 2px; color: #9B8EA0; margin-bottom: 0.6rem; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LLM PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════
LLM_PROVIDERS = {
    "Groq — LLaMA 3.1 8B (fastest)": {
        "provider"  : "groq",
        "model"     : "llama-3.1-8b-instant",
        "key_hint"  : "gsk_...",
        "key_link"  : "https://console.groq.com",
        "limit"     : "14,400 req/day free",
        "max_chars" : 6000,       # ← reduced from 12000
        "ctx_window": 8192,
    },
    "Groq — LLaMA 3.3 70B (smarter)": {
        "provider"  : "groq",
        "model"     : "llama-3.3-70b-versatile",
        "key_hint"  : "gsk_...",
        "key_link"  : "https://console.groq.com",
        "limit"     : "1,000 req/day free",
        "max_chars" : 8000,       # ← reduced from 16000
        "ctx_window": 32768,
    },
    "Groq — Gemma 2 9B": {
        "provider"  : "groq",
        "model"     : "gemma2-9b-it",
        "key_hint"  : "gsk_...",
        "key_link"  : "https://console.groq.com",
        "limit"     : "14,400 req/day free",
        "max_chars" : 6000,       # ← reduced from 12000
        "ctx_window": 8192,
    },
    "Groq — Mixtral 8x7B": {
        "provider"  : "groq",
        "model"     : "mixtral-8x7b-32768",
        "key_hint"  : "gsk_...",
        "key_link"  : "https://console.groq.com",
        "limit"     : "14,400 req/day free",
        "max_chars" : 10000,      # ← reduced from 20000
        "ctx_window": 32768,
    },
    "Together AI — LLaMA 3 8B (free tier)": {
        "provider"  : "together",
        "model"     : "meta-llama/Llama-3-8b-chat-hf",
        "key_hint"  : "...",
        "key_link"  : "https://api.together.xyz",
        "limit"     : "$25 free credits",
        "max_chars" : 6000,       # ← reduced from 12000
        "ctx_window": 8192,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# FIX: Robust token-safe LLM call with proper error handling
# ══════════════════════════════════════════════════════════════════════════════
def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters for English text."""
    return len(text) // 4


def call_llm(prompt: str, api_key: str, provider_config: dict, max_tokens: int = 4000) -> str:
    provider   = provider_config["provider"]
    model      = provider_config["model"]
    max_chars  = provider_config.get("max_chars", 6000)
    ctx_window = provider_config.get("ctx_window", 8192)

    # ── Enforce hard character limit ──
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]

    # ── Ensure prompt + max_tokens fits in context window ──
    prompt_tokens = estimate_tokens(prompt)
    safe_max_tokens = min(max_tokens, ctx_window - prompt_tokens - 200)   # 200 token buffer
    if safe_max_tokens < 500:
        # Prompt is still too big — aggressively trim
        prompt = prompt[: (ctx_window - max_tokens - 200) * 4]
        safe_max_tokens = max_tokens

    # Guard against empty or whitespace-only prompts
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty after trimming.")

    if provider == "groq":
        try:
            client   = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model       = model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = safe_max_tokens,
                temperature = 0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_msg = str(e).lower()
            # If it's a token / length error, retry with a much shorter prompt
            if any(kw in err_msg for kw in ["token", "length", "too long", "maximum", "context"]):
                trimmed = prompt[: len(prompt) // 2]
                client  = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model       = model,
                    messages    = [{"role": "user", "content": trimmed}],
                    max_tokens  = min(safe_max_tokens, 2000),
                    temperature = 0.3,
                )
                return response.choices[0].message.content
            raise   # re-raise non-token errors

    elif provider == "together":
        import urllib.request as ur
        body = json.dumps({
            "model"      : model,
            "messages"   : [{"role": "user", "content": prompt}],
            "max_tokens" : safe_max_tokens,
            "temperature": 0.3,
        }).encode()
        req = ur.Request(
            "https://api.together.xyz/v1/chat/completions",
            data    = body,
            headers = {"Authorization": f"Bearer {api_key}",
                       "Content-Type": "application/json"},
        )
        with ur.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown provider: {provider}")


# ══════════════════════════════════════════════════════════════════════════════
# SAFE EXCEL SHEET NAME
# ══════════════════════════════════════════════════════════════════════════════
def safe_sheet_name(name: str, fallback: str = "TestCases") -> str:
    if not name or not str(name).strip():
        return fallback
    s = re.sub(r'[:\\/?*\[\]]', '_', str(name))
    s = s.strip(" '")
    s = re.sub(r'_+', '_', s)
    s = s[:31].strip(" '_")
    return s if s else fallback


# ══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
class SimpleVectorStore:
    def __init__(self):
        self.documents : list = []
        self.embeddings: list = []
        self.ids       : set  = set()

    def add(self, documents, embeddings, ids):
        for doc, emb, id_ in zip(documents, embeddings, ids):
            if id_ not in self.ids:
                self.documents.append(doc)
                self.embeddings.append(emb)
                self.ids.add(id_)

    def count(self): return len(self.documents)

    def query(self, query_embeddings, n_results=10):
        if not self.documents:
            return {"documents": [[]]}
        qe = query_embeddings[0]
        def cos(a, b):
            d = sum(x*y for x, y in zip(a, b))
            return d / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)) + 1e-9)
        scored = sorted(
            [(cos(qe, e), d) for e, d in zip(self.embeddings, self.documents)],
            reverse=True
        )
        return {"documents": [[d for _, d in scored[:n_results]]]}

    def clear(self):
        self.documents = []; self.embeddings = []; self.ids = set()


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

@st.cache_resource
def get_vector_store():
    return SimpleVectorStore()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "extracted_text"    : None,
    "feature_understand": None,
    "test_cases_parsed" : [],
    "excel_bytes"       : None,
    "stage"             : 0,
    "tc_count"          : 0,
    "pos_count"         : 0,
    "neg_count"         : 0,
    "edge_count"        : 0,
    "rag_loaded"        : False,
    "rag_count"         : 0,
    "rag_docs"          : [],
    "last_tc_id"        : 3000,
    "ocr_warnings"      : [],
    "llm_used"          : "",
    "feature_name"      : "Feature",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# OCR
# ══════════════════════════════════════════════════════════════════════════════
def run_ocr(image: Image.Image) -> tuple:
    def try_ocr(img, cfg):
        try:
            t = pytesseract.image_to_string(img, config=cfg).strip()
            return t if len(t) > 30 else ""
        except Exception:
            return ""
    img = image.convert("RGB")
    w, h = img.size
    if w < 1000:
        img = img.resize((1000, int(h * 1000 / w)), Image.LANCZOS)
    img  = ImageEnhance.Contrast(img).enhance(2.0)
    img  = ImageEnhance.Sharpness(img).enhance(2.0)
    gray = img.convert("L")
    for cfg in ["--psm 6 --oem 3", "--psm 11", "--psm 3"]:
        t = try_ocr(gray, cfg)
        if t: return t, []
    t = try_ocr(image.convert("RGB"), "--psm 6")
    if t: return t, []
    return "", ["⚠️ OCR failed. Please paste text in the Text tab."]


def extract_docx_text(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            if "word/document.xml" not in z.namelist(): return ""
            xml = z.read("word/document.xml")
        NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        tree = ET.fromstring(xml)
        paras = []
        for p in tree.iter(f"{{{NS}}}p"):
            t = "".join(n.text or "" for n in p.iter(f"{{{NS}}}t"))
            if t.strip(): paras.append(t.strip())
        return "\n".join(paras)
    except Exception:
        return ""


def extract_txt_text(file_bytes: bytes) -> str:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try: return file_bytes.decode(enc).strip()
        except Exception: pass
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# RAG
# ══════════════════════════════════════════════════════════════════════════════
def load_excel_to_rag(excel_bytes: bytes) -> tuple:
    embed = load_embed_model(); store = get_vector_store()
    total = 0; last_id = 3000
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    for sname in wb.sheetnames:
        ws = wb[sname]
        if ws.max_row < 3: continue
        headers = [str(c.value).strip().lower() if c.value else f"col{i}" for i, c in enumerate(ws[1])]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not any(row): continue
            for num in re.findall(r'\d+', str(row[0] or "")): last_id = max(last_id, int(num))
            row_text = "\n".join(f"{headers[i]}: {v}" for i, v in enumerate(row) if v is not None and i < len(headers))
            if not row_text.strip(): continue
            st.session_state.rag_docs.append(row_text)
            emb = embed.encode(row_text).tolist()
            try: store.add([row_text], [emb], [f"tc_{sname}_{total}"]); total += 1
            except Exception: pass
    st.session_state.rag_count = total; st.session_state.rag_loaded = True; st.session_state.last_tc_id = last_id
    return total, last_id


def rag_retrieve(query: str) -> str:
    embed = load_embed_model(); store = get_vector_store()
    if store.count() == 0 and st.session_state.rag_docs:
        for i, doc in enumerate(st.session_state.rag_docs):
            emb = embed.encode(doc).tolist()
            try: store.add([doc], [emb], [f"tc_r_{i}"])
            except Exception: pass
    if store.count() == 0: return "No past test cases loaded."
    emb = embed.encode(query[:300]).tolist()
    results = store.query([emb], n_results=min(3, store.count()))
    docs = results["documents"][0] if results["documents"] else []
    # Cap RAG context strictly to prevent token overflow
    combined = "\n---\n".join(docs) if docs else "No similar cases found."
    return combined[:800]


# ══════════════════════════════════════════════════════════════════════════════
# JSON PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_json_response(raw: str) -> list:
    for fn in [
        lambda s: json.loads(s),
        lambda s: json.loads(re.search(r'\[[\s\S]*\]', s).group()),
        lambda s: json.loads(re.sub(r'```json|```', '', s).strip()),
        lambda s: json.loads(re.search(r'\[[\s\S]*\]', re.sub(r',\s*([}\]])', r'\1', re.sub(r"'", '"', s))).group()),
    ]:
        try:
            r = fn(raw)
            if isinstance(r, list) and r: return r
        except Exception: pass
    return []


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE TEST CASES — with safe, compact prompts
# ══════════════════════════════════════════════════════════════════════════════
def generate_test_cases(combined_text, feature_understand, similar_cases,
                        num_cases, auto_mode, last_tc_id,
                        api_key, provider_config, include_neg, include_edge) -> list:

    start_id = last_tc_id + 1

    if auto_mode:
        target_count = 20
        type_rule    = "40% Positive, 35% Negative, 25% Edge."
    else:
        target_count = num_cases
        if include_neg and include_edge:
            pos = num_cases // 3 + (num_cases % 3); neg = num_cases // 3; edge = num_cases - pos - neg
        elif include_neg:
            pos = num_cases // 2 + (num_cases % 2); neg = num_cases - pos; edge = 0
        elif include_edge:
            pos = num_cases // 2 + (num_cases % 2); neg = 0; edge = num_cases - pos
        else:
            pos, neg, edge = num_cases, 0, 0
        type_rule = f"{pos} Positive + {neg} Negative + {edge} Edge."

    # ── Keep ALL inputs SHORT — this is the critical fix ──
    fu  = (feature_understand or "")[:500]
    ct  = (combined_text or "")[:400]
    rag = (similar_cases or "")[:400]

    prompt = f"""You are a Senior QA Architect. Generate exactly {target_count} test cases.
Types: {type_rule}
TC IDs start from TC-{start_id:04d}.

FEATURE SUMMARY:
{fu}

SIMILAR PAST CASES (match style):
{rag}

RULES:
- prerequisites: "User logged in, module accessible"
- steps: numbered 1-5, each one UI action
- expected: exact visible UI outcome
- type: Positive / Negative / Edge
- priority: High / Medium / Low
- related_tc: None

Return ONLY a valid JSON array. No text before [ or after ].
Each object has keys: id, title, prerequisites, steps, expected, type, priority, related_tc

Feature: {ct}"""

    # ── Attempt 1 ──
    try:
        raw = call_llm(prompt, api_key, provider_config, max_tokens=3500)
        result = parse_json_response(raw)
        if result:
            return result
    except Exception as e:
        st.warning(f"⚠️ Attempt 1 failed: {str(e)[:120]}. Retrying with shorter prompt...")

    # ── Attempt 2 — much simpler prompt ──
    simple_count = min(target_count, 10)
    simple = f"""Generate {simple_count} QA test cases as a JSON array.
Feature: {ct[:250]}
Each object needs: id (start TC-{start_id:04d}), title, prerequisites, steps, expected, type (Positive/Negative/Edge), priority (High/Medium/Low), related_tc (None).
Return ONLY JSON. Start with ["""

    try:
        raw = call_llm(simple, api_key, provider_config, max_tokens=3000)
        if not raw.strip().startswith("["):
            raw = "[" + raw
        result = parse_json_response(raw)
        if result:
            return result
    except Exception as e:
        st.warning(f"⚠️ Attempt 2 failed: {str(e)[:120]}. Trying minimal prompt...")

    # ── Attempt 3 — ultra minimal ──
    ultra = f"""Create 5 QA test cases as JSON array for: {ct[:150]}
Keys: id(TC-{start_id:04d}+), title, prerequisites, steps, expected, type(Positive/Negative/Edge), priority(High/Medium/Low), related_tc(None)
["""

    try:
        raw = call_llm(ultra, api_key, provider_config, max_tokens=2000)
        if not raw.strip().startswith("["):
            raw = "[" + raw
        result = parse_json_response(raw)
        if result:
            return result
    except Exception as e:
        st.error(f"❌ All 3 attempts failed. Last error: {str(e)[:150]}")

    return []


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_excel(test_cases: list, feature_name: str) -> bytes:
    wb     = openpyxl.Workbook()
    h_fill = PatternFill("solid", fgColor="1A1A2E")
    h_font = Font(bold=True, color="E8F4FD", name="Calibri", size=11)
    ca     = Alignment(horizontal="center", vertical="center", wrap_text=True)
    la     = Alignment(horizontal="left",   vertical="top",   wrap_text=True)
    thin   = Side(style="thin", color="CCCCCC")
    bdr    = Border(left=thin, right=thin, top=thin, bottom=thin)

    pos_c  = sum(1 for t in test_cases if "positive" in str(t.get("type", "")).lower())
    neg_c  = sum(1 for t in test_cases if "negative" in str(t.get("type", "")).lower())
    edge_c = sum(1 for t in test_cases if "edge"     in str(t.get("type", "")).lower())

    # SUMMARY
    ws_s = wb.active; ws_s.title = "SUMMARY"
    for c, h in enumerate(["Feature / Module Name", "Total Test Cases", "Positive Count",
                            "Negative Count", "Edge Case Count", "ACs Covered", "Gaps"], 1):
        cell = ws_s.cell(row=1, column=c, value=h)
        cell.fill = h_fill; cell.font = h_font; cell.alignment = ca
        ws_s.column_dimensions[get_column_letter(c)].width = 22
    ws_s.cell(row=2, column=1, value=feature_name)
    ws_s.cell(row=2, column=2, value=len(test_cases))
    ws_s.cell(row=2, column=3, value=pos_c)
    ws_s.cell(row=2, column=4, value=neg_c)
    ws_s.cell(row=2, column=5, value=edge_c)
    ws_s.cell(row=2, column=6, value="Extracted from feature input")
    ws_s.cell(row=2, column=7, value="Legacy integration, Performance")
    ws_s.freeze_panes = "A2"

    # TEST CASES
    ws = wb.create_sheet(title=safe_sheet_name(feature_name, "TestCases"))

    hdrs   = ["Test Case ID", "Test Case Title", "Prerequisites", "Test Steps",
              "Expected Result", "Test Type", "Priority", "Related TC ID"]
    widths = [14, 35, 32, 55, 42, 12, 10, 14]
    fills  = {
        "pos" : PatternFill("solid", fgColor="E2EFDA"),
        "posa": PatternFill("solid", fgColor="D0E8C5"),
        "neg" : PatternFill("solid", fgColor="FCE4D6"),
        "nega": PatternFill("solid", fgColor="F0D0B8"),
        "edg" : PatternFill("solid", fgColor="FFF2CC"),
        "edga": PatternFill("solid", fgColor="EFE8AA"),
    }

    for c, (h, w) in enumerate(zip(hdrs, widths), 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = h_fill; cell.font = h_font; cell.alignment = ca; cell.border = bdr
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"

    for ri, tc in enumerate(test_cases, 2):
        t = str(tc.get("type", "")).lower(); alt = ri % 2 == 0
        if "negative" in t:   fill = fills["nega"] if alt else fills["neg"]
        elif "edge"   in t:   fill = fills["edga"] if alt else fills["edg"]
        else:                 fill = fills["posa"] if alt else fills["pos"]

        # Safely convert steps to string
        steps_val = tc.get("steps", "")
        if isinstance(steps_val, list):
            steps_val = "\n".join(str(s) for s in steps_val)

        vals = [tc.get("id", f"TC-{3000+ri:04d}"), tc.get("title", ""),
                tc.get("prerequisites", ""), steps_val, tc.get("expected", ""),
                tc.get("type", "Positive"), tc.get("priority", "Medium"), tc.get("related_tc", "None")]
        for ci, val in enumerate(vals, 1):
            cell = ws.cell(row=ri, column=ci, value=str(val) if val is not None else "")
            cell.font = Font(name="Calibri", size=10); cell.border = bdr; cell.fill = fill
            cell.alignment = ca if ci in [1, 6, 7, 8] else la
        ws.row_dimensions[ri].height = 80

    buf = io.BytesIO(); wb.save(buf); return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="app-title">QA Test Case Generator</div>
    <div class="app-sub">Enterprise test cases powered by AI — switch model if rate limit hits</div>
    <div style="margin-top:0.8rem">
        <span class="tag">GROQ AI</span><span class="tag">TOGETHER AI</span>
        <span class="tag">RAG</span><span class="tag">EXCEL EXPORT</span>
    </div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🤖 Select AI Model</div>', unsafe_allow_html=True)
    st.caption("Switch model if you hit the rate limit")
    selected_llm = st.selectbox("AI Model", list(LLM_PROVIDERS.keys()), 0, label_visibility="hidden")
    provider_cfg = LLM_PROVIDERS[selected_llm]
    st.markdown(f"""<div class="info-box">📊 Limit: <strong>{provider_cfg['limit']}</strong><br>
        🔑 Get key: <a href="{provider_cfg['key_link']}" target="_blank">{provider_cfg['key_link'].split('/')[2]}</a>
    </div>""", unsafe_allow_html=True)
    api_key = st.text_input("API Key", type="password", placeholder=provider_cfg["key_hint"], label_visibility="hidden")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Existing Test Cases (RAG)</div>', unsafe_allow_html=True)
    st.caption("Upload master Excel → avoids duplicates")
    past_excel = st.file_uploader("Upload existing test cases", type=["xlsx", "xls"], label_visibility="hidden")
    if past_excel and not st.session_state.rag_loaded:
        with st.spinner("Loading into RAG..."):
            count, last_id = load_excel_to_rag(past_excel.read())
        st.success(f"✅ {count} test cases loaded | Next: TC-{last_id+1:04d}")
    elif past_excel and st.session_state.rag_loaded:
        st.markdown(f'<div class="rag-ok">✅ {st.session_state.rag_count} TCs in RAG | Next: TC-{st.session_state.last_tc_id+1:04d}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rag-warn">⚠️ Upload master Excel for better quality</div>', unsafe_allow_html=True)
    if st.session_state.rag_loaded:
        if st.button("🗑️ Clear RAG & Upload New"):
            st.session_state.rag_loaded = False; st.session_state.rag_count = 0
            st.session_state.rag_docs = []; st.session_state.last_tc_id = 3000
            get_vector_store().clear(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🎛️ Settings</div>', unsafe_allow_html=True)
    auto_mode = st.toggle("🤖 Auto — generate ALL scenarios", value=False)
    if auto_mode:
        num_cases = 999
        st.markdown('<div class="info-box">AI generates 20-30 TCs for full coverage</div>', unsafe_allow_html=True)
    else:
        num_cases = st.number_input("Number of test cases", min_value=5, max_value=100, value=15, step=5)
    include_neg  = st.toggle("Include negative tests", value=True)
    include_edge = st.toggle("Include edge cases",     value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">📋 Pipeline Status</div>', unsafe_allow_html=True)
    stage = st.session_state.stage or 0
    for i, label in enumerate(["🔍 Extract Text", "🧠 Parse Feature", "📚 RAG Search", "⚙️ Generate JSON", "📊 Build Excel"]):
        color = "#2E7D32" if i < stage else ("#1565C0" if i == stage and stage > 0 else "#9B8EA0")
        icon  = "✅"      if i < stage else ("⏳"      if i == stage and stage > 0 else "○")
        st.markdown(f'<div style="color:{color};font-size:0.85rem;padding:0.2rem 0;">{icon} {label}</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-label">📤 Feature Input</div>', unsafe_allow_html=True)
    tab_img, tab_doc, tab_txt = st.tabs(["🖼️ Images", "📄 Documents", "📝 Text / Jira Story"])

    with tab_img:
        st.markdown('<div class="info-box">💡 If OCR fails, paste text in the Text tab.</div>', unsafe_allow_html=True)
        images = st.file_uploader("Upload feature images", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                  accept_multiple_files=True, label_visibility="hidden")
        if images:
            for img in images: st.image(img, caption=img.name, use_container_width=True)

    with tab_doc:
        st.markdown('<div class="info-box">💡 Supports .docx and .txt files.</div>', unsafe_allow_html=True)
        doc_files = st.file_uploader("Upload .docx or .txt", type=["docx", "txt"],
                                     accept_multiple_files=True, label_visibility="hidden")

    with tab_txt:
        manual_text = st.text_area("Feature description", height=240,
            placeholder="Paste your Jira story, acceptance criteria, or feature description here...",
            label_visibility="hidden")

    st.markdown("**🔗 VD / Figma / Confluence link:**")
    vd_link = st.text_input("VD Link", placeholder="https://...", label_visibility="hidden")
    if vd_link:
        st.markdown('<div class="warn-box">⚠️ Open the link, copy text (Ctrl+A → Ctrl+C), paste in Text tab.</div>', unsafe_allow_html=True)

    has_image = bool(images)
    has_doc   = bool(doc_files)
    has_text  = bool(manual_text and manual_text.strip())
    has_input = has_image or has_doc or has_text
    can_run   = has_input and bool(api_key)

    if not api_key:     st.caption("🔑 Add your API key in the sidebar")
    elif not has_input: st.caption("⬆️ Upload images, documents, or paste text above")

    if st.button("⚡ GENERATE TEST CASES", disabled=not can_run):
        combined_text = ""; ocr_warnings = []

        # Stage 1
        st.session_state.stage = 1
        if has_image:
            with st.spinner("🔍 Extracting text from images..."):
                for img_file in images:
                    text, warns = run_ocr(Image.open(img_file))
                    ocr_warnings.extend(warns)
                    combined_text += (f"\n\n[Image: {img_file.name}]\n{text}" if text else "")
                    if not text: ocr_warnings.append(f"⚠️ No text from '{img_file.name}'.")

        if has_doc:
            with st.spinner("📄 Reading documents..."):
                for df in doc_files:
                    raw = df.read()
                    text = (extract_docx_text(raw) if df.name.endswith(".docx")
                            else extract_txt_text(raw))
                    combined_text += (f"\n\n[Doc: {df.name}]\n{text}" if text else "")
                    if not text: ocr_warnings.append(f"⚠️ Could not extract from '{df.name}'.")

        if has_text:
            combined_text += f"\n\n[Feature Description]\n{manual_text.strip()}"

        combined_text = combined_text.strip()
        st.session_state.ocr_warnings = ocr_warnings

        if not combined_text:
            st.error("❌ No text extracted. Please paste content in the Text tab."); st.stop()

        st.session_state.extracted_text = combined_text

        # Determine feature name
        feature_name = "Feature"
        if images:      feature_name = os.path.splitext(images[0].name)[0]
        elif doc_files: feature_name = os.path.splitext(doc_files[0].name)[0]
        elif manual_text:
            fl = manual_text.strip().split("\n")[0]
            feature_name = fl[:50].replace("Feature:", "").strip() or "Feature"
        st.session_state.feature_name = feature_name

        # Stage 2 — Feature understanding with SHORT input
        st.session_state.stage = 2
        with st.spinner("🧠 Parsing feature..."):
            fu_prompt = f"""Parse this feature input. Return structured understanding.

Feature Name, User Roles, Screens, UI Components, Acceptance Criteria, Business Rules, Error States.

Input:
{combined_text[:800]}"""
            try:
                feature_understand = call_llm(fu_prompt, api_key, provider_cfg, max_tokens=500)
            except Exception as e:
                feature_understand = f"Feature Name: {feature_name}\nInput summary: {combined_text[:200]}"
                st.warning(f"⚠️ Feature parsing error: {str(e)[:80]}. Using raw input.")
            st.session_state.feature_understand = feature_understand

        # Stage 3
        st.session_state.stage = 3
        with st.spinner("📚 Searching similar test cases..."):
            similar = rag_retrieve(feature_understand[:200])

        # Stage 4
        st.session_state.stage = 4
        with st.spinner("⚙️ Generating test cases..."):
            test_cases = generate_test_cases(
                combined_text=combined_text, feature_understand=feature_understand,
                similar_cases=similar, num_cases=num_cases, auto_mode=auto_mode,
                last_tc_id=st.session_state.last_tc_id or 3000,
                api_key=api_key, provider_config=provider_cfg,
                include_neg=include_neg, include_edge=include_edge,
            )
            if not test_cases:
                st.error("❌ Could not generate test cases.\n\n"
                         "**Try:** Switch AI model in sidebar, reduce test case count, or wait 30s and retry.")
                st.stop()
            st.session_state.test_cases_parsed = test_cases
            st.session_state.tc_count   = len(test_cases)
            st.session_state.pos_count  = sum(1 for t in test_cases if "positive" in t.get("type", "").lower())
            st.session_state.neg_count  = sum(1 for t in test_cases if "negative" in t.get("type", "").lower())
            st.session_state.edge_count = sum(1 for t in test_cases if "edge"     in t.get("type", "").lower())
            st.session_state.llm_used   = selected_llm

        # Stage 5
        st.session_state.stage = 5
        with st.spinner("📊 Building Excel..."):
            excel = build_excel(test_cases, feature_name)
            st.session_state.excel_bytes = excel

        st.rerun()

with col_right:
    st.markdown('<div class="card-label">📋 Results</div>', unsafe_allow_html=True)
    for w in (st.session_state.ocr_warnings or []):
        st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

    if not st.session_state.test_cases_parsed:
        st.markdown("""<div style="background:#FFFFFF;border:1px solid #E8E5DF;border-radius:12px;
                    padding:4rem 2rem;text-align:center;">
            <div style="font-size:3rem">⏳</div>
            <div style="margin-top:0.8rem;font-size:0.9rem;color:#9B8EA0;">
                Results will appear here after generation</div></div>""", unsafe_allow_html=True)
    else:
        tc = st.session_state.tc_count or 0; pos = st.session_state.pos_count or 0
        neg = st.session_state.neg_count or 0; edge = st.session_state.edge_count or 0
        rag = st.session_state.rag_count or 0; llm = st.session_state.llm_used or ""
        if llm: st.markdown(f'<span class="llm-badge">Generated by: {llm}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="success-box">✅ &nbsp; <strong>{tc} test cases generated!</strong></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-strip">
            <div class="metric"><div class="metric-val" style="color:#2E7D32">{pos}</div><div class="metric-lbl">✅ Positive</div></div>
            <div class="metric"><div class="metric-val" style="color:#C00000">{neg}</div><div class="metric-lbl">❌ Negative</div></div>
            <div class="metric"><div class="metric-val" style="color:#F59E0B">{edge}</div><div class="metric-lbl">⚠️ Edge</div></div>
            <div class="metric"><div class="metric-val">{rag}</div><div class="metric-lbl">📚 RAG</div></div>
        </div>""", unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["📊 Test Cases", "🧠 Feature Understanding", "🔍 Extracted Text"])
        with t1:
            tcs = st.session_state.test_cases_parsed
            preview = "\n\n".join(
                f"{t.get('id', '')} | {t.get('type', '')} | {t.get('priority', '')}\n"
                f"Title: {t.get('title', '')}\nPrerequisites: {t.get('prerequisites', '')}\n"
                f"Steps:\n{t.get('steps', '')}\nExpected: {t.get('expected', '')}"
                for t in tcs[:5])
            if len(tcs) > 5: preview += f"\n\n... and {len(tcs)-5} more in the Excel file"
            st.markdown(f'<div class="result-preview">{preview}</div>', unsafe_allow_html=True)
        with t2:
            st.markdown(f'<div class="result-preview">{st.session_state.feature_understand or "—"}</div>', unsafe_allow_html=True)
        with t3:
            st.markdown(f'<div class="result-preview">{(st.session_state.extracted_text or "—")[:2000]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        fname = safe_sheet_name(st.session_state.feature_name or "Feature", "Feature")
        st.download_button(
            label="⬇️ DOWNLOAD EXCEL TEST CASES",
            data=st.session_state.excel_bytes,
            file_name=f"{fname}_TestCases.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        if st.button("🔄 Generate New"):
            for k in ["extracted_text", "feature_understand", "test_cases_parsed", "excel_bytes",
                      "tc_count", "pos_count", "neg_count", "edge_count", "ocr_warnings", "llm_used", "feature_name"]:
                st.session_state[k] = [] if k in ["test_cases_parsed", "ocr_warnings"] else None
            st.session_state.stage = 0; st.rerun()
