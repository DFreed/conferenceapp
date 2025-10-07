import os
import io
import time
import json
import re
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import requests
import streamlit as st
from urllib.parse import urlencode
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Safety & compliance notes
# ---------------------------
# - This app does NOT automate logins or scrape protected pages.
# - It supports vendor APIs (ZoomInfo, NewsAPI, etc.) that you configure.
# - Store secrets in environment variables or Streamlit secrets—not in code.
# - Respect each provider’s ToS and rate limits; obtain necessary permissions.

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Conference → Target List Prioritizer", layout="wide")
st.title("Conference → Target List Prioritizer (CSV + Local LLM)")

with st.sidebar:
    st.header("Integrations (Optional)")
    # CSV-only flow; API keys optional/unused if you don't have them
    ZOOMINFO_API_KEY = st.text_input("ZoomInfo API Key (optional)", type="password")
    NEWSAPI_KEY = st.text_input("NewsAPI Key (optional)", type="password")
    st.divider()
    st.subheader("Local LLM (free via Ollama)")
    USE_OLLAMA = st.checkbox("Use local LLM for intent & role classification", value=True,
                             help="Requires Ollama running at http://localhost:11434")
    
    # Get available models from Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            available_models = [m.get("name") for m in models_data if m.get("name")]
        else:
            available_models = ["llama3.1:8b"]  # fallback
    except:
        available_models = ["llama3.1:8b"]  # fallback
    
    if available_models:
        _choice = st.selectbox(
            "Ollama model",
            options=available_models + ["Other..."],
            index=0,
            help="Choose from your installed models. Select \"Other...\" to enter a custom tag."
        )
        if _choice == "Other...":
            OLLAMA_MODEL = st.text_input(
                "Custom model tag",
                value="llama3.1:8b",
                help="Enter any model tag you've pulled with `ollama pull <tag>`."
            )
        else:
            OLLAMA_MODEL = _choice
    else:
        st.warning("No models found. Run `ollama pull llama3.1:8b` to download a model.")
        OLLAMA_MODEL = st.text_input(
            "Model tag",
            value="llama3.1:8b",
            help="Enter any model tag you've pulled with `ollama pull <tag>`."
        )
    OLLAMA_CHUNK = st.slider("LLM batch size", min_value=10, max_value=100, value=25, step=5,
                             help="Titles classified per LLM call (smaller = faster)")
    # keep chunk size in session so worker functions can read it
    st.session_state["OLLAMA_CHUNK"] = OLLAMA_CHUNK
    st.caption("Batching & caching make this fast; smaller models (e.g., qwen2.5:3b-instruct) run quickest.")
    
    # Test Ollama connection
    if st.button("Test Ollama Connection"):
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                st.success("✅ Ollama is running and accessible!")
                models = response.json().get("models", [])
                if models:
                    st.write("Available models:", [m.get("name") for m in models])
                else:
                    st.warning("No models found. Run `ollama pull llama3.1:8b` to download a model.")
            else:
                st.error(f"❌ Ollama returned status code: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot connect to Ollama: {e}")
            st.info("Make sure Ollama is running: `ollama serve`")

    st.caption("No credentials are stored server-side by this app. Restart clears them.")

st.markdown("""
This tool ingests your attendee list and ranks contacts against your query.
**No scraping or password automation.** Optionally, a local open-source LLM (via Ollama) improves intent understanding & role classification.
""")

# ---------------------------
# Helpers
# ---------------------------

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def full_name(row: pd.Series) -> str:
    fn = row.get("first_name", "") or ""
    ln = row.get("last_name", "") or ""
    if fn or ln:
        return f"{fn} {ln}".strip()
    return str(row.get("name", "")).strip()

def combine_text_fields(row: pd.Series) -> str:
    fields = []
    for key in ["first_name", "last_name", "name", "company", "title", "email", "linkedin_url", "recent_mentions"]:
        val = row.get(key, "")
        if pd.notnull(val) and str(val).strip():
            fields.append(str(val))
    return " | ".join(fields)

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# ---------------------------
# Local LLM via Ollama (optional, free)
# ---------------------------

def _ollama_chat(messages: List[Dict[str,str]], model: str, options: Dict[str,Any]=None) -> str:
    """
    Minimal client for Ollama's /api/generate. Returns assistant text or raises.
    """
    try:
        # Convert messages to a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        
        # Debug: print the request details
        print(f"DEBUG: Making request to Ollama with model: {model}")
        print(f"DEBUG: Prompt length: {len(prompt)}")
        
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 5,  # Limit response length
                    "num_ctx": 256,    # Smaller context window
                }
            },
            timeout=15,  # Shorter timeout
        )
        
        print(f"DEBUG: Response status: {r.status_code}")
        print(f"DEBUG: Response content: {r.text[:200]}...")
        
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"DEBUG: Error details: {e}")
        raise RuntimeError(f"Ollama generate failed: {e}")

def llm_interpret_query_ollama(query: str, model: str) -> dict:
    """
    Convert free text to strict JSON:
    {
      "must_include": [..],
      "must_exclude": [..],
      "target_functions": [..],
      "seniority": [..],
      "notes": "..."
    }
    """
    sys = ("You translate prospecting queries into STRICT JSON with fields: "
           "must_include, must_exclude, target_functions, seniority, notes. "
           "Do not add extra keys. Return ONLY JSON.")
    user = f"Query: {query}\nReturn ONLY JSON."
    try:
        out = _ollama_chat(
            [{"role":"system","content":sys},{"role":"user","content":user}],
            model=model
        )
        obj = json.loads(out)
        return {
            "must_include": [s.strip() for s in obj.get("must_include", [])],
            "must_exclude": [s.strip() for s in obj.get("must_exclude", [])],
            "target_functions": obj.get("target_functions", []),
            "seniority": obj.get("seniority", []),
            "notes": obj.get("notes", "")
        }
    except Exception:
        # Safe default when LLM unavailable
        return {
            "must_include": [],
            "must_exclude": ["sales","consulting","business development"],
            "target_functions": [],
            "seniority": [],
            "notes": "fallback"
        }

# Fast rule-first role guess (zero-cost); LLM only for ambiguous
ROLE_LABELS = ["Sales","Consulting","Procurement/Buying","Engineering/Technical",
               "Operations/Field","Executive","Regulatory/Policy","Other"]

_RULES = [
    ("Sales", r"\b(sales|account (exec|mgr)|business development|bd|growth|inside sales|outside sales|pre-?sales|sales engineer)\b"),
    ("Consulting", r"\b(consult(ant|ing)|advis(or|ory))\b"),
    ("Procurement/Buying", r"\b(procurement|purchas(ing|er)|sourc(ing|e)|supply chain|category manager)\b"),
    ("Engineering/Technical", r"\b(engineer|protection|relay|power systems|substation|transmission|distribution|grid|electrical)\b"),
    ("Operations/Field", r"\b(operations|maintenance|field|plant|asset management|system planning)\b"),
    ("Executive", r"\b(ceo|cto|cfo|coo|chief|evp|svp|vp|head|director|founder)\b"),
    ("Regulatory/Policy", r"\b(regulatory|policy|compliance)\b"),
]

def rule_role(title: str, company: str) -> str:
    txt = f"{title} {company}".lower()
    for label, pat in _RULES:
        if re.search(pat, txt):
            return label
    return "Other"

# Tiny disk cache so repeat runs are instant
_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".role_cache.json")
try:
    with open(_CACHE_PATH, "r") as fh:
        ROLE_CACHE = json.load(fh)
except Exception:
    ROLE_CACHE = {}

def _cache_get(key: str):
    return ROLE_CACHE.get(key)

def _cache_set(key: str, val: str):
    ROLE_CACHE[key] = val

def _cache_flush():
    try:
        with open(_CACHE_PATH, "w") as fh:
            json.dump(ROLE_CACHE, fh)
    except Exception:
        pass

def llm_classify_role_ollama_batch(rows: List[Dict[str,str]], model: str) -> List[str]:
    """
    Batch classify many rows in one call. rows: [{id,title,company},...]
    Returns labels aligned to input order.
    """
    # Build compact prompt → strict JSON array of labels
    lines = [f'{r["title"]}|{r.get("company","")}' for r in rows]  # Remove index for shorter prompt
    sys = ("Classify each line into one label: Sales, Consulting, Procurement/Buying, Engineering/Technical, Operations/Field, Executive, Regulatory/Policy, Other. Return JSON array only.")
    user = "\n".join(lines)
    out = _ollama_chat(
        [{"role":"system","content":sys},{"role":"user","content":user}],
        model=model
    )
    try:
        labels = json.loads(out)
        if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
            # sanitize
            return [x if x in ROLE_LABELS else "Other" for x in labels]
    except Exception:
        pass
    # Fallback: all Other (should be rare)
    return ["Other"] * len(rows)

# ---------------------------
# Enrichment stubs (APIs only)
# ---------------------------

def enrich_with_zoominfo(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Example stub. Replace the endpoint/payload with ZoomInfo's official API docs.
    This function tries to enrich company/title/etc. for each row that has an email or name+company.
    """
    if not api_key:
        return df

    df = df.copy()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Collect rows needing enrichment
    need_rows = df.index.tolist()
    enriched_rows = []
    for idx in need_rows:
        row = df.loc[idx]
        payload = {}
        # Prefer email match if available
        if pd.notnull(row.get("email")) and str(row["email"]).strip():
            payload = {"email": str(row["email"]).strip()}
        else:
            # fallback: name + company search (exact fields depend on ZoomInfo API)
            name_val = full_name(row)
            company_val = str(row.get("company", "")).strip()
            if name_val and company_val:
                payload = {"firstNameLastName": name_val, "company": company_val}
            else:
                continue

        try:
            # NOTE: Replace with the correct ZoomInfo endpoint
            # resp = requests.post("https://api.zoominfo.com/people/match", json=payload, headers=headers, timeout=15)
            # Mock behavior here; uncomment above and map to actual fields per API docs.
            # For safety in the scaffold, we skip real calls:
            # if resp.status_code == 200:
            #     data = resp.json()
            #     # Map fields as appropriate:
            #     df.at[idx, "title"] = data.get("jobTitle", df.at[idx, "title"] if "title" in df.columns else None)
            #     df.at[idx, "company"] = data.get("company", df.at[idx, "company"] if "company" in df.columns else None)
            pass
        except Exception:
            # Keep going even if some queries fail
            continue
        enriched_rows.append(idx)

    return df

def fetch_recent_mentions(name: str, company: str, newsapi_key: str, lookback_days: int = 30) -> List[str]:
    """Use NewsAPI (or similar) to find recent public mentions of a person/company."""
    if not newsapi_key:
        return []

    q_parts = []
    if name: q_parts.append(f"\"{name}\"")
    if company: q_parts.append(f"\"{company}\"")
    if not q_parts:
        return []

    q = " AND ".join(q_parts)
    from_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    params = {
        "q": q,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": newsapi_key
    }
    url = f"https://newsapi.org/v2/everything?{urlencode(params)}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            articles = data.get("articles", [])
            out = []
            for a in articles:
                title = safe_get(a, "title", default="")
                source = safe_get(a, "source", "name", default="")
                link = safe_get(a, "url", default="")
                published = safe_get(a, "publishedAt", default="")
                if title and link:
                    out.append(f"{title} — {source} ({published}) | {link}")
            return out
    except Exception:
        pass
    return []

def fetch_utility_dive(company: str, lookback_days: int = 60) -> List[str]:
    """
    Lightweight RSS search via Utility Dive vertical feeds would require known feed URLs.
    As a generic placeholder, we return [] in the scaffold and recommend configuring
    per-vertical feeds + keyword filter on the server side.
    """
    # IMPLEMENTATION IDEA:
    #  - Maintain a small set of RSS feeds (e.g., power/utility verticals).
    #  - Fetch and cache items, filter for company keyword and date <= lookback_days.
    return []

# ---------------------------
# Ranking
# ---------------------------

def rank_people(df: pd.DataFrame, query: str, extra_weight_title: float = 1.25, extra_weight_company: float = 1.1) -> pd.DataFrame:
    """
    Simple, transparent TF-IDF cosine similarity using concatenated text fields,
    with light weighting for title/company when present.
    """
    df = df.copy()

    # Build a corpus with optional weights
    def row_text(row):
        bits = []
        nm = full_name(row)
        if nm: bits.append(nm)
        company = str(row.get("company", "") or "")
        title = str(row.get("title", "") or "")
        mentions = str(row.get("recent_mentions", "") or "")
        li = str(row.get("linkedin_url", "") or "")

        # Weighting by repeating key fields
        if title: bits.extend([title] * int(extra_weight_title * 2))
        if company: bits.extend([company] * int(extra_weight_company * 2))

        # Base info
        for k in ["email"]:
            v = str(row.get(k, "") or "")
            if v: bits.append(v)
        if li: bits.append(li)
        if mentions: bits.append(mentions)

        return " | ".join([b for b in bits if b])

    texts = [row_text(r) for _, r in df.iterrows()]
    corpus = texts + [query]

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(corpus)
    sims = cosine_similarity(X[:-1], X[-1])

    df["relevance_score"] = sims.flatten()

    # Simple tie-breakers: seniority keyword bump, exact company mentions
    seniority_terms = ["founder", "ceo", "svp", "evp", "vp", "director", "head", "chief", "cto", "cfo", "coo"]
    def seniority_bump(title: str) -> float:
        if not title: return 0.0
        t = title.lower()
        return 0.05 * sum(1 for s in seniority_terms if s in t)

    bumps = []
    for _, r in df.iterrows():
        bump = seniority_bump(str(r.get("title", "")))
        bumps.append(bump)

    df["relevance_score"] = df["relevance_score"] + np.array(bumps)
    df = df.sort_values("relevance_score", ascending=False)
    return df

# ---------------------------
# Negation-aware ranking (uses Ollama if enabled, otherwise a strong fallback)
# ---------------------------

EXCLUDE_TITLES_FALLBACK = [
    "sales","account executive","account manager","business development","bd",
    "consulting","consultant","advisor","advisory","customer success",
    "marketing","partnerships","recruiter","talent acquisition",
    "pre-sales","solutions consultant","sales engineer","growth"
]
INCLUDE_HINTS = [
    "high voltage","transmission","substation","distribution","grid","relay","protection",
    "procurement","purchasing","sourcing","supply chain",
    "engineering","electrical engineer","power systems",
    "operations","maintenance","asset management","system planning","utility"
]

def contains_any(txt: str, terms: List[str]) -> bool:
    t = (txt or "").lower()
    return any(term in t for term in terms)

def row_text_all(r: pd.Series) -> str:
    bits = []
    fn = str(r.get("first_name","") or "")
    ln = str(r.get("last_name","") or "")
    nm = (fn + " " + ln).strip() or str(r.get("name",""))
    if nm: bits.append(nm)
    for k in ["title","company","email","linkedin_url","recent_mentions"]:
        v = str(r.get(k,"") or "")
        if v: bits.append(v)
    return " | ".join(bits)

def negation_aware_rank(df: pd.DataFrame, user_query: str, use_ollama: bool, ollama_model: str) -> pd.DataFrame:
    df = df.copy()
    t_start = time.time()
    # status / progress UI
    phase = st.empty()
    bar = st.progress(0, text="Preparing…")
    # 1) Interpret query
    if use_ollama:
        plan = llm_interpret_query_ollama(user_query, ollama_model)
        must_exclude = [s.lower() for s in plan.get("must_exclude", [])]
        must_include = [s.lower() for s in plan.get("must_include", [])]
    else:
        # Fallback: parse common negations
        plan = {}
        must_exclude = ["sales","consulting","business development"]
        must_include = []
        if "high voltage" in user_query.lower():
            must_include.append("high voltage")

    # 2) Pre-filter obvious exclusions by keywords (fast path)
    total_n = len(df)
    title_text = df["title"].astype(str).str.lower().fillna("")
    mask_ex_kw_pre = title_text.apply(lambda t: contains_any(t, EXCLUDE_TITLES_FALLBACK) or contains_any(t, must_exclude))
    if must_include:
        mask_in_pre = df.apply(lambda r: contains_any(f"{r.get('title','')} {r.get('company','')}", must_include), axis=1)
    else:
        mask_in_pre = pd.Series([True]*len(df), index=df.index)
    pre_drop = int((mask_ex_kw_pre & mask_in_pre).sum())
    df = df[~(mask_ex_kw_pre) & mask_in_pre].copy()
    phase.markdown(f"**Phase A:** Keyword pre-filter removed **{pre_drop}** of **{total_n}** rows.")
    bar.progress(10, text="Keyword pre-filter complete")

    # 3) Role classification:
    #    (a) fast rule pass
    #    (b) cache lookup
    #    (c) batch LLM only for ambiguous
    labels = []
    ambiguous_idx = []
    batch_payload = []
    cached_ct = 0
    rule_ct = 0
    scanned = 0
    scan_n = len(df)
    for i, r in df.iterrows():
        title = str(r.get("title",""))
        company = str(r.get("company",""))
        key = (title + " | " + company).strip().lower()
        cached = _cache_get(key)
        if cached:
            labels.append(cached)
            cached_ct += 1
        else:
            rule = rule_role(title, company)
            if rule in ["Sales","Consulting","Procurement/Buying","Engineering/Technical","Operations/Field","Executive","Regulatory/Policy"]:
                labels.append(rule)
                _cache_set(key, rule)
                rule_ct += 1
            else:
                labels.append(None)  # mark for LLM
                ambiguous_idx.append(i)
                batch_payload.append({"title": title, "company": company})
        scanned += 1
        if scanned % max(1, scan_n // 10) == 0:
            pct = 10 + int(30 * scanned / max(1, scan_n))
            bar.progress(min(pct, 40), text=f"Scanning titles… {scanned}/{scan_n}")

    # Batch LLM for ambiguous (if enabled)
    if use_ollama and batch_payload:
        chunk = st.session_state.get("OLLAMA_CHUNK") or 150
        phase.markdown(f"**Phase B:** Cache hits **{cached_ct}**, rule-resolved **{rule_ct}**, sending **{len(batch_payload)}** to LLM in batches of **{chunk}**.")
        llm_done = 0
        total_llm = len(batch_payload)
        for start in range(0, total_llm, chunk):
            part = batch_payload[start:start+chunk]
            preds = llm_classify_role_ollama_batch(part, ollama_model)
            for j, pred in enumerate(preds):
                df_idx = ambiguous_idx[start + j]
                title = df.loc[df_idx, "title"]
                company = df.loc[df_idx, "company"]
                key = (str(title) + " | " + str(company)).strip().lower()
                _cache_set(key, pred)
                # write label into labels list at the matching position
                pos = list(df.index).index(df_idx)
                labels[pos] = pred
            llm_done += len(part)
            pct = 40 + int(40 * llm_done / max(1, total_llm))
            bar.progress(min(pct, 80), text=f"LLM classifying… {llm_done}/{total_llm}")
        _cache_flush()
    else:
        # If LLM off, unresolved become "Other"
        labels = [x if x is not None else "Other" for x in labels]

    df["llm_role"] = labels
    mask_ex_role = df["llm_role"].isin(["Sales","Consulting"])

    mask_ex_kw = df["title"].str.lower().fillna("").apply(lambda t: contains_any(t, EXCLUDE_TITLES_FALLBACK) or contains_any(t, must_exclude))
    if must_include:
        mask_in = df.apply(lambda r: contains_any(f"{r.get('title','')} {r.get('company','')}", must_include), axis=1)
    else:
        mask_in = pd.Series([True]*len(df), index=df.index)

    before_final = len(df)
    df = df[~(mask_ex_role | mask_ex_kw) & mask_in].copy()
    final_drop = before_final - len(df)
    bar.progress(90, text="Final filters applied")

    # 4) Base TF-IDF similarity to query
    if len(df) == 0:
        # No rows left after filtering
        bar.progress(100, text="No results after filtering")
        phase.markdown(
            f"**Phase C:** All rows were filtered out.  \n"
            f"**Summary:** start **{total_n}**, pre-filtered **{pre_drop}**, cache **{cached_ct}**, rules **{rule_ct}**, "
            f"LLM **{len(batch_payload)}**, result **0**.  \n"
            f"**Elapsed:** {time.time() - t_start:.1f}s"
        )
        return pd.DataFrame()  # Return empty DataFrame
    
    corpus = [row_text_all(r) for _, r in df.iterrows()] + [user_query]
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(corpus)
    sims = cosine_similarity(X[:-1], X[-1]).flatten()
    df["base_score"] = sims

    # 4) Heuristic boosts (buyers/operators/engineers/utilities)
    boosts = []
    for _, r in df.iterrows():
        t = f"{r.get('title','')} {r.get('company','')}".lower()
        boost = 0.0
        if contains_any(t, INCLUDE_HINTS): boost += 0.25
        if any(w in t for w in ["utility","power","electric","transmission","substation","grid"]): boost += 0.10
        if r.get("llm_role") in ["Procurement/Buying","Engineering/Technical","Operations/Field"]:
            boost += 0.15
        boosts.append(boost)
    df["score"] = df["base_score"] + np.array(boosts)
    out = df.sort_values("score", ascending=False)
    elapsed = time.time() - t_start
    bar.progress(100, text="Done")
    phase.markdown(
        f"**Phase C:** Final filter removed **{final_drop}** more rows.  \n"
        f"**Summary:** start **{total_n}**, pre-filtered **{pre_drop}**, cache **{cached_ct}**, rules **{rule_ct}**, "
        f"LLM **{len(batch_payload)}**, result **{len(out)}**.  \n"
        f"**Elapsed:** {elapsed:.1f}s"
    )
    return out

# ---------------------------
# UI
# ---------------------------

st.subheader("1) Upload your conference attendee list (CSV)")
csv_file = st.file_uploader("Upload CSV", type=["csv"])
data_df = None

if csv_file is not None:
    try:
        data_df = pd.read_csv(csv_file)
    except Exception:
        csv_file.seek(0)
        data_df = pd.read_csv(csv_file, encoding="latin-1")
    data_df = clean_cols(data_df)
    # Ensure expected columns exist
    for col in ["first_name", "last_name", "company", "title", "email", "linkedin_url"]:
        if col not in data_df.columns:
            data_df[col] = None
    st.success(f"Loaded {len(data_df)} attendees.")
    st.dataframe(data_df.head(25), use_container_width=True)

st.subheader("2) Optional enrichment (approved APIs only)")
colA, colB = st.columns(2)

with colA:
    do_zoominfo = st.checkbox("Enrich with ZoomInfo (official API)", value=False, help="Requires API key in sidebar. No scraping.")
with colB:
    do_news_mentions = st.checkbox("Fetch recent mentions (NewsAPI)", value=False, help="Looks for person+company mentions in public news.")

if st.button("Run Enrichment"):
    if data_df is None:
        st.warning("Please upload a CSV first.")
    else:
        df = data_df.copy()

        if do_zoominfo:
            if not ZOOMINFO_API_KEY:
                st.error("ZoomInfo API Key missing.")
            else:
                with st.spinner("Enriching via ZoomInfo…"):
                    df = enrich_with_zoominfo(df, ZOOMINFO_API_KEY)
                    st.success("ZoomInfo enrichment pass completed (see notes in code to map fields).")

        if do_news_mentions:
            if not NEWSAPI_KEY:
                st.error("NewsAPI key missing.")
            else:
                with st.spinner("Fetching recent public mentions…"):
                    mentions_all = []
                    for i, r in df.iterrows():
                        nm = full_name(r)
                        comp = str(r.get("company", "") or "")
                        mentions = fetch_recent_mentions(nm, comp, NEWSAPI_KEY)
                        # Merge Utility Dive RSS matches (placeholder returns [])
                        mentions.extend(fetch_utility_dive(comp))
                        mentions_all.append(" || ".join(mentions) if mentions else "")
                    df["recent_mentions"] = mentions_all
                    st.success("Mentions fetched.")
        st.session_state["df_enriched"] = df

st.subheader("3) Prioritize by your query")
user_query = st.text_input("Describe what you’re looking for (e.g., 'grid-scale storage, US Southeast utilities, hydrogen blending pilots')")

if st.button("Rank & Export"):
    df_base = st.session_state.get("df_enriched", data_df)
    if df_base is None:
        st.warning("Please upload a CSV first.")
    elif not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Scoring (negation-aware)…"):
            ranked = negation_aware_rank(df_base, user_query, USE_OLLAMA, OLLAMA_MODEL)
            
            if len(ranked) == 0:
                st.warning("No results found after filtering. Try:")
                st.markdown("- **Less restrictive query** (e.g., 'engineers' instead of 'high voltage transmission engineers')")
                st.markdown("- **Disable LLM filtering** (uncheck the LLM checkbox)")
                st.markdown("- **Check your CSV data** (ensure titles and companies are populated)")
            else:
                cols = ["first_name","last_name","company","title","email","linkedin_url","recent_mentions",
                        "llm_role","score","base_score"]
                for c in cols:
                    if c not in ranked.columns:
                        ranked[c] = None
                out = ranked[cols]
                st.success(f"Ranked {len(out)} people. (Local LLM={'on' if USE_OLLAMA else 'off'})")
                st.dataframe(out.head(50), use_container_width=True)

                # Export
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download prioritized CSV", data=csv_bytes, file_name="prioritized_contacts.csv", mime="text/csv")
