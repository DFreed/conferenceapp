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
# API Test Functions
# ---------------------------

def test_apollo_connection(api_key: str):
    """Test Apollo API connection with a simple request"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Test with a simple request
    test_payload = {
        "details": [
            {
                "name": "John Doe",
                "title": "Software Engineer",
                "organization_name": "Example Corp"
            }
        ],
        "reveal_personal_emails": False
    }
    
    try:
        st.info("Testing Apollo API connection...")
        resp = requests.post(
            "https://api.apollo.io/api/v1/people/bulk_match",
            headers=headers,
            json=test_payload,
            timeout=30,
        )
        
        st.info(f"Apollo API Test: Status code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            st.success("✅ Apollo API connection successful!")
            st.json(data)
        elif resp.status_code == 401:
            st.error("❌ Apollo API: Unauthorized - Check your API key")
        elif resp.status_code == 403:
            st.error("❌ Apollo API: Forbidden - Check your account permissions")
        elif resp.status_code == 429:
            st.warning("⚠️ Apollo API: Rate limited - Try again later")
        else:
            st.error(f"❌ Apollo API: Error {resp.status_code}")
            if resp.content:
                try:
                    error_data = resp.json()
                    st.error(f"Error details: {error_data}")
                except:
                    st.error(f"Error response: {resp.text[:200]}")
                    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Apollo API: Network error - {str(e)}")
    except Exception as e:
        st.error(f"❌ Apollo API: Unexpected error - {str(e)}")

def test_newsapi_connection(api_key: str):
    """Test NewsAPI connection with a simple request"""
    try:
        st.info("Testing NewsAPI connection...")
        
        # Test with a simple query
        params = {
            "q": "technology",
            "language": "en",
            "pageSize": 1,
            "apiKey": api_key
        }
        url = f"https://newsapi.org/v2/everything?{urlencode(params)}"
        
        resp = requests.get(url, timeout=15)
        st.info(f"NewsAPI Test: Status code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            st.success("✅ NewsAPI connection successful!")
            articles = data.get("articles", [])
            st.info(f"Found {len(articles)} articles in test")
            if articles:
                st.json(articles[0])
        elif resp.status_code == 401:
            st.error("❌ NewsAPI: Unauthorized - Check your API key")
        elif resp.status_code == 429:
            st.warning("⚠️ NewsAPI: Rate limited - Try again later")
        else:
            st.error(f"❌ NewsAPI: Error {resp.status_code}")
            if resp.content:
                try:
                    error_data = resp.json()
                    st.error(f"Error details: {error_data}")
                except:
                    st.error(f"Error response: {resp.text[:200]}")
                    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ NewsAPI: Network error - {str(e)}")
    except Exception as e:
        st.error(f"❌ NewsAPI: Unexpected error - {str(e)}")

def test_lusha_connection(api_key: str):
    """Test Lusha API connection with a simple request"""
    headers = {
        "api_key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Test with a simple request using the bulk endpoint
    test_payload = {
        "contacts": [
            {
                "contactId": "1",
                "fullName": "John Doe",
                "email": "john@example.com",
                "companies": [
                    {
                        "name": "Example Corp",
                        "domain": "example.com",
                        "isCurrent": True
                    }
                ]
            }
        ],
        "metadata": {
            "revealEmails": True,
            "revealPhones": True
        }
    }
    
    try:
        st.info("Testing Lusha API connection...")
        resp = requests.post(
            "https://api.lusha.com/v2/person",
            headers=headers,
            json=test_payload,
            timeout=30,
        )
        
        st.info(f"Lusha API Test: Status code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            st.success("✅ Lusha API connection successful!")
            st.json(data)
        elif resp.status_code == 401:
            st.error("❌ Lusha API: Unauthorized - Check your API key")
        elif resp.status_code == 403:
            st.error("❌ Lusha API: Forbidden - Check your account permissions or plan")
        elif resp.status_code == 429:
            st.warning("⚠️ Lusha API: Rate limited - Try again later")
        else:
            st.error(f"❌ Lusha API: Error {resp.status_code}")
            if resp.content:
                try:
                    error_data = resp.json()
                    st.error(f"Error details: {error_data}")
                except:
                    st.error(f"Error response: {resp.text[:200]}")
                    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Lusha API: Network error - {str(e)}")
    except Exception as e:
        st.error(f"❌ Lusha API: Unexpected error - {str(e)}")

def test_apollo_with_examples():
    """Test Apollo API with specific examples provided by user"""
    api_key = "lfzgyWLZw6JLj4T18w0z5g"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Test with the two specific examples
    test_payload = {
        "details": [
            {
                "name": "Diane Barr",
                "organization_name": "Camas"
            },
            {
                "name": "James Barbis", 
                "organization_name": "Geosyntec Consultants"
            }
        ],
        "reveal_personal_emails": False
    }
    
    try:
        st.info("Testing Apollo API with specific examples...")
        st.info(f"Request payload: {test_payload}")
        
        resp = requests.post(
            "https://api.apollo.io/api/v1/people/bulk_match",
            headers=headers,
            json=test_payload,
            timeout=30,
        )
        
        st.info(f"Apollo API Test: Status code: {resp.status_code}")
        st.info(f"Apollo API Test: Response headers: {dict(resp.headers)}")
        
        if resp.status_code == 200:
            data = resp.json()
            st.success("✅ Apollo API connection successful!")
            st.info(f"Response type: {type(data)}")
            st.info(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            st.json(data)
            
            # Try to extract emails
            if isinstance(data, list):
                for i, person in enumerate(data):
                    st.info(f"Person {i+1}: {person}")
                    email = person.get("email") or person.get("work_email") or person.get("primary_email")
                    if email:
                        st.success(f"Found email: {email}")
                    else:
                        st.warning("No email found")
            elif isinstance(data, dict):
                # Check for common response structures
                for key in ["matches", "people", "enrichments", "data", "results"]:
                    if key in data and isinstance(data[key], list):
                        st.info(f"Found data under key '{key}': {len(data[key])} items")
                        for i, person in enumerate(data[key]):
                            st.info(f"Person {i+1}: {person}")
                            email = person.get("email") or person.get("work_email") or person.get("primary_email")
                            if email:
                                st.success(f"Found email: {email}")
                            else:
                                st.warning("No email found")
                        break
                        
        elif resp.status_code == 401:
            st.error("❌ Apollo API: Unauthorized - Check your API key")
            st.error("This usually means the API key is invalid or expired")
        elif resp.status_code == 403:
            st.error("❌ Apollo API: Forbidden - Check your account permissions")
            st.error("This usually means your account doesn't have access to this endpoint")
        elif resp.status_code == 429:
            st.warning("⚠️ Apollo API: Rate limited - Try again later")
        else:
            st.error(f"❌ Apollo API: Error {resp.status_code}")
            if resp.content:
                try:
                    error_data = resp.json()
                    st.error(f"Error details: {error_data}")
                except:
                    st.error(f"Error response: {resp.text[:500]}")
                    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Apollo API: Network error - {str(e)}")
    except Exception as e:
        st.error(f"❌ Apollo API: Unexpected error - {str(e)}")

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
    APOLLO_API_KEY = st.text_input("Apollo API Key (for bulk email enrichment)", type="password",
                                   help="Header: Authorization: Bearer <token>")
    LUSHA_API_KEY = st.text_input("Lusha API Key (for bulk contact enrichment)", type="password",
                                  help="Header: api_key: <your_key>")
    
    # Lusha API limits
    st.subheader("Lusha API Limits")
    LUSHA_MAX_CONTACTS = st.number_input(
        "Max contacts to enrich with Lusha", 
        min_value=1, 
        max_value=1000, 
        value=100,
        help="Limit the number of contacts sent to Lusha API to control costs and rate limits"
    )
    st.caption(f"💡 Lusha processes up to 100 contacts per batch. With {LUSHA_MAX_CONTACTS} contacts, you'll make {(LUSHA_MAX_CONTACTS + 99) // 100} API calls.")
    
    # Apollo API limits
    APOLLO_MAX_CONTACTS = st.number_input(
        "Max contacts to enrich with Apollo", 
        min_value=1, 
        max_value=1000, 
        value=50,
        help="Limit the number of contacts sent to Apollo API to control costs and rate limits"
    )
    st.caption(f"💡 Apollo processes up to 10 contacts per batch. With {APOLLO_MAX_CONTACTS} contacts, you'll make {(APOLLO_MAX_CONTACTS + 9) // 10} API calls.")
    
    # Cost estimation
    if LUSHA_MAX_CONTACTS > 0 or APOLLO_MAX_CONTACTS > 0:
        st.info("💰 **Estimated API Costs:**")
        if LUSHA_MAX_CONTACTS > 0:
            st.caption(f"• Lusha: ~{LUSHA_MAX_CONTACTS} credits (varies by plan)")
        if APOLLO_MAX_CONTACTS > 0:
            st.caption(f"• Apollo: ~{APOLLO_MAX_CONTACTS} credits (varies by plan)")
        st.caption("💡 Check your API provider's pricing for exact costs")
    
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
    
    # Test API connections
    st.subheader("Test API Connections")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Apollo API"):
            if APOLLO_API_KEY:
                test_apollo_connection(APOLLO_API_KEY)
            else:
                st.error("Please enter Apollo API key first")
        
        # Quick test with provided examples
        if st.button("Test Apollo with Examples"):
            test_apollo_with_examples()
    
    with col2:
        if st.button("Test NewsAPI"):
            if NEWSAPI_KEY:
                test_newsapi_connection(NEWSAPI_KEY)
            else:
                st.error("Please enter NewsAPI key first")
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Test Lusha API"):
            if LUSHA_API_KEY:
                test_lusha_connection(LUSHA_API_KEY)
            else:
                st.error("Please enter Lusha API key first")

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
    
    # Map common column variations to standard names
    column_mapping = {
        "employer_(parsed)": "company",
        "employer_parsed": "company",
        "employer": "company",
        "organization": "company",
        "org": "company",
        "firm": "company",
        "business": "company"
    }
    
    # Rename columns if they exist
    mapped_columns = []
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
            mapped_columns.append(f"{old_name} → {new_name}")
    
    # Store mapping info for display
    if mapped_columns:
        df.attrs['column_mappings'] = mapped_columns
    
    return df

def _split_name_full(name: str) -> Dict[str, str]:
    """
    Split a full name into first_name / last_name.
    Rules:
      - If a comma exists: 'Last, First Middle Suffix' → last from left, first from right
      - Else: 'First Middle Last' (support particles like 'de', 'de la', 'van', 'von', etc.)
      - Strip prefixes (Dr., Mr., Ms., Prof.) and suffixes (Jr., Sr., II/III/IV, PhD, PE, MBA)
      - Preserve hyphenated and multi-word last names
    """
    if not isinstance(name, str):
        return {"first_name": "", "last_name": ""}
    n = " ".join(name.replace("\u00A0", " ").split())  # normalize spaces
    if not n:
        return {"first_name": "", "last_name": ""}

    prefixes = {"dr", "mr", "mrs", "ms", "mx", "prof"}
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v", "phd", "pe", "mba", "esq"}
    # multi-word particles should be checked first (ordered by length desc)
    multi_particles = [("de la",), ("del",), ("van der",), ("van de",), ("von der",)]
    particles = {"da","de","di","du","dos","das","van","von","bin","al","la","le","st.","saint"}

    def _strip_prefix_suffix(tokens: list) -> list:
        toks = [t.strip(".").lower() for t in tokens]
        # strip prefix
        if toks and toks[0] in prefixes:
            tokens = tokens[1:]
            toks = toks[1:]
        # strip suffix (may be multiple)
        while toks and toks[-1].strip(".").lower() in suffixes:
            tokens = tokens[:-1]
            toks = toks[:-1]
        return tokens

    # Case 1: "Last, First Middle Suffix"
    if "," in n:
        left, right = [x.strip() for x in n.split(",", 1)]
        left_tokens = _strip_prefix_suffix(left.split())
        right_tokens = _strip_prefix_suffix(right.split())
        last_tokens = left_tokens if left_tokens else []
        first_tokens = right_tokens if right_tokens else []
        first = first_tokens[0] if first_tokens else ""
        last = " ".join(last_tokens) if last_tokens else ""
        return {"first_name": first, "last_name": last}

    # Case 2: "First Middle Last" with particles
    tokens = _strip_prefix_suffix(n.split())
    if not tokens:
        return {"first_name": "", "last_name": ""}
    if len(tokens) == 1:
        return {"first_name": tokens[0], "last_name": ""}

    # Build last name from right, capturing particles
    # Start with last token, then include preceding particle(s) or multi-word sequences.
    t_lower = [t.lower() for t in tokens]
    i_last = len(tokens) - 1
    last_start = i_last

    # Check multi-word particles (e.g., 'de la', 'van der')
    joined = " ".join(t_lower)
    for (mp,) in multi_particles:
        if joined.endswith(" " + mp + " " + t_lower[-1]) or joined.endswith(mp + " " + t_lower[-1]):
            mp_len = len(mp.split())
            last_start = max(0, i_last - mp_len)
            break

    # Include single-word particles immediately before last name
    while last_start - 1 >= 0 and t_lower[last_start - 1] in particles:
        last_start -= 1

    last_tokens = tokens[last_start:]
    first_tokens = tokens[:last_start]
    # First name is the first of the remaining tokens
    first = first_tokens[0] if first_tokens else ""
    last = " ".join(last_tokens)
    return {"first_name": first, "last_name": last}

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
            timeout=30,  # Longer timeout for email generation
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
# API Test Functions
# ---------------------------

def test_apollo_connection(api_key: str):
    """Test Apollo API connection with a simple request"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Test with a simple request
    test_payload = {
        "details": [
            {
                "name": "John Doe",
                "title": "Software Engineer",
                "organization_name": "Example Corp"
            }
        ],
        "reveal_personal_emails": False
    }
    
    try:
        st.info("Testing Apollo API connection...")
        resp = requests.post(
            "https://api.apollo.io/api/v1/people/bulk_match",
            headers=headers,
            json=test_payload,
            timeout=30,
        )
        
        st.info(f"Apollo API Test: Status code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            st.success("✅ Apollo API connection successful!")
            st.json(data)
        elif resp.status_code == 401:
            st.error("❌ Apollo API: Unauthorized - Check your API key")
        elif resp.status_code == 403:
            st.error("❌ Apollo API: Forbidden - Check your account permissions")
        elif resp.status_code == 429:
            st.warning("⚠️ Apollo API: Rate limited - Try again later")
        else:
            st.error(f"❌ Apollo API: Error {resp.status_code}")
            if resp.content:
                try:
                    error_data = resp.json()
                    st.error(f"Error details: {error_data}")
                except:
                    st.error(f"Error response: {resp.text[:200]}")
                    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Apollo API: Network error - {str(e)}")
    except Exception as e:
        st.error(f"❌ Apollo API: Unexpected error - {str(e)}")

def test_newsapi_connection(api_key: str):
    """Test NewsAPI connection with a simple request"""
    try:
        st.info("Testing NewsAPI connection...")
        
        # Test with a simple query
        params = {
            "q": "technology",
            "language": "en",
            "pageSize": 1,
            "apiKey": api_key
        }
        url = f"https://newsapi.org/v2/everything?{urlencode(params)}"
        
        resp = requests.get(url, timeout=15)
        st.info(f"NewsAPI Test: Status code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            st.success("✅ NewsAPI connection successful!")
            articles = data.get("articles", [])
            st.info(f"Found {len(articles)} articles in test")
            if articles:
                st.json(articles[0])
        elif resp.status_code == 401:
            st.error("❌ NewsAPI: Unauthorized - Check your API key")
        elif resp.status_code == 429:
            st.warning("⚠️ NewsAPI: Rate limited - Try again later")
        else:
            st.error(f"❌ NewsAPI: Error {resp.status_code}")
            if resp.content:
                try:
                    error_data = resp.json()
                    st.error(f"Error details: {error_data}")
                except:
                    st.error(f"Error response: {resp.text[:200]}")
                    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ NewsAPI: Network error - {str(e)}")
    except Exception as e:
        st.error(f"❌ NewsAPI: Unexpected error - {str(e)}")

def test_apollo_with_examples():
    """Test Apollo API with specific examples provided by user"""
    api_key = "lfzgyWLZw6JLj4T18w0z5g"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Test with the two specific examples
    test_payload = {
        "details": [
            {
                "name": "Diane Barr",
                "organization_name": "Camas"
            },
            {
                "name": "James Barbis", 
                "organization_name": "Geosyntec Consultants"
            }
        ],
        "reveal_personal_emails": False
    }
    
    try:
        st.info("Testing Apollo API with specific examples...")
        st.info(f"Request payload: {test_payload}")
        
        resp = requests.post(
            "https://api.apollo.io/api/v1/people/bulk_match",
            headers=headers,
            json=test_payload,
            timeout=30,
        )
        
        st.info(f"Apollo API Test: Status code: {resp.status_code}")
        st.info(f"Apollo API Test: Response headers: {dict(resp.headers)}")
        
        if resp.status_code == 200:
            data = resp.json()
            st.success("✅ Apollo API connection successful!")
            st.info(f"Response type: {type(data)}")
            st.info(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            st.json(data)
            
            # Try to extract emails
            if isinstance(data, list):
                for i, person in enumerate(data):
                    st.info(f"Person {i+1}: {person}")
                    email = person.get("email") or person.get("work_email") or person.get("primary_email")
                    if email:
                        st.success(f"Found email: {email}")
                    else:
                        st.warning("No email found")
            elif isinstance(data, dict):
                # Check for common response structures
                for key in ["matches", "people", "enrichments", "data", "results"]:
                    if key in data and isinstance(data[key], list):
                        st.info(f"Found data under key '{key}': {len(data[key])} items")
                        for i, person in enumerate(data[key]):
                            st.info(f"Person {i+1}: {person}")
                            email = person.get("email") or person.get("work_email") or person.get("primary_email")
                            if email:
                                st.success(f"Found email: {email}")
                            else:
                                st.warning("No email found")
                        break
                        
        elif resp.status_code == 401:
            st.error("❌ Apollo API: Unauthorized - Check your API key")
            st.error("This usually means the API key is invalid or expired")
        elif resp.status_code == 403:
            st.error("❌ Apollo API: Forbidden - Check your account permissions")
            st.error("This usually means your account doesn't have access to this endpoint")
        elif resp.status_code == 429:
            st.warning("⚠️ Apollo API: Rate limited - Try again later")
        else:
            st.error(f"❌ Apollo API: Error {resp.status_code}")
            if resp.content:
                try:
                    error_data = resp.json()
                    st.error(f"Error details: {error_data}")
                except:
                    st.error(f"Error response: {resp.text[:500]}")
                    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Apollo API: Network error - {str(e)}")
    except Exception as e:
        st.error(f"❌ Apollo API: Unexpected error - {str(e)}")

def test_lusha_connection(api_key: str):
    """Test Lusha API connection with a simple request"""
    headers = {
        "api_key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Test with a simple request using the bulk endpoint
    test_payload = {
        "contacts": [
            {
                "contactId": "1",
                "fullName": "John Doe",
                "email": "john@example.com",
                "companies": [
                    {
                        "name": "Example Corp",
                        "domain": "example.com",
                        "isCurrent": True
                    }
                ]
            }
        ],
        "metadata": {
            "revealEmails": True,
            "revealPhones": True
        }
    }
    
    try:
        st.info("Testing Lusha API connection...")
        resp = requests.post(
            "https://api.lusha.com/v2/person",
            headers=headers,
            json=test_payload,
            timeout=30,
        )
        
        st.info(f"Lusha API Test: Status code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            st.success("✅ Lusha API connection successful!")
            st.json(data)
        elif resp.status_code == 401:
            st.error("❌ Lusha API: Unauthorized - Check your API key")
        elif resp.status_code == 403:
            st.error("❌ Lusha API: Forbidden - Check your account permissions or plan")
        elif resp.status_code == 429:
            st.warning("⚠️ Lusha API: Rate limited - Try again later")
        else:
            st.error(f"❌ Lusha API: Error {resp.status_code}")
            if resp.content:
                try:
                    error_data = resp.json()
                    st.error(f"Error details: {error_data}")
                except:
                    st.error(f"Error response: {resp.text[:200]}")
                    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Lusha API: Network error - {str(e)}")
    except Exception as e:
        st.error(f"❌ Lusha API: Unexpected error - {str(e)}")

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

# Apollo.io — Bulk People Enrichment (name/title[/company] -> email only)
# Docs: POST https://api.apollo.io/api/v1/people/bulk_match (up to 10 per call)
# - Uses details[] array; more info improves match rate
# - By default, personal emails/phones are not returned (we do not request phones)
# - Consumes credits; bulk per-minute rate ≈ 50% of single endpoint

def enrich_emails_with_apollo_bulk(df: pd.DataFrame, api_key: str, reveal_personal_emails: bool = False, max_contacts: int = 50) -> pd.DataFrame:
    if not api_key or df.empty:
        st.warning("Apollo API: No API key provided or empty dataframe")
        return df
    
    st.info("🚀 Apollo API: Starting bulk email enrichment...")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    df = df.copy()
    # work on rows missing email
    need_idx = df.index[df["email"].isna() | (df["email"].astype(str).str.strip() == "")].tolist()
    if not need_idx:
        st.info("Apollo API: No rows need email enrichment (all have emails)")
        return df

    # Limit the number of contacts to process
    if len(need_idx) > max_contacts:
        st.info(f"Apollo API: Limiting enrichment to first {max_contacts} contacts (out of {len(need_idx)} total)")
        need_idx = need_idx[:max_contacts]
    else:
        st.info(f"Apollo API: Found {len(need_idx)} rows needing email enrichment")

    # Build payloads in batches of 10
    BATCH = 10
    enriched_ct = 0
    skipped_ct = 0
    phase = st.empty()
    bar = st.progress(0, text="Apollo bulk enrichment starting…")
    total_batches = (len(need_idx) + BATCH - 1) // BATCH
    
    for bi, start in enumerate(range(0, len(need_idx), BATCH), start=1):
        batch_idx = need_idx[start:start+BATCH]
        details = []
        row_map = []  # keep mapping to write results back
        for ridx in batch_idx:
            r = df.loc[ridx]
            # Prefer a full name if present; else build from first/last
            full_name = (str(r.get("first_name") or "").strip() + " " + str(r.get("last_name") or "").strip()).strip()
            if not full_name and "name" in df.columns:
                full_name = str(r.get("name") or "").strip()
            title = str(r.get("title") or "").strip()
            company = str(r.get("company") or "").strip()
            # Require at least a name OR (first+last)
            if not full_name:
                skipped_ct += 1
                continue
            item = {"name": full_name}
            if title:
                item["title"] = title
            if company:
                # Including company helps matching even though user asked name+title;
                # if you prefer strictly name+title, comment this line out.
                item["organization_name"] = company
            details.append(item)
            row_map.append(ridx)
        if not details:
            continue

        payload = {
            "details": details,
            # We want email only; phones remain default (False). Personal emails optional:
            "reveal_personal_emails": bool(reveal_personal_emails),
            # You can also pass "reveal_phone_number": False (default)
        }
        
        st.info(f"Apollo API: Making request for batch {bi}/{total_batches} with {len(details)} people")
        try:
            resp = requests.post(
                "https://api.apollo.io/api/v1/people/bulk_match",
                headers=headers,
                json=payload,
                timeout=30,
            )
            
            st.info(f"Apollo API: Response status code: {resp.status_code}")
            
            if resp.status_code == 429:
                st.warning("Apollo API: Rate limited, retrying after delay...")
                # simple backoff and retry once
                time.sleep(2.0)
                resp = requests.post(
                    "https://api.apollo.io/api/v1/people/bulk_match",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                st.info(f"Apollo API: Retry response status code: {resp.status_code}")
            
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            
            st.info(f"Apollo API: Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Based on Apollo API docs, the response should contain the enriched data directly
            # The response format should be an array of enriched person objects
            results = data if isinstance(data, list) else []
            
            # If it's not a list, try to find the data in common response fields
            if not results and isinstance(data, dict):
                for key in ["matches", "people", "enrichments", "data", "results"]:
                    if isinstance(data.get(key), list):
                        results = data[key]
                        st.info(f"Apollo API: Found results under key '{key}' with {len(results)} items")
                        break
            
            if not results:
                st.warning("Apollo API: No results found in response")
                st.info(f"Apollo API: Full response: {data}")
            else:
                st.info(f"Apollo API: Processing {len(results)} results")

            # Map each returned item to the corresponding row by order
            for i, item in enumerate(results):
                if i >= len(row_map):
                    break
                ridx = row_map[i]
                # Try common email fields (prefer work/professional email)
                email = (
                    item.get("email") or
                    item.get("work_email") or
                    item.get("professional_email") or
                    item.get("primary_email") or
                    item.get("contact_email") or
                    (item.get("emails")[0] if isinstance(item.get("emails"), list) and item["emails"] else None) or
                    None
                )
                
                # Log the item structure for debugging
                st.info(f"Apollo API: Item {i} keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}")
                if email and str(email).strip():
                    df.at[ridx, "email"] = str(email).strip()
                    enriched_ct += 1
                    st.info(f"Apollo API: Found email for row {ridx}: {email}")
                    
        except requests.exceptions.RequestException as e:
            st.error(f"Apollo API: Request failed for batch {bi}: {str(e)}")
            continue
        except Exception as e:
            st.error(f"Apollo API: Unexpected error for batch {bi}: {str(e)}")
            continue

        pct = int(100 * bi / max(1, total_batches))
        bar.progress(min(pct, 100), text=f"Apollo enrichment: batch {bi}/{total_batches}")
        phase.markdown(f"Enriched emails so far: **{enriched_ct}** · skipped (no name): **{skipped_ct}**")

        # polite pacing to respect rate limits (bulk is throttled vs single endpoint)
        time.sleep(0.3)

    phase.markdown(f"**Apollo bulk enrichment complete.** New emails: **{enriched_ct}** · Skipped: **{skipped_ct}**")
    bar.progress(100, text="Apollo enrichment done")
    return df

def enrich_contacts_with_lusha_bulk(df: pd.DataFrame, api_key: str, reveal_emails: bool = True, reveal_phones: bool = True, max_contacts: int = 100) -> pd.DataFrame:
    """
    Enrich contacts using Lusha's bulk person enrichment API.
    Supports up to 100 contacts per request.
    """
    if not api_key or df.empty:
        st.warning("Lusha API: No API key provided or empty dataframe")
        return df
    
    st.info("🚀 Lusha API: Starting bulk contact enrichment...")
    headers = {
        "api_key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    df = df.copy()
    
    # Work on all rows to enrich missing data, but limit to max_contacts
    need_idx = df.index.tolist()
    if not need_idx:
        st.info("Lusha API: No rows to enrich")
        return df

    # Limit the number of contacts to process
    if len(need_idx) > max_contacts:
        st.info(f"Lusha API: Limiting enrichment to first {max_contacts} contacts (out of {len(need_idx)} total)")
        need_idx = need_idx[:max_contacts]
    else:
        st.info(f"Lusha API: Found {len(need_idx)} rows to enrich")

    # Build payloads in batches of 100 (Lusha's limit)
    BATCH = 100
    enriched_ct = 0
    skipped_ct = 0
    phase = st.empty()
    bar = st.progress(0, text="Lusha bulk enrichment starting…")
    total_batches = (len(need_idx) + BATCH - 1) // BATCH
    
    for bi, start in enumerate(range(0, len(need_idx), BATCH), start=1):
        batch_idx = need_idx[start:start+BATCH]
        contacts = []
        row_map = []  # keep mapping to write results back
        
        for ridx in batch_idx:
            r = df.loc[ridx]
            
            # Build contact data - Lusha requires either LinkedIn URL, email, or firstName+lastName+(companyName OR companyDomain)
            contact_data = {"contactId": str(ridx)}
            
            # Try to get full name
            full_name = (str(r.get("first_name") or "").strip() + " " + str(r.get("last_name") or "").strip()).strip()
            if not full_name and "name" in df.columns:
                full_name = str(r.get("name") or "").strip()
            
            # Add available data
            if full_name:
                contact_data["fullName"] = full_name
            
            email = str(r.get("email") or "").strip()
            if email:
                contact_data["email"] = email
            
            linkedin_url = str(r.get("linkedin_url") or "").strip()
            if linkedin_url:
                contact_data["linkedinUrl"] = linkedin_url
            
            company = str(r.get("company") or "").strip()
            if company:
                # Try to extract domain from company name or use a placeholder
                company_domain = company.lower().replace(" ", "").replace(".", "") + ".com"
                contact_data["companies"] = [{
                    "name": company,
                    "domain": company_domain,
                    "isCurrent": True
                }]
            
            # Skip if we don't have enough data for Lusha's requirements
            if not (contact_data.get("linkedinUrl") or contact_data.get("email") or 
                   (contact_data.get("fullName") and contact_data.get("companies"))):
                skipped_ct += 1
                continue
                
            contacts.append(contact_data)
            row_map.append(ridx)
        
        if not contacts:
            continue

        payload = {
            "contacts": contacts,
            "metadata": {
                "revealEmails": reveal_emails,
                "revealPhones": reveal_phones,
                "partialProfile": True  # Allow partial matches
            }
        }
        
        st.info(f"Lusha API: Making request for batch {bi}/{total_batches} with {len(contacts)} contacts")
        try:
            resp = requests.post(
                "https://api.lusha.com/v2/person",
                headers=headers,
                json=payload,
                timeout=30,
            )
            
            st.info(f"Lusha API: Response status code: {resp.status_code}")
            
            if resp.status_code == 429:
                st.warning("Lusha API: Rate limited, retrying after delay...")
                time.sleep(2.0)
                resp = requests.post(
                    "https://api.lusha.com/v2/person",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                st.info(f"Lusha API: Retry response status code: {resp.status_code}")
            
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            
            st.info(f"Lusha API: Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Process the response - Lusha returns contacts and companies separately
            contacts_data = data.get("contacts", {})
            companies_data = data.get("companies", {})
            
            if not contacts_data:
                st.warning("Lusha API: No contact data found in response")
                continue
            
            # Map each returned contact to the corresponding row
            for contact_id, contact_info in contacts_data.items():
                if contact_info.get("error"):
                    st.warning(f"Lusha API: Error for contact {contact_id}: {contact_info['error']}")
                    continue
                
                contact_data = contact_info.get("data", {})
                if not contact_data:
                    continue
                
                # Find the corresponding row index
                try:
                    ridx = int(contact_id)
                    if ridx not in row_map:
                        continue
                except ValueError:
                    continue
                
                # Extract and update data
                # Email
                email = (contact_data.get("email") or 
                        contact_data.get("workEmail") or 
                        contact_data.get("personalEmail") or
                        None)
                if email and str(email).strip():
                    df.at[ridx, "email"] = str(email).strip()
                    enriched_ct += 1
                
                # Phone
                phone = (contact_data.get("phone") or 
                        contact_data.get("workPhone") or 
                        contact_data.get("personalPhone") or
                        None)
                if phone and str(phone).strip():
                    df.at[ridx, "phone"] = str(phone).strip()
                
                # LinkedIn URL
                linkedin = contact_data.get("linkedinUrl")
                if linkedin and str(linkedin).strip():
                    df.at[ridx, "linkedin_url"] = str(linkedin).strip()
                
                # Title/Job Title
                title = contact_data.get("title") or contact_data.get("jobTitle")
                if title and str(title).strip():
                    df.at[ridx, "title"] = str(title).strip()
                
                # Company information
                company_id = contact_data.get("companyId")
                if company_id and company_id in companies_data:
                    company_info = companies_data[company_id].get("data", {})
                    if company_info:
                        company_name = company_info.get("name")
                        if company_name and str(company_name).strip():
                            df.at[ridx, "company"] = str(company_name).strip()
                        
                        # Add company domain if available
                        company_domain = company_info.get("domain")
                        if company_domain and str(company_domain).strip():
                            df.at[ridx, "company_domain"] = str(company_domain).strip()
                
                # Location
                location = contact_data.get("location")
                if location and str(location).strip():
                    df.at[ridx, "location"] = str(location).strip()
                
                st.info(f"Lusha API: Enriched contact {ridx}")
                    
        except requests.exceptions.RequestException as e:
            st.error(f"Lusha API: Request failed for batch {bi}: {str(e)}")
            continue
        except Exception as e:
            st.error(f"Lusha API: Unexpected error for batch {bi}: {str(e)}")
            continue

        pct = int(100 * bi / max(1, total_batches))
        bar.progress(min(pct, 100), text=f"Lusha enrichment: batch {bi}/{total_batches}")
        phase.markdown(f"Enriched contacts so far: **{enriched_ct}** · skipped (insufficient data): **{skipped_ct}**")

        # Polite pacing to respect rate limits
        time.sleep(0.5)

    phase.markdown(f"**Lusha bulk enrichment complete.** New data enriched: **{enriched_ct}** · Skipped: **{skipped_ct}**")
    bar.progress(100, text="Lusha enrichment done")
    return df

def fetch_recent_mentions(name: str, company: str, newsapi_key: str, lookback_days: int = 30) -> List[str]:
    """Use NewsAPI (or similar) to find recent public mentions of a person/company."""
    if not newsapi_key:
        st.warning("NewsAPI: No API key provided")
        return []

    q_parts = []
    if name: q_parts.append(f"\"{name}\"")
    if company: q_parts.append(f"\"{company}\"")
    if not q_parts:
        st.warning("NewsAPI: No name or company provided for search")
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
    
    st.info(f"NewsAPI: Searching for '{q}' from {from_date}")
    st.info(f"NewsAPI: Request URL: {url}")
    
    try:
        r = requests.get(url, timeout=15)
        st.info(f"NewsAPI: Response status code: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            st.info(f"NewsAPI: Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            articles = data.get("articles", [])
            st.info(f"NewsAPI: Found {len(articles)} articles")
            
            out = []
            for i, a in enumerate(articles):
                title = safe_get(a, "title", default="")
                source = safe_get(a, "source", "name", default="")
                link = safe_get(a, "url", default="")
                published = safe_get(a, "publishedAt", default="")
                if title and link:
                    out.append(f"{title} — {source} ({published}) | {link}")
                    st.info(f"NewsAPI: Article {i+1}: {title[:50]}...")
            
            st.info(f"NewsAPI: Returning {len(out)} valid articles")
            return out
        else:
            st.error(f"NewsAPI: Request failed with status {r.status_code}")
            if r.content:
                try:
                    error_data = r.json()
                    st.error(f"NewsAPI: Error response: {error_data}")
                except:
                    st.error(f"NewsAPI: Error response (raw): {r.text[:200]}")
                    
    except requests.exceptions.RequestException as e:
        st.error(f"NewsAPI: Request failed: {str(e)}")
    except Exception as e:
        st.error(f"NewsAPI: Unexpected error: {str(e)}")
    
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

# ---------------------------
# Email draft generation (template + optional LLM rewrite)
# ---------------------------

class _DefaultDict(dict):
    # allows {missing} placeholders to become "" instead of KeyError
    def __missing__(self, key): return ""

def _clean_text(text: str) -> str:
    """Clean up common encoding issues and funny characters"""
    if not text:
        return ""
    
    # Simple approach: just return the text as-is for now
    # The LLM will handle character cleaning in the prompt
    return text

def _render_template_row(row: pd.Series, template: str, user_query: str) -> str:
    data = {
        "first_name": _clean_text(str(row.get("first_name","") or "")),
        "last_name": _clean_text(str(row.get("last_name","") or "")),
        "name": _clean_text((str(row.get("first_name","") or "") + " " + str(row.get("last_name","") or "")).strip() or str(row.get("name","") or "")),
        "title": _clean_text(str(row.get("title","") or "")),
        "company": _clean_text(str(row.get("company","") or "")),
        "email": _clean_text(str(row.get("email","") or "")),
        "linkedin_url": _clean_text(str(row.get("linkedin_url","") or "")),
        "recent_mentions": _clean_text(str(row.get("recent_mentions","") or "")),
        "query": _clean_text(str(user_query or "")),
    }
    # Basic helpful extras
    if not data["first_name"] and data["name"]:
        data["first_name"] = data["name"].split(" ")[0]
    return template.format_map(_DefaultDict(data))

def _ollama_rewrite_email_batch(drafts: List[Dict[str,str]], model: str, style: str = "concise") -> List[str]:
    """
    drafts: [{ 'draft': <text>, 'first_name':..., 'title':..., 'company':..., 'recent_mentions':..., 'linkedin_url':... }, ...]
    Returns: list of rewritten emails aligned to input order.
    """
    # Keep prompt tight; ask for plain email text.
    style_hint = "Keep to ~120 words, specific and respectful, non-salesy, one clear ask."
    if style == "brief": style_hint = "Keep to ~80 words, specific, non-salesy, one clear ask."
    if style == "detailed": style_hint = "Up to ~160 words, still focused, one clear ask."

    # Build one chat with all items; return a JSON array of strings.
    lines = []
    for i, d in enumerate(drafts):
        # compact JSON-ish line to reduce tokens
        lines.append(json.dumps({
            "i": i,
            "first_name": d.get("first_name",""),
            "title": d.get("title",""),
            "company": d.get("company",""),
            "linkedin_url": d.get("linkedin_url",""),
            "recent_mentions": d.get("recent_mentions","")[:800],  # cap
            "draft": d.get("draft","")[:1200],  # cap
        }, ensure_ascii=False))

    sys = (
        "You refine cold outreach emails to utility/industrial buyers. "
        "Rewrite each draft with the given context (title, company, LinkedIn URL, recent mentions). "
        "Use a professional, non-salesy tone suited to engineers/procurement. "
        f"{style_hint} If there's no subject line, add a short subject like 'High-voltage equipment for [company name]'. "
        "IMPORTANT: Remove any duplicate phrases or repetitive content. Clean up any funny characters like â€, â€™, â€œ, â€, etc. from titles and text. "
        "Replace them with proper punctuation (', ", ", -, etc.). Ensure the final email is clean and professional. "
        "Output STRICT JSON array of strings, each the final email body (subject line on first line like 'Subject: ...')."
    )
    user = "ITEMS:\n" + "\n".join(lines)
    out = _ollama_chat(
        [{"role":"system","content":sys},{"role":"user","content":user}],
        model=model,
        options={"num_predict": 128}
    )
    try:
        arr = json.loads(out)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr
    except Exception:
        pass
    # Fallback to originals on parse failure
    return [d["draft"] for d in drafts]

def generate_email_drafts(df: pd.DataFrame, template: str, user_query: str,
                          use_ollama: bool, ollama_model: str, style: str = "concise",
                          batch_size: int = 10) -> pd.DataFrame:
    """
    Adds 'email_draft' column. First renders the user template per row, then optionally rewrites via Ollama in batches.
    """
    if df.empty or not template.strip():
        return df
    df = df.copy()
    
    # Step 1: render base drafts
    base = []
    for _, r in df.iterrows():
        base.append(_render_template_row(r, template, user_query))
    df["email_draft"] = base

    # Step 2: optional LLM rewrite
    if not use_ollama:
        return df

    phase = st.empty()
    bar = st.progress(0, text="Rewriting email drafts with local LLM…")
    items = []
    for _, r in df.iterrows():
        items.append({
            "draft": r["email_draft"],
            "first_name": _clean_text(str(r.get("first_name","") or "")),
            "title": _clean_text(str(r.get("title","") or "")),
            "company": _clean_text(str(r.get("company","") or "")),
            "linkedin_url": _clean_text(str(r.get("linkedin_url","") or "")),
            "recent_mentions": _clean_text(str(r.get("recent_mentions","") or "")),
        })
    
    out_texts = []
    total = len(items)
    done = 0
    
    for start in range(0, total, batch_size):
        chunk = items[start:start+batch_size]
        try:
            outs = _ollama_rewrite_email_batch(chunk, ollama_model, style=style)
            out_texts.extend(outs)
        except Exception as e:
            st.warning(f"⚠️ LLM rewrite failed for batch {start//batch_size + 1}: {str(e)[:100]}...")
            # Fallback to original drafts for this batch
            out_texts.extend([item["draft"] for item in chunk])
        
        done += len(chunk)
        pct = min(100, int(100 * done / max(1, total)))
        bar.progress(pct)
        phase.markdown(f"📧 Rewrote {done}/{total} drafts ({pct}%)")
        
        # Small delay to prevent overwhelming the LLM
        if done < total:
            time.sleep(0.5)
    
    df["email_draft"] = out_texts[:len(df)]
    bar.progress(100, text="✅ Email drafts ready")
    phase.markdown(f"✅ Generated {len(out_texts)} email drafts")
    return df

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
    
    # Show column mappings if any
    if hasattr(data_df, 'attrs') and 'column_mappings' in data_df.attrs:
        st.info(f"📋 Column mappings: {', '.join(data_df.attrs['column_mappings'])}")
    
    # Ensure expected columns exist
    for col in ["first_name", "last_name", "company", "title", "email", "linkedin_url"]:
        if col not in data_df.columns:
            data_df[col] = None
    
    # If a 'name' column exists, auto-split into first/last where missing
    if "name" in data_df.columns:
        filled_ct = [0]  # Use list to make it mutable
        
        def _maybe_split(row):
            fn = str(row.get("first_name") or "").strip()
            ln = str(row.get("last_name") or "").strip()
            full = str(row.get("name") or "").strip()
            if full and (not fn or not ln):
                parts = _split_name_full(full)
                # only fill blanks; don't overwrite existing content
                if not fn and parts["first_name"]:
                    row["first_name"] = parts["first_name"]
                    filled_ct[0] += 1
                if not ln and parts["last_name"]:
                    row["last_name"] = parts["last_name"]
                    filled_ct[0] += 1
            return row
            
        data_df = data_df.apply(_maybe_split, axis=1)
        if filled_ct[0]:
            st.info(f"Split 'name' into first/last for {filled_ct[0]} fields.")
    st.success(f"Loaded {len(data_df)} attendees.")
    st.dataframe(data_df.head(25), use_container_width=True)

st.subheader("2) Optional enrichment (approved APIs only)")
colA, colB = st.columns(2)

with colA:
    do_zoominfo = st.checkbox("Enrich with ZoomInfo (official API)", value=False, help="Requires API key in sidebar. No scraping.")
with colB:
    do_news_mentions = st.checkbox("Fetch recent mentions (NewsAPI)", value=False, help="Looks for person+company mentions in public news.")

colC, colD = st.columns(2)
with colC:
    do_apollo = st.checkbox("Apollo: Bulk People Enrichment (fill missing emails only)", value=False,
                            help="Sends name/title[/company] in batches of 10; writes back email. Consumes Apollo credits.")
with colD:
    apollo_personal = st.checkbox("Allow personal emails (Apollo)", value=False,
                                  help="If enabled, sets reveal_personal_emails=true.")

colE, colF = st.columns(2)
with colE:
    do_lusha = st.checkbox("Lusha: Bulk Contact Enrichment (comprehensive data)", value=False,
                           help="Enriches emails, phones, titles, companies, LinkedIn URLs. Supports up to 100 contacts per batch.")
with colF:
    lusha_reveal_phones = st.checkbox("Reveal phone numbers (Lusha)", value=True,
                                      help="If enabled, includes phone numbers in enrichment (requires Unified Credits plan).")

# Enrichment section with completion tracking
enrichment_complete = st.session_state.get("enrichment_complete", False)

if st.button("Run Enrichment", disabled=enrichment_complete):
    if data_df is None:
        st.warning("Please upload a CSV first.")
    else:
        df = data_df.copy()
        enrichment_errors = []
        enrichment_success = []

        if do_zoominfo:
            if not ZOOMINFO_API_KEY:
                st.error("ZoomInfo API Key missing.")
                enrichment_errors.append("ZoomInfo: Missing API key")
            else:
                with st.spinner("Enriching via ZoomInfo…"):
                    try:
                        df = enrich_with_zoominfo(df, ZOOMINFO_API_KEY)
                        enrichment_success.append("ZoomInfo: Enrichment completed")
                        st.success("ZoomInfo enrichment pass completed (see notes in code to map fields).")
                    except Exception as e:
                        st.error(f"ZoomInfo enrichment failed: {str(e)}")
                        enrichment_errors.append(f"ZoomInfo: {str(e)}")

        if do_news_mentions:
            if not NEWSAPI_KEY:
                st.error("NewsAPI key missing.")
                enrichment_errors.append("NewsAPI: Missing API key")
            else:
                with st.spinner("Fetching recent public mentions…"):
                    try:
                        mentions_all = []
                        for i, r in df.iterrows():
                            nm = full_name(r)
                            comp = str(r.get("company", "") or "")
                            mentions = fetch_recent_mentions(nm, comp, NEWSAPI_KEY)
                            # Merge Utility Dive RSS matches (placeholder returns [])
                            mentions.extend(fetch_utility_dive(comp))
                            mentions_all.append(" || ".join(mentions) if mentions else "")
                        df["recent_mentions"] = mentions_all
                        enrichment_success.append("NewsAPI: Mentions fetched")
                        st.success("Mentions fetched.")
                    except Exception as e:
                        st.error(f"NewsAPI enrichment failed: {str(e)}")
                        enrichment_errors.append(f"NewsAPI: {str(e)}")

        if do_apollo:
            if not APOLLO_API_KEY:
                st.error("Apollo API Key missing.")
                enrichment_errors.append("Apollo: Missing API key")
            else:
                with st.spinner("Enriching emails via Apollo bulk…"):
                    try:
                        df_before = df.copy()
                        df = enrich_emails_with_apollo_bulk(df, APOLLO_API_KEY, reveal_personal_emails=apollo_personal, max_contacts=APOLLO_MAX_CONTACTS)
                        
                        # Check if any emails were actually enriched
                        emails_before = df_before["email"].notna().sum()
                        emails_after = df["email"].notna().sum()
                        new_emails = emails_after - emails_before
                        
                        if new_emails > 0:
                            enrichment_success.append(f"Apollo: {new_emails} new emails found")
                            st.success(f"Apollo email enrichment completed. Found {new_emails} new emails.")
                        else:
                            st.warning("Apollo enrichment completed but no new emails were found.")
                            st.info("**Possible reasons for no results:**")
                            st.markdown("""
                            - **Insufficient data**: Apollo needs at least name + title/company for matching
                            - **No matches found**: The person may not be in Apollo's database
                            - **API rate limits**: Try again in a few minutes
                            - **Invalid API key**: Check your Apollo API key is correct and active
                            - **Account limits**: Your Apollo plan may have reached its monthly limit
                            - **Data quality**: Names/titles may need to be more specific or standardized
                            """)
                            enrichment_errors.append("Apollo: No new emails found")
                    except Exception as e:
                        st.error(f"Apollo enrichment failed: {str(e)}")
                        enrichment_errors.append(f"Apollo: {str(e)}")
                        st.info("**Common Apollo API issues:**")
                        st.markdown("""
                        - **401 Unauthorized**: Invalid or expired API key
                        - **429 Too Many Requests**: Rate limit exceeded, wait and retry
                        - **403 Forbidden**: Account suspended or insufficient credits
                        - **400 Bad Request**: Invalid request format or missing required fields
                        - **Network timeout**: Apollo servers may be slow, try smaller batches
                        """)

        if do_lusha:
            if not LUSHA_API_KEY:
                st.error("Lusha API Key missing.")
                enrichment_errors.append("Lusha: Missing API key")
            else:
                with st.spinner("Enriching contacts via Lusha bulk…"):
                    try:
                        df_before = df.copy()
                        df = enrich_contacts_with_lusha_bulk(df, LUSHA_API_KEY, reveal_emails=True, reveal_phones=lusha_reveal_phones, max_contacts=LUSHA_MAX_CONTACTS)
                        
                        # Check if any data was actually enriched
                        emails_before = df_before["email"].notna().sum()
                        emails_after = df["email"].notna().sum()
                        new_emails = emails_after - emails_before
                        
                        phones_before = df_before.get("phone", pd.Series()).notna().sum()
                        phones_after = df.get("phone", pd.Series()).notna().sum()
                        new_phones = phones_after - phones_before
                        
                        if new_emails > 0 or new_phones > 0:
                            enrichment_success.append(f"Lusha: {new_emails} new emails, {new_phones} new phones found")
                            st.success(f"Lusha contact enrichment completed. Found {new_emails} new emails and {new_phones} new phone numbers.")
                        else:
                            st.warning("Lusha enrichment completed but no new data was found.")
                            st.info("**Possible reasons for no results:**")
                            st.markdown("""
                            - **Insufficient data**: Lusha needs at least LinkedIn URL, email, or name+company for matching
                            - **No matches found**: The contacts may not be in Lusha's database
                            - **API rate limits**: Try again in a few minutes
                            - **Invalid API key**: Check your Lusha API key is correct and active
                            - **Account limits**: Your Lusha plan may have reached its limits
                            - **Plan restrictions**: Phone numbers require Unified Credits plan
                            """)
                            enrichment_errors.append("Lusha: No new data found")
                    except Exception as e:
                        st.error(f"Lusha enrichment failed: {str(e)}")
                        enrichment_errors.append(f"Lusha: {str(e)}")
                        st.info("**Common Lusha API issues:**")
                        st.markdown("""
                        - **401 Unauthorized**: Invalid or expired API key
                        - **429 Too Many Requests**: Rate limit exceeded, wait and retry
                        - **403 Forbidden**: Account suspended, insufficient credits, or plan restrictions
                        - **400 Bad Request**: Invalid request format or missing required fields
                        - **Network timeout**: Lusha servers may be slow, try smaller batches
                        """)

        # Store results and mark enrichment as complete
        st.session_state["df_enriched"] = df
        st.session_state["enrichment_complete"] = True
        st.session_state["enrichment_errors"] = enrichment_errors
        st.session_state["enrichment_success"] = enrichment_success

# Show enrichment status
if enrichment_complete:
    st.success("✅ Enrichment completed!")
    
    if st.session_state.get("enrichment_success"):
        st.info(f"**Successful enrichments:** {', '.join(st.session_state['enrichment_success'])}")
    
    if st.session_state.get("enrichment_errors"):
        st.warning(f"**Issues encountered:** {', '.join(st.session_state['enrichment_errors'])}")
    
    if st.button("Reset Enrichment", help="Clear enrichment results and allow re-running"):
        st.session_state["enrichment_complete"] = False
        st.session_state["enrichment_errors"] = []
        st.session_state["enrichment_success"] = []
        st.rerun()

st.subheader("3) Prioritize by your query")
user_query = st.text_input("Describe what you’re looking for (e.g., 'grid-scale storage, US Southeast utilities, hydrogen blending pilots')")

# Email draft options
with st.expander("Optional: generate per-contact email drafts"):
    MAKE_DRAFTS = st.checkbox("Add an email draft column to the CSV", value=False)
    
    if MAKE_DRAFTS:
        MAX_EMAILS = st.number_input(
            "Limit email generation to top N ranked contacts", 
            min_value=1, 
            max_value=1000, 
            value=50,
            help="Only generate emails for the top N highest-ranked contacts to save time and API costs"
        )
        
        DEFAULT_TEMPLATE = (
            "Subject: Exploring high-voltage equipment fit for {company}\n\n"
            "Hi {first_name},\n\n"
            "I work with industrial innovators enabling high-voltage reliability and safety. Given your role as {title} at {company}, "
            "I thought this might be relevant. {recent_mentions}\n\n"
            "If helpful, here's a short brief on how we support utilities/substations similar to yours. "
            "Would you be open to a quick call next week?\n\n"
            "Best,\n"
            "[Your Name]\n"
            "[Your Company]\n"
            "[Your Contact Info]\n"
            "{linkedin_url}\n"
        )
        EMAIL_TEMPLATE = st.text_area(
            "Email template (use placeholders like {first_name}, {title}, {company}, {recent_mentions}, {linkedin_url}, {query})",
            value=DEFAULT_TEMPLATE,
            height=220,
        )
        USE_LLM_REWRITE = st.checkbox("Refine each draft with local LLM (Ollama)", value=True,
                                      help="Uses your model choice in the sidebar to tailor tone and include relevant context.")
        LLM_STYLE = st.selectbox("Refinement style", options=["brief","concise","detailed"], index=1)
        DRAFT_BATCH = st.slider("Draft rewrite batch size", min_value=5, max_value=50, value=10, step=5,
                               help="Smaller batches are more reliable and less likely to timeout")
    else:
        MAX_EMAILS = 0
        EMAIL_TEMPLATE = ""
        USE_LLM_REWRITE = False
        LLM_STYLE = "concise"
        DRAFT_BATCH = 10

# Check if enrichment is required but not complete
enrichment_required = do_zoominfo or do_news_mentions or do_apollo or do_lusha
can_proceed = not enrichment_required or enrichment_complete

if not can_proceed:
    st.info("⏳ Please complete enrichment first before ranking and exporting.")

if st.button("Rank & Export", disabled=not can_proceed):
    df_base = st.session_state.get("df_enriched", data_df)
    if df_base is None:
        st.warning("Please upload a CSV first.")
    elif not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Scoring (negation-aware)…"):
            ranked = negation_aware_rank(df_base, user_query, USE_OLLAMA, OLLAMA_MODEL)
            
            # Optionally add email drafts (only for top N contacts)
            if MAKE_DRAFTS and len(ranked) > 0:
                # Limit to top N contacts for email generation
                top_contacts = ranked.head(MAX_EMAILS)
                st.info(f"📧 Generating emails for top {len(top_contacts)} contacts (out of {len(ranked)} total)")
                
                ranked_with_emails = generate_email_drafts(
                    top_contacts, EMAIL_TEMPLATE, user_query,
                    use_ollama=USE_LLM_REWRITE and USE_OLLAMA,
                    ollama_model=OLLAMA_MODEL,
                    style=LLM_STYLE,
                    batch_size=DRAFT_BATCH
                )
                
                # Merge back with the full ranked list
                ranked = ranked.merge(
                    ranked_with_emails[['email_draft']], 
                    left_index=True, 
                    right_index=True, 
                    how='left'
                )
            
            if len(ranked) == 0:
                st.warning("No results found after filtering. Try:")
                st.markdown("- **Less restrictive query** (e.g., 'engineers' instead of 'high voltage transmission engineers')")
                st.markdown("- **Disable LLM filtering** (uncheck the LLM checkbox)")
                st.markdown("- **Check your CSV data** (ensure titles and companies are populated)")
            else:
                cols = ["first_name","last_name","company","title","email","linkedin_url","recent_mentions",
                        "llm_role","score","base_score"]
                # Add Lusha-specific columns if they exist
                for lusha_col in ["phone", "location", "company_domain"]:
                    if lusha_col in ranked.columns:
                        cols.append(lusha_col)
                if "email_draft" in ranked.columns:
                    cols.append("email_draft")
                for c in cols:
                    if c not in ranked.columns:
                        ranked[c] = None
                out = ranked[cols]
                st.success(f"Ranked {len(out)} people. (Local LLM={'on' if USE_OLLAMA else 'off'})")
                st.dataframe(out.head(50), use_container_width=True)

                # Export
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download prioritized CSV",
                    data=csv_bytes,
                    file_name="prioritized_contacts.csv",
                    mime="text/csv"
                )
