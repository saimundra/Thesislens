# yourapp/services/gemini_client.py
import os
import time
import re
import json
from typing import List, Dict, Optional

# Attempt imports for different SDK names (install whichever works for you)
try:
    from google import genai        # newer packaging might be "google-genai"
    GenAI_SDK = "google.genai"
except Exception:
    try:
        import google_generativeai as genai  # fallback package names
        GenAI_SDK = "google_generativeai"
    except Exception:
        genai = None
        GenAI_SDK = None

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-mini")  # change to model you have access to
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0        # seconds multiplier
CHUNK_TOKENS_APPROX = 2500 # approximate characters per chunk (tune as needed)

# Initialize client (SDK surfaces differ; adjust if your installed package uses different API)
client = None
if genai is not None:
    try:
        # new genai style
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=API_KEY)
        else:
            # older style (module-level configure)
            genai.configure(api_key=API_KEY)
            client = genai
    except Exception:
        client = None

# --- Helper utilities ---
def _safe_find_json(text: str) -> Optional[Dict]:
    """
    Try to extract and parse the first JSON object in text robustly.
    Returns parsed dict or None.
    """
    # Find first '{' to last '}' region heuristically
    json_candidate = None
    # Use regex to find {...} blocks, pick the largest JSON-like block
    matches = list(re.finditer(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL))
    if not matches:
        # fallback: find first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_candidate = text[start:end+1]
    else:
        # choose the longest match (often the real JSON)
        longest = max(matches, key=lambda m: m.end() - m.start())
        json_candidate = longest.group(0)

    if not json_candidate:
        return None

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        # Try cleaning trailing commas, simple fixes
        cleaned = re.sub(r",\s*}", "}", json_candidate)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

def _call_model(prompt: str, model: str = DEFAULT_MODEL, max_output_tokens: int = 1024, temperature: float = 0.0) -> str:
    """
    Call the Gemini API with retries. Returns text output (string).
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized. Make sure SDK is installed and GEMINI_API_KEY set.")

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # NOTE: SDK method names differ. Try common patterns:
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                # genai.Client style
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
                # Different SDKs put text in different places
                if hasattr(resp, "text"):
                    return resp.text
                if isinstance(resp, dict) and "candidates" in resp:
                    # some responses use candidates
                    return resp["candidates"][0].get("content", "")
                return str(resp)

            elif hasattr(genai, "generate"):
                # alternate older style
                resp = genai.generate(model=model, prompt=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
                # adapt depending on structure
                if isinstance(resp, dict):
                    return resp.get("output", "") or resp.get("text", "")
                return str(resp)

            else:
                # generic attempt
                resp = client.generate(prompt)  # may fail depending on SDK
                return str(resp)
        except Exception as e:
            last_exc = e
            wait = RETRY_BACKOFF ** attempt
            time.sleep(wait)
            continue

    # after retries, raise
    raise RuntimeError(f"Failed to call model after {MAX_RETRIES} attempts. Last error: {last_exc}")

def chunk_text_by_chars(text: str, size: int = CHUNK_TOKENS_APPROX) -> List[str]:
    """
    Simple character-based chunking. Prefer semantic chunking if possible.
    """
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        # try to split at nearest newline before end for nicer chunks
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# --- Prompt templates ---
ANALYSIS_PROMPT = """
You are an expert academic editor. Given the following thesis section, respond ONLY with valid JSON (no extra text) with the keys:
 - "summary": a concise 3-4 sentence summary.
 - "grammar": a short list (bullet or numbered) describing the most important grammar/style issues and suggested corrections.
 - "citations": noticeable citation/reference problems (missing citations, inconsistent formats, ambiguous refs). Provide short examples or locations.
 - "improvements": 6 prioritized points to improve clarity, structure, argument strength, and readability (short bullets).
 - "suggestions": 6 short actionable suggestions the student can apply (e.g., add figure, rephrase, tighten conclusion).

Text:
\"\"\"
{chunk_text}
\"\"\"
"""

COMBINE_PROMPT = """
You are an academic assistant. Given the list of chunk summaries / findings below (JSON objects from earlier), produce a single consolidated JSON object with the keys:
 - "summary" (3-5 sentences) — combine chunk summaries.
 - "grammar" — consolidated list of repeated/important grammar issues (top 8).
 - "citations" — consolidated citation issues.
 - "improvements" — top 6 prioritized improvement points across chunks.
 - "suggestions" — top 6 actionable suggestions across chunks.

Input (one JSON per line):
{chunk_results}
"""

# --- High-level functions ---
def analyze_text_single(text: str) -> Dict[str, str]:
    """
    Send entire text to model (for small texts).
    Returns dict with keys summary, grammar, citations, improvements, suggestions.
    """
    prompt = ANALYSIS_PROMPT.format(chunk_text=text)
    raw = _call_model(prompt, model=DEFAULT_MODEL, max_output_tokens=1600, temperature=0.0)
    parsed = _safe_find_json(raw)
    if not parsed:
        # fallback: put whole text into 'summary' if parsing failed
        return {
            "summary": raw[:1500],
            "grammar": "Could not parse structured grammar output.",
            "citations": "Could not parse structured citation output.",
            "improvements": "",
            "suggestions": "",
        }
    # normalize keys to strings
    return {
        "summary": parsed.get("summary", "").strip(),
        "grammar": parsed.get("grammar", "").strip(),
        "citations": parsed.get("citations", "").strip(),
        "improvements": parsed.get("improvements", "").strip(),
        "suggestions": parsed.get("suggestions", "").strip(),
    }

def analyze_text_chunked(full_text: str) -> Dict[str, str]:
    """
    For long documents: chunk, analyze each chunk, then combine results.
    """
    chunks = chunk_text_by_chars(full_text, size=CHUNK_TOKENS_APPROX)
    chunk_results = []
    for i, c in enumerate(chunks):
        prompt = ANALYSIS_PROMPT.format(chunk_text=c)
        raw = _call_model(prompt, model=DEFAULT_MODEL, max_output_tokens=1200, temperature=0.0)
        parsed = _safe_find_json(raw)
        if parsed is None:
            # if parsing failed, put raw as summary fallback
            parsed = {
                "summary": raw[:800],
                "grammar": "Could not parse grammar.",
                "citations": "",
                "improvements": "",
                "suggestions": "",
            }
        # add chunk index to help combine reasoning if needed
        parsed["_chunk_index"] = i
        chunk_results.append(parsed)

    # Combine chunk results
    chunk_json_lines = "\n".join(json.dumps(r) for r in chunk_results)
    combine_prompt = COMBINE_PROMPT.format(chunk_results=chunk_json_lines)
    raw_combined = _call_model(combine_prompt, model=DEFAULT_MODEL, max_output_tokens=1500, temperature=0.0)
    parsed_combined = _safe_find_json(raw_combined)
    if parsed_combined is None:
        # fallback: stitch summaries
        stitched_summary = " ".join(r.get("summary", "") for r in chunk_results)
        return {"summary": stitched_summary, "grammar": "", "citations": "", "improvements": "", "suggestions": ""}

    return {
        "summary": parsed_combined.get("summary", "").strip(),
        "grammar": parsed_combined.get("grammar", "").strip(),
        "citations": parsed_combined.get("citations", "").strip(),
        "improvements": parsed_combined.get("improvements", "").strip(),
        "suggestions": parsed_combined.get("suggestions", "").strip(),
    }

def analyze_thesis_text(text: str) -> Dict[str, str]:
    """
    High-level entry point. Decides whether to chunk and returns normalized dict.
    """
    if not text or text.strip() == "":
        return {"summary": "", "grammar": "", "citations": "", "improvements": "", "suggestions": ""}

    # if small, do single call
    if len(text) < CHUNK_TOKENS_APPROX * 1.1:
        return analyze_text_single(text)
    else:
        return analyze_text_chunked(text)
