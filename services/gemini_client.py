import os
import time
import re
import json
from typing import List, Dict, Optional

try:
    from google import genai
    GenAI_SDK = "google.genai"
except Exception:
    try:
        import google_generativeai as genai
        GenAI_SDK = "google_generativeai"
    except Exception:
        genai = None
        GenAI_SDK = None

API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
def list_available_models():
    if client is None or not hasattr(client, "models") or not hasattr(client.models, "list"):
        print("Gemini client not initialized or models.list not available.")
        return []
    try:
        models = client.models.list()
        print("Available Gemini models:")
        for m in models:
            print("-", getattr(m, "name", str(m)))
        return models
    except Exception as e:
        print("Error listing models:", e)
        return []
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
CHUNK_TOKENS_APPROX = 2500

client = None
if genai is not None:
    try:
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=API_KEY)
        else:
            genai.configure(api_key=API_KEY)
            client = genai
    except Exception:
        client = None

print("DEBUG: Gemini client initialized:", client is not None)

def _safe_find_json(text: str) -> Optional[Dict]:
    json_candidate = None
    # Python's re does not support (?R) recursion, so use a simple approach
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_candidate = text[start:end+1]
    if not json_candidate:
        return None
    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*}", "}", json_candidate)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

def _call_model(prompt: str, model: str = DEFAULT_MODEL, max_output_tokens: int = 1200) -> str:
    if client is None:
        raise RuntimeError("Gemini client not initialized. Make sure SDK is installed and GEMINI_API_KEY set.")

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                # Try calling with max_output_tokens if supported, else fallback
                try:
                
                    resp = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        max_output_tokens=max_output_tokens
                    )
                except TypeError:
                    resp = client.models.generate_content(
                        model=model,
                        contents=prompt
                    )
                except TypeError:
                    resp = client.models.generate_content(
                        model=model,
                        contents=prompt
                    )
                if hasattr(resp, "text"):
                    return resp.text
                if isinstance(resp, dict) and "candidates" in resp:
                    return resp["candidates"][0].get("content", "")
                return str(resp)
            elif hasattr(genai, "generate"):
                resp = genai.generate(model=model, prompt=prompt, max_output_tokens=max_output_tokens)
                if isinstance(resp, dict):
                    return resp.get("output", "") or resp.get("text", "")
                return str(resp)
            else:
                resp = client.generate(prompt)
                return str(resp)
        except Exception as e:
            last_exc = e
            wait = RETRY_BACKOFF ** attempt
            time.sleep(wait)
            continue
    raise RuntimeError(f"Failed to call model after {MAX_RETRIES} attempts. Last error: {last_exc}")

def chunk_text_by_chars(text: str, size: int = CHUNK_TOKENS_APPROX) -> List[str]:
    if not isinstance(text, str):
        raise TypeError(f"chunk_text_by_chars expects a string, got {type(text)}")
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl
        chunks.append(text[start:end].strip())
        start = end
    return chunks

ANALYSIS_PROMPT = """
You are an expert academic editor. Given the following thesis section, respond ONLY with valid JSON (no extra text) with the keys:
 - "summary": a concise 3-4 sentence summary.
 - "grammar": a short list describing grammar/style issues.
 - "citations": noticeable citation/reference problems.
 - "improvements": 6 prioritized points to improve clarity, structure, argument strength, readability.
 - "suggestions": 6 actionable suggestions.

Text:
\"\"\"
{chunk_text}
\"\"\"
"""

COMBINE_PROMPT = """
You are an academic assistant. Given the list of chunk summaries / findings below, produce a single consolidated JSON object with the keys:
 - "summary"
 - "grammar"
 - "citations"
 - "improvements"
 - "suggestions"

Input (one JSON per line):
{chunk_results}
"""

def analyze_text_single(text: str) -> Dict[str, str]:
    prompt = ANALYSIS_PROMPT.format(chunk_text=text)
    raw = _call_model(prompt, model=DEFAULT_MODEL, max_output_tokens=1600)
    parsed = _safe_find_json(raw)
    if not parsed:
        return {"summary": raw[:1500], "grammar": "", "citations": "", "improvements": "", "suggestions": ""}
    return {
        "summary": parsed.get("summary", "").strip(),
        "grammar": parsed.get("grammar", "").strip(),
        "citations": parsed.get("citations", "").strip(),
        "improvements": parsed.get("improvements", "").strip(),
        "suggestions": parsed.get("suggestions", "").strip(),
    }

def analyze_text_chunked(full_text: str) -> Dict[str, str]:
    chunks = chunk_text_by_chars(full_text, size=CHUNK_TOKENS_APPROX)
    chunk_results = []
    for i, c in enumerate(chunks):
        prompt = ANALYSIS_PROMPT.format(chunk_text=c)
        raw = _call_model(prompt, model=DEFAULT_MODEL, max_output_tokens=1200)
        parsed = _safe_find_json(raw) or {"summary": raw[:800], "grammar": "", "citations": "", "improvements": "", "suggestions": ""}
        parsed["_chunk_index"] = i
        chunk_results.append(parsed)
    chunk_json_lines = "\n".join(json.dumps(r) for r in chunk_results)
    raw_combined = _call_model(COMBINE_PROMPT.format(chunk_results=chunk_json_lines), model=DEFAULT_MODEL, max_output_tokens=1500)
    parsed_combined = _safe_find_json(raw_combined)
    if parsed_combined is None:
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
    if not isinstance(text, str):
        print(f"DEBUG: Input to analyze_thesis_text is not a string: {type(text)}")
        return {"summary": "Input is not a string.", "grammar": "", "citations": "", "improvements": "", "suggestions": ""}
    if not text.strip():
        print("DEBUG: Text is empty")
        return {"summary": "", "grammar": "", "citations": "", "improvements": "", "suggestions": ""}

    print(f"DEBUG: Text length = {len(text)}")
    
    if len(text) < CHUNK_TOKENS_APPROX * 1.1:
        result = analyze_text_single(text)
        print("DEBUG: Using analyze_text_single")
    else:
        result = analyze_text_chunked(text)
        print("DEBUG: Using analyze_text_chunked")
    
    # Debug the parsed results
    print("DEBUG: Analysis result keys and lengths")
    for k, v in result.items():
        print(f"  {k}: {len(v) if isinstance(v, str) else v}")

    return result

