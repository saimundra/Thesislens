import os
import time
import re
import json
import logging
from typing import List, Dict, Optional, Union, Any
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiConfig:
    """Configuration for the Gemini client."""
    
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0
    CHUNK_TOKENS_APPROX = 2500
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.default_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")


class GeminiClient:
    """Client for interacting with Google's Gemini AI."""
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.client = None
        self.sdk_name = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Gemini client based on available SDK."""
        try:
            from google import genai
            self.sdk_name = "google.genai"
            if hasattr(genai, "Client"):
                self.client = genai.Client(api_key=self.config.api_key)
            else:
                genai.configure(api_key=self.config.api_key)
                self.client = genai
        except ImportError:
            try:
                import google_generativeai as genai
                self.sdk_name = "google_generativeai"
                genai.configure(api_key=self.config.api_key)
                self.client = genai
            except ImportError:
                logger.warning("Google Generative AI SDK not found. Please install it to use this tool.")
            except Exception as e:
                logger.error(f"Error initializing Google Generative AI: {e}")
        except Exception as e:
            logger.error(f"Error initializing Google GenAI: {e}")
        
        logger.info(f"Gemini client initialized: {self.client is not None}, SDK: {self.sdk_name}")
    
    def list_available_models(self) -> List:
        """List available Gemini models."""
        if self.client is None or not hasattr(self.client, "models") or not hasattr(self.client.models, "list"):
            logger.warning("Gemini client not initialized or models.list not available.")
            return []
        
        try:
            models = self.client.models.list()
            logger.info("Available Gemini models:")
            for m in models:
                logger.info(f"- {getattr(m, 'name', str(m))}")
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def call_model(self, prompt: str, model: str = None, max_output_tokens: int = 1200) -> str:
        """Call the Gemini model with the given prompt."""
        if self.client is None:
            raise RuntimeError("Gemini client not initialized. Make sure SDK is installed and GEMINI_API_KEY set.")
        
        model = model or self.config.default_model
        last_exc = None
        
        for attempt in range(1, self.config.MAX_RETRIES + 1):
            try:
                if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
                    # Try calling with max_output_tokens if supported, else fallback
                    try:
                        resp = self.client.models.generate_content(
                            model=model,
                            contents=prompt,
                            max_output_tokens=max_output_tokens
                        )
                    except TypeError:
                        resp = self.client.models.generate_content(
                            model=model,
                            contents=prompt
                        )
                    
                    if hasattr(resp, "text"):
                        return resp.text
                    if isinstance(resp, dict) and "candidates" in resp:
                        return resp["candidates"][0].get("content", "")
                    return str(resp)
                elif hasattr(self.client, "generate"):
                    resp = self.client.generate(model=model, prompt=prompt, max_output_tokens=max_output_tokens)
                    if isinstance(resp, dict):
                        return resp.get("output", "") or resp.get("text", "")
                    return str(resp)
                else:
                    resp = self.client.generate(prompt)
                    return str(resp)
            except Exception as e:
                last_exc = e
                wait = self.config.RETRY_BACKOFF ** attempt
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait} seconds...")
                time.sleep(wait)
        
        raise RuntimeError(f"Failed to call model after {self.config.MAX_RETRIES} attempts. Last error: {last_exc}")


class TextProcessor:
    """Utility class for text processing operations."""
    
    @staticmethod
    def chunk_text_by_chars(text: str, size: int = GeminiConfig.CHUNK_TOKENS_APPROX) -> List[str]:
        """Split text into chunks of approximately the specified size."""
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
                # Try to break at a newline to avoid cutting sentences
                nl = text.rfind("\n", start, end)
                if nl > start:
                    end = nl
            chunks.append(text[start:end].strip())
            start = end
        
        return chunks
    
    @staticmethod
    def safe_find_json(text: str) -> Optional[Dict]:
        """Safely extract and parse JSON from text."""
        if not text:
            return None
            
        # Try to find JSON in the text
        start = text.find('{')
        end = text.rfind('}')
        
        if start == -1 or end == -1 or end <= start:
            return None
        
        json_candidate = text[start:end+1]
        
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            # Try to clean up common JSON issues
            cleaned = re.sub(r",\s*}", "}", json_candidate)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from model response")
                return None
    
    @staticmethod
    def ensure_string(value: Any) -> str:
        """Ensure the value is a string, converting if necessary."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        return str(value)


class TextAnalyzer:
    """Main class for analyzing text using Gemini AI."""
    
    ANALYSIS_PROMPT = """
    You are an expert academic editor. Given the following thesis section, respond ONLY with valid JSON (no extra text) with the keys:
     - "summary": a concise 3-4 sentence summary.
     - "grammar": a short list describing grammar/style issues .
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
    
    def __init__(self, config: GeminiConfig = None):
        self.config = config or GeminiConfig()
        self.client = GeminiClient(self.config)
        self.processor = TextProcessor()
    
    def analyze_text_single(self, text: str) -> Dict[str, str]:
        """Analyze text using a single API call."""
        prompt = self.ANALYSIS_PROMPT.format(chunk_text=text)
        raw = self.client.call_model(prompt, model=self.config.default_model, max_output_tokens=1600)
        parsed = self.processor.safe_find_json(raw)
        
        if not parsed:
            logger.warning("Failed to parse JSON from model response, using raw text for summary")
            return {"summary": raw[:1500], "grammar": "", "citations": "", "improvements": "", "suggestions": ""}
        
        return {
            "summary": self.processor.ensure_string(parsed.get("summary")).strip(),
            "grammar": self.processor.ensure_string(parsed.get("grammar")).strip(),
            "citations": self.processor.ensure_string(parsed.get("citations")).strip(),
            "improvements": self.processor.ensure_string(parsed.get("improvements")).strip(),
            "suggestions": self.processor.ensure_string(parsed.get("suggestions")).strip(),
        }
    
    def analyze_text_chunked(self, full_text: str) -> Dict[str, str]:
        """Analyze text by breaking it into chunks and then combining the results."""
        chunks = self.processor.chunk_text_by_chars(full_text, size=self.config.CHUNK_TOKENS_APPROX)
        chunk_results = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            prompt = self.ANALYSIS_PROMPT.format(chunk_text=chunk)
            raw = self.client.call_model(prompt, model=self.config.default_model, max_output_tokens=1200)
            parsed = self.processor.safe_find_json(raw) or {
                "summary": raw[:800], 
                "grammar": "", 
                "citations": "", 
                "improvements": "", 
                "suggestions": ""
            }
            parsed["_chunk_index"] = i
            chunk_results.append(parsed)
        
        chunk_json_lines = "\n".join(json.dumps(r) for r in chunk_results)
        raw_combined = self.client.call_model(
            self.COMBINE_PROMPT.format(chunk_results=chunk_json_lines), 
            model=self.config.default_model, 
            max_output_tokens=1500
        )
        parsed_combined = self.processor.safe_find_json(raw_combined)
        
        if parsed_combined is None:
            logger.warning("Failed to parse combined JSON, stitching summaries together")
            stitched_summary = " ".join(r.get("summary", "") for r in chunk_results)
            return {"summary": stitched_summary, "grammar": "", "citations": "", "improvements": "", "suggestions": ""}
        
        try:
            return {
                "summary": self.processor.ensure_string(parsed_combined.get("summary")).strip(),
                "grammar": self.processor.ensure_string(parsed_combined.get("grammar")).strip(),
                "citations": self.processor.ensure_string(parsed_combined.get("citations")).strip(),
                "improvements": self.processor.ensure_string(parsed_combined.get("improvements")).strip(),
                "suggestions": self.processor.ensure_string(parsed_combined.get("suggestions")).strip(),
            }
        except Exception as e:
            logger.error(f"Error processing combined results: {e}")
            logger.error(f"Combined result type: {type(parsed_combined)}")
            logger.error(f"Combined result: {parsed_combined}")
            # Fallback to stitched summaries
            stitched_summary = " ".join(r.get("summary", "") for r in chunk_results)
            return {"summary": stitched_summary, "grammar": "", "citations": "", "improvements": "", "suggestions": ""}
    
    def analyze_thesis_text(self, text: str) -> Dict[str, str]:
        """Analyze thesis text, either as a whole or in chunks depending on length."""
        if not isinstance(text, str):
            logger.error(f"Input to analyze_thesis_text is not a string: {type(text)}")
            return {"summary": "Input is not a string.", "grammar": "", "citations": "", "improvements": "", "suggestions": ""}
        
        if not text.strip():
            logger.warning("Text is empty")
            return {"summary": "", "grammar": "", "citations": "", "improvements": "", "suggestions": ""}

        logger.info(f"Text length = {len(text)}")
        
        try:
            if len(text) < self.config.CHUNK_TOKENS_APPROX * 1.1:
                logger.info("Using analyze_text_single")
                result = self.analyze_text_single(text)
            else:
                logger.info("Using analyze_text_chunked")
                result = self.analyze_text_chunked(text)
            
            # Log the parsed results
            logger.info("Analysis result keys and lengths")
            for k, v in result.items():
                logger.info(f"  {k}: {len(v) if isinstance(v, str) else v}")

            return result
        except Exception as e:
            logger.error(f"Error in analyze_thesis_text: {e}")
            return {"summary": f"Error analyzing text: {str(e)}", "grammar": "", "citations": "", "improvements": "", "suggestions": ""}


# Convenience function for backward compatibility
def analyze_thesis_text(text: str) -> Dict[str, str]:
    """Analyze thesis text using the TextAnalyzer class."""
    analyzer = TextAnalyzer()
    return analyzer.analyze_thesis_text(text)