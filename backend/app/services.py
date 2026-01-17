"""
Services for quiz generation using LangChain and LLM
"""
import json
import logging
from typing import Dict, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from app.prompts import QUIZ_GENERATION_PROMPT, RELATED_TOPICS_PROMPT, SUMMARY_PROMPT
from app.config import settings

logger = logging.getLogger(__name__)


class QuizGenerationService:
    """Service for generating quizzes using LLM"""
    
    def __init__(self):
        """Initialize the LLM"""
        # Allow model to be configured via environment (useful if quota prevents a model)
        model = getattr(settings, "LLM_MODEL", "gemini-2.0-flash")
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.7,
            max_tokens=2000
        )
    
    def generate_quiz(self, title: str, content: str) -> Dict:
        """
        Generate a quiz from article content using LLM
        
        Args:
            title: Article title
            content: Article content text
            
        Returns:
            Dictionary with quiz questions
        """
        try:
            # Generate quiz questions
            prompt_value = QUIZ_GENERATION_PROMPT.invoke(
                {"title": title, "content": content[:8000]}  # Limit content to avoid token limits
            )
            
            response = self.llm.invoke(prompt_value)
            response_text = response.content.strip()
            
            # Parse JSON response
            # Try to extract JSON from response
            quiz_data = self._parse_json_response(response_text)
            
            if not quiz_data or "questions" not in quiz_data:
                logger.error(f"Invalid quiz response format. Response length: {len(response_text)}")
                logger.error(f"Response preview (first 500 chars): {response_text[:500]}")
                logger.error(f"Parsed quiz_data: {quiz_data}")
                raise ValueError("Failed to generate valid quiz format")
            
            return quiz_data
            
        except Exception as e:
            # Detect quota / ResourceExhausted style errors and propagate a clear message
            msg = str(e)
            low = msg.lower()
            if "quota" in low or "resourceexhausted" in low or "exceeded" in low:
                logger.error(f"LLM quota error: {e}")
                raise RuntimeError("LLM_QUOTA_EXCEEDED: Gemini quota exhausted or model unavailable. Check GEMINI_API_KEY, billing, or switch LLM_MODEL in config.")
            logger.error(f"Error generating quiz: {e}")
            raise
    
    def generate_related_topics(self, title: str, content: str) -> List[str]:
        """
        Extract related topics from article content
        
        Args:
            title: Article title
            content: Article content text
            
        Returns:
            List of related topics
        """
        try:
            prompt_value = RELATED_TOPICS_PROMPT.invoke(
                {"title": title, "content": content[:8000]}
            )
            
            response = self.llm.invoke(prompt_value)
            response_text = response.content.strip()
            
            # Parse JSON response
            topics_data = self._parse_json_response(response_text)
            
            if topics_data and "related_topics" in topics_data:
                return topics_data["related_topics"]
            
            return []
            
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "resourceexhausted" in msg or "exceeded" in msg:
                logger.error(f"LLM quota error while generating topics: {e}")
                # propagate to caller to surface a 503
                raise RuntimeError("LLM_QUOTA_EXCEEDED: Gemini quota exhausted or model unavailable.")
            logger.error(f"Error generating related topics: {e}")
            return []
    
    def generate_summary(self, title: str, content: str) -> str:
        """
        Generate a brief summary of the article
        
        Args:
            title: Article title
            content: Article content text
            
        Returns:
            Summary text
        """
        try:
            prompt_value = SUMMARY_PROMPT.invoke(
                {"title": title, "content": content[:4000]}
            )
            
            response = self.llm.invoke(prompt_value)
            return response.content.strip()
            
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "resourceexhausted" in msg or "exceeded" in msg:
                logger.error(f"LLM quota error while generating summary: {e}")
                raise RuntimeError("LLM_QUOTA_EXCEEDED: Gemini quota exhausted or model unavailable.")
            logger.error(f"Error generating summary: {e}")
            return f"Article about {title}"
    
    @staticmethod
    def _parse_json_response(response_text: str) -> Dict:
        """
        Parse JSON from LLM response
        
        Tries to extract JSON from various formats that LLM might return
        """
        import re

        text = response_text.strip()

        # 1) Remove surrounding markdown fences/backticks if present
        if text.startswith("```"):
            m = re.match(r"^```[a-zA-Z0-9_-]*\n(.*)\n```$", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
            else:
                text = text.strip('`').strip()

        # 2) Fix escaped square brackets often returned by some models (e.g., \[ and \])
        text_fixed = text.replace('\\[', '[').replace('\\]', ']')

        # 3) Escape raw newlines within JSON string literals
        def escape_newlines_in_strings(s: str) -> str:
            out = []
            in_str = False
            esc = False
            for ch in s:
                if in_str:
                    if esc:
                        out.append(ch)
                        esc = False
                    else:
                        if ch == '\\':
                            out.append(ch)
                            esc = True
                        elif ch == '"':
                            out.append(ch)
                            in_str = False
                        elif ch == '\r':
                            # drop carriage return; handle when next is \n
                            continue
                        elif ch == '\n':
                            out.append('\\n')
                        else:
                            out.append(ch)
                else:
                    if ch == '"':
                        out.append(ch)
                        in_str = True
                    else:
                        out.append(ch)
            return ''.join(out)

        text_fixed2 = escape_newlines_in_strings(text_fixed)

        # 4) Try direct JSON parsing with increasingly fixed candidates
        for candidate in (text_fixed2, text_fixed, text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 5) Remove trailing commas globally and try again
        no_trailing = re.sub(r',\s*([}\]])', r'\1', text_fixed2)
        try:
            return json.loads(no_trailing)
        except json.JSONDecodeError:
            pass

        # 6) Extract first top-level JSON object or array substring and parse
        def extract_json_blob(s: str) -> str | None:
            start_candidates = [i for i in [s.find('{'), s.find('[')] if i != -1]
            if not start_candidates:
                return None
            start = min(start_candidates)
            stack = []
            i = start
            while i < len(s):
                ch = s[i]
                if ch in '{[':
                    stack.append(ch)
                elif ch in '}]':
                    if not stack:
                        return None
                    open_ch = stack.pop()
                    if (open_ch == '{' and ch != '}') or (open_ch == '[' and ch != ']'):
                        pass
                    if not stack:
                        return s[start:i+1]
                elif ch == '"':
                    # skip strings to avoid counting braces inside strings
                    j = i + 1
                    escaped = False
                    while j < len(s):
                        c2 = s[j]
                        if escaped:
                            escaped = False
                        else:
                            if c2 == '\\':
                                escaped = True
                            elif c2 == '"':
                                break
                        j += 1
                    i = j
                i += 1
            return None

        blob = extract_json_blob(no_trailing) or extract_json_blob(text_fixed)
        if blob:
            candidates = [
                blob,
                blob.replace('\\[', '[').replace('\\]', ']'),
                re.sub(r',\s*([}\]])', r'\1', blob),
            ]
            for cand in candidates:
                try:
                    return json.loads(cand)
                except json.JSONDecodeError:
                    continue

        logger.error(f"Could not parse JSON from response. Length: {len(response_text)}")
        logger.error(f"Response preview (first 500 chars): {response_text[:500]}")
        return {}
