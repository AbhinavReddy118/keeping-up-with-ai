"""
Versioned prompts and templates for LLM article analysis.
"""

ANALYSIS_PROMPT_VERSION = "1.0"

SYSTEM_PROMPT = """You are an AI intelligence analyst extracting structured facts from news metadata.

CRITICAL INSTRUCTIONS:
1. Ground your analysis ONLY in the provided article title, summary, and metadata.
2. DO NOT hallucinate, assume, or invent facts not stated or directly implied.
3. If an entity, company, model, or person is not mentioned, return an empty list: [].
4. If a field cannot be determined with confidence, indicate uncertainty (e.g. novelty="unknown", event_type="unknown").
5. Return ONLY valid, raw JSON with NO markdown formatting, NO backticks, and NO commentary.
6. The JSON must strictly conform to the expected schema.
"""

USER_PROMPT_TEMPLATE = """Analyze the following AI article and produce a structured JSON object.

ARTICLE METADATA:
- Title: {title}
- Source: {source}
- Published Date: {published_at}
- Article Content / Summary: {summary}

EXPECTED JSON SCHEMA:
{{
  "relevance_score": <float between 0.0 and 1.0 indicating relevance to artificial intelligence>,
  "summary": "<concise, 1-3 sentence factual summary of the article's specific development>",
  "key_points": ["<factual point 1>", "<factual point 2>"],
  "topics": ["<broad topic e.g. NLP, Computer Vision, Robotics, AI Safety, Hardware>"],
  "technologies": ["<specific technology, technique, or framework mentioned>"],
  "companies": ["<company or institution mentioned, e.g. OpenAI, Google, Meta>"],
  "people": ["<person mentioned by name>"],
  "models": ["<specific AI model name mentioned, e.g. GPT-4, Claude 3, Llama 3>"],
  "event_type": "<one of: model_release, research_paper, tool_release, business_partnership, regulation_policy, industry_news, opinion_editorial, unknown>",
  "novelty": "<one of: breakthrough, significant, incremental, unknown>",
  "confidence": "<float between 0.0 and 1.0 indicating your confidence in this extraction>"
}}
"""


def build_analysis_prompt(title: str, source: str, published_at: str | None, summary: str | None) -> str:
    """Render the user analysis prompt with sanitized article metadata."""
    clean_title = title.strip() if title else "Untitled"
    clean_source = source.strip() if source else "Unknown Source"
    clean_date = published_at.strip() if published_at else "Unknown Date"
    clean_summary = summary.strip() if summary else "No summary provided in source feed."

    return USER_PROMPT_TEMPLATE.format(
        title=clean_title,
        source=clean_source,
        published_at=clean_date,
        summary=clean_summary,
    )
