"""
LangChain prompt templates for quiz and related topics generation
"""
from langchain_core.prompts import ChatPromptTemplate

# Prompt for generating quiz questions
QUIZ_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert quiz generator. Your task is to create a structured quiz based on Wikipedia article content.

IMPORTANT RULES:
1. Generate 5-10 questions of varying difficulty levels (easy, medium, hard)
2. Each question must be directly supported by the provided article content
3. Create 4 multiple choice options (A, B, C, D) for each question
4. Clearly identify the correct answer
5. Provide a brief explanation referencing the article
6. Ensure questions cover different topics/sections of the article
7. Do NOT make up or hallucinate facts - only use information from the article
8. Format output as valid JSON
9. Do NOT use markdown code fences or backticks. Output must be raw JSON only (no ```json).
10. Do not include any commentary before or after the JSON.

Return ONLY a valid JSON object with this structure:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "Correct option text",
      "difficulty": "easy|medium|hard",
      "explanation": "Why this answer is correct, referencing the article"
    }}
  ]
}}"""),
    ("human", "Article Title: {title}\n\nArticle Content:\n{content}\n\nGenerate a quiz with 5-10 questions based on ONLY the provided content above.")
])

# Prompt for extracting related topics
RELATED_TOPICS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at identifying key topics from Wikipedia articles.

Your task is to extract 5-8 important related topics/concepts from the provided article content.
These should be:
1. Key entities mentioned in the article
2. Concepts central to understanding the topic
3. Related fields or disciplines
4. Important historical periods or events

Return ONLY a JSON object with this structure. Do NOT use markdown code fences or backticks; output must be raw JSON only:
{{
  "related_topics": ["Topic 1", "Topic 2", "Topic 3", ...]
}}"""),
    ("human", "Article Title: {title}\n\nArticle Content:\n{content}\n\nIdentify 5-8 related topics from this article.")
])

# Prompt for article summary
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at summarizing Wikipedia articles.

Provide a brief 2-3 sentence summary of the article that captures the main subject and key points.

Return ONLY the summary text, no JSON or additional formatting."""),
    ("human", "Article Title: {title}\n\nArticle Content:\n{content}\n\nProvide a brief summary of this article.")
])
