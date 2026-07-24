import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_language_filter(
    model: str | None = None,
    base_url: str | None = None,
):
    llm = ChatOllama(
        model=model or os.getenv("BEAR_LANGUAGE_MODEL", "phi3:mini"),
        base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        format="json",
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là bộ lọc ngôn ngữ cho trẻ em.
CHỈ đánh dấu unsafe nếu tin nhắn chứa chửi thề, ngôn ngữ thô tục rõ ràng.
Các chủ đề bình thường như động vật, thiên nhiên, màu sắc, học tập = LUÔN safe.
Trả về JSON duy nhất, không giải thích: {{"safe": true}} hoặc {{"safe": false, "reason": "lý do", "words": []}}"""),
        ("human", "{message}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain
