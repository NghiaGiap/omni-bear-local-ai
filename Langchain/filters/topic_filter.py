import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_topic_filter(
    model: str | None = None,
    base_url: str | None = None,
):
    llm = ChatOllama(
        model=model or os.getenv("BEAR_TOPIC_MODEL", "phi3:mini"),
        base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        format="json",
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là bộ lọc chủ đề cho trẻ em.
CHỈ đánh dấu unsafe nếu tin nhắn hỏi RÕ RÀNG về: vũ khí, bạo lực thực sự, nội dung người lớn 18+, ma túy.
Các chủ đề này LUÔN safe: động vật, thiên nhiên, cầu vồng, màu sắc, em bé, gia đình, học tập, khoa học, ăn uống, trò chơi.
Trả về JSON duy nhất, không giải thích: {{"safe": true}} hoặc {{"safe": false, "category": "lý do"}}"""),
        ("human", "{message}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain
