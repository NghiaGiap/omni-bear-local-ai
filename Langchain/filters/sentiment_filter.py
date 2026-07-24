import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def create_sentiment_filter(
    model: str | None = None,
    base_url: str | None = None,
):
    llm = ChatOllama(
        model=model or os.getenv("BEAR_SENTIMENT_MODEL", "phi3:mini"),
        base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        format="json",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Phân tích cảm xúc trong tin nhắn của trẻ em.
Trả về JSON duy nhất, không giải thích:
{{"emotion": "happy|sad|scared|angry|curious|neutral", "needs_comfort": true|false}}"""),
        ("human", "{message}"),
    ])

    return prompt | llm | StrOutputParser()
