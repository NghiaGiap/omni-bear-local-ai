"""
=============================================================
  Gấu Bông AI — Speech Filter Pipeline v2
  Thể hiện rõ các concept LangChain:
    1. PromptTemplate      — quản lý prompt có biến
    2. LCEL Chain (|)      — nối các bước thành pipeline
    3. RunnableParallel    — chạy Model 2 & 3 song song
    4. RunnableLambda      — nhúng logic Python vào chain
    5. .with_fallbacks()   — tự chuyển model dự phòng khi lỗi
    6. ConversationMemory  — ghi nhớ lịch sử hội thoại
=============================================================

Cài đặt:
  pip install langchain langchain-ollama
  ollama pull gemma2:2b
  ollama pull phi3:mini
"""

import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

BASE_URL = "http://localhost:11434"

# ═══════════════════════════════════════════════════════════
#  BƯỚC 0: Khởi tạo các LLM — có fallback chain
#  Concept: .with_fallbacks() — nếu model chính lỗi/timeout
#           → tự động thử model dự phòng, không crash
# ═══════════════════════════════════════════════════════════

primary_llm   = OllamaLLM(model="gemma2:2b",  base_url=BASE_URL, temperature=0)
fallback_llm  = OllamaLLM(model="phi3:mini",  base_url=BASE_URL, temperature=0)

# .with_fallbacks() là tính năng thuần LangChain:
# Nếu primary_llm raise exception → tự gọi fallback_llm
robust_llm = primary_llm.with_fallbacks([fallback_llm])

topic_llm    = OllamaLLM(model="phi3:mini", base_url=BASE_URL, temperature=0)
language_llm = OllamaLLM(model="phi3:mini", base_url=BASE_URL, temperature=0)
decision_llm = OllamaLLM(model="gemma2:2b", base_url=BASE_URL, temperature=0)

# ═══════════════════════════════════════════════════════════
#  BƯỚC 1: PromptTemplate + Chain cho Model 1 (Cleaner)
#  Concept: PromptTemplate — định nghĩa prompt có {biến}
#           LCEL | operator — nối Prompt → LLM → Parser
# ═══════════════════════════════════════════════════════════

# PromptTemplate quản lý template, tự inject biến khi invoke()
cleaner_prompt = PromptTemplate(
    input_variables=["input_text"],  # khai báo biến tường minh
    template="""Chuẩn hóa văn bản sau: sửa lỗi chính tả, viết tắt, ký tự rác.
Chỉ trả về văn bản đã sạch, không giải thích.

Văn bản gốc: {input_text}
Văn bản đã chuẩn hóa:"""
)

# LCEL: dùng | để nối thành pipeline
# Khi chain.invoke({"input_text": "..."}) được gọi:
#   1. cleaner_prompt nhận dict → format thành string prompt
#   2. robust_llm nhận prompt → gọi Ollama → trả raw string
#   3. StrOutputParser() trả về string thuần (strip metadata)
cleaner_chain = cleaner_prompt | robust_llm | StrOutputParser()

# ═══════════════════════════════════════════════════════════
#  BƯỚC 2 & 3: RunnableParallel — chạy 2 model CÙNG LÚC
#  Concept: RunnableParallel — thay vì đợi Model2 xong mới
#           chạy Model3, cả hai chạy song song → giảm latency
# ═══════════════════════════════════════════════════════════

topic_prompt = PromptTemplate(
    input_variables=["cleaned_text"],
    template="""Kiểm tra văn bản có chứa chủ đề nhạy cảm không:
bạo lực, ma túy, nội dung người lớn, tự làm hại bản thân.

Văn bản: {cleaned_text}

Chỉ trả về JSON (không markdown, không giải thích):
{{"sensitive_topic": true/false, "reason": "...", "type": "..."}}"""
)

language_prompt = PromptTemplate(
    input_variables=["cleaned_text"],
    template="""Kiểm tra văn bản có chứa từ thô tục, xúc phạm, bắt nạt không.

Văn bản: {cleaned_text}

Chỉ trả về JSON (không markdown, không giải thích):
{{"offensive_language": true/false, "reason": "...", "words": []}}"""
)

# RunnableParallel nhận 1 input, phân phối cho nhiều chain chạy đồng thời
# Output là dict: {"topic": <kết quả Model2>, "language": <kết quả Model3>}
parallel_checkers = RunnableParallel(
    topic    = topic_prompt    | topic_llm    | StrOutputParser(),
    language = language_prompt | language_llm | StrOutputParser(),
    # RunnablePassthrough: giữ nguyên input để truyền xuống bước sau
    original = RunnablePassthrough(),
)

# ═══════════════════════════════════════════════════════════
#  BƯỚC 3.5: RunnableLambda — nhúng Python logic vào chain
#  Concept: RunnableLambda biến hàm Python thường thành
#           một "Runnable" có thể nối vào chain bằng |
# ═══════════════════════════════════════════════════════════

def parse_checker_results(parallel_output: dict) -> dict:
    """
    RunnableLambda wrapper: parse JSON output từ hai model checker.
    Đây là logic Python thuần — LangChain cho phép nhúng thẳng vào chain.
    """
    def safe_json(text: str, fallback: dict) -> dict:
        try:
            s = text.find("{"); e = text.rfind("}") + 1
            return json.loads(text[s:e]) if s != -1 and e > s else fallback
        except Exception:
            return fallback

    topic = safe_json(
        parallel_output.get("topic", "{}"),
        {"sensitive_topic": False, "reason": "parse error", "type": "none"}
    )
    language = safe_json(
        parallel_output.get("language", "{}"),
        {"offensive_language": False, "reason": "parse error", "words": []}
    )
    return {
        "original":   parallel_output.get("original", {}).get("cleaned_text", ""),
        "topic":      topic,
        "language":   language,
    }

# RunnableLambda: bọc hàm Python → có thể dùng | trong chain
parse_step = RunnableLambda(parse_checker_results)

# ═══════════════════════════════════════════════════════════
#  BƯỚC 4: Decision chain
# ═══════════════════════════════════════════════════════════

decision_prompt = PromptTemplate(
    input_variables=["original", "topic_result", "language_result"],
    template="""Bạn là bộ quyết định an toàn cho gấu bông AI trẻ em.

Văn bản: {original}
Kiểm tra chủ đề: {topic_result}
Kiểm tra từ ngữ: {language_result}

Quy tắc: BLOCK nếu sensitive_topic=true HOẶC offensive_language=true.

Chỉ trả về JSON:
{{"decision": "PASS/BLOCK", "reason": "...", "safe_response": "câu thay thế hoặc null"}}"""
)

def prepare_decision_input(parsed: dict) -> dict:
    """RunnableLambda: chuẩn bị input cho decision prompt."""
    return {
        "original":        parsed["original"],
        "topic_result":    json.dumps(parsed["topic"],    ensure_ascii=False),
        "language_result": json.dumps(parsed["language"], ensure_ascii=False),
    }

decision_chain = (
    RunnableLambda(prepare_decision_input)
    | decision_prompt
    | decision_llm
    | StrOutputParser()
)

# ═══════════════════════════════════════════════════════════
#  BƯỚC 5: Chat History (thay thế ConversationBufferWindowMemory)
#  Concept: Dùng list[HumanMessage | AIMessage] — cách hiện đại
#           LangChain v0.3+ khuyến khích tự quản lý history.
#           MessagesPlaceholder tự inject list này vào prompt.
#  Dùng cho AI chính của gấu bông (sau khi speech đã PASS filter)
# ═══════════════════════════════════════════════════════════

MEMORY_K = 5  # nhớ tối đa 5 lượt hội thoại gần nhất
teddy_history: list = []  # lưu HumanMessage + AIMessage xen kẽ

teddy_prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là gấu bông dễ thương, nói chuyện vui vẻ với trẻ em. Trả lời ngắn gọn, thân thiện."),
    MessagesPlaceholder(variable_name="history"),  # <- inject list history vào đây
    ("human", "{input}"),
])

teddy_chain = teddy_prompt | robust_llm | StrOutputParser()

def teddy_chat(user_input: str) -> str:
    """Chat với gấu bông — có memory, nhớ hội thoại trước."""
    response = teddy_chain.invoke({"input": user_input, "history": teddy_history})
    # Lưu lượt này vào history
    teddy_history.append(HumanMessage(content=user_input))
    teddy_history.append(AIMessage(content=response))
    # Giữ tối đa MEMORY_K lượt (mỗi lượt = 2 messages)
    if len(teddy_history) > MEMORY_K * 2:
        del teddy_history[0:2]
    return response

# ═══════════════════════════════════════════════════════════
#  FULL PIPELINE: kết hợp tất cả
# ═══════════════════════════════════════════════════════════

def run_full_pipeline(user_speech: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  📥 Câu nói: \"{user_speech}\"")
    print(f"{'═'*60}")

    # ── Model 1: Làm sạch ──────────────────────────────────
    print("\n  🔧 [Model 1] Cleaner đang chạy...")
    cleaned = cleaner_chain.invoke({"input_text": user_speech}).strip()
    print(f"     → \"{cleaned}\"")

    # ── Model 2 & 3: Parallel checkers ────────────────────
    print("\n  ⚡ [Model 2 + 3] Chạy SONG SONG (RunnableParallel)...")
    parallel_output = parallel_checkers.invoke({"cleaned_text": cleaned})
    parsed = parse_step.invoke(parallel_output)

    topic_flag = "🚨" if parsed["topic"].get("sensitive_topic") else "✅"
    lang_flag  = "🚨" if parsed["language"].get("offensive_language") else "✅"
    print(f"     → Topic:    {topic_flag} {parsed['topic'].get('reason','')}")
    print(f"     → Language: {lang_flag} {parsed['language'].get('reason','')}")

    # ── Model 4: Quyết định ────────────────────────────────
    print("\n  🧠 [Model 4] Decision Maker đang tổng hợp...")
    parsed["original"] = cleaned
    decision_raw = decision_chain.invoke(parsed)

    try:
        s = decision_raw.find("{"); e = decision_raw.rfind("}") + 1
        decision = json.loads(decision_raw[s:e])
    except Exception:
        decision = {"decision": "BLOCK", "reason": "parse error", "safe_response": "Hãy hỏi câu khác nhé!"}

    print(f"\n  {'─'*56}")
    verdict = decision.get("decision", "BLOCK")
    if verdict == "PASS":
        print(f"  ✅ KẾT QUẢ: PASS")
        print(f"  📝 {decision.get('reason','')}")
        # ── Memory: gấu bông trả lời và nhớ hội thoại ─────
        print(f"\n  🐻 [Memory] Gấu bông đang trả lời (nhớ lịch sử)...")
        teddy_reply = teddy_chat(cleaned)
        print(f"  💬 Gấu bông: \"{teddy_reply}\"")
        print(f"  🧠 Memory hiện tại: {len(teddy_history) // 2} lượt hội thoại được nhớ")
    else:
        print(f"  🚫 KẾT QUẢ: BLOCK")
        print(f"  📝 {decision.get('reason','')}")
        print(f"  💬 Gấu bông nói: \"{decision.get('safe_response','')}\"")
    print(f"  {'─'*56}\n")


# ═══════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║        Gấu Bông AI — LangChain Demo v2                  ║
║  Các concept được dùng:                                  ║
║  ✦ PromptTemplate   — quản lý prompt có {biến}           ║
║  ✦ LCEL |           — nối Prompt | LLM | Parser          ║
║  ✦ RunnableParallel — Model 2 & 3 chạy song song         ║
║  ✦ RunnableLambda   — nhúng Python logic vào chain       ║
║  ✦ .with_fallbacks()— tự chuyển model dự phòng khi lỗi  ║
║  ✦ Memory           — gấu bông nhớ hội thoại             ║
╚══════════════════════════════════════════════════════════╝
""")

    test_cases = [
        "Hôm nay con học bài xong rồi ạ gấu bông ơi!",         # PASS
        "kể chuyện cô bé quàng khăn đỏ đi gấu bông",           # PASS (test memory)
        "hôm qua bn kể chuyện gì vậy?",                        # PASS (test memory recall)
        "tao ghét mày, vcl luôn ý",                            # BLOCK — từ thô tục
        "làm sao đánh nhau mà ko bị phát hiện",                # BLOCK — bạo lực
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n  ── TEST {i}/{len(test_cases)} ──")
        run_full_pipeline(test)
        if i < len(test_cases):
            input("  ↩ Nhấn Enter để tiếp tục...\n")