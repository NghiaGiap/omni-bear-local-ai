import asyncio
import json
import os
import re
import sys
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from omnibear_config import OmniBearConfigProvider, format_omnibear_context


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _fold_vietnamese(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return without_marks.replace("đ", "d")


@dataclass(frozen=True)
class BearAIConfig:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    answer_model: str = os.getenv("BEAR_MODEL", "gemma2:2b")
    fallback_model: str = os.getenv("BEAR_FALLBACK_MODEL", "phi3:mini")
    cleaner_model: str = os.getenv("BEAR_CLEANER_MODEL", "gemma2:2b")
    topic_model: str = os.getenv("BEAR_TOPIC_MODEL", "phi3:mini")
    language_model: str = os.getenv("BEAR_LANGUAGE_MODEL", "phi3:mini")
    sentiment_model: str = os.getenv("BEAR_SENTIMENT_MODEL", "phi3:mini")
    memory_turns: int = _env_int("BEAR_MEMORY_TURNS", 5)
    num_ctx: int = _env_int("BEAR_NUM_CTX", 1024)
    num_predict: int = _env_int("BEAR_NUM_PREDICT", 96)
    use_llm_cleaner: bool = _env_bool("BEAR_USE_LLM_CLEANER", False)
    use_llm_filters: bool = _env_bool("BEAR_USE_LLM_FILTERS", False)
    use_llm_sentiment: bool = _env_bool("BEAR_USE_LLM_SENTIMENT", False)
    use_omnibear_config: bool = _env_bool("BEAR_USE_OMNIBEAR_CONFIG", True)
    verbose: bool = _env_bool("BEAR_VERBOSE", False)


class BearAIPipeline:
    """Local multi-model LangChain pipeline for the teddy bear assistant."""

    OFFENSIVE_PATTERNS = (
        r"\bvcl\b",
        r"\bvl\b",
        r"\bdm\b",
        r"\bdcm\b",
        r"địt",
        r"đụ",
        r"đéo",
        r"lồn",
        r"cặc",
    )
    OFFENSIVE_FOLDED_PATTERNS = (
        r"\bvcl\b",
        r"\bvl\b",
        r"\bdm\b",
        r"\bdcm\b",
        r"\bdit\b",
        r"\bdu\b",
        r"\bdeo\b",
        r"\bcac\b",
    )
    TOPIC_GUARDS = (
        ("self_harm", (
            r"tự\s+tử",
            r"tự\s+hại",
            r"muốn\s+chết",
            r"không\s+muốn\s+sống",
        )),
        ("dangerous", (
            r"chế\s+bom",
            r"làm\s+bom",
            r"chế\s+súng",
            r"làm\s+súng",
            r"giết\s+người",
        )),
        ("drugs", (
            r"ma\s+túy",
            r"cần\s+sa",
            r"thuốc\s+lắc",
        )),
        ("adult", (
            r"nội\s+dung\s+18\+?",
            r"\bsex\b",
            r"khiêu\s+dâm",
        )),
    )
    TOPIC_FOLDED_GUARDS = (
        ("self_harm", (
            r"tu\s+tu",
            r"tu\s+hai",
            r"muon\s+chet",
            r"khong\s+muon\s+song",
        )),
        ("dangerous", (
            r"che\s+bom",
            r"lam\s+bom",
            r"che\s+sung",
            r"lam\s+sung",
            r"giet\s+nguoi",
            r"dam\s+nguoi",
            r"danh\s+nhau",
        )),
        ("drugs", (
            r"ma\s+tuy",
            r"can\s+sa",
            r"thuoc\s+lac",
        )),
        ("adult", (
            r"noi\s+dung\s+18\+?",
            r"\bsex\b",
            r"khieu\s+dam",
        )),
    )

    def __init__(self, config: BearAIConfig | None = None):
        self.config = config or BearAIConfig()
        self.history: deque[HumanMessage | AIMessage] = deque(
            maxlen=max(1, self.config.memory_turns) * 2
        )
        self.omnibear_config = (
            OmniBearConfigProvider() if self.config.use_omnibear_config else None
        )

        self.answer_llm = self._chat_llm(
            self.config.answer_model,
            temperature=0.65,
        ).with_fallbacks([
            self._chat_llm(self.config.fallback_model, temperature=0.65)
        ])
        self.cleaner_llm = self._chat_llm(
            self.config.cleaner_model,
            temperature=0,
        ).with_fallbacks([
            self._chat_llm(self.config.fallback_model, temperature=0)
        ])
        self.topic_llm = self._chat_llm(
            self.config.topic_model,
            temperature=0,
            json_mode=True,
        )
        self.language_llm = self._chat_llm(
            self.config.language_model,
            temperature=0,
            json_mode=True,
        )
        self.sentiment_llm = self._chat_llm(
            self.config.sentiment_model,
            temperature=0,
            json_mode=True,
        )

        self.cleaner_chain = self._build_cleaner_chain()
        self.topic_chain = self._build_topic_chain()
        self.language_chain = self._build_language_chain()
        self.sentiment_chain = self._build_sentiment_chain()
        self.answer_chain = self._build_answer_chain()
        self.guidance_chain = self._build_guidance_chain()

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message)

    def _chat_llm(
        self,
        model: str,
        *,
        temperature: float,
        json_mode: bool = False,
    ) -> ChatOllama:
        return ChatOllama(
            model=model,
            base_url=self.config.ollama_base_url,
            temperature=temperature,
            num_ctx=self.config.num_ctx,
            num_predict=self.config.num_predict,
            keep_alive="10m",
            format="json" if json_mode else None,
        )

    def _build_cleaner_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn chuẩn hóa lời nói từ STT của trẻ em.
Nhiệm vụ:
- Sửa lỗi chính tả nhẹ, viết tắt phổ biến, ký tự rác.
- Giữ nguyên ý của trẻ, không tự trả lời câu hỏi.
- Nếu câu đã rõ thì trả lại gần như nguyên văn.
- Chỉ trả về một câu văn bản sạch, không giải thích."""),
            ("human", "{message}"),
        ])
        return prompt | self.cleaner_llm | StrOutputParser()

    def _build_topic_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là bộ lọc chủ đề cho gấu bông AI dành cho trẻ em.
Chỉ đánh dấu unsafe nếu tin nhắn hỏi rõ ràng về vũ khí, bạo lực thực tế, nội dung người lớn, ma túy, tự làm hại bản thân, hoặc cách làm điều nguy hiểm.
Các chủ đề như động vật, học tập, gia đình, đồ chơi, cảm xúc, khoa học đơn giản, truyện kể, ăn uống là safe.
Trả về JSON hợp lệ duy nhất theo schema:
{{"safe": true|false, "category": "none|violence|adult|drugs|self_harm|dangerous", "reason": "ngắn gọn"}}"""),
            ("human", "{message}"),
        ])
        return prompt | self.topic_llm | StrOutputParser()

    def _build_language_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là bộ lọc ngôn ngữ cho trẻ em.
Chỉ đánh dấu unsafe nếu tin nhắn có chửi thề, xúc phạm, bắt nạt, hoặc lời nói thô tục rõ ràng.
Không đánh dấu unsafe với câu hỏi bình thường hoặc cảm xúc tiêu cực không xúc phạm ai.
Trả về JSON hợp lệ duy nhất theo schema:
{{"safe": true|false, "reason": "ngắn gọn", "words": []}}"""),
            ("human", "{message}"),
        ])
        return prompt | self.language_llm | StrOutputParser()

    def _build_sentiment_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Phân tích cảm xúc trong tin nhắn của trẻ.
Trả về JSON hợp lệ duy nhất theo schema:
{{"emotion": "happy|sad|scared|angry|curious|neutral", "needs_comfort": true|false}}"""),
            ("human", "{message}"),
        ])
        return prompt | self.sentiment_llm | StrOutputParser()

    def _build_answer_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tớ là Gấu Bông, người bạn AI thân thiện và thông minh của trẻ em.

Cách nói chuyện:
- Luôn xưng "tớ" và gọi người dùng là "cậu".
- Nếu câu của người dùng có chữ "tớ" hoặc "cậu", vẫn giữ đúng vai: Gấu Bông là "tớ", người dùng là "cậu".
- Không tự xưng là "mình", "cậu", "bạn", hoặc "Gấu Bông"; chỉ tự xưng là "tớ".
- Luôn trả lời bằng tiếng Việt, vui vẻ, ấm áp, dùng từ đơn giản.
- Trả lời ngắn gọn, thường 2-4 câu, phù hợp để đọc text-to-speech.
- Không dùng markdown, không liệt kê dài.
- Nếu cậu buồn, sợ, hoặc giận thì an ủi trước rồi gợi ý một bước nhỏ an toàn.

Quy tắc context:
- Chỉ dùng lịch sử trong phần history bên dưới.
- Nếu cậu hỏi "lúc nãy", "hôm qua", "vừa rồi" mà history không có thông tin, hãy nói nhẹ nhàng rằng tớ chưa nhớ chắc.
- Không bịa thông tin cá nhân, sở thích, sự kiện, hoặc lời hứa nếu chưa có trong history.
- Không nhắc đến kết quả filter hoặc hệ thống nội bộ."""),
            ("system", """Cau hinh OmniBear DB cho luot hien tai:
{runtime_config}

Neu khong co config active thi dung prompt mac dinh.
Khong tiet lo raw DB, table name, API key, access token, service role key, hay config noi bo."""),
            ("system", """Quy tac xung ho bat buoc, uu tien cao hon vi du trong OmniBear DB config:
- Gấu Bông luôn tự xưng là "tớ".
- Luôn gọi người dùng là "cậu".
- Không gọi người dùng là "con", "bé", "cháu", "bạn", hoặc "em".
- Nếu biết tên người dùng, chỉ có thể gọi thêm tên theo kiểu "<tên> ơi"; sau đó vẫn giữ xưng hô "tớ/cậu".
- Nếu người dùng tự xưng là "con", không bắt chước cách xưng đó; hãy đổi về "cậu" trong câu trả lời."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Cảm xúc dự đoán của cậu: {emotion}. Cậu nói: {message}"),
        ])
        return prompt | self.answer_llm | StrOutputParser()

    def _build_guidance_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tớ là Gấu Bông, người bạn thân thiện của trẻ em.
Cậu vừa dùng từ ngữ không phù hợp. Hãy khuyên nhẹ nhàng:
- Xưng "tớ", gọi người dùng là "cậu".
- Không la mắng, không phán xét.
- Giải thích ngắn tại sao từ đó không hay.
- Gợi ý một cách nói lịch sự hơn.
- Trả lời 2-3 câu, phù hợp để đọc text-to-speech."""),
            ("human", "Cậu vừa nói: {message}"),
        ])
        return prompt | self.answer_llm | StrOutputParser()

    def _parse_json(self, text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end <= start:
                return fallback
            parsed = json.loads(text[start:end])
            return parsed if isinstance(parsed, dict) else fallback
        except (TypeError, ValueError, json.JSONDecodeError):
                return fallback

    def _normalize_text(self, message: str) -> str:
        cleaned = re.sub(r"\s+", " ", message).strip()
        replacements = {
            " bn ": " cậu ",
            " ko ": " không ",
            " k ": " không ",
            " hok ": " không ",
        }
        padded = f" {cleaned} "
        lowered = padded.lower()
        for source, target in replacements.items():
            lowered = lowered.replace(source, target)
        return lowered.strip() if cleaned.islower() else cleaned

    def _regex_hit(
        self,
        message: str,
        patterns: tuple[str, ...],
        *,
        folded: bool = False,
    ) -> str | None:
        lowered = _fold_vietnamese(message) if folded else message.lower()
        for pattern in patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                return pattern
        return None

    def _topic_hit(self, message: str) -> tuple[str, str] | None:
        for category, patterns in self.TOPIC_GUARDS:
            hit = self._regex_hit(message, patterns)
            if hit:
                return category, hit

        for category, patterns in self.TOPIC_FOLDED_GUARDS:
            hit = self._regex_hit(message, patterns, folded=True)
            if hit:
                return category, hit

        return None

    def _rule_based_safety(self, message: str) -> tuple[dict[str, Any], dict[str, Any]]:
        offensive_hit = self._regex_hit(message, self.OFFENSIVE_PATTERNS)
        if not offensive_hit:
            offensive_hit = self._regex_hit(
                message,
                self.OFFENSIVE_FOLDED_PATTERNS,
                folded=True,
            )

        if offensive_hit:
            language = {
                "safe": False,
                "reason": "keyword_guard",
                "words": [offensive_hit],
            }
        else:
            language = {
                "safe": True,
                "reason": "no_keyword_guard_hit",
                "words": [],
            }

        topic_hit = self._topic_hit(message)
        if topic_hit:
            category, pattern = topic_hit
            topic = {
                "safe": False,
                "category": category,
                "reason": "keyword_guard",
                "pattern": pattern,
            }
        else:
            topic = {
                "safe": True,
                "category": "none",
                "reason": "no_keyword_guard_hit",
            }

        return topic, language

    def _rule_based_sentiment(self, message: str) -> dict[str, Any]:
        lowered = message.lower()
        folded = _fold_vietnamese(message)

        def has_any(words: tuple[str, ...]) -> bool:
            return any(word in lowered or word in folded for word in words)

        if has_any(("sợ", "lo quá", "ác mộng", "so", "lo qua", "ac mong")):
            return {"emotion": "scared", "needs_comfort": True, "reason": "rule_based"}
        if has_any(("giận", "tức", "bực", "gian", "tuc", "buc")):
            return {"emotion": "angry", "needs_comfort": True, "reason": "rule_based"}
        if has_any(("buồn", "khóc", "cô đơn", "chán", "buon", "khoc", "co don", "chan")):
            return {"emotion": "sad", "needs_comfort": True, "reason": "rule_based"}
        if has_any(("vui", "thích", "hạnh phúc", "thich", "hanh phuc")):
            return {"emotion": "happy", "needs_comfort": False, "reason": "rule_based"}
        if "?" in message or has_any(("tại sao", "vì sao", "là gì", "kể", "tai sao", "vi sao", "la gi", "ke")):
            return {"emotion": "curious", "needs_comfort": False, "reason": "rule_based"}

        return {"emotion": "neutral", "needs_comfort": False, "reason": "rule_based"}

    async def _clean_message(self, message: str) -> str:
        normalized = self._normalize_text(message)
        if not self.config.use_llm_cleaner:
            return normalized
        try:
            cleaned = (await self.cleaner_chain.ainvoke({"message": normalized})).strip()
            return cleaned or normalized
        except Exception as exc:
            self._log(f"  Cleaner lỗi, dùng câu gốc: {exc}")
            return normalized

    async def _run_filters(self, message: str) -> dict[str, Any]:
        topic, language = self._rule_based_safety(message)
        sentiment = self._rule_based_sentiment(message)
        raw: dict[str, Any] = {
            "topic": "rule_based",
            "language": "rule_based",
            "sentiment": "rule_based",
        }

        if self.config.use_llm_filters:
            self._log("  Đang chạy LLM filter bổ sung...")
            try:
                topic_raw, language_raw = await asyncio.gather(
                    self.topic_chain.ainvoke({"message": message}),
                    self.language_chain.ainvoke({"message": message}),
                )
                raw["topic"] = topic_raw
                raw["language"] = language_raw

                model_topic = self._parse_json(
                    topic_raw,
                    {"safe": True, "category": "none", "reason": "parse_error"},
                )
                model_language = self._parse_json(
                    language_raw,
                    {"safe": True, "reason": "parse_error", "words": []},
                )

                if bool(topic.get("safe", True)) and not bool(model_topic.get("safe", True)):
                    topic = {
                        **model_topic,
                        "reason": f"llm_guard:{model_topic.get('reason', 'unsafe')}",
                    }
                if bool(language.get("safe", True)) and not bool(model_language.get("safe", True)):
                    language = {
                        **model_language,
                        "reason": f"llm_guard:{model_language.get('reason', 'unsafe')}",
                    }
            except Exception as exc:
                self._log(f"  LLM filter lỗi, dùng rule-based: {exc}")

        if self.config.use_llm_sentiment:
            self._log("  Đang chạy LLM sentiment bổ sung...")
            try:
                sentiment_raw = await self.sentiment_chain.ainvoke({"message": message})
                raw["sentiment"] = sentiment_raw
                model_sentiment = self._parse_json(
                    sentiment_raw,
                    {"emotion": "neutral", "needs_comfort": False},
                )
                allowed_emotions = {"happy", "sad", "scared", "angry", "curious", "neutral"}
                if model_sentiment.get("emotion") in allowed_emotions:
                    sentiment = model_sentiment
            except Exception as exc:
                self._log(f"  LLM sentiment lỗi, dùng rule-based: {exc}")

        return {
            "topic": topic,
            "language": language,
            "sentiment": sentiment,
            "raw": raw,
        }

    def _remember(self, user_message: str, assistant_message: str) -> None:
        self.history.append(HumanMessage(content=user_message))
        self.history.append(AIMessage(content=assistant_message))

    def _last_user_message(self) -> str | None:
        for item in reversed(self.history):
            if isinstance(item, HumanMessage):
                return item.content
        return None

    def _try_context_recall(self, message: str) -> str | None:
        lowered = message.lower()
        asks_recent_context = any(
            marker in lowered
            for marker in ("lúc nãy", "vừa rồi", "hồi nãy", "ban nãy", "nãy")
        ) and any(
            verb in lowered
            for verb in ("nói", "hỏi", "kể", "nhắc")
        )
        if not asks_recent_context:
            return None

        last_user_message = self._last_user_message()
        if not last_user_message:
            return (
                "Tớ chưa có ký ức nào trước đó trong cuộc trò chuyện này, "
                "nên tớ chưa nhớ chắc cậu vừa nói gì."
            )
        return f"Lúc nãy cậu nói: \"{last_user_message}\". Cậu muốn nói tiếp về điều đó nhé."

    def _topic_redirect(self, category: str) -> str:
        if category == "self_harm":
            return (
                "Tớ nghe thấy cậu đang có điều rất nặng trong lòng. "
                "Cậu hãy tìm một người lớn đáng tin ở gần cậu ngay bây giờ nhé, "
                "và tớ sẽ ở đây cùng cậu trong lúc cậu nói ra cảm giác của cậu."
            )
        return (
            "Câu này có vẻ hơi nguy hiểm với trẻ em, nên tớ sẽ không hướng dẫn theo cách đó nhé. "
            "Cậu có thể đổi sang một cách an toàn hơn, như kể chuyện, học điều thú vị, "
            "hoặc nghĩ cách giải quyết bằng lời nói nhẹ nhàng."
        )

    def _language_guidance(self) -> str:
        return (
            "Tớ nghe thấy cậu vừa dùng một từ chưa lịch sự lắm. "
            "Cậu thử nói nhẹ nhàng hơn nhé, vì lời nói ấm áp sẽ làm mọi người thấy an toàn và vui hơn."
        )

    def _clean_response(self, response: str) -> str:
        response = re.split(
            r"(?im)\n?\s*(?:[-_=]{3,}\s*)?\(?\s*history\s*\)?\s*:?\s*$",
            response,
            maxsplit=1,
        )[0]
        for marker in (
            "**Explanation:**",
            "Explanation:",
            "Let me know",
            "Giải thích:",
            "History:",
            "history:",
        ):
            if marker in response:
                response = response.split(marker, 1)[0]
        response = re.sub(r"[*_`#>]+", "", response)
        response = re.sub(r"(?m)\n?\s*[-_=]{3,}\s*$", "", response)
        response = re.sub(r"\bMình\b", "Tớ", response)
        response = re.sub(r"\bmình\b", "tớ", response)
        return re.sub(r"\s+\n", "\n", response).strip()

    def reset_memory(self) -> None:
        self.history.clear()

    async def process(self, message: str) -> str:
        message = message.strip()
        if not message:
            return "Tớ chưa nghe rõ cậu nói gì. Cậu thử nói lại chậm hơn một chút nhé."

        self._log(f"Nhận tin nhắn: {message}")
        cleaned = await self._clean_message(message)
        self._log(f"  Câu đã chuẩn hóa: {cleaned}")

        recall_response = self._try_context_recall(cleaned)
        if recall_response:
            self._remember(cleaned, recall_response)
            return recall_response

        filter_results = await self._run_filters(cleaned)
        topic = filter_results["topic"]
        language = filter_results["language"]
        sentiment = filter_results["sentiment"]

        self._log(f"  Topic filter: {topic}")
        self._log(f"  Language filter: {language}")
        self._log(f"  Sentiment: {sentiment}")

        if not bool(language.get("safe", True)):
            response = self._language_guidance()
            self._remember(cleaned, response)
            return response

        if not bool(topic.get("safe", True)):
            response = self._topic_redirect(str(topic.get("category", "dangerous")))
            self._remember(cleaned, response)
            return response

        emotion = str(sentiment.get("emotion", "neutral"))
        runtime_config = "No OmniBear DB config is active for this run."
        if self.omnibear_config is not None:
            try:
                config_snapshot = await self.omnibear_config.get_config()
                runtime_config = format_omnibear_context(config_snapshot)
                self._log(f"  OmniBear config source: {config_snapshot.source}")
            except Exception as exc:
                self._log(f"  OmniBear config loi, dung prompt mac dinh: {exc}")

        response = await self.answer_chain.ainvoke({
            "message": cleaned,
            "history": list(self.history),
            "emotion": emotion,
            "runtime_config": runtime_config,
        })
        response = self._clean_response(response)
        self._remember(cleaned, response)
        return response
