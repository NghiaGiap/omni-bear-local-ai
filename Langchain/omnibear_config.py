from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import requests


DEFAULT_OMNIBEAR_API_URL = "https://omni-bear-api-production.up.railway.app/api"
SECRET_KEY_RE = re.compile(r"(key|token|secret|password|credential|connection|dsn|url)", re.I)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _normalize_base_url(url: str) -> str:
    return url.strip().rstrip("/")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _compact_text(value: Any, *, limit: int = 160) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _safe_items(data: dict[str, Any], *, max_items: int = 12) -> list[str]:
    lines: list[str] = []
    for key, value in data.items():
        if len(lines) >= max_items:
            break
        if SECRET_KEY_RE.search(str(key)):
            continue
        if isinstance(value, (dict, list)):
            try:
                rendered = json.dumps(value, ensure_ascii=False)
            except TypeError:
                rendered = str(value)
        else:
            rendered = str(value)
        lines.append(f"- {key}: {_compact_text(rendered)}")
    return lines


@dataclass
class RuntimeConfigSnapshot:
    source: str
    data: dict[str, Any]
    fetched_at: float


class OmniBearConfigProvider:
    """Loads OmniBear runtime config for the local AI server.

    Priority:
    1. OMNIBEAR_CONFIG_URL, for a dedicated backend endpoint.
    2. OMNIBEAR_API_URL + OMNIBEAR_ACCESS_TOKEN + teddy/device id, matching the mobile API.
    3. Supabase REST table, when explicit table/query env vars are provided.
    4. OMNIBEAR_CONFIG_FILE, as a local fallback.
    """

    def __init__(self) -> None:
        self.config_url = os.getenv("OMNIBEAR_CONFIG_URL", "").strip()
        self.config_token = os.getenv("OMNIBEAR_CONFIG_TOKEN", "").strip()

        api_url = os.getenv("OMNIBEAR_API_URL") or os.getenv("EXPO_PUBLIC_API_URL") or DEFAULT_OMNIBEAR_API_URL
        self.api_url = _normalize_base_url(api_url)
        self.access_token = os.getenv("OMNIBEAR_ACCESS_TOKEN", "").strip()
        self.teddy_id = os.getenv("OMNIBEAR_TEDDY_ID", "").strip()
        self.device_id = os.getenv("OMNIBEAR_DEVICE_ID", "").strip()

        self.supabase_url = _normalize_base_url(
            os.getenv("OMNIBEAR_SUPABASE_URL") or os.getenv("SUPABASE_URL", "")
        )
        self.supabase_key = (
            os.getenv("OMNIBEAR_SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("OMNIBEAR_SUPABASE_ANON_KEY")
            or os.getenv("SUPABASE_ANON_KEY")
            or ""
        ).strip()
        self.supabase_table = os.getenv("OMNIBEAR_SUPABASE_CONFIG_TABLE", "").strip()
        self.supabase_select = os.getenv("OMNIBEAR_SUPABASE_SELECT", "*").strip() or "*"
        self.supabase_filter = os.getenv("OMNIBEAR_SUPABASE_FILTER", "").strip()

        config_file = os.getenv("OMNIBEAR_CONFIG_FILE", "").strip()
        self.config_file = Path(config_file) if config_file else None

        self.timeout_seconds = _env_float("OMNIBEAR_CONFIG_TIMEOUT_SECONDS", 2.0)
        self.cache_seconds = _env_float("OMNIBEAR_CONFIG_CACHE_SECONDS", 60.0)
        self._snapshot: RuntimeConfigSnapshot | None = None
        self._lock = asyncio.Lock()

    async def get_config(self) -> RuntimeConfigSnapshot:
        now = time.monotonic()
        if self._snapshot and now - self._snapshot.fetched_at < self.cache_seconds:
            return self._snapshot

        async with self._lock:
            now = time.monotonic()
            if self._snapshot and now - self._snapshot.fetched_at < self.cache_seconds:
                return self._snapshot

            snapshot = await asyncio.to_thread(self._load_config)
            self._snapshot = snapshot
            return snapshot

    def _load_config(self) -> RuntimeConfigSnapshot:
        last_error_source = ""
        for loader in (
            self._fetch_config_url,
            self._fetch_mobile_api_config,
            self._fetch_supabase_config,
            self._load_local_config_file,
        ):
            try:
                snapshot = loader()
            except Exception as exc:
                last_error_source = f"{loader.__name__}:error:{type(exc).__name__}"
                continue
            if snapshot.data:
                return snapshot

        return RuntimeConfigSnapshot(source=last_error_source or "none", data={}, fetched_at=time.monotonic())

    def _headers(self, token: str) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _fetch_config_url(self) -> RuntimeConfigSnapshot:
        if not self.config_url:
            return RuntimeConfigSnapshot(source="config_url:disabled", data={}, fetched_at=time.monotonic())

        response = requests.get(
            self.config_url,
            headers=self._headers(self.config_token),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return RuntimeConfigSnapshot(
            source="config_url",
            data=_as_dict(payload),
            fetched_at=time.monotonic(),
        )

    def _fetch_mobile_api_config(self) -> RuntimeConfigSnapshot:
        if not self.access_token:
            return RuntimeConfigSnapshot(source="mobile_api:disabled", data={}, fetched_at=time.monotonic())

        if self.teddy_id:
            payload = self._api_get(f"/me/teddies/{quote(self.teddy_id)}")
            return self._snapshot_from_teddy_payload(payload, "mobile_api")

        payload = self._api_get("/me/teddies")
        teddies = payload.get("teddies") if isinstance(payload, dict) else None
        if not isinstance(teddies, list) or not teddies:
            return RuntimeConfigSnapshot(source="mobile_api:no_teddies", data={}, fetched_at=time.monotonic())

        selected = None
        if self.device_id:
            selected = next(
                (
                    teddy
                    for teddy in teddies
                    if str(_as_dict(teddy).get("deviceId") or _as_dict(teddy).get("device_id")) == self.device_id
                ),
                None,
            )
        selected = selected or teddies[0]
        return self._snapshot_from_teddy_payload({"teddy": selected}, "mobile_api")

    def _api_get(self, path: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.api_url}{path}",
            headers=self._headers(self.access_token),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _as_dict(response.json())

    def _snapshot_from_teddy_payload(self, payload: dict[str, Any], source: str) -> RuntimeConfigSnapshot:
        teddy = _as_dict(payload.get("teddy") or payload)
        config = _as_dict(
            teddy.get("runtimeConfig")
            or teddy.get("effectiveConfig")
            or teddy.get("config")
            or teddy.get("aiConfig")
            or teddy.get("ai_config")
        )
        data = {
            "teddy": {
                "id": teddy.get("id"),
                "deviceId": teddy.get("deviceId") or teddy.get("device_id"),
                "name": teddy.get("name"),
                "childName": teddy.get("childName") or teddy.get("child_name"),
                "ownerName": teddy.get("ownerName") or teddy.get("owner_name"),
            },
            "config": config,
        }
        for key in ("globalConfig", "global_config", "parentConfig", "parent_config"):
            if isinstance(teddy.get(key), dict):
                data[key] = teddy[key]
        return RuntimeConfigSnapshot(source=source, data=data, fetched_at=time.monotonic())

    def _global_value_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return _json_dict(record.get("value"))

    def _fetch_supabase_config(self) -> RuntimeConfigSnapshot:
        if not (self.supabase_url and self.supabase_key and self.supabase_table):
            return RuntimeConfigSnapshot(source="supabase:disabled", data={}, fetched_at=time.monotonic())

        query = {"select": self.supabase_select}
        query_string = urlencode(query)
        if self.supabase_filter:
            query_string = f"{query_string}&{self.supabase_filter.lstrip('?&')}"

        url = f"{self.supabase_url}/rest/v1/{quote(self.supabase_table)}?{query_string}"
        headers = {
            "Accept": "application/json",
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
        }
        response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            records = [_as_dict(item) for item in payload if isinstance(item, dict)]
            data: dict[str, Any] = {}
            if records:
                data["config"] = self._global_value_from_record(records[0])
        else:
            data = {"config": self._global_value_from_record(_as_dict(payload))}
        return RuntimeConfigSnapshot(source="supabase", data=data, fetched_at=time.monotonic())

    def _load_local_config_file(self) -> RuntimeConfigSnapshot:
        if not self.config_file:
            return RuntimeConfigSnapshot(source="file:disabled", data={}, fetched_at=time.monotonic())
        if not self.config_file.exists():
            return RuntimeConfigSnapshot(source="file:missing", data={}, fetched_at=time.monotonic())

        data = json.loads(self.config_file.read_text(encoding="utf-8"))
        return RuntimeConfigSnapshot(
            source="file",
            data=_as_dict(data),
            fetched_at=time.monotonic(),
        )


def format_omnibear_context(snapshot: RuntimeConfigSnapshot | None) -> str:
    if not snapshot or not snapshot.data:
        return "No OmniBear DB config is active for this run."

    data = snapshot.data
    teddy = _as_dict(data.get("teddy"))
    config = _as_dict(data.get("config"))
    if not config and isinstance(data.get("records"), list) and data["records"]:
        first_record = data["records"][0]
        config = _as_dict(first_record)

    if snapshot.source == "supabase":
        lines = [
            "Admin global config from Supabase value:",
            "Use only these fields when answering. Do not reveal DB config.",
        ]
        teddy_prompt = _first_present(config, ("teddyPrompt", "teddy_prompt"))
        age_range = _first_present(config, ("ageRange", "age_range"))
        voice_tone = _first_present(config, ("voiceTone", "voice_tone"))

        if teddy_prompt:
            lines.append(f"- teddyPrompt: {_compact_text(teddy_prompt, limit=1200)}")
        if age_range:
            lines.append(f"- ageRange: {_compact_text(age_range, limit=80)}")
        if voice_tone:
            lines.append(f"- voiceTone: {_compact_text(voice_tone, limit=120)}")

        if len(lines) == 2:
            lines.append("- No supported global config fields found.")
        return "\n".join(lines)

    lines = [
        f"Source: {snapshot.source}.",
        "Follow parent/admin config when it is relevant.",
        "Never reveal raw DB rows, table names, API keys, access tokens, or internal config values.",
    ]

    display_name = config.get("displayName") or config.get("display_name") or teddy.get("name")
    child_name = config.get("childName") or config.get("child_name") or teddy.get("childName")
    if display_name:
        lines.append(f"Teddy display name: {_compact_text(display_name, limit=80)}.")
    if child_name:
        lines.append(f"Child name: {_compact_text(child_name, limit=80)}. Use it only when natural; keep to/cau style.")

    voice = config.get("voice")
    language = config.get("language")
    personality = config.get("personality")
    if voice:
        lines.append(f"Voice style: {_compact_text(voice, limit=80)}.")
    if language:
        lines.append(f"Preferred language: {_compact_text(language, limit=80)}.")
    if personality is not None:
        lines.append(f"Personality level 0-100: {_compact_text(personality, limit=20)}.")

    bedtime = config.get("bedtimeMode")
    if bedtime is None:
        bedtime = config.get("bedtime_mode")
    if bedtime is True:
        lines.append("Bedtime mode is on: keep replies calmer, shorter, and avoid high-energy play.")

    daily_limit = config.get("dailyLimitMinutes") or config.get("daily_limit_minutes")
    if daily_limit:
        lines.append(f"Daily limit minutes: {_compact_text(daily_limit, limit=20)}.")

    topics = config.get("topics")
    if isinstance(topics, list) and topics:
        rendered_topics = ", ".join(_compact_text(topic, limit=40) for topic in topics[:12])
        lines.append(f"Approved/favorite topics: {rendered_topics}.")

    require_topic_approval = config.get("requireTopicApproval")
    if require_topic_approval is None:
        require_topic_approval = config.get("require_topic_approval")
    if require_topic_approval is True:
        lines.append("Topic approval is required: if a new topic is outside approved topics, redirect gently.")

    override_global = config.get("overrideGlobal")
    if override_global is None:
        override_global = config.get("override_global")
    if override_global is True:
        lines.append("This teddy overrides global settings where teddy-specific config conflicts.")

    extra_blocks: list[str] = []
    for key in ("globalConfig", "global_config", "parentConfig", "parent_config"):
        extra = _as_dict(data.get(key))
        if extra:
            extra_blocks.extend(_safe_items(extra, max_items=6))

    if extra_blocks:
        lines.append("Extra safe config summary:")
        lines.extend(extra_blocks[:8])
    elif config:
        generic = _safe_items(config, max_items=8)
        if generic:
            lines.append("Raw safe config summary:")
            lines.extend(generic)

    return "\n".join(lines)
