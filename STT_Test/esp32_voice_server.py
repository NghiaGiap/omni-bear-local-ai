import argparse
import asyncio
import base64
import hashlib
import json
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from opus_utils import decode_opus_frames_to_wav, encode_wav_to_opus_frames
from stt_tester import load_config, load_stt_model, transcribe_wav


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
LANGCHAIN_DIR = ROOT_DIR / "Langchain"
TEMP_DIR = BASE_DIR / "esp32_temp"
DEFAULT_TTS_URL = "http://localhost:5001"
DEFAULT_OTA_PATH = "/xiaozhi/ota/"
DEFAULT_WS_PATH = "/xiaozhi/ws/"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

if str(LANGCHAIN_DIR) not in sys.path:
    sys.path.insert(0, str(LANGCHAIN_DIR))

from bear_chain import BearAIPipeline  # noqa: E402


class CaseInsensitiveHeaders:
    def __init__(self, pairs: list[tuple[str, str]]):
        self._values = {name.lower(): value for name, value in pairs}

    def get(self, name: str, default: str = "") -> str:
        return self._values.get(name.lower(), default)


@dataclass
class HttpRequest:
    method: str
    target: str
    path: str
    version: str
    headers: CaseInsensitiveHeaders
    body: bytes


def now_ms() -> int:
    return int(time.time() * 1000)


async def send_json(websocket, payload: dict[str, Any]) -> None:
    await websocket.send(json.dumps(payload, ensure_ascii=False))


def parse_json_message(message: str) -> dict[str, Any]:
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return {"type": "text", "text": message}
    return data if isinstance(data, dict) else {"type": "unknown"}


def normalize_path(path: str) -> str:
    normalized = "/" + path.strip("/")
    if path.endswith("/"):
        normalized += "/"
    return normalized


def same_path(left: str, right: str) -> bool:
    return left.rstrip("/") == right.rstrip("/")


def public_http_to_ws(url: str) -> str:
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


def synthesize_tts_wav(
    *,
    text: str,
    tts_url: str,
    output_wav: Path,
    ref_audio: str | None,
    ref_text: str | None,
) -> Path:
    payload: dict[str, Any] = {"text": text}
    if ref_audio:
        payload["ref_audio"] = ref_audio
    if ref_text:
        payload["ref_text"] = ref_text

    response = requests.post(f"{tts_url}/tts", json=payload, timeout=60)
    response.raise_for_status()
    output_wav.write_bytes(response.content)
    return output_wav


async def read_http_request(reader: asyncio.StreamReader) -> HttpRequest | None:
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = await reader.read(4096)
        if not chunk:
            return None
        buffer += chunk
        if len(buffer) > 65536:
            raise ValueError("HTTP request header is too large")

    header_bytes, body = buffer.split(b"\r\n\r\n", 1)
    lines = header_bytes.decode("iso-8859-1").split("\r\n")
    method, target, version = lines[0].split(" ", 2)

    pairs: list[tuple[str, str]] = []
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        pairs.append((name.strip(), value.strip()))

    headers = CaseInsensitiveHeaders(pairs)
    content_length = int(headers.get("Content-Length", "0") or "0")
    while len(body) < content_length:
        chunk = await reader.read(content_length - len(body))
        if not chunk:
            break
        body += chunk

    parsed = urlsplit(target)
    return HttpRequest(
        method=method.upper(),
        target=target,
        path=parsed.path or "/",
        version=version,
        headers=headers,
        body=body[:content_length],
    )


async def write_http_response(
    writer: asyncio.StreamWriter,
    status_code: int,
    body: str | bytes,
    headers: dict[str, str] | None = None,
) -> None:
    reasons = {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        405: "Method Not Allowed",
        426: "Upgrade Required",
        500: "Internal Server Error",
    }
    body_bytes = body.encode("utf-8") if isinstance(body, str) else body
    response_headers = {
        "Content-Length": str(len(body_bytes)),
        "Connection": "close",
        "Server": "esp32-voice-server",
    }
    if headers:
        response_headers.update(headers)

    reason = reasons.get(status_code, "OK")
    lines = [f"HTTP/1.1 {status_code} {reason}"]
    lines.extend(f"{name}: {value}" for name, value in response_headers.items())
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body_bytes)
    await writer.drain()
    writer.close()
    await writer.wait_closed()


class SimpleWebSocket:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.recv()
        if message is None:
            raise StopAsyncIteration
        return message

    async def recv(self) -> str | bytes | None:
        fragments: list[bytes] = []
        fragment_opcode: int | None = None

        while True:
            header = await self.reader.readexactly(2)
            first, second = header
            fin = bool(first & 0x80)
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F

            if length == 126:
                length = struct.unpack("!H", await self.reader.readexactly(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", await self.reader.readexactly(8))[0]

            mask = await self.reader.readexactly(4) if masked else b""
            payload = await self.reader.readexactly(length) if length else b""
            if masked:
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))

            if opcode == 0x8:
                await self.close()
                return None
            if opcode == 0x9:
                await self._send_frame(payload, 0xA)
                continue
            if opcode == 0xA:
                continue

            if opcode in {0x1, 0x2}:
                if fin:
                    return payload.decode("utf-8", errors="replace") if opcode == 0x1 else payload
                fragment_opcode = opcode
                fragments = [payload]
                continue

            if opcode == 0x0 and fragment_opcode is not None:
                fragments.append(payload)
                if fin:
                    data = b"".join(fragments)
                    if fragment_opcode == 0x1:
                        return data.decode("utf-8", errors="replace")
                    return data

    async def send(self, message: str | bytes) -> None:
        if isinstance(message, str):
            await self._send_frame(message.encode("utf-8"), 0x1)
        else:
            await self._send_frame(bytes(message), 0x2)

    async def ping(self) -> None:
        await self._send_frame(b"", 0x9)

    async def _send_frame(self, payload: bytes, opcode: int) -> None:
        if self.closed:
            return

        first = 0x80 | opcode
        length = len(payload)
        if length < 126:
            header = bytes([first, length])
        elif length < 65536:
            header = bytes([first, 126]) + struct.pack("!H", length)
        else:
            header = bytes([first, 127]) + struct.pack("!Q", length)

        self.writer.write(header + payload)
        await self.writer.drain()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await self._send_frame(b"", 0x8)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except ConnectionError:
            pass


class Esp32VoiceSession:
    def __init__(self, websocket: SimpleWebSocket, server: "Esp32VoiceServer"):
        self.websocket = websocket
        self.server = server
        self.session_id = f"{now_ms()}"
        self.frames: list[bytes] = []
        self.receiving_audio = False
        self.audio_closed = False
        self.processing_audio = False
        self.ignored_audio_frames = 0
        self.auto_stop_task: asyncio.Task | None = None

    def cancel_auto_stop(self) -> None:
        current_task = asyncio.current_task()
        if self.auto_stop_task and not self.auto_stop_task.done() and self.auto_stop_task is not current_task:
            self.auto_stop_task.cancel()
        if self.auto_stop_task is not current_task:
            self.auto_stop_task = None

    def start_auto_stop(self) -> None:
        self.cancel_auto_stop()
        if self.server.max_record_seconds > 0:
            self.auto_stop_task = asyncio.create_task(self.auto_stop_after_delay())

    async def auto_stop_after_delay(self) -> None:
        try:
            await asyncio.sleep(self.server.max_record_seconds)
            if self.receiving_audio and not self.processing_audio:
                print(
                    f"WS auto-stop: session={self.session_id} "
                    f"after={self.server.max_record_seconds:.1f}s frames={len(self.frames)}"
                )
                await self.process_audio_frames(source="auto")
        except asyncio.CancelledError:
            pass
        finally:
            if self.auto_stop_task is asyncio.current_task():
                self.auto_stop_task = None

    async def send_hello(self) -> None:
        await send_json(
            self.websocket,
            {
                "type": "hello",
                "transport": "websocket",
                "session_id": self.session_id,
                "audio_params": {
                    "format": "opus",
                    "sample_rate": 16000,
                    "channels": 1,
                    "frame_duration": 60,
                },
            },
        )

    async def send_stt_result(self, text: str) -> None:
        await send_json(
            self.websocket,
            {
                "session_id": self.session_id,
                "type": "stt",
                "text": text,
            },
        )

    async def send_tts_stop(self) -> None:
        await send_json(
            self.websocket,
            {
                "session_id": self.session_id,
                "type": "tts",
                "state": "stop",
            },
        )

    async def send_alert(self, status: str, message: str, emotion: str = "neutral") -> None:
        await send_json(
            self.websocket,
            {
                "session_id": self.session_id,
                "type": "alert",
                "status": status,
                "message": message,
                "emotion": emotion,
            },
        )

    async def handle_json(self, payload: dict[str, Any]) -> None:
        message_type = str(payload.get("type", "")).lower()

        if message_type == "hello":
            print(f"WS JSON: session={self.session_id} type=hello")
            await self.send_hello()
            return

        if message_type == "listen":
            state = str(payload.get("state", "")).lower()
            print(
                f"WS JSON: session={self.session_id} type=listen "
                f"state={state} frames={len(self.frames)}"
            )
            if state in {"start", "detect"}:
                self.frames.clear()
                self.receiving_audio = True
                self.audio_closed = False
                self.ignored_audio_frames = 0
                self.start_auto_stop()
                return
            if state == "stop":
                self.cancel_auto_stop()
                self.receiving_audio = False
                await self.process_audio_frames(source="client")
                return
            return

        if message_type == "abort":
            self.cancel_auto_stop()
            self.frames.clear()
            self.receiving_audio = False
            self.audio_closed = True
            await self.send_tts_stop()
            return

        if message_type == "goodbye":
            self.cancel_auto_stop()
            await self.websocket.close()
            return

        if message_type == "ping":
            await send_json(
                self.websocket,
                {
                    "session_id": self.session_id,
                    "type": "pong",
                    "ts": now_ms(),
                },
            )
            return

        await self.send_alert("Unsupported message", f"unknown type: {message_type}")

    async def handle_binary(self, frame: bytes) -> None:
        if self.audio_closed or self.processing_audio:
            self.ignored_audio_frames += 1
            if self.ignored_audio_frames == 1 or self.ignored_audio_frames % 50 == 0:
                print(
                    f"WS audio ignored: session={self.session_id} "
                    f"frames={self.ignored_audio_frames}"
                )
            return
        if not self.receiving_audio and not self.frames:
            self.receiving_audio = True
            self.audio_closed = False
            self.start_auto_stop()
        self.frames.append(bytes(frame))
        if len(self.frames) == 1 or len(self.frames) % 50 == 0:
            print(f"WS audio: session={self.session_id} frames={len(self.frames)}")

    async def process_audio_frames(self, source: str) -> None:
        if self.processing_audio:
            print(f"WS process skipped: session={self.session_id} source={source} already_processing=true")
            return

        self.cancel_auto_stop()
        self.processing_audio = True
        self.receiving_audio = False
        self.audio_closed = True
        frames = self.frames
        self.frames = []

        if not frames:
            self.processing_audio = False
            await self.send_alert("No audio", "no opus frames")
            await self.send_tts_stop()
            return

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        decoded_wav = TEMP_DIR / f"esp32_in_{self.session_id}.wav"

        try:
            decode_start = time.perf_counter()
            await asyncio.to_thread(decode_opus_frames_to_wav, frames, decoded_wav)
            decode_seconds = time.perf_counter() - decode_start

            stt_start = time.perf_counter()
            transcript = await asyncio.to_thread(transcribe_wav, decoded_wav, self.server.stt_config)
            stt_seconds = time.perf_counter() - stt_start

            print(
                f"STT: source={source} frames={len(frames)} decode={decode_seconds:.3f}s "
                f"stt={stt_seconds:.3f}s text={transcript!r}"
            )
            await self.send_stt_result(transcript)

            if not transcript:
                await self.send_tts_stop()
                return

            await self.process_text(transcript)
        finally:
            self.processing_audio = False

    async def process_text(self, text: str) -> None:
        ai_start = time.perf_counter()
        if self.server.no_ai:
            response_text = f"To nghe thay: {text}"
        else:
            response_text = await self.server.ai.process(text)
        ai_seconds = time.perf_counter() - ai_start

        print(f"AI: {ai_seconds:.3f}s text={response_text!r}")
        await send_json(
            self.websocket,
            {
                "session_id": self.session_id,
                "type": "llm",
                "emotion": "neutral",
                "text": response_text,
            },
        )

        if self.server.no_tts:
            await self.send_tts_stop()
            return

        await self.send_tts_response(response_text)

    async def send_tts_response(self, text: str) -> None:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f"tts_{self.session_id}_",
            suffix=".wav",
            dir=TEMP_DIR,
            delete=False,
        ) as tmp_file:
            tts_wav = Path(tmp_file.name)

        tts_start = time.perf_counter()
        try:
            await asyncio.to_thread(
                synthesize_tts_wav,
                text=text,
                tts_url=self.server.tts_url,
                output_wav=tts_wav,
                ref_audio=self.server.ref_audio,
                ref_text=self.server.ref_text,
            )
            tts_seconds = time.perf_counter() - tts_start

            encode_start = time.perf_counter()
            frames, _headers, _normalized_wav, debug_ogg = await asyncio.to_thread(
                encode_wav_to_opus_frames,
                tts_wav,
                TEMP_DIR,
            )
            encode_seconds = time.perf_counter() - encode_start

            print(
                f"TTS: synth={tts_seconds:.3f}s encode={encode_seconds:.3f}s "
                f"frames={len(frames)}"
            )
            await send_json(
                self.websocket,
                {
                    "session_id": self.session_id,
                    "type": "tts",
                    "state": "start",
                },
            )
            await send_json(
                self.websocket,
                {
                    "session_id": self.session_id,
                    "type": "tts",
                    "state": "sentence_start",
                    "text": text,
                },
            )

            delay = self.server.tts_frame_delay_ms / 1000.0
            for frame in frames:
                await self.websocket.send(frame)
                if delay > 0:
                    await asyncio.sleep(delay)

            if self.server.debug_save:
                print(f"TTS debug opus: {debug_ogg}")
            await self.send_tts_stop()
        except Exception as exc:
            await self.send_alert("TTS error", str(exc), emotion="sad")
            await self.send_tts_stop()
        finally:
            if not self.server.debug_save:
                try:
                    tts_wav.unlink(missing_ok=True)
                except OSError:
                    pass


class Esp32VoiceServer:
    def __init__(self, args: argparse.Namespace):
        self.host = args.host
        self.port = args.port
        self.tts_url = args.tts_url.rstrip("/")
        self.ref_audio = args.ref_audio
        self.ref_text = args.ref_text
        self.debug_save = args.debug_save
        self.no_ai = args.no_ai
        self.no_tts = args.no_tts
        self.tts_frame_delay_ms = args.tts_frame_delay_ms
        self.max_record_seconds = args.max_record_seconds
        self.ota_path = normalize_path(args.ota_path)
        self.ws_path = normalize_path(args.ws_path)
        self.public_url = args.public_url.rstrip("/") if args.public_url else ""
        self.websocket_token = args.websocket_token
        self.websocket_version = args.websocket_version

        self.stt_config: dict[str, Any] = {}
        self.ai = None

        self.stt_config = load_config()
        if args.engine:
            self.stt_config["engine"] = args.engine
        if args.model:
            self.stt_config["model"] = args.model

        print("Loading STT model...")
        start = time.perf_counter()
        loaded = load_stt_model(self.stt_config)
        print(
            f"STT ready: {loaded['engine']} {self.stt_config.get('model', 'small')} "
            f"({loaded.get('device', 'auto')}) in {time.perf_counter() - start:.2f}s"
        )

        self.ai = None if self.no_ai else BearAIPipeline()

    def _public_base_url(self, request: HttpRequest) -> str:
        if self.public_url:
            return self.public_url

        host = request.headers.get("Host", f"localhost:{self.port}")
        proto = request.headers.get("X-Forwarded-Proto", "http")
        if proto == "https" or "ngrok-free" in host:
            return f"https://{host}"
        return f"http://{host}"

    def _ota_response(self, request: HttpRequest) -> dict[str, Any]:
        public_base = self._public_base_url(request).rstrip("/")
        websocket_url = public_http_to_ws(public_base) + self.ws_path
        return {
            "websocket": {
                "url": websocket_url,
                "token": self.websocket_token,
                "version": self.websocket_version,
            },
            "server_time": {
                "timestamp": now_ms(),
                "timezone_offset": 7 * 60,
            },
        }

    async def handle_http_ota(self, writer: asyncio.StreamWriter, request: HttpRequest) -> None:
        body = json.dumps(self._ota_response(request), ensure_ascii=False)
        print(f"OTA {request.method} {request.path} -> 200")
        await write_http_response(
            writer,
            200,
            body,
            {
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
        )

    async def accept_websocket(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        request: HttpRequest,
    ) -> None:
        key = request.headers.get("Sec-WebSocket-Key")
        if not key:
            await write_http_response(writer, 400, "Missing Sec-WebSocket-Key\n")
            return

        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode("ascii")).digest()).decode("ascii")
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "Server: esp32-voice-server\r\n"
            "\r\n"
        )
        writer.write(response.encode("ascii"))
        await writer.drain()
        await self.handler(SimpleWebSocket(reader, writer))

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request = await read_http_request(reader)
            if request is None:
                writer.close()
                await writer.wait_closed()
                return

            if same_path(request.path, self.ota_path):
                await self.handle_http_ota(writer, request)
                return

            activate_path = self.ota_path.rstrip("/") + "/activate"
            if same_path(request.path, activate_path):
                await write_http_response(
                    writer,
                    200,
                    '{"ok":true}',
                    {"Content-Type": "application/json; charset=utf-8"},
                )
                return

            upgrade = request.headers.get("Upgrade").lower()
            if upgrade == "websocket":
                if not same_path(request.path, self.ws_path):
                    await write_http_response(writer, 404, f"Unknown WebSocket path: {request.path}\n")
                    return
                await self.accept_websocket(reader, writer, request)
                return

            public_base = self._public_base_url(request).rstrip("/")
            body = (
                "ESP32 Xiaozhi OTA server is running.\n"
                f"OTA URL: {public_base}{self.ota_path}\n"
                f"WebSocket URL: {public_http_to_ws(public_base)}{self.ws_path}\n"
            )
            await write_http_response(writer, 200, body, {"Content-Type": "text/plain; charset=utf-8"})
        except Exception as exc:
            print(f"HTTP/WebSocket connection error: {exc}")
            if not writer.is_closing():
                await write_http_response(writer, 500, f"{exc}\n")

    async def handler(self, websocket: SimpleWebSocket) -> None:
        session = Esp32VoiceSession(websocket, self)
        print(f"ESP32 connected: session={session.session_id}")
        keepalive_task = asyncio.create_task(self.keepalive(websocket, session.session_id))
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await session.handle_binary(message)
                else:
                    await session.handle_json(parse_json_message(message))
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
            await websocket.close()
            print(f"ESP32 disconnected: session={session.session_id}")

    async def keepalive(self, websocket: SimpleWebSocket, session_id: str) -> None:
        while True:
            await asyncio.sleep(20)
            try:
                await websocket.ping()
                print(f"WS ping: session={session_id}")
            except Exception as exc:
                print(f"WS ping failed: session={session_id} error={exc}")
                return

    async def run(self) -> None:
        local_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        local_base = f"http://{local_host}:{self.port}"
        public_base = self.public_url or local_base

        print(f"Local server: {local_base}")
        print(f"OTA URL: {public_base}{self.ota_path}")
        print(f"WebSocket URL in OTA: {public_http_to_ws(public_base)}{self.ws_path}")
        print(f"Auto-stop recording after: {self.max_record_seconds:.1f}s")
        print("Protocol: Xiaozhi OTA + WebSocket")

        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        async with server:
            await server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ESP32 Xiaozhi OTA/WebSocket voice server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--engine", choices=["phowhisper", "faster", "whisper"])
    parser.add_argument("--model")
    parser.add_argument("--tts-url", default=DEFAULT_TTS_URL)
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--tts-frame-delay-ms", type=float, default=0.0)
    parser.add_argument("--max-record-seconds", type=float, default=10.0)
    parser.add_argument("--public-url", help="Public HTTP(S) base URL, e.g. https://name.ngrok-free.dev")
    parser.add_argument("--ota-path", default=DEFAULT_OTA_PATH)
    parser.add_argument("--ws-path", default=DEFAULT_WS_PATH)
    parser.add_argument("--websocket-token", default="local-dev-token")
    parser.add_argument("--websocket-version", type=int, default=1)
    parser.add_argument("--debug-save", action="store_true")
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    server = Esp32VoiceServer(args)
    asyncio.run(server.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
