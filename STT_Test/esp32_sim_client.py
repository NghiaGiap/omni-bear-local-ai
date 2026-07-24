import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import requests
import websockets

from opus_utils import (
    decode_opus_frames_to_wav,
    encode_wav_to_opus_frames,
    write_opus_frame_stream,
)


BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "esp32_temp"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def parse_json_message(message: str) -> dict:
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return {"type": "text", "text": message}
    return data if isinstance(data, dict) else {"type": "unknown"}


def resolve_uri_from_ota(ota_url: str) -> str:
    response = requests.get(
        ota_url,
        headers={"ngrok-skip-browser-warning": "true"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    websocket_config = data.get("websocket", {})
    uri = websocket_config.get("url", "")
    if not uri:
        raise RuntimeError(f"OTA response has no websocket.url: {data}")
    print(f"OTA endpoint: {ota_url}")
    print(f"OTA websocket URL: {uri}")
    return str(uri)


async def send_wav_as_xiaozhi_audio(
    *,
    uri: str,
    wav_path: Path,
    delay_ms: float,
    output_wav: Path,
) -> int:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    frames, _headers, normalized_wav, encoded_ogg = encode_wav_to_opus_frames(
        wav_path,
        TEMP_DIR,
    )
    stream_path = TEMP_DIR / f"{wav_path.stem}_client_sent.opusframes"
    write_opus_frame_stream(frames, stream_path)

    print(f"Connect: {uri}")
    print(f"Input WAV: {wav_path}")
    print(f"Normalized WAV: {normalized_wav}")
    print(f"Client debug Ogg: {encoded_ogg}")
    print(f"Client sent frame stream: {stream_path}")
    print(f"Sending Opus frames: {len(frames)}")

    tts_frames: list[bytes] = []
    receiving_tts = False
    started = time.perf_counter()

    try:
        websocket = await websockets.connect(uri, max_size=None)
    except websockets.exceptions.InvalidStatus as exc:
        response = getattr(exc, "response", None)
        if response is not None:
            reason = getattr(response, "reason_phrase", "")
            print(f"WebSocket handshake rejected: HTTP {response.status_code} {reason}")
            body = getattr(response, "body", b"")
            if body:
                if isinstance(body, bytes):
                    body = body.decode("utf-8", errors="replace")
                print(str(body)[:1000])
        else:
            print(f"WebSocket handshake rejected: {exc}")
        return 1
    except Exception as exc:
        print(f"WebSocket connection failed: {type(exc).__name__}: {exc}")
        return 1

    try:
        await websocket.send(
            json.dumps(
                {
                    "type": "hello",
                    "version": 1,
                    "features": {"mcp": False, "aec": False},
                    "transport": "websocket",
                    "audio_params": {
                        "format": "opus",
                        "sample_rate": 16000,
                        "channels": 1,
                        "frame_duration": 60,
                    },
                },
                ensure_ascii=False,
            )
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "listen",
                    "state": "start",
                    "mode": "manual",
                },
                ensure_ascii=False,
            )
        )

        delay = delay_ms / 1000.0
        for frame in frames:
            await websocket.send(frame)
            if delay > 0:
                await asyncio.sleep(delay)
        await websocket.send(json.dumps({"type": "listen", "state": "stop"}))

        async for message in websocket:
            if isinstance(message, bytes):
                if receiving_tts:
                    tts_frames.append(bytes(message))
                continue

            payload = parse_json_message(message)
            message_type = payload.get("type")
            print(f"<- {json.dumps(payload, ensure_ascii=False)}")

            if message_type == "tts" and payload.get("state") == "start":
                receiving_tts = True
                tts_frames.clear()
            elif message_type == "tts" and payload.get("state") == "stop":
                receiving_tts = False
                if tts_frames:
                    decode_opus_frames_to_wav(tts_frames, output_wav)
                    print(f"Saved TTS response WAV: {output_wav}")
                    print(f"Received TTS frames: {len(tts_frames)}")
                print(f"Total round-trip: {time.perf_counter() - started:.2f}s")
                return 0
            elif message_type == "alert":
                print(f"Server alert: {payload.get('status', '')} {payload.get('message', '')}")
    finally:
        await websocket.close()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate Xiaozhi ESP32 WebSocket Opus client")
    parser.add_argument("--uri", default="ws://127.0.0.1:8765/xiaozhi/ws/")
    parser.add_argument("--ota-url", help="Call OTA first, then connect to websocket.url")
    parser.add_argument("--wav", required=True, help="WAV/audio file to send")
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=0.0,
        help="Delay between Opus frames. Use 60 to simulate real time.",
    )
    parser.add_argument(
        "--output-wav",
        default=str(TEMP_DIR / "tts_response_from_server.wav"),
        help="Where to save decoded TTS response audio",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.is_absolute():
        wav_path = (Path.cwd() / wav_path).resolve()

    uri = resolve_uri_from_ota(args.ota_url) if args.ota_url else args.uri

    return asyncio.run(
        send_wav_as_xiaozhi_audio(
            uri=uri,
            wav_path=wav_path,
            delay_ms=args.delay_ms,
            output_wav=Path(args.output_wav),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
