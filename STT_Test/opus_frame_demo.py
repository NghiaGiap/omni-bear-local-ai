import argparse
import json
import time
from pathlib import Path

from opus_utils import (
    decode_opus_frames_to_wav,
    encode_wav_to_opus_frames,
    read_opus_frame_stream,
    write_opus_frame_stream,
)
from stt_tester import load_config, load_stt_model, print_words, transcribe_wav


BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "opus_temp"


def describe_frames(frames: list[bytes]) -> dict:
    sizes = [len(frame) for frame in frames]
    return {
        "count": len(frames),
        "total_payload_bytes": sum(sizes),
        "min_frame_bytes": min(sizes) if sizes else 0,
        "max_frame_bytes": max(sizes) if sizes else 0,
        "first_10_frame_bytes": sizes[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate ESP32: WAV -> Opus frames -> receive/decode -> STT."
    )
    parser.add_argument("--wav", required=True, help="Input WAV/audio file")
    parser.add_argument(
        "--engine",
        choices=["phowhisper", "faster", "whisper"],
        default=None,
        help="STT engine",
    )
    parser.add_argument("--model", default=None, help="STT model")
    parser.add_argument("--no-stt", action="store_true", help="Only encode/decode frames")
    parser.add_argument("--json-out", help="Save details JSON")
    args = parser.parse_args()

    config = load_config()
    if args.engine:
        config["engine"] = args.engine
    if args.model:
        config["model"] = args.model

    wav_path = Path(args.wav)
    if not wav_path.is_absolute():
        wav_path = (Path.cwd() / wav_path).resolve()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    frame_stream_path = TEMP_DIR / f"{wav_path.stem}.opusframes"
    decoded_wav = TEMP_DIR / f"{wav_path.stem}_from_received_frames.wav"

    print("Step 1: WAV -> 16 kHz mono -> Opus frames")
    encode_start = time.perf_counter()
    frames, _headers, normalized_wav, encoded_ogg = encode_wav_to_opus_frames(
        wav_path,
        TEMP_DIR,
    )
    encode_seconds = time.perf_counter() - encode_start
    frame_info = describe_frames(frames)
    write_opus_frame_stream(frames, frame_stream_path)

    print(f"Input WAV: {wav_path}")
    print(f"Normalized WAV: {normalized_wav}")
    print(f"Debug Ogg Opus: {encoded_ogg}")
    print(f"Frame stream sent: {frame_stream_path}")
    print(f"Opus frames: {frame_info['count']}")
    print(f"Payload bytes: {frame_info['total_payload_bytes']}")
    print(
        "Frame bytes: "
        f"min={frame_info['min_frame_bytes']} "
        f"max={frame_info['max_frame_bytes']} "
        f"first10={frame_info['first_10_frame_bytes']}"
    )
    print(f"Encode/frame split: {encode_seconds:.2f}s")

    print("\nStep 2: STT side receives Opus frames -> decode WAV")
    receive_start = time.perf_counter()
    received_frames = read_opus_frame_stream(frame_stream_path)
    decode_opus_frames_to_wav(received_frames, decoded_wav)
    decode_seconds = time.perf_counter() - receive_start
    print(f"Received frames: {len(received_frames)}")
    print(f"Decoded WAV for STT: {decoded_wav}")
    print(f"Receive/decode: {decode_seconds:.2f}s")

    result = {
        "input_wav": str(wav_path),
        "normalized_wav": str(normalized_wav),
        "encoded_ogg_debug": str(encoded_ogg),
        "frame_stream": str(frame_stream_path),
        "decoded_wav": str(decoded_wav),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        **frame_info,
    }

    if not args.no_stt:
        print("\nStep 3: Decoded audio -> STT")
        load_start = time.perf_counter()
        loaded = load_stt_model(config)
        load_seconds = time.perf_counter() - load_start
        print(
            f"Model ready: {loaded['engine']} {config.get('model', 'small')} "
            f"({loaded.get('device', 'auto')}) in {load_seconds:.2f}s"
        )

        stt_start = time.perf_counter()
        transcript = transcribe_wav(decoded_wav, config)
        stt_seconds = time.perf_counter() - stt_start
        print(f"STT latency: {stt_seconds:.2f}s")
        print_words(transcript)
        result.update(
            {
                "model_load_seconds": load_seconds,
                "stt_seconds": stt_seconds,
                "transcript": transcript,
            }
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
