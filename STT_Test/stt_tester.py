import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
os.environ.setdefault("HF_HOME", str(BASE_DIR / "models" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_MODEL_CACHE: dict[tuple, object] = {}


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_device(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    needle = str(value).lower()
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0 and needle in device["name"].lower():
            return index
    return None


def device_label(device_index: int | None) -> str:
    if device_index is None:
        device_index = sd.default.device[0]
    try:
        device = sd.query_devices(device_index)
        return f"{device_index}: {device['name']}"
    except Exception:
        return str(device_index)


def list_devices() -> None:
    print(f"Default [input, output]: {sd.default.device}")
    print("\nInput devices:")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            print(
                f"  {index:>2}: {device['name']} "
                f"(channels={device['max_input_channels']}, "
                f"default_sr={int(device['default_samplerate'])})"
            )


def scan_devices(seconds: float = 0.8) -> None:
    results = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] <= 0:
            continue
        sample_rate = int(device["default_samplerate"])
        try:
            audio = sd.rec(
                int(sample_rate * seconds),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=index,
            )
            sd.wait()
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            rms = float(np.sqrt(np.mean(np.square(audio))))
            peak = float(np.max(np.abs(audio)))
            results.append((peak, rms, index, device["name"], sample_rate, None))
        except Exception as exc:
            results.append((-1.0, -1.0, index, device["name"], sample_rate, exc))

    print("Input signal scan, sorted by peak:")
    for peak, rms, index, name, sample_rate, exc in sorted(results, reverse=True):
        if exc:
            status = f"ERROR: {exc}"
        elif peak >= 0.006 or rms >= 0.0015:
            status = "ACTIVE?"
        else:
            status = "quiet"
        print(
            f"{index:>2}: peak={peak:.6f} rms={rms:.6f} "
            f"sr={sample_rate:<5} {status:<8} {name}"
        )


def save_config(config: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def config_bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def model_cache_dir(config: dict, engine: str) -> Path:
    key = f"{engine}_cache_dir"
    value = config.get(key) or config.get("model_cache_dir") or f"models/{engine}"
    path = Path(str(value))
    if not path.is_absolute():
        path = BASE_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def decode_options(config: dict) -> dict:
    return {
        "beam_size": int(config.get("beam_size", 5)),
        "best_of": int(config.get("best_of", 5)),
        "temperature": float(config.get("temperature", 0.0)),
        "initial_prompt": str(
            config.get("initial_prompt", "Tiếng Việt rõ ràng, có dấu đầy đủ.")
        ).strip(),
    }


def normalize_audio_for_stt(audio: np.ndarray, config: dict) -> np.ndarray:
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if audio.size == 0 or not config_bool(config, "normalize_recording", True):
        return audio

    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio

    target_peak = float(config.get("target_peak", 0.92))
    max_gain = float(config.get("max_gain", 12.0))
    gain = min(target_peak / peak, max_gain)
    if gain > 1.0:
        audio = audio * gain
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def record_until_silence(config: dict) -> np.ndarray:
    input_device = resolve_device(config.get("input_device"))
    if input_device is None:
        input_device = sd.default.device[0]

    sample_rate = config.get("sample_rate")
    if not sample_rate:
        sample_rate = int(sd.query_devices(input_device)["default_samplerate"])

    vad_threshold = float(config.get("vad_threshold", 0.0015))
    peak_threshold = float(config.get("peak_threshold", 0.006))
    start_timeout = float(config.get("start_timeout", 8.0))
    silence_timeout = float(config.get("silence_timeout", 0.9))
    max_record_seconds = float(config.get("max_record_seconds", 12.0))
    chunk_seconds = float(config.get("chunk_seconds", 0.18))

    chunk_size = max(1, int(sample_rate * chunk_seconds))
    start_chunks_limit = max(1, int(start_timeout / chunk_seconds))
    silence_chunks_needed = max(1, int(silence_timeout / chunk_seconds))
    max_chunks = max(1, int(max_record_seconds / chunk_seconds))

    print(f"Input: {device_label(input_device)}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Threshold: rms={vad_threshold:.4f}, peak={peak_threshold:.4f}")
    print("Listening now...")

    chunks: list[np.ndarray] = []
    pre_roll: list[np.ndarray] = []
    started = False
    silent_chunks = 0
    max_rms = 0.0
    max_peak = 0.0
    level_every = max(1, int(0.6 / chunk_seconds))

    for chunk_index in range(max_chunks):
        chunk = sd.rec(
            chunk_size,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=input_device,
        )
        sd.wait()
        chunk = np.nan_to_num(np.asarray(chunk, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        peak = float(np.max(np.abs(chunk)))
        max_rms = max(max_rms, rms)
        max_peak = max(max_peak, peak)

        if chunk_index % level_every == 0:
            print(f"level rms={rms:.4f} peak={peak:.4f}")

        is_voice = rms >= vad_threshold or peak >= peak_threshold
        if not started:
            pre_roll.append(chunk)
            if len(pre_roll) > 3:
                pre_roll.pop(0)
            if is_voice:
                started = True
                chunks.extend(pre_roll)
                pre_roll.clear()
                print(f"Voice detected rms={rms:.4f} peak={peak:.4f}")
            elif chunk_index >= start_chunks_limit:
                print("No clear voice detected.")
                print(f"Max seen: rms={max_rms:.4f}, peak={max_peak:.4f}")
                return np.empty((0, 1), dtype=np.float32)
            continue

        chunks.append(chunk)
        if is_voice:
            silent_chunks = 0
        else:
            silent_chunks += 1
            if silent_chunks >= silence_chunks_needed:
                break

    if not chunks:
        return np.empty((0, 1), dtype=np.float32)

    audio = np.concatenate(chunks, axis=0)
    audio = normalize_audio_for_stt(audio, config)
    duration = audio.shape[0] / sample_rate
    print(f"Recorded {duration:.2f}s")

    if config.get("save_wav", True):
        wav_path = BASE_DIR / str(config.get("wav_path", "last_recording.wav"))
        sf.write(wav_path, audio, sample_rate)
        print(f"Saved WAV: {wav_path}")

    return audio


def load_stt_model(config: dict) -> dict:
    engine = str(config.get("engine", "phowhisper")).lower()
    model_name = str(config.get("model", "small"))
    device = str(config.get("device", "cuda"))
    compute_type = str(config.get("compute_type", "float16"))

    if engine in {"pho", "phowhisper", "vinai"}:
        import torch
        from transformers import pipeline

        aliases = {
            "tiny": "vinai/PhoWhisper-tiny",
            "base": "vinai/PhoWhisper-base",
            "small": "vinai/PhoWhisper-small",
            "medium": "vinai/PhoWhisper-medium",
            "large": "vinai/PhoWhisper-large",
        }
        model_id = aliases.get(model_name.lower(), model_name)
        use_cuda = device == "cuda" and torch.cuda.is_available()
        key = ("phowhisper", model_id, "cuda" if use_cuda else "cpu")
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch.float16 if use_cuda else torch.float32,
                device=0 if use_cuda else -1,
            )
        return {"engine": "phowhisper", "model": _MODEL_CACHE[key], "device": key[2]}

    if engine == "faster":
        from faster_whisper import WhisperModel

        download_root = model_cache_dir(config, "faster")
        attempts: list[tuple[str, str]] = []
        if device == "cuda":
            attempts.append(("cuda", compute_type))
        cpu_compute = str(config.get("cpu_compute_type", "int8"))
        if device == "cuda":
            attempts.append(("cpu", cpu_compute))
        elif device == "cpu":
            attempts.append(("cpu", cpu_compute if compute_type == "float16" else compute_type))
        else:
            attempts.append((device, compute_type))

        last_error: Exception | None = None
        for attempt_device, attempt_compute in attempts:
            key = ("faster", model_name, attempt_device, attempt_compute, str(download_root))
            if key in _MODEL_CACHE:
                return {
                    "engine": "faster",
                    "model": _MODEL_CACHE[key],
                    "device": attempt_device,
                    "compute_type": attempt_compute,
                }
            try:
                _MODEL_CACHE[key] = WhisperModel(
                    model_name,
                    device=attempt_device,
                    compute_type=attempt_compute,
                    download_root=str(download_root),
                )
                return {
                    "engine": "faster",
                    "model": _MODEL_CACHE[key],
                    "device": attempt_device,
                    "compute_type": attempt_compute,
                }
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Cannot load faster-whisper: {last_error}")

    if engine in {"whisper", "openai-whisper"}:
        import torch
        import whisper

        use_cuda = device == "cuda" and torch.cuda.is_available()
        cache_dir = model_cache_dir(config, "whisper")
        key = ("whisper", model_name, "cuda" if use_cuda else "cpu", str(cache_dir))
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = whisper.load_model(
                model_name,
                device="cuda" if use_cuda else "cpu",
                download_root=str(cache_dir),
            )
        return {
            "engine": "whisper",
            "model": _MODEL_CACHE[key],
            "device": key[2],
            "use_cuda": use_cuda,
        }

    raise ValueError(f"Unknown engine: {engine}")


def transcribe_wav(path: Path, config: dict) -> str:
    loaded = load_stt_model(config)
    options = decode_options(config)

    if loaded["engine"] == "phowhisper":
        result = loaded["model"](
            str(path),
            generate_kwargs={
                "task": "transcribe",
                "language": "vi",
                "num_beams": options["beam_size"],
            },
        )
        return str(result.get("text", "")).strip()

    if loaded["engine"] == "faster":
        try:
            return transcribe_faster(path, config, loaded, options)
        except Exception as exc:
            if loaded.get("device") != "cuda":
                raise
            print(f"faster-whisper CUDA failed during inference, fallback CPU/int8: {exc}")
            cpu_config = dict(config)
            cpu_config["device"] = "cpu"
            cpu_config["compute_type"] = str(config.get("cpu_compute_type", "int8"))
            cpu_loaded = load_stt_model(cpu_config)
            return transcribe_faster(path, cpu_config, cpu_loaded, options)

    if loaded["engine"] == "whisper":
        whisper_options = {
            "language": "vi",
            "task": "transcribe",
            "fp16": bool(loaded.get("use_cuda")),
            "condition_on_previous_text": False,
            "temperature": options["temperature"],
            "initial_prompt": options["initial_prompt"] or None,
        }
        if options["temperature"] == 0.0:
            whisper_options["beam_size"] = options["beam_size"]
        else:
            whisper_options["best_of"] = options["best_of"]
        result = loaded["model"].transcribe(str(path), **whisper_options)
        return str(result.get("text", "")).strip()

    raise ValueError(f"Unknown engine: {loaded['engine']}")


def transcribe_faster(path: Path, config: dict, loaded: dict, options: dict) -> str:
    vad_parameters = {
        "min_silence_duration_ms": int(config.get("vad_min_silence_ms", 500)),
        "speech_pad_ms": int(config.get("vad_speech_pad_ms", 200)),
    }
    segments, _info = loaded["model"].transcribe(
        str(path),
        language="vi",
        task="transcribe",
        beam_size=options["beam_size"],
        best_of=options["best_of"],
        temperature=options["temperature"],
        vad_filter=config_bool(config, "stt_vad_filter", True),
        vad_parameters=vad_parameters,
        condition_on_previous_text=False,
        initial_prompt=options["initial_prompt"] or None,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def print_words(text: str) -> None:
    print(f"\nTranscript: {text or '(empty)'}")
    words = re.findall(r"\S+", text)
    if not words:
        return
    print("\nWords:")
    for index, word in enumerate(words, 1):
        print(f"  word[{index:02d}] {word}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List input devices")
    parser.add_argument("--scan", action="store_true", help="Measure all input devices")
    parser.add_argument("--device", help="Override and save input device id/name")
    parser.add_argument(
        "--engine",
        choices=["phowhisper", "faster", "whisper"],
        help="Override STT engine",
    )
    parser.add_argument("--model", help="Override STT model")
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument("--transcribe", help="Transcribe an existing WAV file")
    args = parser.parse_args()

    config = load_config()

    if args.device:
        config["input_device"] = args.device
        save_config(config)
        print(f"Saved input_device={args.device} to {CONFIG_PATH}")

    if args.engine:
        config["engine"] = args.engine
        save_config(config)
    if args.model:
        config["model"] = args.model
        save_config(config)

    if args.list:
        list_devices()
        return 0

    if args.scan:
        scan_devices()
        return 0

    if args.transcribe:
        wav_path = Path(args.transcribe)
        load_start = time.perf_counter()
        load_stt_model(config)
        print(f"Model load/warmup seconds: {time.perf_counter() - load_start:.2f}")
        start = time.perf_counter()
        text = transcribe_wav(wav_path, config)
        print(f"STT seconds: {time.perf_counter() - start:.2f}")
        print_words(text)
        return 0

    audio = record_until_silence(config)
    if audio.size == 0 or args.record_only:
        return 0

    wav_path = BASE_DIR / str(config.get("wav_path", "last_recording.wav"))
    load_start = time.perf_counter()
    load_stt_model(config)
    print(f"\nModel load/warmup seconds: {time.perf_counter() - load_start:.2f}")
    start = time.perf_counter()
    text = transcribe_wav(wav_path, config)
    print(f"\nSTT seconds: {time.perf_counter() - start:.2f}")
    print_words(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
