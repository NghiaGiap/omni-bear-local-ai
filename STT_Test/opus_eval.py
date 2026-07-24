import argparse
import json
import time
from pathlib import Path

from stt_tester import load_config, load_stt_model, print_words, transcribe_wav


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "test_cases.json"
TEMP_DIR = BASE_DIR / "opus_temp"


def load_cases(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        raise ValueError("test_cases.json must contain a list")
    return data


def transcribe_any(wav_path: Path, config: dict, engine: str, model: str) -> str:
    local_config = dict(config)
    local_config["engine"] = engine
    local_config["model"] = model
    return transcribe_wav(wav_path, local_config)


def score_wer(reference: str, hypothesis: str) -> float | None:
    if not reference:
        return None
    try:
        from jiwer import wer
    except Exception:
        return None
    return float(wer(reference, hypothesis))


def run_one(
    wav_path: Path,
    reference: str,
    config: dict,
    engine: str,
    model: str,
) -> dict:
    from opus_utils import opus_roundtrip_wav

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    decoded_wav = TEMP_DIR / f"{wav_path.stem}_decoded.wav"

    start_roundtrip = time.perf_counter()
    decoded_wav, frame_count = opus_roundtrip_wav(wav_path, decoded_wav)
    roundtrip_seconds = time.perf_counter() - start_roundtrip

    start_stt = time.perf_counter()
    hypothesis = transcribe_any(decoded_wav, config, engine, model)
    stt_seconds = time.perf_counter() - start_stt

    wer_score = score_wer(reference, hypothesis)
    return {
        "file": str(wav_path),
        "decoded_wav": str(decoded_wav),
        "opus_frames": frame_count,
        "reference": reference,
        "hypothesis": hypothesis,
        "wer": wer_score,
        "roundtrip_seconds": roundtrip_seconds,
        "stt_seconds": stt_seconds,
    }


def print_result(result: dict) -> None:
    print(f"\nFile: {result['file']}")
    print(f"Decoded WAV: {result['decoded_wav']}")
    print(f"Opus frames: {result['opus_frames']}")
    print(f"Opus round-trip: {result['roundtrip_seconds']:.2f}s")
    print(f"STT latency: {result['stt_seconds']:.2f}s")
    if result["reference"]:
        print(f"Reference: {result['reference']}")
    print_words(result["hypothesis"])
    if result["wer"] is not None:
        print(f"WER: {result['wer']:.2%}")
    elif result["reference"]:
        print("WER: unavailable. Install jiwer to compute WER.")


def summarize(results: list[dict]) -> None:
    if not results:
        return
    avg_latency = sum(item["stt_seconds"] for item in results) / len(results)
    latencies = sorted(item["stt_seconds"] for item in results)
    p95_index = min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))
    print("\n=== Summary ===")
    print(f"Cases: {len(results)}")
    print(f"Average STT latency: {avg_latency:.2f}s")
    print(f"P95 STT latency: {latencies[p95_index]:.2f}s")

    wer_values = [item["wer"] for item in results if item["wer"] is not None]
    if wer_values:
        avg_wer = sum(wer_values) / len(wer_values)
        print(f"Average WER: {avg_wer:.2%}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", help="Single WAV/audio file to test")
    parser.add_argument("--text", default="", help="Ground truth transcript for --wav")
    parser.add_argument("--cases", default=str(CASES_PATH), help="JSON list of test cases")
    parser.add_argument(
        "--engine",
        choices=["phowhisper", "faster", "whisper"],
        default=None,
        help="STT engine to use",
    )
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--json-out", help="Save detailed results JSON")
    args = parser.parse_args()

    config = load_config()
    engine = args.engine or str(config.get("engine", "phowhisper"))
    model = args.model or str(config.get("model", "small"))
    run_config = dict(config)
    run_config["engine"] = engine
    run_config["model"] = model

    jobs: list[tuple[Path, str]] = []
    if args.wav:
        jobs.append((Path(args.wav), args.text))
    else:
        for case in load_cases(Path(args.cases)):
            wav_file = case.get("file") or case.get("wav")
            if wav_file:
                jobs.append((Path(wav_file), str(case.get("text", ""))))

    if not jobs:
        print("No test cases. Use --wav file.wav or edit STT_Test/test_cases.json.")
        return 1

    load_start = time.perf_counter()
    loaded = load_stt_model(run_config)
    print(
        f"Model ready: {loaded['engine']} {model} "
        f"({loaded.get('device', 'auto')}) in {time.perf_counter() - load_start:.2f}s"
    )

    results = []
    for wav_path, reference in jobs:
        if not wav_path.is_absolute():
            wav_path = (Path.cwd() / wav_path).resolve()
        result = run_one(wav_path, reference, run_config, engine, model)
        print_result(result)
        results.append(result)

    summarize(results)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
