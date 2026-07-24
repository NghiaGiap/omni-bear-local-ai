import argparse
import json
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

CONFIG_PATH = Path(__file__).resolve().with_name("voice_config.json")


def list_devices() -> None:
    print(f"Default device [input, output]: {sd.default.device}")
    print("\nInput devices:")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            print(
                f"  {index:>2}: {device['name']} "
                f"(channels={device['max_input_channels']}, "
                f"default_sr={int(device['default_samplerate'])})"
            )


def record_level(device_index: int, seconds: float) -> None:
    device = sd.query_devices(device_index)
    sample_rate = int(device["default_samplerate"])
    print(f"\nTesting input {device_index}: {device['name']}")
    print(f"Speak into this mic for {seconds:.1f}s...")
    time.sleep(0.5)

    audio = sd.rec(
        int(sample_rate * seconds),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))

    print(f"RMS : {rms:.5f}")
    print(f"Peak: {peak:.5f}")
    if peak < 0.003:
        print("Result: very low signal. This is probably not the active mic.")
    elif peak < 0.030:
        print("Result: weak signal. Try lowering VOICE_VAD_THRESHOLD or moving closer.")
    else:
        print("Result: good signal. Use this device id as VOICE_INPUT_DEVICE.")


def save_input_device(device_index: int) -> None:
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config = {}

    config["input_device"] = device_index
    config.setdefault("output_device", "Headphones")
    config.setdefault("vad_threshold", 0.003)
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved input_device={device_index} to {CONFIG_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("device", nargs="?", type=int, help="Input device id to test")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--save", action="store_true", help="Save this input device")
    args = parser.parse_args()

    list_devices()
    if args.device is None:
        print("\nUsage: python audio_check.py <device_id>")
        return 0

    record_level(args.device, args.seconds)
    if args.save:
        save_input_device(args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
