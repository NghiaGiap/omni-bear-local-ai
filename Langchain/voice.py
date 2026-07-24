import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parent / ".cache" / "huggingface"),
)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

VOICE_CONFIG_PATH = Path(__file__).resolve().with_name("voice_config.json")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _load_voice_config() -> dict:
    try:
        return json.loads(VOICE_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _resolve_device(value: str | None, *, input_device: bool) -> int | None:
    if not value:
        return None

    value = value.strip()
    try:
        return int(value)
    except ValueError:
        pass

    lowered = value.lower()
    for index, device in enumerate(sd.query_devices()):
        channels_key = "max_input_channels" if input_device else "max_output_channels"
        if device[channels_key] > 0 and lowered in device["name"].lower():
            return index
    return None


def _auto_select_device(*, input_device: bool) -> int | None:
    if input_device:
        preferred = (
            "headset",
            "microphone (realtek",
            "realtek hd audio mic input",
            "mic input",
            "microphone array (intel",
            "primary sound capture",
        )
        rejected = ("voice.ai", "stereo mix", "pc speaker", "speaker")
        channels_key = "max_input_channels"
    else:
        preferred = (
            "headphones",
            "headset",
            "speakers (realtek",
            "primary sound driver",
        )
        rejected = ("voice.ai",)
        channels_key = "max_output_channels"

    candidates: list[tuple[int, int]] = []
    for index, device in enumerate(sd.query_devices()):
        name = device["name"].lower()
        if device[channels_key] <= 0 or any(word in name for word in rejected):
            continue
        for rank, word in enumerate(preferred):
            if word in name:
                candidates.append((rank, index))
                break

    if candidates:
        return sorted(candidates)[0][1]

    default_index = sd.default.device[0 if input_device else 1]
    return default_index if default_index is not None and default_index >= 0 else None


def _device_label(device_index: int | None, *, input_device: bool) -> str:
    if device_index is None:
        default_index = sd.default.device[0 if input_device else 1]
        if default_index is None or default_index < 0:
            return "system default"
        device_index = default_index

    try:
        return f"{device_index}: {sd.query_devices(device_index)['name']}"
    except Exception:
        return str(device_index)


class SpeechToText:
    def __init__(self):
        self.voice_config = _load_voice_config()
        self.verbose = _env_bool("VOICE_VERBOSE", False)
        self.engine = os.getenv(
            "STT_ENGINE",
            str(self.voice_config.get("engine", "faster")),
        ).strip().lower()
        self.model_name = os.getenv(
            "STT_MODEL",
            str(self.voice_config.get("model", "small")),
        ).strip()
        self.device = os.getenv(
            "STT_DEVICE",
            str(self.voice_config.get("device", "cuda")),
        ).strip()
        self.compute_type = os.getenv(
            "STT_COMPUTE_TYPE",
            str(self.voice_config.get("compute_type", "float16")),
        ).strip()
        self.cpu_compute_type = os.getenv(
            "STT_CPU_COMPUTE_TYPE",
            str(self.voice_config.get("cpu_compute_type", "int8")),
        ).strip()
        self.beam_size = _env_int(
            "STT_BEAM_SIZE",
            int(self.voice_config.get("beam_size", 5)),
        ) or 5
        self.best_of = _env_int(
            "STT_BEST_OF",
            int(self.voice_config.get("best_of", 5)),
        ) or 5
        self.temperature = _env_float(
            "STT_TEMPERATURE",
            float(self.voice_config.get("temperature", 0.0)),
        )
        self.initial_prompt = os.getenv(
            "STT_INITIAL_PROMPT",
            str(self.voice_config.get("initial_prompt", "Tiếng Việt rõ ràng, có dấu đầy đủ.")),
        ).strip()
        self.stt_vad_filter = _env_bool(
            "STT_VAD_FILTER",
            bool(self.voice_config.get("stt_vad_filter", True)),
        )
        self.vad_min_silence_ms = _env_int(
            "STT_VAD_MIN_SILENCE_MS",
            int(self.voice_config.get("vad_min_silence_ms", 500)),
        ) or 500
        self.vad_speech_pad_ms = _env_int(
            "STT_VAD_SPEECH_PAD_MS",
            int(self.voice_config.get("vad_speech_pad_ms", 200)),
        ) or 200
        self.normalize_recording = _env_bool(
            "VOICE_NORMALIZE_RECORDING",
            bool(self.voice_config.get("normalize_recording", True)),
        )
        self.target_peak = _env_float(
            "VOICE_TARGET_PEAK",
            float(self.voice_config.get("target_peak", 0.92)),
        )
        self.max_gain = _env_float(
            "VOICE_MAX_GAIN",
            float(self.voice_config.get("max_gain", 12.0)),
        )
        input_setting = (
            self.voice_config.get("input_device")
            or os.getenv("VOICE_INPUT_DEVICE")
        )
        self.input_device = _resolve_device(
            str(input_setting) if input_setting is not None else None,
            input_device=True,
        )
        if self.input_device is None:
            self.input_device = _auto_select_device(input_device=True)
        self.sample_rate = _env_int("VOICE_SAMPLE_RATE", None)
        if self.sample_rate is None:
            device_index = self.input_device
            if device_index is None:
                device_index = sd.default.device[0]
            self.sample_rate = int(sd.query_devices(device_index)["default_samplerate"])

        self.model = None
        self.backend = None
        self.word_log = _env_bool(
            "VOICE_WORD_LOG",
            bool(self.voice_config.get("word_log", True)),
        )
        self.level_log = _env_bool(
            "VOICE_LEVEL_LOG",
            bool(self.voice_config.get("level_log", True)),
        )
        print(f"🎙️ Mic input: {_device_label(self.input_device, input_device=True)}")
        print(f"🎙️ Sample rate: {self.sample_rate} Hz")
        self._load_model()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _cache_dir(self, key: str, default_value: str) -> Path:
        value = self.voice_config.get(key) or self.voice_config.get("model_cache_dir") or default_value
        path = Path(str(value))
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_phowhisper_model_name(self) -> str:
        configured = os.getenv("PHOWHISPER_MODEL", self.model_name).strip()
        aliases = {
            "tiny": "vinai/PhoWhisper-tiny",
            "base": "vinai/PhoWhisper-base",
            "small": "vinai/PhoWhisper-small",
            "medium": "vinai/PhoWhisper-medium",
            "large": "vinai/PhoWhisper-large",
        }
        return aliases.get(configured.lower(), configured)

    def _load_phowhisper(self) -> bool:
        try:
            import torch
            from transformers import pipeline
        except Exception as exc:
            print(f"⚠️ Chưa cài đủ PhoWhisper/transformers: {exc}")
            print("↪️ Chạy: .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt")
            return False

        model_id = self._resolve_phowhisper_model_name()
        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        torch_dtype = torch.float16 if use_cuda else torch.float32
        device = 0 if use_cuda else -1

        try:
            print(f"🎤 STT: PhoWhisper {model_id} ({'cuda' if use_cuda else 'cpu'})")
            self.model = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=device,
            )
            self.backend = "phowhisper"
            print("✅ STT tiếng Việt sẵn sàng")
            return True
        except Exception as exc:
            print(f"⚠️ Không load được PhoWhisper {model_id}: {exc}")
            return False

    def _load_model(self) -> None:
        fallback_to_faster = False
        if self.engine in {"pho", "phowhisper", "vinai"}:
            if self._load_phowhisper():
                return
            print("↪️ Thử fallback faster-whisper...")
            fallback_to_faster = True

        if self.engine == "faster" or fallback_to_faster:
            try:
                from faster_whisper import WhisperModel

                download_root = self._cache_dir("faster_cache_dir", "../STT_Test/models/faster-whisper")
                print(
                    f"🎤 STT: faster-whisper {self.model_name} "
                    f"({self.device}/{self.compute_type})"
                )
                self.model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(download_root),
                )
                self.backend = "faster"
                print("✅ STT sẵn sàng")
                return
            except Exception as exc:
                print(f"⚠️ Không load được faster-whisper GPU: {exc}")
                print("↪️ Thử fallback faster-whisper CPU/int8...")
                try:
                    from faster_whisper import WhisperModel

                    self.model = WhisperModel(
                        self.model_name,
                        device="cpu",
                        compute_type=self.cpu_compute_type,
                        download_root=str(self._cache_dir("faster_cache_dir", "../STT_Test/models/faster-whisper")),
                    )
                    self.backend = "faster"
                    print("✅ STT sẵn sàng trên CPU/int8")
                    return
                except Exception as fallback_exc:
                    print(f"⚠️ Fallback faster-whisper cũng lỗi: {fallback_exc}")

        import whisper

        fallback_model = os.getenv("OPENAI_WHISPER_MODEL", "small").strip()
        print(f"🎤 STT: openai-whisper {fallback_model}")
        self.model = whisper.load_model(
            fallback_model,
            download_root=str(self._cache_dir("whisper_cache_dir", "../STT_Test/models/whisper")),
        )
        self.backend = "openai"
        print("✅ STT sẵn sàng")

    def _record_until_silence(self) -> np.ndarray:
        threshold = _env_float(
            "VOICE_VAD_THRESHOLD",
            float(self.voice_config.get("vad_threshold", 0.0015)),
        )
        peak_threshold = _env_float(
            "VOICE_PEAK_THRESHOLD",
            float(self.voice_config.get("peak_threshold", 0.006)),
        )
        start_timeout = _env_float(
            "VOICE_START_TIMEOUT",
            float(self.voice_config.get("start_timeout", 8.0)),
        )
        silence_timeout = _env_float(
            "VOICE_SILENCE_TIMEOUT",
            float(self.voice_config.get("silence_timeout", 0.7)),
        )
        max_record_seconds = _env_float(
            "VOICE_MAX_SECONDS",
            float(self.voice_config.get("max_record_seconds", 10.0)),
        )
        chunk_seconds = _env_float(
            "VOICE_CHUNK_SECONDS",
            float(self.voice_config.get("chunk_seconds", 0.12)),
        )
        pre_roll_chunks = int(
            _env_float("VOICE_PREROLL_CHUNKS", float(self.voice_config.get("pre_roll_chunks", 3)))
        )

        chunk_size = max(1, int(self.sample_rate * chunk_seconds))
        silence_chunks_needed = max(1, int(silence_timeout / chunk_seconds))
        start_chunks_limit = max(1, int(start_timeout / chunk_seconds))
        max_chunks = max(1, int(max_record_seconds / chunk_seconds))

        print("🎤 Tớ đang nghe đây...")

        chunks: list[np.ndarray] = []
        pre_roll: list[np.ndarray] = []
        started = False
        silent_chunks = 0
        max_rms = 0.0
        max_peak = 0.0
        level_log_every = max(1, int(0.6 / chunk_seconds))

        for chunk_index in range(max_chunks):
            chunk = sd.rec(
                chunk_size,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.input_device,
            )
            sd.wait()
            chunk = np.nan_to_num(np.asarray(chunk, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            peak = float(np.max(np.abs(chunk)))
            max_rms = max(max_rms, rms)
            max_peak = max(max_peak, peak)
            self._log(f"  mic rms={rms:.4f}")
            if self.level_log and chunk_index % level_log_every == 0:
                print(
                    f"🎚️ Mic level rms={rms:.4f} peak={peak:.4f} "
                    f"(ngưỡng rms={threshold:.4f}, peak={peak_threshold:.4f})"
                )

            if not started:
                pre_roll.append(chunk)
                if len(pre_roll) > pre_roll_chunks:
                    pre_roll.pop(0)

                if rms >= threshold or peak >= peak_threshold:
                    started = True
                    print(f"✅ Mic đã nhận giọng nói (rms={rms:.4f}, peak={peak:.4f})")
                    chunks.extend(pre_roll)
                    pre_roll.clear()
                elif chunk_index >= start_chunks_limit:
                    print("⚠️ Tớ chưa nghe thấy giọng nói rõ.")
                    print(
                        f"   Mic cao nhất: rms={max_rms:.4f}, peak={max_peak:.4f}; "
                        f"ngưỡng rms={threshold:.4f}, peak={peak_threshold:.4f}; "
                        f"đang dùng {_device_label(self.input_device, input_device=True)}"
                    )
                    return np.empty((0, 1), dtype=np.float32)
                continue

            chunks.append(chunk)
            if rms < threshold and peak < peak_threshold:
                silent_chunks += 1
                if silent_chunks >= silence_chunks_needed:
                    break
            else:
                silent_chunks = 0

        if not chunks:
            return np.empty((0, 1), dtype=np.float32)
        audio = np.concatenate(chunks, axis=0)
        duration = audio.shape[0] / self.sample_rate
        print(f"✅ Đã thu {duration:.2f}s âm thanh")
        return audio

    def _prepare_audio_for_stt(self, audio: np.ndarray) -> np.ndarray:
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0 or not self.normalize_recording:
            return audio

        audio = audio - float(np.mean(audio))
        peak = float(np.max(np.abs(audio)))
        if peak <= 0:
            return audio

        gain = min(self.target_peak / peak, self.max_gain)
        if gain > 1.0:
            audio = audio * gain
        return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)

    def _transcribe_file(self, path: str) -> str:
        if self.backend == "phowhisper":
            result = self.model(
                path,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "vi",
                    "num_beams": self.beam_size,
                },
            )
            return str(result.get("text", "")).strip()

        if self.backend == "faster":
            vad_parameters = {
                "min_silence_duration_ms": self.vad_min_silence_ms,
                "speech_pad_ms": self.vad_speech_pad_ms,
            }
            try:
                segments, _info = self.model.transcribe(
                    path,
                    language="vi",
                    task="transcribe",
                    beam_size=self.beam_size,
                    best_of=self.best_of,
                    temperature=self.temperature,
                    vad_filter=self.stt_vad_filter,
                    vad_parameters=vad_parameters,
                    condition_on_previous_text=False,
                    initial_prompt=self.initial_prompt or None,
                )
            except Exception as exc:
                if self.device != "cuda":
                    raise
                print(f"âš ï¸ faster-whisper CUDA lá»—i khi cháº¡y, fallback CPU/int8: {exc}")
                from faster_whisper import WhisperModel

                self.model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type=self.cpu_compute_type,
                    download_root=str(self._cache_dir("faster_cache_dir", "../STT_Test/models/faster-whisper")),
                )
                self.device = "cpu"
                segments, _info = self.model.transcribe(
                    path,
                    language="vi",
                    task="transcribe",
                    beam_size=self.beam_size,
                    best_of=self.best_of,
                    temperature=self.temperature,
                    vad_filter=self.stt_vad_filter,
                    vad_parameters=vad_parameters,
                    condition_on_previous_text=False,
                    initial_prompt=self.initial_prompt or None,
                )
            return " ".join(segment.text.strip() for segment in segments).strip()

        transcribe_options = {
            "language": "vi",
            "task": "transcribe",
            "fp16": self.device == "cuda",
            "condition_on_previous_text": False,
            "temperature": self.temperature,
            "initial_prompt": self.initial_prompt or None,
        }
        if self.temperature == 0.0:
            transcribe_options["beam_size"] = self.beam_size
        else:
            transcribe_options["best_of"] = self.best_of
        result = self.model.transcribe(path, **transcribe_options)
        return result["text"].strip()

    def _print_word_log(self, text: str) -> None:
        if not self.word_log:
            return

        words = re.findall(r"\S+", text)
        if not words:
            print("🧩 STT chưa tách được từ nào.")
            return

        print("🧩 STT từng từ:")
        for index, word in enumerate(words, 1):
            print(f"  word[{index:02d}] {word}")

    def listen(self, seconds=None) -> str:
        audio = self._record_until_silence()
        if audio.size == 0:
            return ""

        print("🔎 Đang nhận diện...")
        tmp_path = ""
        try:
            audio = self._prepare_audio_for_stt(audio)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                sf.write(tmp_path, audio, self.sample_rate)

            text = self._transcribe_file(tmp_path)
            print(f"📝 Cậu nói: {text or '(trống)'}")
            self._print_word_log(text)
            return text
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass


class TextToSpeech:
    def __init__(self, server_url="http://localhost:5001", ref_audio=None, ref_text=None):
        self.voice_config = _load_voice_config()
        self.server_url = server_url
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.verbose = _env_bool("VOICE_VERBOSE", False)
        output_setting = (
            self.voice_config.get("output_device")
            or os.getenv("VOICE_OUTPUT_DEVICE")
        )
        self.output_device = _resolve_device(
            str(output_setting) if output_setting is not None else None,
            input_device=False,
        )
        if self.output_device is None:
            self.output_device = _auto_select_device(input_device=False)
        print(f"🔊 Audio output: {_device_label(self.output_device, input_device=False)}")

        try:
            res = requests.get(f"{server_url}/health", timeout=3)
            if res.json().get("status") == "ok":
                print("✅ KhanhTTS sẵn sàng")
                if ref_audio and self.verbose:
                    print(f"🎙️ Dùng giọng custom: {ref_audio}")
        except Exception:
            print("⚠️ Chưa kết nối được KhanhTTS. Hãy chạy khanh_tts_server.py trước.")

    def speak(self, text: str):
        clean = re.sub(
            r"[^\w\s,.!?áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]",
            "",
            text,
        )

        try:
            res = requests.post(
                f"{self.server_url}/tts",
                json={
                    "text": clean,
                    "ref_audio": self.ref_audio,
                    "ref_text": self.ref_text,
                },
                timeout=30,
            )
            res.raise_for_status()
            audio_bytes = io.BytesIO(res.content)
            data, samplerate = sf.read(audio_bytes)
            sd.play(data, samplerate=samplerate, device=self.output_device)
            sd.wait()
        except Exception as e:
            print(f"❌ Lỗi TTS: {e}")
