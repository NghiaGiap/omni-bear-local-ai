import wave
import struct
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


# Xiaozhi/ESP32 audio format: 16 kHz, mono, 60 ms Opus frames.
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SIZE = 960
SAMPLE_WIDTH = 2
OPUS_FRAME_STREAM_MAGIC = b"OPUSFRM1"


def _load_opuslib():
    import opuslib

    return opuslib


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = audio.shape[0] / source_rate
    source_times = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_count = max(1, int(round(duration * target_rate)))
    target_times = np.linspace(0.0, duration, num=target_count, endpoint=False)
    return np.interp(target_times, source_times, audio).astype(np.float32)


def normalize_wav_for_opus(input_path: str | Path, output_path: str | Path) -> Path:
    """Convert an arbitrary readable audio file to 16 kHz mono 16-bit WAV."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    audio, sample_rate = sf.read(input_path, always_2d=False)
    audio = _to_mono_float32(audio)
    audio = _resample_linear(audio, int(sample_rate), SAMPLE_RATE)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")
    return output_path


def decode_opus_frames(opus_frames: list[bytes]) -> bytes:
    opuslib = _load_opuslib()
    decoder = opuslib.Decoder(SAMPLE_RATE, CHANNELS)
    pcm_chunks = []
    for frame in opus_frames:
        pcm = decoder.decode(frame, FRAME_SIZE)
        pcm_chunks.append(pcm)
    return b"".join(pcm_chunks)


def save_wav(pcm_data: bytes, path: str | Path) -> Path:
    path = Path(path)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    return path


def encode_to_opus_frames(wav_path: str | Path) -> list[bytes]:
    opuslib = _load_opuslib()
    encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"WAV must be {SAMPLE_RATE} Hz")
        if wf.getnchannels() != CHANNELS:
            raise ValueError("WAV must be mono")
        if wf.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError("WAV must be 16-bit PCM")

        frames = []
        while True:
            raw = wf.readframes(FRAME_SIZE)
            if len(raw) < FRAME_SIZE * SAMPLE_WIDTH:
                break
            frames.append(encoder.encode(raw, FRAME_SIZE))
        return frames


def _run_ffmpeg(args: list[str]) -> None:
    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg failed")


def _read_ogg_packets(path: str | Path) -> list[bytes]:
    data = Path(path).read_bytes()
    offset = 0
    packets: list[bytes] = []
    current = bytearray()

    while offset < len(data):
        if data[offset : offset + 4] != b"OggS":
            raise ValueError(f"Invalid Ogg page at byte {offset}")

        segment_count = data[offset + 26]
        segment_start = offset + 27
        segment_end = segment_start + segment_count
        segments = data[segment_start:segment_end]
        payload_start = segment_end
        payload_end = payload_start + sum(segments)
        payload = data[payload_start:payload_end]

        cursor = 0
        for segment_size in segments:
            current.extend(payload[cursor : cursor + segment_size])
            cursor += segment_size
            if segment_size < 255:
                packets.append(bytes(current))
                current.clear()

        offset = payload_end

    if current:
        packets.append(bytes(current))
    return packets


def _ogg_crc(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc


def _ogg_lacing(packet: bytes) -> bytes:
    remaining = len(packet)
    segments = []
    while remaining >= 255:
        segments.append(255)
        remaining -= 255
    segments.append(remaining)
    return bytes(segments)


def _write_ogg_page(
    output,
    packet: bytes,
    *,
    serial: int,
    sequence: int,
    granule_position: int,
    header_type: int,
) -> None:
    lacing = _ogg_lacing(packet)
    header = (
        b"OggS"
        + bytes([0, header_type])
        + struct.pack("<QIIIB", granule_position, serial, sequence, 0, len(lacing))
        + lacing
    )
    page = header + packet
    checksum = _ogg_crc(page)
    header = header[:22] + struct.pack("<I", checksum) + header[26:]
    output.write(header + packet)


def default_opus_headers() -> list[bytes]:
    opus_head = (
        b"OpusHead"
        + bytes([1, CHANNELS])
        + struct.pack("<HIhB", 312, SAMPLE_RATE, 0, 0)
    )
    vendor = b"STT_Test"
    opus_tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)
    return [opus_head, opus_tags]


def write_ogg_opus(
    opus_frames: list[bytes],
    output_path: str | Path,
    header_packets: list[bytes] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = header_packets or default_opus_headers()
    serial = 0x53545431
    sequence = 0
    granule_position = 0
    granule_step = int(48000 * FRAME_SIZE / SAMPLE_RATE)

    with output_path.open("wb") as output:
        _write_ogg_page(
            output,
            headers[0],
            serial=serial,
            sequence=sequence,
            granule_position=0,
            header_type=2,
        )
        sequence += 1
        _write_ogg_page(
            output,
            headers[1],
            serial=serial,
            sequence=sequence,
            granule_position=0,
            header_type=0,
        )
        sequence += 1

        for index, frame in enumerate(opus_frames):
            granule_position += granule_step
            header_type = 4 if index == len(opus_frames) - 1 else 0
            _write_ogg_page(
                output,
                frame,
                serial=serial,
                sequence=sequence,
                granule_position=granule_position,
                header_type=header_type,
            )
            sequence += 1

    return output_path


def encode_wav_to_opus_frames(
    input_wav: str | Path,
    work_dir: str | Path,
) -> tuple[list[bytes], list[bytes], Path, Path]:
    """Normalize a WAV/audio file and return raw Opus packets like ESP32 frames."""
    input_wav = Path(input_wav)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    normalized_wav = work_dir / f"{input_wav.stem}_normalized.wav"
    opus_ogg = work_dir / f"{input_wav.stem}_encoded.ogg"

    normalize_wav_for_opus(input_wav, normalized_wav)

    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(normalized_wav),
            "-c:a",
            "libopus",
            "-application",
            "voip",
            "-frame_duration",
            "60",
            "-b:a",
            "24k",
            "-vbr",
            "off",
            str(opus_ogg),
        ]
    )

    packets = _read_ogg_packets(opus_ogg)
    if len(packets) < 3:
        raise ValueError("Ogg Opus output did not contain audio frames")
    return packets[2:], packets[:2], normalized_wav, opus_ogg


def write_opus_frame_stream(opus_frames: list[bytes], output_path: str | Path) -> Path:
    """Write length-prefixed Opus frames, similar to frames received from ESP32."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        output.write(OPUS_FRAME_STREAM_MAGIC)
        output.write(struct.pack("<III", SAMPLE_RATE, CHANNELS, len(opus_frames)))
        for frame in opus_frames:
            output.write(struct.pack("<I", len(frame)))
            output.write(frame)
    return output_path


def read_opus_frame_stream(path: str | Path) -> list[bytes]:
    data = Path(path).read_bytes()
    if data[: len(OPUS_FRAME_STREAM_MAGIC)] != OPUS_FRAME_STREAM_MAGIC:
        raise ValueError("Invalid Opus frame stream")
    offset = len(OPUS_FRAME_STREAM_MAGIC)
    sample_rate, channels, frame_count = struct.unpack_from("<III", data, offset)
    offset += 12
    if sample_rate != SAMPLE_RATE or channels != CHANNELS:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz mono Opus frames")

    frames = []
    for _ in range(frame_count):
        frame_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        frames.append(data[offset : offset + frame_size])
        offset += frame_size
    return frames


def decode_opus_frames_to_wav(
    opus_frames: list[bytes],
    decoded_wav: str | Path,
    header_packets: list[bytes] | None = None,
) -> Path:
    decoded_wav = Path(decoded_wav)
    received_ogg = decoded_wav.with_suffix(".received.ogg")
    write_ogg_opus(opus_frames, received_ogg, header_packets=header_packets)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(received_ogg),
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(decoded_wav),
        ]
    )
    return decoded_wav


def opus_roundtrip_wav(input_wav: str | Path, decoded_wav: str | Path) -> tuple[Path, int]:
    """Normalize input, encode to Opus frames, decode frames, then save WAV."""
    decoded_wav = Path(decoded_wav)
    frames, headers, _normalized_wav, _opus_ogg = encode_wav_to_opus_frames(
        input_wav,
        decoded_wav.parent,
    )
    decode_opus_frames_to_wav(frames, decoded_wav, header_packets=headers)
    return decoded_wav, len(frames)
