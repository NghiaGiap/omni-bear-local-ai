from pathlib import Path
from threading import Lock
from typing import Optional
import io

from fastapi import FastAPI
from fastapi.responses import Response
from omnivoice import OmniVoice
from pydantic import BaseModel
import soundfile as sf
import torch
import uvicorn


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REF_AUDIO = BASE_DIR / "nobita_voice.wav"
DEFAULT_REF_TEXT = (
    "Đúng cái mình đang cần luôn, đã là đàn ông con trai mình không thể thua "
    "cuộc chuyến này được, đô ra ê mon ơi, dùng gương này để nhân bản ra thật "
    "nhiều bánh rán."
)

voice_prompt_cache = {}
voice_prompt_cache_lock = Lock()

print("Loading KhanhTTS-OmniVoice...")
model = OmniVoice.from_pretrained(
    "kjanh/KhanhTTS-OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16,
)
model.eval()
print("KhanhTTS ready!")


class TTSRequest(BaseModel):
    text: str
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None


def resolve_ref_audio(ref_audio: Optional[str]) -> str:
    if not ref_audio:
        return str(DEFAULT_REF_AUDIO.resolve())

    path = Path(ref_audio)
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path.resolve())


def get_voice_prompt(ref_audio: Optional[str], ref_text: Optional[str]):
    audio_path = resolve_ref_audio(ref_audio)
    text = ref_text or DEFAULT_REF_TEXT
    cache_key = (audio_path, text)

    with voice_prompt_cache_lock:
        if cache_key not in voice_prompt_cache:
            print(f"Creating voice cache: {Path(audio_path).name}")
            with torch.inference_mode():
                voice_prompt_cache[cache_key] = model.create_voice_clone_prompt(
                    ref_audio=audio_path,
                    ref_text=text,
                )

        return voice_prompt_cache[cache_key]


default_voice_prompt = get_voice_prompt(str(DEFAULT_REF_AUDIO), DEFAULT_REF_TEXT)


@app.post("/tts")
async def synthesize(req: TTSRequest):
    voice_prompt = (
        default_voice_prompt
        if not req.ref_audio and not req.ref_text
        else get_voice_prompt(req.ref_audio, req.ref_text)
    )

    with torch.inference_mode():
        audio = model.generate(text=req.text, voice_clone_prompt=voice_prompt)

    buf = io.BytesIO()
    sf.write(buf, audio[0], getattr(model, "sampling_rate", 24000), format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


@app.get("/health")
def health():
    return {"status": "ok", "voice_cache_size": len(voice_prompt_cache)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
