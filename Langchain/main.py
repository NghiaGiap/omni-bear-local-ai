import asyncio
import time
import sys
from pathlib import Path

from bear_chain import BearAIPipeline
from voice import SpeechToText, TextToSpeech

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
REF_AUDIO = ROOT_DIR / "Khanhtts" / "nobita_voice.wav"

async def main():
    print("🔄 Đang khởi động Gấu Bông AI...")

    pipeline = BearAIPipeline()

    stt = SpeechToText()

    tts = TextToSpeech(
        server_url="http://localhost:5001",
        ref_audio=str(REF_AUDIO),         # đường dẫn tuyệt đối để KhanhTTS đọc được
        ref_text="Đúng cái mình đang cần luôn, đã là đàn ông con trai mình không thể thua cuộc chuyến này được, đô ra ê mon ơi, dùng gương này để nhân bản ra thật nhiều bánh rán."  # ← nội dung trong file wav
    )

    print("\n🐻 Sẵn sàng!")
    print("  Enter trống: nói qua mic | Gõ chữ: chat text | quit: thoát")
    print("="*50)

    while True:
        try:
            user_input = input("\nBạn (Enter=mic / gõ=text): ").strip()
        except EOFError:
            break

        if user_input.lower() == "quit":
            tts.speak("Tạm biệt cậu nhé!")
            print("🐻 Tạm biệt! ✨")
            break

        if not user_input:
            stt_start = time.perf_counter()
            user_input = stt.listen(seconds=5)
            print(f"⏱️ STT: {time.perf_counter() - stt_start:.2f}s")
            if not user_input:
                print("❌ Chưa nghe rõ, cậu thử lại nhé.")
                continue

        ai_start = time.perf_counter()
        response = await pipeline.process(user_input)
        print(f"⏱️ AI response: {time.perf_counter() - ai_start:.2f}s")
        print(f"\n🐻 Gấu Bông: {response}")
        tts_start = time.perf_counter()
        tts.speak(response)
        print(f"⏱️ TTS + playback: {time.perf_counter() - tts_start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
