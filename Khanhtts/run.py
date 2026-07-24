from omnivoice import OmniVoice
import soundfile as sf
import numpy as np
import torch

data, samplerate = sf.read("VietHoang.wav")
if data.ndim == 2:
    data = data.mean(axis=1)
sf.write("clone.wav", data, samplerate)
print("Đã chuyển xong stereo → mono")

model = OmniVoice.from_pretrained(
    "kjanh/KhanhTTS-OmniVoice",
)
model = model.to("cuda")        # ← giữ lại
# model.half()                  # ← bỏ dòng này đi

audio = model.generate(
    text="Tôi là biên tập viên Việt Hoàng, Phóng viên của đài truyền hình Việt Nam, As always I go to work, sometime I feel so lonely, I need someone to talk to.",
    ref_audio="clone.wav",
    ref_text="Ngay khi cô phóng viên của chúng tôi đang dẫn hiện trường về vấn đề này, thì ở phía sau lưng, vẫn có những người vô tư tụt khẩu trang rồi tụ tập trà đá vỉa hè với nhau.",
)

sf.write("output.wav", audio[0], 24000)
print("Xong! Nghe file output.wav")