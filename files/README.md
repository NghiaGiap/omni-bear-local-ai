# 🐻 Gấu Bông AI — Speech Filter Pipeline

Pipeline lọc speech gồm **4 model nhỏ** chạy tuần tự qua LangChain + Ollama.

## Kiến trúc

```
Câu nói của trẻ
      │
      ▼
┌─────────────────────┐
│  Model 1: Cleaner   │  gemma2:2b — Chuẩn hóa, sửa lỗi, loại ký tự rác
└─────────┬───────────┘
          │ cleaned_text
          ▼
┌─────────────────────┐   ┌─────────────────────┐
│  Model 2: Topic     │   │  Model 3: Language   │  (chạy song song*)
│  phi3:mini          │   │  phi3:mini           │
│  → chủ đề nhạy cảm │   │  → từ ngữ thô tục    │
└─────────┬───────────┘   └──────────┬──────────┘
          └──────────┬───────────────┘
                     ▼
          ┌─────────────────────┐
          │  Model 4: Decision  │  gemma2:2b — Tổng hợp → PASS / BLOCK
          └─────────────────────┘
```

(*) Demo này chạy tuần tự cho dễ đọc; có thể dùng `asyncio.gather` để song song hóa.

## Cài đặt

### Bước 1: Cài Ollama
Tải tại https://ollama.com và cài đặt.

### Bước 2: Pull các model
```bash
ollama pull gemma2:2b
ollama pull phi3:mini
```

### Bước 3: Cài Python dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Chạy demo
```bash
python main.py
```

## Tùy chỉnh

Để đổi model, sửa các biến ở đầu `main.py`:
```python
MODEL_CLEANER   = "gemma2:2b"   # hoặc "llama3.2:1b"
MODEL_TOPIC     = "phi3:mini"   # hoặc "mistral:7b"
MODEL_LANGUAGE  = "phi3:mini"
MODEL_DECISION  = "gemma2:2b"
```

## Tích hợp vào dự án

```python
from main import TeddyFilterPipeline

pipeline = TeddyFilterPipeline()
result = pipeline.run("câu nói của trẻ")

if result["decision"]["decision"] == "PASS":
    # Gửi cleaned text vào AI chính của gấu bông
    send_to_teddy_ai(result["cleaned"])
else:
    # Trả về safe_response cho trẻ
    speak_to_child(result["decision"]["safe_response"])
```
