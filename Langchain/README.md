# Gấu Bông AI LangChain

Thư mục này là lớp điều phối AI local cho gấu bông. Luồng hiện tại dùng Ollama ở
`http://localhost:11434` và chia tác vụ cho nhiều model nhỏ:

1. Cleaner: chuẩn hóa nhẹ câu từ STT. Mặc định dùng rule để không làm mất ngữ cảnh; có thể bật LLM cleaner bằng `BEAR_USE_LLM_CLEANER=1`.
2. Topic filter: lọc chủ đề nguy hiểm hoặc không phù hợp trẻ em.
3. Language filter: lọc chửi thề, xúc phạm, bắt nạt.
4. Sentiment: đoán cảm xúc để trả lời mềm hơn.
5. Answer model: gấu bông trả lời có memory ngắn hạn.

## Chạy local AI

```powershell
ollama pull gemma2:2b
ollama pull phi3:mini
ollama serve
```

Kiểm tra Ollama:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

## Chạy TTS

Mở terminal ở thư mục `Khanhtts` rồi chạy:

```powershell
.\venv\Scripts\python.exe khanh_tts_server.py
```

Khi server sẵn sàng, endpoint này phải trả `{"status":"ok"}`:

```powershell
Invoke-RestMethod http://localhost:5001/health
```

## Chạy gấu bông

Mở terminal ở thư mục `Langchain` rồi chạy:

```powershell
.\venv\Scripts\python.exe main.py
```

## Cấu hình model

Có thể set biến môi trường trước khi chạy:

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:BEAR_MODEL="gemma2:2b"
$env:BEAR_FALLBACK_MODEL="phi3:mini"
$env:BEAR_CLEANER_MODEL="gemma2:2b"
$env:BEAR_TOPIC_MODEL="phi3:mini"
$env:BEAR_LANGUAGE_MODEL="phi3:mini"
$env:BEAR_SENTIMENT_MODEL="phi3:mini"
$env:BEAR_MEMORY_TURNS="5"
$env:BEAR_NUM_CTX="1024"
$env:BEAR_NUM_PREDICT="96"
$env:BEAR_USE_LLM_CLEANER="0"
$env:BEAR_USE_LLM_FILTERS="0"
$env:BEAR_USE_LLM_SENTIMENT="0"
$env:BEAR_VERBOSE="0"
$env:VOICE_VERBOSE="0"
$env:STT_ENGINE="phowhisper"
$env:STT_MODEL="small"
$env:STT_DEVICE="cuda"
$env:STT_COMPUTE_TYPE="float16"
```

Nếu muốn đổi sang model local khác, chỉ cần model đó có trong `ollama list` và
sửa các biến tương ứng.

Mặc định topic/language/sentiment filter dùng rule-based tiếng Việt để phản hồi nhanh
và tránh việc model nhỏ hiểu sai câu an toàn. Chỉ bật `BEAR_USE_LLM_FILTERS=1` hoặc
`BEAR_USE_LLM_SENTIMENT=1` khi cần thử chế độ kiểm tra nghiêm hơn.

Mặc định STT dùng PhoWhisper để ưu tiên nhận diện tiếng Việt rõ và chuẩn:

```powershell
$env:STT_ENGINE="phowhisper"
$env:STT_MODEL="small"
```

Có thể nâng lên `medium` khi muốn chính xác hơn và còn đủ VRAM:

```powershell
$env:STT_MODEL="medium"
```

## OmniBear DB config cho AI local

AI local co the doc config cua OmniBear truoc khi tra loi. ESP32 khong can biet
token nao; token chi dat tren may chay server local.

Cach nen dung neu backend OmniBear da co API nhu mobile app:

```powershell
$env:OMNIBEAR_API_URL="https://omni-bear-api-production.up.railway.app/api"
$env:OMNIBEAR_ACCESS_TOKEN="<parent-or-admin-access-token>"
$env:OMNIBEAR_TEDDY_ID="<teddy-id>"
```

Neu chua biet `teddy-id` nhung biet `deviceId` cua ESP32:

```powershell
$env:OMNIBEAR_DEVICE_ID="<esp32-device-id>"
```

Neu muon doc truc tiep Supabase REST sau khi biet table trong dashboard:

```powershell
$env:OMNIBEAR_SUPABASE_URL="https://rmvfestzknjhkhwbzsxv.supabase.co"
$env:OMNIBEAR_SUPABASE_SERVICE_ROLE_KEY="<service-role-key-only-on-local-server>"
$env:OMNIBEAR_SUPABASE_CONFIG_TABLE="<table-name>"
$env:OMNIBEAR_SUPABASE_FILTER="scope=eq.global&is_active=eq.true"
```

Voi table `global_configs`, AI local chi quan tam den `value.teddyPrompt`,
`value.ageRange`, va `value.voiceTone`. Cac field khac nhu `language`,
`updated_at`, `updated_by` se khong dua vao prompt.

Config duoc cache mac dinh 60 giay de khong lam cham AI response:

```powershell
$env:OMNIBEAR_CONFIG_CACHE_SECONDS="60"
$env:OMNIBEAR_CONFIG_TIMEOUT_SECONDS="2"
```

Tat doc DB config neu can debug:

```powershell
$env:BEAR_USE_OMNIBEAR_CONFIG="0"
```

## Cấu hình STT cho RTX 4050

`voice.py` mặc định dùng `vinai/PhoWhisper-small`. Nếu muốn quay lại `faster-whisper`
để ưu tiên tốc độ, có thể set:

```powershell
$env:STT_ENGINE="faster"
$env:STT_MODEL="small"
$env:STT_DEVICE="cuda"
$env:STT_COMPUTE_TYPE="float16"
```

Đây là cấu hình nhanh và cân bằng cho RTX 4050 Laptop 6GB vì KhanhTTS cũng đang dùng GPU.
Nếu dùng PhoWhisper và muốn tiếng Việt chính xác hơn, có thể thử:

```powershell
$env:STT_MODEL="medium"
```

Mic/output mặc định được đọc từ `voice_config.json`, nên bình thường chỉ cần chạy
`main.py`. Nếu auto chọn sai mic, kiểm tra mic nào đang có tín hiệu:

```powershell
.\venv\Scripts\python.exe audio_check.py
.\venv\Scripts\python.exe audio_check.py 25
```

Thiết bị nào báo `good signal` thì lưu một lần:

```powershell
.\venv\Scripts\python.exe audio_check.py 25 --save
```

Lần chạy sau `main.py` sẽ tự dùng mic đã lưu, không cần nhập `$env`.
