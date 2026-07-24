# STT Test

Thu muc nay dung de test rieng microphone va speech-to-text, khong can chay
Langchain, Ollama hay KhanhTTS.

## Chay bang venv hien co

Tu workspace root:

```powershell
cd D:\FPT_University_Study\FPT_SUMMER2026\SWD392
.\Langchain\venv\Scripts\python.exe STT_Test\stt_tester.py --list
```

## Quet tin hieu cac mic

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\stt_tester.py --scan
```

Thiet bi nao co `peak`/`rms` cao khi ban noi la mic dang co tin hieu.

## Luu mic dung

Vi du dung device id 8:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\stt_tester.py --device 8
```

## Thu am va transcribe

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\stt_tester.py
```

Script se:

- nghe mic den khi co giong noi,
- dung khi ban im,
- luu `last_recording.wav`,
- chay STT,
- in transcript va tung tu.

Mac dinh hien tai dung `faster-whisper small` tren CPU/int8 de on dinh tren
Windows nay. Day van la Whisper local, khong dung Groq/API ben ngoai.

## Doi engine STT

PhoWhisper, uu tien tieng Viet:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\stt_tester.py --engine phowhisper --model small
```

Faster-whisper, uu tien toc do:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\stt_tester.py --engine faster --model small
```

OpenAI Whisper local:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\stt_tester.py --engine whisper --model small
```

Ghi chu toc do tren file `Khanhtts\VietHoang.wav` sau khi model da san sang:

- `faster-whisper small` CPU/int8: khoang 3.7s.
- `openai-whisper small` CPU: khoang 13.6s.

Neu can chinh xac hon nua, thu `--model medium`, nhung se cham hon.

## Chi thu am, chua transcribe

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\stt_tester.py --record-only
```

## Transcribe file da thu

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\stt_tester.py --transcribe STT_Test\last_recording.wav
```

## Test dung format Opus cua ESP32/Xiaozhi

Muc tieu cua buoc nay la gia lap audio ma firmware se gui:

- Opus
- 16 kHz
- mono
- frame 60 ms, 960 samples

### Cai package bo sung

```powershell
.\Langchain\venv\Scripts\python.exe -m pip install -r STT_Test\requirements-opus.txt
```

### Thu am mau bang mic

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\stt_tester.py --record-only
```

File se duoc luu tai:

```text
STT_Test\last_recording.wav
```

### Round-trip WAV -> Opus frames -> WAV -> STT local

PhoWhisper:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\opus_eval.py --wav STT_Test\last_recording.wav --engine phowhisper --model small
```

### Demo sat luong ESP32 frame

Lenh nay tach ro 3 buoc:

```text
WAV -> Opus frames -> server nhan frame/decode -> STT
```

Chay:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\opus_frame_demo.py --wav Khanhtts\VietHoang.wav
```

Script se tao file debug trong `STT_Test\opus_temp`:

- `*.opusframes`: cac Opus frame length-prefixed, gia lap goi ESP32 gui len.
- `*_from_received_frames.wav`: audio da decode lai tu frame de dua vao STT.
- `*_encoded.ogg`: file debug de nghe/kiem tra container Opus.

## OTA + WebSocket server cho firmware Xiaozhi

Server that su cho luong:

```text
ESP32 boot -> OTA endpoint -> server tra websocket.url
ESP32 mic -> Opus frames -> WebSocket -> STT -> AI local -> KhanhTTS -> Opus frames -> ESP32 speaker
```

Chay KhanhTTS truoc de co audio tra loi:

```powershell
cd D:\FPT_University_Study\FPT_SUMMER2026\SWD392\Khanhtts
..\Langchain\venv\Scripts\python.exe khanh_tts_server.py
```

Mo terminal khac, chay voice server:

Neu muon AI local doc OmniBear config, set cac bien nay truoc khi chay server
(ESP32 khong can token):

```powershell
$env:OMNIBEAR_API_URL="https://omni-bear-api-production.up.railway.app/api"
$env:OMNIBEAR_ACCESS_TOKEN="<parent-or-admin-access-token>"
$env:OMNIBEAR_TEDDY_ID="<teddy-id>"
```

Neu doc truc tiep Supabase cua project `omni-bear`:

```powershell
$env:OMNIBEAR_SUPABASE_URL="https://rmvfestzknjhkhwbzsxv.supabase.co"
$env:OMNIBEAR_SUPABASE_SERVICE_ROLE_KEY="<service-role-key-only-on-local-server>"
$env:OMNIBEAR_SUPABASE_CONFIG_TABLE="<global-config-table>"
```

```powershell
cd D:\FPT_University_Study\FPT_SUMMER2026\SWD392
.\Langchain\venv\Scripts\python.exe STT_Test\esp32_voice_server.py --host 127.0.0.1 --port 8765 --public-url http://untainted-helping-janitor.ngrok-free.dev --debug-save --tts-frame-delay-ms 60 --max-record-seconds 10
```

Mo terminal khac, chay ngrok:

```powershell
.\tools\ngrok\ngrok.exe http 8765 --url http://untainted-helping-janitor.ngrok-free.dev --config .\tools\ngrok\ngrok.yml
```

Firmware Xiaozhi can OTA URL, khong phai WebSocket URL:

```text
http://untainted-helping-janitor.ngrok-free.dev/xiaozhi/ota/
```

OTA response se tra WebSocket URL:

```text
ws://untainted-helping-janitor.ngrok-free.dev/xiaozhi/ws/
```

Kiem tra dung kieu firmware Xiaozhi goi OTA. Firmware dung `POST`, nen khong
chi test bang browser/GET:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/xiaozhi/ota/" -Method POST -Body "{}" -ContentType "application/json"
Invoke-RestMethod -Uri "http://untainted-helping-janitor.ngrok-free.dev/xiaozhi/ota/" -Method POST -Body "{}" -ContentType "application/json"
```

Neu chi muon test WebSocket + STT, chua can AI/TTS, van dung OTA nhu cu
nhung them `--no-ai --no-tts`:

```powershell
cd D:\FPT_University_Study\FPT_SUMMER2026\SWD392
.\Langchain\venv\Scripts\python.exe STT_Test\esp32_voice_server.py --host 127.0.0.1 --port 8765 --public-url http://untainted-helping-janitor.ngrok-free.dev --no-ai --no-tts --max-record-seconds 10
```

Protocol Xiaozhi:

- ESP32 -> server JSON `hello` voi `transport=websocket`.
- Server -> ESP32 JSON `hello`.
- ESP32 -> server JSON `listen/start`.
- ESP32 -> server binary Opus frames 16 kHz mono 60 ms.
- ESP32 -> server JSON `listen/stop`.
- Server -> ESP32 JSON `stt`, `llm`, `tts/start`, `tts/sentence_start`, `tts/stop`.
- Server -> ESP32 binary Opus frames TTS de phat loa.

Test khong can ESP32 bang client gia lap, di qua OTA truoc:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\esp32_sim_client.py --ota-url http://untainted-helping-janitor.ngrok-free.dev/xiaozhi/ota/ --wav Khanhtts\VietHoang.wav
```

faster-whisper:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\opus_eval.py --wav STT_Test\last_recording.wav --engine faster --model small
```

OpenAI Whisper local:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\opus_eval.py --wav STT_Test\last_recording.wav --engine whisper --model small
```

Neu muon chinh xac hon nhung cham hon:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\opus_eval.py --wav STT_Test\last_recording.wav --engine whisper --model medium
```

### Test co ground truth de tinh WER

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\opus_eval.py --wav STT_Test\last_recording.wav --text "gau oi ke chuyen cho toi nghe" --engine phowhisper --model small
```

Neu co nhieu mau, sua `STT_Test\test_cases.json`, moi item gom:

```json
{
  "file": "STT_Test/samples/sample1.wav",
  "text": "gau oi ke chuyen cho toi nghe"
}
```

Sau do chay:

```powershell
.\Langchain\venv\Scripts\python.exe STT_Test\opus_eval.py --cases STT_Test\test_cases.json --engine phowhisper --model small
```
