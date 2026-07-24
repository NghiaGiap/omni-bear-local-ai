# Omni Bear Local AI

Local voice AI server for the OmniBear prototype.

This repository contains the local pipeline used to connect:

- ESP32 Xiaozhi firmware over OTA + WebSocket
- Opus audio frames
- Vietnamese STT
- Local LangChain/Ollama AI response
- KhanhTTS speech output
- Supabase `global_configs` runtime configuration

## Main Folders

- `Langchain/` - local AI chain, OmniBear Supabase config reader, voice utilities
- `STT_Test/` - STT tests, ESP32 WebSocket server, Opus frame tooling
- `Khanhtts/` - local TTS server scripts
- `files/` - small auxiliary test files

## Runtime Config

Secrets are intentionally loaded from environment variables and are not committed.

Required for Supabase-backed OmniBear config:

```powershell
$env:OMNIBEAR_SUPABASE_URL="https://rmvfestzknjhkhwbzsxv.supabase.co"
$env:OMNIBEAR_SUPABASE_SERVICE_ROLE_KEY="<service-role-key>"
$env:OMNIBEAR_SUPABASE_CONFIG_TABLE="global_configs"
$env:OMNIBEAR_SUPABASE_FILTER="limit=1"
```

The local AI currently reads only these fields from `global_configs.value`:

- `teddyPrompt`
- `ageRange`
- `voiceTone`

## Smoke Checks

```powershell
.\Langchain\venv\Scripts\python.exe -m py_compile Langchain\omnibear_config.py Langchain\bear_chain.py STT_Test\esp32_voice_server.py
.\Langchain\venv\Scripts\python.exe Langchain\test_omnibear_config.py
```

See `Langchain/README.md` and `STT_Test/README.md` for full run commands.
