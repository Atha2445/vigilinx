# Frigate on a Windows PC without a graphics card

Docker Desktop setup that runs Frigate on the CPU (OpenVINO; works on Intel and AMD).
Ollama runs as the normal Windows app, not in Docker.

Full step-by-step instructions, including running the Vigilinx app itself:
**[SETUP_WINDOWS.md](../../../SETUP_WINDOWS.md)** (Parts 5–7 cover this folder).

| File | Purpose |
|---|---|
| `docker-compose.yml` | Frigate 0.18.0 (standard image) + Mosquitto |
| `config/config.example.yml` | CPU config: YOLOv9-tiny at 320px, 5 fps; validated against Frigate 0.18.0 |
| `scripts/export_yolov9.ps1` | Builds the model with Docker (PowerShell) |
| `.env.example` | Camera username and password |
