# Soul of Waifu — Linux (Arch-first)

Портирование и «Unix-Way»-перестройка Windows-приложения
[jofizcd/Soul-of-Waifu](https://github.com/jofizcd/Soul-of-Waifu).

## Установка (один раз)

Нужен Python 3.11 (uv скачает сам), ffmpeg системный, NVIDIA/CUDA опционально.

```bash
git clone https://github.com/Ganoda/Soul-of-Waifu-Linux ~/Soul-of-Waifu-Linux
cd ~/Soul-of-Waifu-Linux

# окружение
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --no-cache torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --extra-index-url https://download.pytorch.org/whl/cu128   # CPU: убери --extra-index-url
uv pip install -r requirements-linux.txt
# отложенные пакеты (конфликты пинов, ставятся без зависимостей):
uv pip install --no-deps \
  "rvc-python @ git+https://github.com/JarodMica/rvc-python@9a67ac77743d749e9f2cbbb02798957d5343fd19" \
  fairseq==0.12.2 pyautogui
uv pip install --no-deps https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
playwright install chromium

# fairseq патчится один раз (dataclass-баг на python>=3.11)
python - <<'EOF'
import re, pathlib
p = pathlib.Path(".venv/lib/python3.11/site-packages/fairseq/dataclass/configs.py")
src = p.read_text()
pat = re.compile(r"^(\s+)(\w+):\s*(\w+Config)\s*=\s*\3\(\)\s*$", re.M)
p.write_text(pat.sub(lambda m: f"{m.group(1)}{m.group(2)}: {m.group(3)} = field(default_factory={m.group(3)})", src))
print("fairseq patched")
EOF
```

## Запуск

```bash
./start.sh          # окно приложения; конфиги в ~/.config/soul-of-waifu/
```

## Конфигурация (Unix Way)

Всё — текстовые файлы в `~/.config/soul-of-waifu/`:

| Файл          | Что задаёт                                              |
|---------------|---------------------------------------------------------|
| `llm.yaml`    | локальный LLM (адрес/модель/сэмплинг), STT-веса Whisper |
| `media.yaml`  | ComfyUI endpoint, порядок TTS/провайдеров, cache-dir    |
| `ui.yaml`     | тема, платформа Qt под Wayland, хоткеи                  |
| `*.json`      | легаси-настройки (мигрируют сами при первом запуске)    |

Дефолты лежат в репо (`config-defaults/`) и сидируются при первом старте;
правь уже свои копии.

Для локального LLM: непустой `llm.yaml → backend` (любой
OpenAI-compatible сервер — llama-server, vLLM, Ollama) значит «ходить
напрямую на этот адрес»; пустой — приложение само запускает
встроенный llama-server с бинарей из `app/utils/ai_clients/backend/`.
Подробный сценарий см. `docs/gemma-preset.md`.

На Linux llama.cpp ставьте системно (Arch: `sudo pacman -S llama.cpp`)
или укажите внешний бэкенд в `llm.yaml`. Встроенный диалог «LLAMA.CPP
Updater» умеет ставить только Windows-архивы апстрима — на Linux кнопка
обновления движка показывает «No compatible binary found» и ничего не
скачивает.

Баннер «доступна новая версия» проверяет релизы этого форка
(Ganoda/Soul-of-Waifu-Linux), а не апстрима.

## Медиа через ComfyUI (workflows as config)

Кладёшь экспорт ComfyUI (**Save (API Format)**) в drop-in папку — пресет
готов без единой строчки кода:

```
app/data/workflows/tts/*.json       # голоса
app/data/workflows/image/*.json     # генерация картинок
app/data/workflows/video/*.json     # видео (video_enabled: true)
```

Маппинг нод — соседний `<name>.meta.yaml` (`text_node_id`, `output_node_id`);
без него сработает автопоиск по типам нод. Подробности в README внутри папок.
ComfyUI недоступен ⇒ TTS автоматически падает на встроенный Kokoro.

Порядок бэкендов задаётся в `media.yaml`:

- `tts_engine_order` — цепочка фолбэков TTS (имена: `comfyui`, `kokoro`,
  `xttsv2`, `edge`, `silero`, `qwen3`, `elevenlabs`). Движок, выбранный
  в GUI, всегда первый; остальные пункты списка — запасные на случай
  ошибки или пустого результата. Пустой/отсутствующий список — старое
  поведение (один выбранный движок; у ComfyUI запас — Kokoro).
- `image_provider_order` — то же для генерации картинок
  (`comfyui-workflow`, `a1111`, `novelai`, `dalle`, `flux`;
  `cloud` = NovelAI → DALL-E → FLUX).
- `video_enabled` пока никуда не подключён (видео-воркфлоу зарезервировано).

## Тесты

```bash
source .venv/bin/activate && pytest
```
