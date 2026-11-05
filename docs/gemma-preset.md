# Gemma 4 Abliterated — локальный пресет

Zero-cloud: всё работает без интернета. Чат-шаблон Gemma уже зашит в
дефолтном `llm.yaml` (см. ниже).

## 1. Получить веса

Нужен GGUF-квант модели `*-abliterated` на базе Gemma (например, поиск на
Hugging Face: `gemma abliterated GGUF Q4_K_M`). Варианты размещения:

- уже работающий сервер: пропусти шаг 2, только проверь адрес/порт;
- файл рядом с приложением: положи в
  `~/Soul-of-Waifu-Linux/assets/local_llm/`.

## 2. Запустить llama-server

```bash
llama-server \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 16384 \
  ~/Soul-of-Waifu-Linux/assets/local_llm/<модель>.gguf
```

## 3. Подключить приложение

Файл `~/.config/soul-of-waifu/llm.yaml`:

```yaml
backend: http://127.0.0.1:8080   # или Ollama/vLLM — любой OpenAI-compatible
api_key: ""                      # пусто или фиктивный ("ollama")
model: gemma4-abliterated        # имя, под которым сервер отдаёт модель
```

Заполненный `backend` означает: приложение ходит на этот адрес напрямую
(`POST {backend}/v1/chat/completions`, имя модели подставляется из
`model`) и **не запускает встроенный llama-server**. Адрес можно писать
и с суффиксом `/v1`, и без него — нормализация автоматическая. Пустой
`backend` ⇒ прежнее поведение: приложение само ищет бинари
`app/utils/ai_clients/backend/{cpu,vulkan,cuda,hip,sycl}/llama-server`
и запускает его на порту 48596.

Если с бэкендом пусто и поставляемых бинарей нет, приложение дополнительно
ищет системный `llama-server` в `PATH` (можно поставить пакет llama.cpp,
например `sudo pacman -S llama.cpp` в Arch) и запускает его так же, как
встроенный — рабочий порт, модель и GPU/CPU-флаги берутся из настроек.
Если llama-server нет нигде (ни бинарь, ни PATH), кнопка «Load Model»
даёт понятную ошибку с этими же двумя вариантами решения вместо
«Local API Connection Error».

`ctx` — параметр самого сервера (смотри команду выше или флаг
`--ctx-size` встроенного запуска), остальные сэмплеры читаются из
llm.yaml и уходят в каждый запрос. Явные настройки сэмплинга в GUI
приоритетнее значений из llm.yaml.

Сервер llama.cpp сам применяет chat-template из GGUF; токены Gemma для
справки (они же — дефолт в llm.yaml):

| Токен               | Назначение            |
|---------------------|-----------------------|
| `<start_of_turn>user`  | начало хода юзера  |
| `<end_of_turn>`        | конец хода         |
| `<start_of_turn>model` | ход модели         |

Шаблон хода целиком: `<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n`.

Параметры сэмплинга (`temperature`, `repeat_penalty`, `stop_tokens`)
правятся там же и читаются при старте.

## 4. STT без сети

`llm.yaml → stt.whisper_model_path` — путь к локальным весам
Faster-Whisper (например `models/whisper/small-fp16`). Пустое значение ⇒
библиотека сама скачает модель при первом запуске (нужен интернет ровно
один раз).

## 5. Проверка zero-cloud

Отключи сеть → запусти `./start.sh`. Ожидаемо: окно открывается,
баннер «new version» может появиться как ошибка сети — это допустимо,
на функциональность не влияет.
