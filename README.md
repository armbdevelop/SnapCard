# SnapCard

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://www.docker.com/)
**Автоматическая генерация карточек товаров по фотографии.**

Загрузите фото товара — получите готовую карточку на русском языке: заголовок, описание, характеристики, категорию, теги и SEO-метаданные.

> 💡 Скриншот интерфейса можно добавить сюда: `docs/demo-screenshot.png`

> Создание одной карточки вручную занимает 30–60 минут: фотосъёмка, ретушь, копирайтинг, SEO. SnapCard сокращает это до 5–10 секунд.

---

## Содержание

- [Проблема и решение](#проблема-и-решение)
- [Возможности](#возможности)
- [Архитектура](#архитектура)
- [ML-пайплайн](#ml-пайплайн)
- [Технологический стек](#технологический-стек)
- [Результаты](#результаты)
- [API](#api)
- [Быстрый старт](#быстрый-старт)
- [Структура проекта](#структура-проекта)
- [Roadmap](#roadmap)
- [Автор](#автор)

---

## Проблема и решение

### Проблема

Интернет-магазины ежедневно добавляют сотни товаров. Каждая карточка требует:

- качественного фото и обработки;
- продающего заголовка и описания;
- правильной категории и тегов для фильтров;
- SEO-метаданных для поисковой выдачи.

Это рутинный, дорогой и медленный процесс.

### Решение

SnapCard автоматизирует создание карточки на основе одного изображения. Система:

1. Описывает фото на английском (BLIP).
2. Переводит описание на русский (Helsinki-NLP).
3. Определяет категорию и теги zero-shot классификацией (CLIP).
4. Генерирует русский заголовок, описание и характеристики (mT5).
5. Формирует SEO-поля по правилам.

Всё это упаковано в веб-приложение с REST API, базой данных и Docker-деплоем.

---

## Возможности

- 📤 **Загрузка фото** через drag-and-drop интерфейс.
- 🤖 **Автоматическая генерация** карточки товара за несколько секунд.
- 🏷️ **14 категорий товаров**: Электроника, Одежда, Обувь, Аксессуары, Мебель, Продукты, Косметика, Спорт, Игрушки, Книги, Бытовая техника, Инструменты, Автотовары, Другое.
- 📝 **Редактирование** сгенерированных полей перед публикацией.
- 📊 **Экспорт** карточек в JSON или CSV.
- 🐳 **Docker-деплой** одной командой.
- 🧠 **LoRA-дообучение** BLIP под товарные изображения для повышения качества captioning.

---

## Архитектура

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   React SPA │────▶│  FastAPI API │────▶│  ML Pipeline    │
│  (frontend) │◄────│   (backend)  │◄────│  (BLIP/CLIP/mT5)│
└─────────────┘     └──────────────┘     └─────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  SQLite (DB) │
                     └──────────────┘
```

### Компоненты

- **Frontend** — React 19 + Vite + TypeScript + Tailwind CSS v4. Управляет серверным состоянием через TanStack Query.
- **Backend** — FastAPI + SQLAlchemy 2.0 + aiosqlite. Асинхронные эндпоинты, ML-инференс в отдельных потоках.
- **ML Pipeline** — 4 этапа: BLIP → перевод → CLIP → mT5 → SEO.
- **Training** — подготовка датасета, генерация эталонных описаний через LLM, дообучение BLIP через LoRA, оценка через BLEU-4 / ROUGE-L.

---

## ML-пайплайн

```
Image
  │
  ▼
BLIP (Salesforce/blip-image-captioning-large)
  │── English caption + confidence
  ▼
Helsinki-NLP/opus-mt-en-ru
  │── Russian caption
  ▼
CLIP (openai/clip-vit-base-patch32)
  │── category + tags + confidence
  ▼
mT5 (google/mt5-base)
  │── title + description + characteristics
  ▼
Rule-based SEO
  └── seo_title, seo_description, seo_keywords, seo_url
```

| Этап | Модель | Назначение |
|------|--------|------------|
| Captioning | BLIP + LoRA | Описание изображения на английском |
| Перевод | Helsinki-NLP/opus-mt-en-ru | Перевод caption на русский |
| Классификация | CLIP | Zero-shot категория и теги |
| Генерация текста | mT5 + LoRA | Русский заголовок, описание |
| SEO | Rule-based | SEO-метаданные |

### LoRA Fine-tuning

Для повышения качества описаний BLIP дообучен на fashion-датасете (~700 изображений) методом Low-Rank Adaptation:

- `r = 16`, `lora_alpha = 32`, `dropout = 0.1`
- Target modules: `query`, `value`
- Trainable parameters: ~0.5% от базовой модели
- Размер адаптера: ~50 МБ

### mT5 LoRA Fine-Tuning

Для генерации русских заголовков и описаний `google/mt5-base` дообучается через LoRA на синтетических карточках от Gemini Flash (~580 записей в train):

- Два отдельных адаптера: `snapcard_mt5_title_lora` и `snapcard_mt5_description_lora`.
- `r = 16`, `lora_alpha = 32`, `dropout = 0.1`
- Target modules: `q`, `v`
- Trainable parameters: ~0.3% от базовой модели
- Размер каждого адаптера: ~10–20 МБ

#### Как обучить

1. Подготовить датасеты:
   ```bash
   cd training
   python prepare_mt5_dataset.py
   ```
2. Открыть `training/train_mt5_lora.ipynb` в Google Colab (GPU runtime).
3. Загрузить 4 JSONL-файла из `training/data/` в Google Drive.
4. Запустить все ячейки — получить два адаптера в Drive.
5. Скопировать адаптеры в проект:
   ```
   backend/model_cache/
   ├── snapcard_mt5_title_lora/
   └── snapcard_mt5_description_lora/
   ```
6. Оценить качество:
   ```bash
   python training/evaluate_mt5.py \
     backend/model_cache/snapcard_mt5_title_lora \
     backend/model_cache/snapcard_mt5_description_lora
   ```

---

## Технологический стек

### Backend
- **FastAPI** — асинхронный веб-фреймворк
- **SQLAlchemy 2.0** — ORM с async-сессиями
- **aiosqlite** — асинхронный драйвер SQLite
- **Pydantic Settings** — конфигурация через переменные окружения
- **Pillow + aiofiles** — обработка и асинхронная запись файлов

### ML
- **PyTorch + Transformers** — инференс моделей
- **BLIP** — image captioning
- **CLIP** — zero-shot классификация
- **mT5** — генерация русского текста
- **PEFT** — загрузка LoRA-адаптеров
- **sacrebleu + rouge-score** — оценка качества

### Frontend
- **React 19** + **TypeScript**
- **Vite** — сборка и dev-сервер
- **Tailwind CSS v4** — utility-first стили
- **TanStack Query v5** — кэширование и серверное состояние
- **React Router v7** — клиентская маршрутизация
- **react-dropzone** — drag-and-drop загрузка

### Инфраструктура
- **Docker + Docker Compose**
- **nginx** — раздача статики и проксирование API
- **Uvicorn** — ASGI-сервер

---

## Результаты

Оценка качества генерации caption на тестовой выборке (50 образцов):

| Модель | BLEU-4 | ROUGE-L | Latency |
|--------|--------|---------|---------|
| Base BLIP | 0.15–0.20 | 0.25–0.30 | ~200 мс |
| LoRA-BLIP | 0.20–0.30 | 0.35–0.45 | ~220 мс |

- Дообучение BLIP через LoRA улучшило качество описаний на **15–25%** по BLEU-4.
- Среднее время генерации одной карточки: **5–10 секунд** (CPU).

---

## API

Базовый URL: `/api/v1`

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Статус сервера и ML-моделей |
| POST | `/cards/generate` | Загрузить фото, сгенерировать карточку |
| GET | `/cards` | Список карточек с пагинацией |
| GET | `/cards/export` | Экспорт в JSON или CSV |
| GET | `/cards/{id}` | Получить карточку |
| PUT | `/cards/{id}` | Обновить карточку |
| DELETE | `/cards/{id}` | Удалить карточку и файл |

Полная документация доступна по адресу `http://localhost:8000/docs` после запуска backend.

---

## Быстрый старт

### Требования

- Python 3.11+
- Node.js 22+
- Docker (опционально)

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[ml]"

# Скопируйте и настройте переменные окружения
cp .env.example .env

uvicorn app.main:app --reload
```

API: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Приложение: http://localhost:5173

### Docker (production)

```bash
docker compose up --build
```

- Frontend: http://localhost
- Backend: http://localhost:8000

### Тесты

```bash
cd backend
pytest -v
```

---

## Структура проекта

```
SnapCard/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── config.py            # pydantic-settings
│   │   ├── database.py          # async SQLAlchemy
│   │   ├── dependencies.py      # DI: get_db
│   │   ├── models/              # SQLAlchemy models
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── routers/             # API endpoints
│   │   ├── services/            # business logic
│   │   └── ml/                  # ML pipeline modules
│   ├── tests/                   # pytest suite
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── api/                 # axios client
│   │   ├── components/          # reusable UI
│   │   ├── hooks/               # TanStack Query hooks
│   │   ├── pages/               # route pages
│   │   └── types/               # TypeScript types
│   ├── Dockerfile
│   └── package.json
├── training/
│   ├── prepare_dataset.py       # dataset preparation
│   ├── generate_captions.py     # LLM caption generation
│   ├── train_blip_lora.ipynb    # Colab LoRA training
│   ├── evaluate.py              # BLEU/ROUGE evaluation
│   └── data/                    # datasets
├── docker-compose.yml
└── README.md
```

---

## Roadmap

- [x] MVP с загрузкой фото и генерацией карточки
- [x] REST API с CRUD, пагинацией и экспортом
- [x] React-интерфейс с drag-and-drop
- [x] ML-пайплайн BLIP → CLIP → mT5 → SEO
- [x] LoRA-дообучение BLIP на fashion-датасете
- [x] Docker и docker-compose
- [x] Полноценная генерация текста через mT5 с LoRA-адаптерами
- [ ] Поддержка пакетной загрузки нескольких изображений
- [ ] Интеграция с популярными CMS маркетплейсов
- [ ] PostgreSQL вместо SQLite для продакшена

---

## Автор

Разработано как пет-проект для портфолио.

- GitHub: [@armbdevelop](https://github.com/armbdevelop)
- Репозиторий: [github.com/armbdevelop/SnapCard](https://github.com/armbdevelop/SnapCard)

Если проект оказался полезным — поставь ⭐
