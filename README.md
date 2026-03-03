# SnapCard

Automatic product card generation from images. Upload a product photo and get a complete card: title, description, characteristics, category, tags, and SEO metadata — all in Russian.

## Stack

- **Backend**: FastAPI + SQLAlchemy + aiosqlite
- **Frontend**: React 19 + Vite + TypeScript + Tailwind CSS v4
- **ML Pipeline**: BLIP (captioning) + CLIP (classification) + mT5 (text generation)
- **Deployment**: Docker + nginx

## ML Pipeline

```
Image → BLIP (caption) → CLIP (category + tags) → mT5 (title + description) → Rules (SEO)
```

| Stage | Model | Purpose |
|-------|-------|---------|
| Captioning | Salesforce/blip-image-captioning-large | English image caption |
| Classification | openai/clip-vit-base-patch32 | Zero-shot into 14 categories |
| Text Generation | google/mt5-base | Russian title, description, characteristics |
| SEO | Rule-based | seo_title, seo_description, keywords |

## Quick Start (Development)

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install fastapi "uvicorn[standard]" sqlalchemy aiosqlite python-multipart pydantic-settings Pillow aiofiles

# Optional: install ML models
pip install torch transformers accelerate

uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173

### Tests

```bash
cd backend
source .venv/bin/activate
pip install pytest pytest-asyncio httpx
pytest -v
```

## Docker

### Production

```bash
docker compose up --build
```

Frontend: http://localhost (port 80)
Backend API: http://localhost:8000

### Development

```bash
docker compose -f docker-compose.dev.yml up --build
```

Frontend: http://localhost:5173
Backend: http://localhost:8000

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/v1/cards/generate | Upload image, run ML, return card |
| GET | /api/v1/cards | List cards (paginated) |
| GET | /api/v1/cards/{id} | Get card |
| PUT | /api/v1/cards/{id} | Update card |
| DELETE | /api/v1/cards/{id} | Delete card + image |
| GET | /api/v1/health | Health + model status |

## Project Structure

```
SnapCard/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, lifespan, CORS
│   │   ├── config.py            # pydantic-settings
│   │   ├── database.py          # async SQLAlchemy + aiosqlite
│   │   ├── dependencies.py      # DI: get_db
│   │   ├── models/product.py    # SQLAlchemy Product model
│   │   ├── schemas/product.py   # Pydantic request/response
│   │   ├── routers/cards.py     # /api/v1/cards endpoints
│   │   ├── routers/health.py    # /api/v1/health
│   │   ├── services/            # CRUD + file management
│   │   └── ml/                  # ML pipeline modules
│   ├── tests/
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/client.ts        # Axios + cardsApi
│   │   ├── hooks/useCards.ts    # React Query hooks
│   │   ├── types/index.ts       # TypeScript interfaces
│   │   ├── pages/               # Upload, CardDetail, CardsList
│   │   └── components/          # Reusable UI components
│   └── Dockerfile
├── docker-compose.yml
└── docker-compose.dev.yml
```
