import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import cards, health
from app.ml.pipeline import MLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ML pipeline
    pipeline = MLPipeline()
    app.state.ml_pipeline = pipeline

    # Load models in background — server starts immediately
    async def _load_models():
        try:
            await asyncio.to_thread(pipeline.load_models)
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.warning(f"ML models not loaded: {e}")

    task = asyncio.create_task(_load_models())

    yield

    # Cancel model loading if still in progress
    task.cancel()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Automatic product card generation from images",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for uploaded images
settings.upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")

# Routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(cards.router, prefix="/api/v1")
