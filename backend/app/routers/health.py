from fastapi import APIRouter, Request

from app.schemas.product import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    models_loaded = False
    models_status = {}

    pipeline = getattr(request.app.state, "ml_pipeline", None)
    if pipeline:
        models_loaded = pipeline.is_loaded
        models_status = pipeline.get_status()

    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        models_status=models_status,
    )
