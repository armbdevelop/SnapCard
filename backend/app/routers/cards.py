import csv
import io
import json
import math
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.schemas.product import (
    ProductResponse,
    ProductListResponse,
    ProductUpdate,
    GenerateResponse,
)
from app.services.card_service import card_service
from app.services.file_service import file_service


class ExportFormat(str, Enum):
    json = "json"
    csv = "csv"

router = APIRouter(prefix="/cards", tags=["cards"])


@router.post("/generate", response_model=GenerateResponse)
async def generate_card(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload an image and generate a product card."""
    # Save file
    image_path, original_filename = await file_service.save_file(file)

    # Run ML pipeline if available
    ml_result = {}
    pipeline = getattr(request.app.state, "ml_pipeline", None)
    if pipeline and pipeline.is_loaded:
        import asyncio
        ml_result = await asyncio.to_thread(pipeline.process, image_path)
    else:
        # Placeholder data when ML is not loaded
        ml_result = {
            "title": "Товар",
            "description": "Описание товара будет сгенерировано после загрузки ML моделей.",
            "category": "Другое",
            "characteristics": {},
            "tags": [],
            "caption": "",
            "caption_ru": "",
            "confidence_score": 0.0,
            "seo_title": "",
            "seo_description": "",
            "seo_keywords": "",
        }

    # Create product record
    product = await card_service.create_product(
        db,
        image_path=image_path,
        original_filename=original_filename,
        **ml_result,
    )

    return product


@router.get("", response_model=ProductListResponse)
async def list_cards(
    page: int = 1,
    per_page: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """List all product cards with pagination."""
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 100:
        per_page = 20

    products, total = await card_service.list_products(db, page=page, per_page=per_page)
    pages = math.ceil(total / per_page) if total > 0 else 1

    return ProductListResponse(
        items=products,
        total=total,
        page=page,
        per_page=per_page,
        pages=pages,
    )


@router.get("/export")
async def export_cards(
    format: ExportFormat = Query(default=ExportFormat.json),
    category: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Export all product cards as JSON or CSV."""
    products = await card_service.list_all_products(db, category=category)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if format == ExportFormat.json:
        items = [ProductResponse.model_validate(p).model_dump(mode="json") for p in products]
        content = json.dumps(items, ensure_ascii=False, indent=2)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="snapcard_export_{timestamp}.json"'},
        )

    # CSV export
    if not products:
        fields = ProductResponse.model_fields.keys()
    else:
        fields = ProductResponse.model_fields.keys()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(fields))
    writer.writeheader()

    for p in products:
        row = ProductResponse.model_validate(p).model_dump(mode="json")
        # Serialize complex fields as JSON strings
        for key in ("characteristics", "tags"):
            if key in row and not isinstance(row[key], str):
                row[key] = json.dumps(row[key], ensure_ascii=False)
        writer.writerow(row)

    content = output.getvalue()
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="snapcard_export_{timestamp}.csv"'},
    )


@router.get("/{card_id}", response_model=ProductResponse)
async def get_card(card_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific product card."""
    product = await card_service.get_product(db, card_id)
    if not product:
        raise HTTPException(status_code=404, detail="Card not found")
    return product


@router.put("/{card_id}", response_model=ProductResponse)
async def update_card(
    card_id: int,
    data: ProductUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a product card."""
    product = await card_service.update_product(db, card_id, data)
    if not product:
        raise HTTPException(status_code=404, detail="Card not found")
    return product


@router.delete("/{card_id}")
async def delete_card(card_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a product card and its image."""
    product = await card_service.get_product(db, card_id)
    if not product:
        raise HTTPException(status_code=404, detail="Card not found")

    await file_service.delete_file(product.image_path)
    await card_service.delete_product(db, card_id)

    return {"detail": "Card deleted"}
