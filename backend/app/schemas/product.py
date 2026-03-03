from datetime import datetime
from pydantic import BaseModel, Field


class ProductBase(BaseModel):
    title: str = ""
    description: str = ""
    category: str = ""
    characteristics: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    seo_title: str = ""
    seo_description: str = ""
    seo_keywords: str = ""


class ProductUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    category: str | None = None
    characteristics: dict | None = None
    tags: list[str] | None = None
    seo_title: str | None = None
    seo_description: str | None = None
    seo_keywords: str | None = None


class ProductResponse(ProductBase):
    id: int
    image_path: str
    original_filename: str
    caption: str = ""
    confidence_score: float = 0.0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProductListResponse(BaseModel):
    items: list[ProductResponse]
    total: int
    page: int
    per_page: int
    pages: int


class GenerateResponse(ProductResponse):
    """Response after generating a product card."""
    pass


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    models_loaded: bool = False
    models_status: dict = Field(default_factory=dict)
