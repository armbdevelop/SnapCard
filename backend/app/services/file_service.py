import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile, HTTPException

from app.config import settings


class FileService:
    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_file(self, file: UploadFile) -> None:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{ext}' not allowed. Allowed: {', '.join(settings.allowed_extensions)}"
            )

    async def save_file(self, file: UploadFile) -> tuple[str, str]:
        self.validate_file(file)

        ext = file.filename.rsplit(".", 1)[-1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        file_path = self.upload_dir / unique_name

        content = await file.read()
        if len(content) > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.max_file_size // (1024*1024)}MB"
            )

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        return str(file_path), file.filename

    async def delete_file(self, file_path: str) -> None:
        path = Path(file_path)
        if path.exists():
            path.unlink()


file_service = FileService()
