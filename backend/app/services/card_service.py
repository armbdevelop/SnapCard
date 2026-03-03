from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product import Product
from app.schemas.product import ProductUpdate


class CardService:
    async def create_product(self, session: AsyncSession, **kwargs) -> Product:
        product = Product(**kwargs)
        session.add(product)
        await session.flush()
        await session.refresh(product)
        return product

    async def get_product(self, session: AsyncSession, product_id: int) -> Product | None:
        result = await session.execute(select(Product).where(Product.id == product_id))
        return result.scalar_one_or_none()

    async def list_products(
        self, session: AsyncSession, page: int = 1, per_page: int = 20
    ) -> tuple[list[Product], int]:
        # Count total
        count_result = await session.execute(select(func.count(Product.id)))
        total = count_result.scalar_one()

        # Get page
        offset = (page - 1) * per_page
        result = await session.execute(
            select(Product)
            .order_by(Product.created_at.desc())
            .offset(offset)
            .limit(per_page)
        )
        products = list(result.scalars().all())
        return products, total

    async def update_product(
        self, session: AsyncSession, product_id: int, data: ProductUpdate
    ) -> Product | None:
        product = await self.get_product(session, product_id)
        if not product:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(product, field, value)

        await session.flush()
        await session.refresh(product)
        return product

    async def delete_product(self, session: AsyncSession, product_id: int) -> bool:
        product = await self.get_product(session, product_id)
        if not product:
            return False
        await session.delete(product)
        return True


card_service = CardService()
