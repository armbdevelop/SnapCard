import io
import pytest
from PIL import Image


def create_test_image() -> bytes:
    """Create a minimal test image in memory."""
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.mark.asyncio
async def test_generate_card(client):
    image_data = create_test_image()
    response = await client.post(
        "/api/v1/cards/generate",
        files={"file": ("test.jpg", image_data, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["original_filename"] == "test.jpg"
    assert "title" in data
    assert "category" in data


@pytest.mark.asyncio
async def test_list_cards_empty(client):
    response = await client.get("/api/v1/cards")
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_generate_and_list(client):
    # Generate a card
    image_data = create_test_image()
    gen_response = await client.post(
        "/api/v1/cards/generate",
        files={"file": ("product.jpg", image_data, "image/jpeg")},
    )
    assert gen_response.status_code == 200
    card_id = gen_response.json()["id"]

    # List cards
    list_response = await client.get("/api/v1/cards")
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1

    # Get specific card
    get_response = await client.get(f"/api/v1/cards/{card_id}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == card_id


@pytest.mark.asyncio
async def test_update_card(client):
    # Generate a card first
    image_data = create_test_image()
    gen_response = await client.post(
        "/api/v1/cards/generate",
        files={"file": ("test.jpg", image_data, "image/jpeg")},
    )
    card_id = gen_response.json()["id"]

    # Update it
    update_response = await client.put(
        f"/api/v1/cards/{card_id}",
        json={"title": "Обновлённый товар", "category": "Электроника"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["title"] == "Обновлённый товар"
    assert update_response.json()["category"] == "Электроника"


@pytest.mark.asyncio
async def test_delete_card(client):
    # Generate a card first
    image_data = create_test_image()
    gen_response = await client.post(
        "/api/v1/cards/generate",
        files={"file": ("test.jpg", image_data, "image/jpeg")},
    )
    card_id = gen_response.json()["id"]

    # Delete it
    del_response = await client.delete(f"/api/v1/cards/{card_id}")
    assert del_response.status_code == 200

    # Verify it's gone
    get_response = await client.get(f"/api/v1/cards/{card_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_get_nonexistent_card(client):
    response = await client.get("/api/v1/cards/9999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_invalid_file_type(client):
    response = await client.post(
        "/api/v1/cards/generate",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_export_json(client):
    # Create 2 cards
    image_data = create_test_image()
    await client.post("/api/v1/cards/generate", files={"file": ("a.jpg", image_data, "image/jpeg")})
    await client.post("/api/v1/cards/generate", files={"file": ("b.jpg", image_data, "image/jpeg")})

    response = await client.get("/api/v1/cards/export?format=json")
    assert response.status_code == 200
    assert "json" in response.headers["content-type"]
    data = response.json()
    assert len(data) == 2


@pytest.mark.asyncio
async def test_export_csv(client):
    image_data = create_test_image()
    await client.post("/api/v1/cards/generate", files={"file": ("a.jpg", image_data, "image/jpeg")})
    await client.post("/api/v1/cards/generate", files={"file": ("b.jpg", image_data, "image/jpeg")})

    response = await client.get("/api/v1/cards/export?format=csv")
    assert response.status_code == 200
    assert "csv" in response.headers["content-type"]
    lines = response.text.strip().split("\n")
    assert "id" in lines[0]  # header row
    assert len(lines) == 3  # header + 2 data rows


@pytest.mark.asyncio
async def test_export_with_category_filter(client):
    image_data = create_test_image()
    # Create card and update category
    r1 = await client.post("/api/v1/cards/generate", files={"file": ("a.jpg", image_data, "image/jpeg")})
    await client.put(f"/api/v1/cards/{r1.json()['id']}", json={"category": "Электроника"})

    r2 = await client.post("/api/v1/cards/generate", files={"file": ("b.jpg", image_data, "image/jpeg")})
    await client.put(f"/api/v1/cards/{r2.json()['id']}", json={"category": "Одежда"})

    response = await client.get("/api/v1/cards/export?format=json&category=Электроника")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["category"] == "Электроника"
