from app.ml.seo_generator import SEOGenerator


def test_seo_title_length():
    seo = SEOGenerator()
    result = seo.generate("Товар", "Описание", "Электроника", [])
    assert len(result["seo_title"]) <= 70


def test_seo_description_length():
    seo = SEOGenerator()
    long_desc = "Очень длинное описание товара. " * 20
    result = seo.generate("Товар", long_desc, "Электроника", [])
    assert len(result["seo_description"]) <= 160


def test_seo_keywords_include_category():
    seo = SEOGenerator()
    result = seo.generate("Красная куртка", "Описание", "Одежда", ["Одежда"])
    assert "одежда" in result["seo_keywords"]
    assert "купить" in result["seo_keywords"]


def test_seo_title_with_category():
    seo = SEOGenerator()
    result = seo.generate("Куртка", "Описание", "Одежда", [])
    assert "Куртка" in result["seo_title"]
    assert "купить" in result["seo_title"]
