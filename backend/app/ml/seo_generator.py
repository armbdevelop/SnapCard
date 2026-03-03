class SEOGenerator:
    """Rule-based SEO metadata generator."""

    def generate(self, title: str, description: str, category: str, tags: list[str]) -> dict[str, str]:
        """Generate SEO metadata.

        Returns:
            Dict with seo_title, seo_description, seo_keywords
        """
        seo_title = self._make_seo_title(title, category)
        seo_description = self._make_seo_description(title, description)
        seo_keywords = self._make_keywords(title, category, tags)

        return {
            "seo_title": seo_title,
            "seo_description": seo_description,
            "seo_keywords": seo_keywords,
        }

    def _make_seo_title(self, title: str, category: str) -> str:
        """Create SEO title, max 70 characters."""
        base = f"{title} — купить"
        if len(base) <= 70:
            suffix = f" | {category}"
            if len(base + suffix) <= 70:
                return base + suffix
        if len(base) > 70:
            return base[:67] + "..."
        return base

    def _make_seo_description(self, title: str, description: str) -> str:
        """Create SEO description, max 160 characters."""
        base = f"{title}. {description}"
        if len(base) <= 160:
            return base
        # Truncate to last complete sentence within 160 chars
        truncated = base[:157]
        last_period = truncated.rfind(".")
        if last_period > 60:
            return truncated[:last_period + 1]
        return truncated + "..."

    def _make_keywords(self, title: str, category: str, tags: list[str]) -> str:
        """Create comma-separated keywords string."""
        keywords: list[str] = []

        # Add title words (skip short words)
        for word in title.split():
            clean = word.strip(".,!?-—\"'()").lower()
            if len(clean) > 2 and clean not in keywords:
                keywords.append(clean)

        # Add category
        cat_lower = category.lower()
        if cat_lower not in keywords:
            keywords.append(cat_lower)

        # Add tags
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in keywords:
                keywords.append(tag_lower)

        # Add common purchase keywords
        purchase_keywords = ["купить", "цена", "заказать"]
        for kw in purchase_keywords:
            if kw not in keywords:
                keywords.append(kw)

        return ", ".join(keywords[:15])
