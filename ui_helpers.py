"""Presentation helpers. No database access or valuation logic lives here."""
import json
import math
import re
from html import escape
from urllib.parse import urlparse


def safe_url(value):
    """Only allow web URLs from database content in HTML and link widgets."""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    try:
        parsed = urlparse(value)
    except ValueError:
        return ""
    return value if parsed.scheme in ("https", "http") and parsed.netloc else ""


def get_photo_urls(photos_json, limit=3):
    """Accept PostgreSQL JSON/JSONB lists or JSON strings; deduplicate frames."""
    if limit is not None and limit <= 0:
        return []
    if isinstance(photos_json, str):
        try:
            photos = json.loads(photos_json)
        except (ValueError, TypeError):
            return []
    else:
        photos = photos_json
    if not isinstance(photos, list):
        return []
    seen, unique = set(), []
    for value in photos:
        url = safe_url(value)
        if not url:
            continue
        match = re.search(r"/(\d+)-\d+x\d+\.\w+$", urlparse(url).path)
        pid = match.group(1) if match else url
        if pid not in seen:
            seen.add(pid)
            unique.append(url)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def photo_html(url, kind="card", alt="Фото квартиры"):
    url = safe_url(url)
    if not url:
        return f'<div class="photo-placeholder {kind}-photo">Фотографии пока нет</div>'
    return (f'<img class="{kind}-photo" src="{escape(url, quote=True)}" '
            f'alt="{escape(alt, quote=True)}" loading="lazy" referrerpolicy="no-referrer">')


def price_label(value):
    return f"{float(value):,.0f}".replace(",", " ") + " ₸"


def discount_label(value):
    if not math.isfinite(float(value)):
        return "Нет оценки", "neutral"
    if value >= 5:
        return f"На {value:.0f}% ниже рынка", "positive"
    if value <= -5:
        return f"На {abs(value):.0f}% выше рынка", "negative"
    return "На уровне рынка", "neutral"


def listing_markup(row, kind="card", photo_index=0):
    photos = get_photo_urls(row["photos"], limit=None)
    url = photos[photo_index % len(photos)] if photos else ""
    label, tone = discount_label(row["скидка_%"])
    return (f'<div class="photo-wrap">{photo_html(url, kind, str(row["title"]))}'
            f'<span class="price-badge {tone}">{label}</span></div>')
