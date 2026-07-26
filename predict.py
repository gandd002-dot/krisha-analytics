"""
Применяем обученную модель ко всем квартирам в базе.

Как работает:
1. Берём квартиры у которых ещё НЕТ renovation_class (не размеченные вручную)
2. Для каждой — получаем embedding через CLIP (усредняем до 3 фото)
3. Загружаем renovation_model_v4.pkl
4. Модель предсказывает GOOD или BAD
5. Записываем результат в новую колонку predicted_class
"""

import torch
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import psycopg2
import pandas as pd
import numpy as np
import json
from io import BytesIO
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     int(os.getenv("DB_PORT")),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_PHOTOS_PER_LISTING = 6


# ==============================
# 1. ЗАГРУЖАЕМ CLIP И ОБУЧЕННУЮ МОДЕЛЬ
# ==============================

print("Загружаем CLIP модель...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP загружен!")

print("Загружаем обученную модель XGBoost...")
with open("renovation_model_v4.pkl", "rb") as f:
    saved = pickle.load(f)
    clf = saved["model"]
    encoder = saved["encoder"]
print("✅ Модель загружена!\n")


# Тексты чтобы отличить фото ЗДАНИЯ от фото ИНТЕРЬЕРА
EXTERIOR_TEXTS = [
    "photo of a building exterior facade",
    "outdoor street view of apartment building",
]
INTERIOR_TEXTS = [
    "photo of a room interior with furniture",
    "indoor photo of kitchen bathroom or bedroom",
]


def is_interior_photo(img):
    """
    Проверяем через CLIP — это фото комнаты или фото здания снаружи?
    Возвращает True если это интерьер (комната).
    """
    texts = EXTERIOR_TEXTS + INTERIOR_TEXTS
    inputs = processor(text=texts, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    probs = out.logits_per_image.softmax(dim=1)[0]

    exterior_score = probs[:len(EXTERIOR_TEXTS)].mean().item()
    interior_score = probs[len(EXTERIOR_TEXTS):].mean().item()

    return interior_score > exterior_score


def fetch_photo_with_fallback(photo_url):
    """
    Пробуем скачать фото. Если конкретный размер даёт 404 —
    пробуем другие распространённые размеры на том же фото.
    """
    # Сначала пробуем как есть
    try:
        resp = requests.get(photo_url, headers=HEADERS, timeout=6)
        if resp.status_code == 200:
            return resp.content
    except requests.RequestException:
        pass

    # Если не вышло — пробуем заменить размер в URL на другие варианты
    import re
    # URL вида .../abcdef/1-750x470.webp — вытаскиваем номер фото и меняем размер
    match = re.match(r"(.+/\d+)-\d+x\d+\.(webp|jpg)$", photo_url)
    if not match:
        return None

    base, ext = match.groups()
    fallback_sizes = ["400x300", "750x470", "280x175", "1024x768"]

    for size in fallback_sizes:
        candidate = f"{base}-{size}.{ext}"
        if candidate == photo_url:
            continue
        try:
            resp = requests.get(candidate, headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                return resp.content
        except requests.RequestException:
            continue

    return None


def get_single_embedding(photo_url, filter_exterior=True):
    content = fetch_photo_with_fallback(photo_url)
    if content is None:
        raise ValueError("Фото недоступно ни в одном размере (все варианты 404)")

    img = Image.open(BytesIO(content)).convert("RGB")

    # Пропускаем фото если это здание снаружи, а не комната
    if filter_exterior and not is_interior_photo(img):
        return None

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)

    if isinstance(out, torch.Tensor):
        embedding_vector = out[0]
    elif hasattr(out, "image_embeds"):
        embedding_vector = out.image_embeds[0]
    elif hasattr(out, "pooler_output"):
        embedding_vector = out.pooler_output[0]
    else:
        raise TypeError(f"Неизвестный формат ответа CLIP: {type(out)}")

    return embedding_vector.detach().cpu().numpy()


from bs4 import BeautifulSoup


def get_fresh_photos(listing_url):
    """
    Ссылки на фото в базе могут устареть — krisha.kz иногда меняет
    пути на CDN даже для активных объявлений.
    Поэтому заходим на страницу заново и берём свежие ссылки.
    """
    try:
        resp = requests.get(listing_url, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        photos = []

        for picture in soup.select("picture"):
            for source in picture.select("source"):
                srcset = source.get("srcset", "")
                if not srcset:
                    continue
                parts = srcset.split(",")
                last = parts[-1].strip().split(" ")[0]
                if last.startswith("http") and last not in photos:
                    photos.append(last)
                break

        return photos
    except requests.RequestException:
        return []


def is_listing_active(listing_url):
    """
    Быстрая проверка — жива ли страница объявления,
    или она в архиве (редирект на главную/404).
    """
    try:
        resp = requests.get(listing_url, headers=HEADERS, timeout=6, allow_redirects=True)
        if resp.status_code != 200:
            return False
        # Если редиректнуло на главную страницу — объявление снято
        if resp.url.rstrip("/") in ["https://krisha.kz", "https://krisha.kz/prodazha", "https://krisha.kz/arenda"]:
            return False
        if "/a/show/" not in resp.url:
            return False
        return True
    except requests.RequestException:
        return False


def get_embedding_avg(photo_urls, max_photos=3, debug=False):
    """
    Берём первые max_photos фото как есть, без попыток угадать
    что на них — здание или комната. Усреднение по нескольким фото
    само сглаживает случайные неудачные кадры.
    """
    embeddings = []
    for url in photo_urls[:max_photos]:
        try:
            emb = get_single_embedding(url, filter_exterior=False)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            if debug:
                print(f"\n    ⚠️ Реальная ошибка на фото: {type(e).__name__}: {e}")
            continue

    if not embeddings:
        return None

    return np.mean(embeddings, axis=0)


# ==============================
# 2. ДОБАВЛЯЕМ КОЛОНКУ ЕСЛИ ЕЁ НЕТ
# ==============================

def ensure_column(conn):
    cur = conn.cursor()
    cur.execute("""
        ALTER TABLE listings
        ADD COLUMN IF NOT EXISTS predicted_class VARCHAR(10)
    """)
    cur.execute("""
        ALTER TABLE listings
        ADD COLUMN IF NOT EXISTS predicted_confidence NUMERIC(4,3)
    """)
    conn.commit()


# ==============================
# 3. ГЛАВНАЯ ФУНКЦИЯ
# ==============================

def run_predict(limit=None):
    conn = psycopg2.connect(**DB_CONFIG)
    ensure_column(conn)

    query = """
        SELECT id, photos, url
        FROM listings
        WHERE predicted_class IS NULL
          AND renovation_class IS NULL
          AND url IS NOT NULL
    """
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, conn)
    print(f"Квартир для предсказания: {len(df)}\n")

    cur = conn.cursor()
    predicted = 0
    failed = 0
    archived = 0

    for i, row in df.iterrows():
        photos = json.loads(row["photos"])
        if not photos:
            continue

        print(f"[{i+1}/{len(df)}] ID {row['id']}...", end="\r")

        # Быстрая проверка — если объявление в архиве, не тратим время на фото
        if row["url"] and not is_listing_active(row["url"]):
            archived += 1
            cur.execute(
                "UPDATE listings SET predicted_class = 'ARCHIVED' WHERE id = %s",
                (row["id"],)
            )
            conn.commit()
            continue

        # Сохранённые ссылки на фото могут быть устаревшими —
        # берём свежие прямо со страницы объявления
        fresh_photos = get_fresh_photos(row["url"]) if row["url"] else []
        photos_to_use = fresh_photos if fresh_photos else photos

        embedding = get_embedding_avg(photos_to_use, debug=(failed < 5))
        if embedding is None or embedding.shape != (512,):
            failed += 1
            if failed <= 5:
                print(f"\n  ⚠️ ID {row['id']}: embedding=None или неправильная форма. Фото: {len(photos)} шт")
            continue

        # Предсказываем класс
        pred_encoded = clf.predict([embedding])[0]
        pred_class = encoder.inverse_transform([pred_encoded])[0]

        # Уверенность модели (вероятность для предсказанного класса)
        proba = clf.predict_proba([embedding])[0]
        confidence = round(float(max(proba)), 3)

        cur.execute(
            "UPDATE listings SET predicted_class = %s, predicted_confidence = %s WHERE id = %s",
            (pred_class, confidence, row["id"])
        )
        conn.commit()
        predicted += 1

        # Каждые 50 квартир — прогресс
        if predicted % 50 == 0:
            print(f"\n--- Обработано: {predicted} | Ошибок: {failed} ---")

    conn.close()
    print(f"\n\n✅ Готово!")
    print(f"   Предсказано: {predicted}")
    print(f"   В архиве (пропущено): {archived}")
    print(f"   Ошибок: {failed}")


if __name__ == "__main__":
    # Начни с limit=100 чтобы проверить как работает,
    # потом убери limit чтобы обработать всю базу
    run_predict(limit=500)