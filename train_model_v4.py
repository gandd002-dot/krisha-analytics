"""
Переобучение XGBoost — версия 4.

Отличие от v2/v3: обучаемся ТОЛЬКО на human_correction —
это чистый датасет где каждая запись прошла проверку глазами
в едином режиме GOOD/BAD (без метаний между A/B/C/D/E).

Смотрим первые 3 фото каждой квартиры — так же как оценивал человек
в review_predictions.py.
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

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
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
MAX_PHOTOS_PER_LISTING = 3  # смотрим первые 3 фото — как и человек при разметке


# ==============================
# 1. ЗАГРУЖАЕМ CLIP
# ==============================

print("Загружаем CLIP модель...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP загружен!\n")


def get_single_embedding(photo_url):
    resp = requests.get(photo_url, headers=HEADERS, timeout=6)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")

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


def get_embedding_avg(photo_urls, max_photos=MAX_PHOTOS_PER_LISTING):
    """Первые 3 фото — так же как смотрел человек при разметке"""
    embeddings = []
    for url in photo_urls[:max_photos]:
        try:
            emb = get_single_embedding(url)
            embeddings.append(emb)
        except Exception:
            continue

    if not embeddings:
        return None

    return np.mean(embeddings, axis=0)


# ==============================
# 2. СОБИРАЕМ ДАННЫЕ — ТОЛЬКО human_correction
# ==============================

from bs4 import BeautifulSoup


def get_fresh_photos(listing_url):
    """
    Ссылки на фото в базе могут устареть — берём свежие
    прямо со страницы объявления, так же как в predict.py
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


def build_dataset():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT id, photos, url, human_correction
        FROM listings
        WHERE human_correction IS NOT NULL
          AND photos != '[]'
    """, conn)
    conn.close()

    print(f"Квартир для обучения (human_correction): {len(df)}\n")

    X, y = [], []
    failed = 0

    for i, row in df.iterrows():
        stored_photos = json.loads(row["photos"])

        print(f"[{i+1}/{len(df)}] Обрабатываем...", end="\r")

        # Сохранённые ссылки могут быть устаревшими — берём свежие с сайта
        fresh_photos = get_fresh_photos(row["url"]) if row["url"] else []
        photos = fresh_photos if fresh_photos else stored_photos

        if not photos:
            failed += 1
            continue

        embedding = get_embedding_avg(photos)
        if embedding is None or embedding.shape != (512,):
            failed += 1
            continue

        X.append(embedding)
        y.append(row["human_correction"])

    print(f"\n\nУспешно обработано: {len(X)}")
    print(f"Пропущено: {failed}\n")

    if len(X) == 0:
        return np.array([]), np.array([])

    return np.vstack(X), np.array(y)


# ==============================
# 3. ОБУЧЕНИЕ С ПОДБОРОМ ПАРАМЕТРОВ
# ==============================

def train():
    X, y = build_dataset()

    if len(X) == 0:
        raise ValueError("Нет данных для обучения")

    np.save("embeddings_X_v4.npy", X)
    np.save("embeddings_y_v4.npy", y)
    print("💾 Embeddings сохранены\n")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print("Классы:", dict(zip(encoder.classes_, range(len(encoder.classes_)))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Обучающая выборка: {len(X_train)} | Тестовая: {len(X_test)}\n")

    param_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 4, 5],
        "learning_rate": [0.05, 0.1],
    }

    print("Подбираем лучшие параметры...\n")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=2,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ Лучшие параметры: {grid_search.best_params_}")
    print(f"✅ Точность на cross-validation: {grid_search.best_score_*100:.1f}%\n")

    best_clf = grid_search.best_estimator_

    y_pred = best_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("="*50)
    print(f"ФИНАЛЬНАЯ ТОЧНОСТЬ НА ТЕСТЕ: {accuracy*100:.1f}%")
    print("="*50)
    print("\nПодробный отчёт:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    with open("renovation_model_v4.pkl", "wb") as f:
        pickle.dump({
            "model": best_clf,
            "encoder": encoder,
            "max_photos": MAX_PHOTOS_PER_LISTING
        }, f)

    print("✅ Модель сохранена в renovation_model_v4.pkl")


if __name__ == "__main__":
    train()
