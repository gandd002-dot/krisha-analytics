import requests
from bs4 import BeautifulSoup
import psycopg2
import time
import json
from datetime import datetime
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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}

BASE_URL = "https://krisha.kz"


# ==============================
# 1. СОЗДАНИЕ ТАБЛИЦ
# ==============================

def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS listings (
            id            VARCHAR(50) PRIMARY KEY,
            url           TEXT,
            title         TEXT,
            price         BIGINT,
            rooms         INTEGER,
            area          NUMERIC(7,2),
            floor         INTEGER,
            floors_total  INTEGER,
            district      TEXT,
            address       TEXT,
            house_type    TEXT,
            year_built    INTEGER,
            description   TEXT,
            photos        TEXT,
            parsed_at     TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id          SERIAL PRIMARY KEY,
            listing_id  VARCHAR(50) REFERENCES listings(id),
            price       BIGINT,
            recorded_at TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("✅ База данных готова!")


# ==============================
# 2. ПАРСЕР КАРТОЧКИ
# ==============================

def parse_listing_card(card):
    try:
        # --- ID объявления ---
        # В HTML: <div class="a-card" data-id="1013046395">
        listing_id = card.get("data-id", "")
        if not listing_id:
            return None

        # --- Ссылка ---
        # В HTML: <a class="a-card__title" href="/a/show/1013046395">
        link_tag = card.select_one("a.a-card__title")
        url = BASE_URL + link_tag.get("href", "") if link_tag else ""

        # --- Заголовок (комнаты, площадь, этаж) ---
        # В HTML: "4-комнатная квартира · 56 м² · 1/5 этаж"
        title = link_tag.get_text(strip=True) if link_tag else ""

        # Парсим комнаты, площадь, этаж из заголовка
        rooms, area, floor, floors_total = None, None, None, None
        if title:
            parts = title.split("·")
            # parts[0] = "4-комнатная квартира"
            # parts[1] = "56 м²"
            # parts[2] = "1/5 этаж"
            for part in parts:
                part = part.strip()
                if "комнатная" in part or "комн" in part:
                    digits = "".join(filter(str.isdigit, part))
                    rooms = int(digits) if digits else None
                elif "м²" in part:
                    try:
                        area = float(part.replace("м²", "").strip())
                    except ValueError:
                        pass
                elif "этаж" in part and "/" in part:
                    floor_part = part.replace("этаж", "").strip()
                    try:
                        floor = int(floor_part.split("/")[0].strip())
                        floors_total = int(floor_part.split("/")[1].strip())
                    except (ValueError, IndexError):
                        pass

        # --- Цена ---
        # В HTML: <div class="a-card__price">36 600 000 <span>₸</span></div>
        price = 0
        price_tag = card.select_one(".a-card__price")
        if price_tag:
            # Убираем дочерние теги (span с ₸) и берём только текст
            price_text = price_tag.get_text(strip=True)
            digits = "".join(filter(str.isdigit, price_text))
            price = int(digits) if digits else 0

        # --- Адрес и район ---
        # В HTML: <div class="a-card__subtitle">Алмалинский р-н, Брусиловского — Дуйсенова</div>
        address, district = "", ""
        addr_tag = card.select_one(".a-card__subtitle")
        if addr_tag:
            address = addr_tag.get_text(strip=True)
            district = address.split(",")[0].strip()

        # --- Описание (тип дома, год, состояние) ---
        # В HTML: <div class="a-card__text-preview">панельный дом, 1979 г.п., состояние...</div>
        description = ""
        house_type = ""
        year_built = None
        desc_tag = card.select_one(".a-card__text-preview")
        if desc_tag:
            description = desc_tag.get_text(strip=True)
            # Вытаскиваем тип дома из описания
            for t in ["панельный", "кирпичный", "монолитный", "блочный", "деревянный"]:
                if t in description.lower():
                    house_type = t
                    break
            # Вытаскиваем год постройки — ищем "1979 г.п." или похожее
            import re
            year_match = re.search(r'(\d{4})\s*г\.?п\.?', description)
            if year_match:
                year_built = int(year_match.group(1))

        # --- Фото ---
        # В HTML: <source srcset="...400x300.webp 1x, ..."> — берём первый srcset
        photos = []
        for source in card.select("picture source"):
            srcset = source.get("srcset", "")
            if srcset and "webp" in srcset:
                # Берём последнюю ссылку в srcset — она самая большая
                last_url = srcset.split(",")[-1].strip().split(" ")[0]
                if last_url.startswith("http") and last_url not in photos:
                    photos.append(last_url)
                    break  # достаточно одного фото с карточки

        return {
            "id":           listing_id,
            "url":          url,
            "title":        title,
            "price":        price,
            "rooms":        rooms,
            "area":         area,
            "floor":        floor,
            "floors_total": floors_total,
            "district":     district,
            "address":      address,
            "house_type":   house_type,
            "year_built":   year_built,
            "description":  description,
            "photos":       json.dumps(photos, ensure_ascii=False),
            "parsed_at":    datetime.now(),
        }

    except Exception as e:
        print(f"  ⚠️ Ошибка карточки: {e}")
        return None


# ==============================
# 3. СОХРАНЕНИЕ В БД
# ==============================

def save_listing(conn, data):
    cur = conn.cursor()
    now = datetime.now()

    cur.execute("SELECT price FROM listings WHERE id = %s", (data["id"],))
    existing = cur.fetchone()

    if existing:
        old_price = existing[0]
        if old_price != data["price"] and data["price"] > 0:
            cur.execute(
                "INSERT INTO price_history (listing_id, price, recorded_at) VALUES (%s, %s, %s)",
                (data["id"], data["price"], now)
            )
            print(f"  📈 Цена изменилась: {old_price:,} → {data['price']:,}")
            cur.execute(
                "UPDATE listings SET price = %s, parsed_at = %s WHERE id = %s",
                (data["price"], now, data["id"])
            )
    else:
        cur.execute("""
            INSERT INTO listings
                (id, url, title, price, rooms, area, floor, floors_total,
                    district, address, house_type, year_built, description, photos, parsed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            data["id"], data["url"], data["title"], data["price"],
            data["rooms"], data["area"], data["floor"], data["floors_total"],
            data["district"], data["address"], data["house_type"],
            data["year_built"], data["description"], data["photos"], data["parsed_at"]
        ))
        if data["price"] > 0:
            cur.execute(
                "INSERT INTO price_history (listing_id, price, recorded_at) VALUES (%s, %s, %s)",
                (data["id"], data["price"], now)
            )

    conn.commit()


# ==============================
# 4. ГЛАВНАЯ ФУНКЦИЯ
# ==============================

def run_parser(city="almaty", category="arenda", max_pages=10):
    init_db()
    conn = psycopg2.connect(**DB_CONFIG)
    total_saved = 0

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/{category}/kvartiry/{city}/?page={page}"
        print(f"\n📄 Страница {page}: {url}")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"  ❌ Статус {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.select(".a-card")

            if not cards:
                print("  ⚠️ Карточки не найдены")
                break

            print(f"  Найдено карточек: {len(cards)}")

            for card in cards:
                data = parse_listing_card(card)
                if not data:
                    continue
                save_listing(conn, data)
                total_saved += 1
                print(f"  ✓ {data['title']} | {data['price']:,} ₸ | {data['district']}")

        except requests.RequestException as e:
            print(f"  ❌ Сетевая ошибка: {e}")

        time.sleep(2)

    conn.close()
    print(f"\n✅ Готово! Сохранено: {total_saved} объявлений")


if __name__ == "__main__":
    run_parser(
        city="almaty",
        category="arenda",
        max_pages=20
    )
