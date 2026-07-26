"""
Скрипт для ручной разметки квартир — GOOD/BAD.

Как работает:
1. Берёт квартиры у которых ещё нет класса (renovation_class IS NULL)
2. Открывает страницу объявления на krisha.kz (с удобной галереей фото)
3. Ты смотришь и решаешь — подходит квартира (хороший ремонт) или нет
4. Записывает в базу, переходит к следующей

Чтобы выйти в любой момент — введи "q"
"""

import psycopg2
import webbrowser
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


def run_labeling(target_per_class=80):
    """
    target_per_class: сколько квартир размечать на каждый класс GOOD/BAD
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        SELECT renovation_class, COUNT(*)
        FROM listings
        WHERE renovation_class IS NOT NULL
        GROUP BY renovation_class
    """)
    current_counts = dict(cur.fetchall())

    print("="*50)
    print("РАЗМЕТКА КВАРТИР — GOOD / BAD")
    print("="*50)
    print("\nТекущий прогресс:")
    for cls in ["GOOD", "BAD"]:
        count = current_counts.get(cls, 0)
        bar = "█" * min(count, 50) + "░" * max(0, min(target_per_class, 50) - count)
        print(f"  {cls:5}: {bar} {count}/{target_per_class}")
    print()

    cur.execute("""
        SELECT id, title, price, area, district, url
        FROM listings
        WHERE renovation_class IS NULL
          AND url IS NOT NULL
        ORDER BY RANDOM()
    """)
    rows = cur.fetchall()

    print(f"Доступно для разметки: {len(rows)} квартир")
    print("\nG или пробел — GOOD (хороший ремонт, можно показывать)")
    print("B             — BAD  (плохой/средний ремонт, не показываем)")
    print("s             — пропустить (skip)")
    print("q             — выйти и сохранить прогресс\n")
    input("Нажми Enter чтобы начать...")

    labeled_this_session = 0

    for listing_id, title, price, area, district, url in rows:

        print("\n" + "-"*50)
        print(f"{title}")
        print(f"💰 {price:,} тг | {area} м² | {district}")
        print(f"🔗 {url}")

        webbrowser.open_new_tab(url)

        while True:
            answer = input("GOOD или BAD? [g/b/s/q]: ").strip().lower()

            if answer == "q":
                conn.close()
                print(f"\n✅ Сессия завершена. Размечено сейчас: {labeled_this_session}")
                return

            if answer == "s":
                break

            if answer in ["g", ""]:
                cls = "GOOD"
            elif answer == "b":
                cls = "BAD"
            else:
                print("  ⚠️ Неверный ввод, попробуй снова (g/b/s/q)")
                continue

            cur.execute(
                "UPDATE listings SET renovation_class = %s WHERE id = %s",
                (cls, listing_id)
            )
            conn.commit()
            labeled_this_session += 1
            current_counts[cls] = current_counts.get(cls, 0) + 1
            print(f"  ✓ Сохранено как {cls}")
            break

        if all(current_counts.get(c, 0) >= target_per_class for c in ["GOOD", "BAD"]):
            print("\n🎉 Оба класса набрали достаточно примеров!")
            break

    conn.close()
    print(f"\n✅ Готово! Размечено в этой сессии: {labeled_this_session}")


if __name__ == "__main__":
    run_labeling(target_per_class=200)