"""
Проверка предсказаний модели — active learning.

Как работает:
1. Показывает квартиры которые модель уже оценила (GOOD/BAD)
2. Ты смотришь на объявление и решаешь — модель права или нет
3. Если права — жмёшь Enter, ничего не меняется
4. Если модель ошиблась — вводишь правильный класс (g/b)
   Это сохраняется в колонку human_correction

Эти исправленные примеры потом используются при переобучении —
они самые ценные, потому что показывают где именно модель путается.
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


def ensure_column(conn):
    cur = conn.cursor()
    cur.execute("""
        ALTER TABLE listings
        ADD COLUMN IF NOT EXISTS human_correction VARCHAR(10)
    """)
    conn.commit()


def run_review():
    conn = psycopg2.connect(**DB_CONFIG)
    ensure_column(conn)
    cur = conn.cursor()

    # Считаем сколько уже проверено
    cur.execute("""
        SELECT COUNT(*) FROM listings
        WHERE predicted_class IS NOT NULL
          AND predicted_class != 'ARCHIVED'
    """)
    total_predicted = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM listings
        WHERE human_correction IS NOT NULL
    """)
    total_corrected = cur.fetchone()[0]

    print("="*55)
    print("ПРОВЕРКА ПРЕДСКАЗАНИЙ МОДЕЛИ")
    print("="*55)
    print(f"\nВсего предсказано моделью: {total_predicted}")
    print(f"Уже найдено ошибок и исправлено: {total_corrected}\n")

    # Случайный порядок — чтобы поймать и "уверенные, но неправильные" случаи,
    # а не только те где модель сама сомневается
    order = "RANDOM()"

    cur.execute(f"""
        SELECT id, title, price, area, district, url, predicted_class, predicted_confidence
        FROM listings
        WHERE predicted_class IN ('BAD')
          AND human_correction IS NULL
        ORDER BY {order}
    """)
    rows = cur.fetchall()

    print(f"Доступно для проверки: {len(rows)}")
    print("\nEnter        — модель права, пропустить")
    print("g            — модель ошиблась, на самом деле GOOD")
    print("b            — модель ошиблась, на самом деле BAD")
    print("s            — не уверен, пропустить не сохраняя")
    print("q            — выйти\n")
    input("Нажми Enter чтобы начать...")

    corrected_this_session = 0
    confirmed_this_session = 0

    for listing_id, title, price, area, district, url, pred_class, confidence in rows:

        print("\n" + "-"*55)
        print(f"{title}")
        print(f"💰 {price:,} тг | {area} м² | {district}")
        print(f"🤖 Модель считает: {pred_class} (уверенность: {confidence})")
        print(f"🔗 {url}")

        webbrowser.open_new_tab(url)

        answer = input("Модель права? [Enter=да / g / b / s / q]: ").strip().lower()

        if answer == "q":
            break

        if answer == "s" or answer == "":
            if answer == "":
                confirmed_this_session += 1
                # Помечаем как подтверждённое (не ошибка, но видели)
                cur.execute(
                    "UPDATE listings SET human_correction = %s WHERE id = %s",
                    (pred_class, listing_id)  # сохраняем то же что предсказала модель — как подтверждение
                )
                conn.commit()
                print("  ✓ Подтверждено")
            continue

        if answer in ["g", "b"]:
            correct_class = "GOOD" if answer == "g" else "BAD"
            if correct_class != pred_class:
                cur.execute(
                    "UPDATE listings SET human_correction = %s WHERE id = %s",
                    (correct_class, listing_id)
                )
                conn.commit()
                corrected_this_session += 1
                print(f"  ✓ Исправлено: было {pred_class} → стало {correct_class}")
            else:
                print("  (это и так был предсказанный класс)")

    conn.close()
    print(f"\n✅ Сессия завершена")
    print(f"   Найдено и исправлено ошибок: {corrected_this_session}")
    print(f"   Подтверждено верных: {confirmed_this_session}")


if __name__ == "__main__":
    run_review()
