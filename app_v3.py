import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import json
import re

DB_CONFIG = {
    "host":     st.secrets["DB_HOST"],
    "port":     int(st.secrets["DB_PORT"]),
    "dbname":   st.secrets["DB_NAME"],
    "user":     st.secrets["DB_USER"],
    "password": st.secrets["DB_PASSWORD"]
}

st.set_page_config(page_title="Krisha Analytics", page_icon="🏠", layout="wide")

st.markdown("""
<style>
.card-photo {
    width: 100%; height: 400px; object-fit: cover;
    border-radius: 8px; display: block;
}
.card-photo-placeholder {
    width: 100%; height: 400px; display: flex; align-items: center;
    justify-content: center; background: #F0F0EC; border-radius: 8px;
}
.card-title-line {
    font-weight: 600; font-size: 15px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; margin-top: 6px;
}
.card-meta-line {
    color: #767672; font-size: 13px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# ДАННЫЕ
# ==============================

@st.cache_data(ttl=300)
def load_data():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT
            id, title, price, area, rooms, district, url, photos,
            predicted_class, predicted_confidence
        FROM listings
        WHERE price > 0
          AND area > 0
          AND rooms IS NOT NULL
          AND district LIKE '%р-н%'
          AND predicted_class IN ('GOOD', 'BAD')
    """, conn)
    conn.close()

    df["цена_м2"] = df["price"] / df["area"]

    # Рыночная цена — внутри ценового сегмента (эконом/средний/премиум)
    # района+комнатности, чтобы премиум и бюджет не мешали друг другу
    df["сегмент"] = pd.qcut(df["цена_м2"], q=3, labels=["эконом", "средний", "премиум"], duplicates="drop")
    market = df.groupby(["district", "rooms", "сегмент"], observed=True)["цена_м2"].mean().reset_index()
    market = market.rename(columns={"цена_м2": "рыночная_цена_м2"})
    df = df.merge(market, on=["district", "rooms", "сегмент"], how="left")

    df["скидка_%"] = ((df["рыночная_цена_м2"] - df["цена_м2"]) / df["рыночная_цена_м2"] * 100).round(1)
    return df


def get_photo_urls(photos_json, limit=3):
    """Дедупликация фото по номеру кадра в имени файла (не по всему URL)."""
    try:
        photos = json.loads(photos_json) if photos_json else []
    except Exception:
        return []

    seen, unique = set(), []
    for url in photos:
        match = re.search(r"/(\d+)-\d+x\d+\.\w+$", url)
        pid = match.group(1) if match else url
        if pid not in seen:
            seen.add(pid)
            unique.append(url)
        if len(unique) >= limit:
            break
    return unique


def compute_scores(df, luxury_mode=False):
    """Векторизованный расчёт score сразу для всего датафрейма — быстро."""
    renovation_score = np.where(
        df["predicted_class"] == "GOOD",
        df["predicted_confidence"],
        1 - df["predicted_confidence"]
    )
    if luxury_mode:
        return (renovation_score * 100).round(1)

    discount_score = (df["скидка_%"] / 40).clip(0, 1)
    return ((renovation_score * 0.55 + discount_score * 0.45) * 100).round(1)


def format_price(price):
    if price >= 1_000_000:
        return f"{price/1_000_000:.2f} млн ₸"
    return f"{price:,.0f} ₸"


def format_price_short(price):
    if price >= 1_000_000:
        return f"{price/1_000_000:.1f}М"
    return f"{price/1000:.0f}К"


# ==============================
# ИНТЕРФЕЙС — заголовок и статистика (нативные st.metric)
# ==============================

st.title("🏠 Krisha Analytics")
st.caption("ИИ анализирует фото и находит квартиры с хорошим ремонтом по цене ниже рынка")

df = load_data()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Квартир в базе", f"{len(df):,}")
m2.metric("Средняя выгода", f"{df[df['скидка_%'] > 0]['скидка_%'].mean():.0f}%")
m3.metric("Средняя экономия", format_price((df[df['скидка_%'] > 0]['скидка_%'] / 100 * df[df['скидка_%'] > 0]['price']).mean()))
m4.metric("С хорошим ремонтом", f"{(df['predicted_class'] == 'GOOD').sum():,}")

st.divider()


# ==============================
# ФИЛЬТРЫ — нативный st.pills (готовый chip-селектор Streamlit,
# без единой строчки кастомного CSS для кнопок)
# ==============================

with st.sidebar:
    st.header("Фильтры")

    districts = sorted(df["district"].unique())
    district_labels = [d.replace(" р-н", "") for d in districts]
    label_to_district = dict(zip(district_labels, districts))

    selected_labels = st.pills(
        "Район", district_labels, selection_mode="multi",
        default=district_labels
    )
    selected_districts = [label_to_district[l] for l in selected_labels] if selected_labels else districts

    st.divider()

    rooms_options = sorted(df["rooms"].unique())
    selected_rooms = st.pills(
        "Комнат", [int(r) for r in rooms_options], selection_mode="multi",
        default=[int(r) for r in rooms_options]
    )
    if not selected_rooms:
        selected_rooms = [int(r) for r in rooms_options]

    st.divider()

    luxury_mode = st.toggle("Бюджет не ограничен")
    if luxury_mode:
        st.caption("Сортировка идёт по качеству ремонта, а не по скидке.")

    price_min_all = int(df["price"].min())
    price_max_all = int(df["price"].quantile(0.98))
    default_high = int(df["price"].quantile(0.75))

    budget_low, budget_high = st.slider(
        "Бюджет, ₸",
        min_value=price_min_all, max_value=price_max_all,
        value=(price_min_all, default_high), step=10000
    )
    st.caption(f"{format_price(budget_low)} — {format_price(budget_high)}")

    st.divider()
    only_good = st.checkbox("Только с хорошим ремонтом", value=luxury_mode)
    min_confidence = st.slider("Мин. уверенность модели", 0.5, 1.0, 0.6, 0.05)


# ==============================
# ФИЛЬТРАЦИЯ И СОРТИРОВКА
# ==============================

filtered = df[
    df["district"].isin(selected_districts) &
    df["rooms"].isin(selected_rooms) &
    (df["price"] >= budget_low) &
    (df["price"] <= budget_high) &
    (df["predicted_confidence"] >= min_confidence)
].copy()

if only_good:
    filtered = filtered[filtered["predicted_class"] == "GOOD"]

filtered["deal_score"] = compute_scores(filtered, luxury_mode)

top_col1, top_col2 = st.columns([3, 1])
top_col1.subheader(f"Найдено: {len(filtered)}")
sort_option = top_col2.selectbox(
    "Сортировка", ["По выгоде", "Дешевле", "Дороже", "Больше площадь"],
    label_visibility="collapsed"
)

sort_map = {
    "По выгоде": ("deal_score", False),
    "Дешевле": ("price", True),
    "Дороже": ("price", False),
    "Больше площадь": ("area", False),
}
sort_col, ascending = sort_map[sort_option]
filtered = filtered.sort_values(sort_col, ascending=ascending)

if "display_count" not in st.session_state:
    st.session_state.display_count = 12


# ==============================
# СЕТКА КАРТОЧЕК — 3 колонки, нативный st.container(border=True)
# ==============================

def render_card(row):
    with st.container(border=True):
        photos = get_photo_urls(row["photos"])
        if photos:
            st.markdown(f'<img class="card-photo" src="{photos[0]}">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card-photo-placeholder">📷 Нет фото</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="card-title-line">{row["title"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-meta-line">📍 {row["district"]} · {row["area"]:.0f} м² · {int(row["rooms"])} комн.</div>',
            unsafe_allow_html=True
        )

        # --- Ремонт (всегда ровно одна строка) ---
        conf = row["predicted_confidence"]
        if row["predicted_class"] == "GOOD":
            (st.success if conf >= 0.85 else st.warning)(f"Ремонт хороший ({conf:.0%})", icon="✅" if conf >= 0.85 else "🟡")
        else:
            (st.error if conf >= 0.85 else st.warning)(f"Ремонт слабый ({conf:.0%})", icon="❌" if conf >= 0.85 else "🟡")

        # --- Цена и скидка ---
        price_col, score_col = st.columns([2, 1])
        with price_col:
            st.markdown(f"### {format_price(row['price'])}")
            st.caption(f"Рынок: {format_price(row['рыночная_цена_м2'] * row['area'])}")
        with score_col:
            st.metric("Score", f"{row['deal_score']:.0f}")

        # --- Статус цены — показываем ВСЕГДА (даже "на уровне рынка"),
        # чтобы у всех карточек было одинаковое число строк и высота ---
        discount = row["скидка_%"]
        if discount >= 5:
            savings = discount / 100 * row["price"]
            st.success(f"↓ Ниже рынка на {discount:.0f}% · экономия {format_price(savings)}", icon="💰")
        elif discount <= -5:
            st.error(f"↑ Выше рынка на {abs(discount):.0f}%", icon="⚠️")
        else:
            st.info("На уровне рынка", icon="⚪")

        st.link_button("Открыть на krisha.kz →", row["url"], use_container_width=True)


cards = filtered.head(st.session_state.display_count)
grid_cols = st.columns(3)

for i, (_, row) in enumerate(cards.iterrows()):
    with grid_cols[i % 3]:
        render_card(row)

# --- Загрузить ещё ---
if st.session_state.display_count < len(filtered):
    st.write("")
    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        remaining = len(filtered) - st.session_state.display_count
        if st.button(f"⬇️ Загрузить ещё ({min(15, remaining)})", use_container_width=True):
            st.session_state.display_count += 15
            st.rerun()