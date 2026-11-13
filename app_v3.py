import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from html import escape
from pathlib import Path

from ui_helpers import get_photo_urls, listing_markup, photo_html, price_label, safe_url

st.set_page_config(page_title="Krisha Analytics", page_icon="🏠", layout="wide")

DB_CONFIG = {
    "host":     st.secrets["DB_HOST"],
    "port":     int(st.secrets["DB_PORT"]),
    "dbname":   st.secrets["DB_NAME"],
    "user":     st.secrets["DB_USER"],
    "password": st.secrets["DB_PASSWORD"]
}


st.markdown("<style>" + (Path(__file__).parent / "assets/styles.css").read_text() + "</style>", unsafe_allow_html=True)

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


# UI state uses listing IDs, so sorting does not change the selected apartment.
for key, default in {
    "selected_listing": None, "photo_index": 0, "preview_open": True,
    "favorites": set(), "favorites_only": False, "display_count": 12,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def select_listing(listing_id):
    st.session_state.selected_listing = listing_id
    st.session_state.photo_index = 0
    st.session_state.preview_open = True


def toggle_favorite(listing_id):
    favorites = set(st.session_state.favorites)
    if listing_id in favorites:
        favorites.remove(listing_id)
    else:
        favorites.add(listing_id)
    st.session_state.favorites = favorites


def toggle_preview():
    st.session_state.preview_open = not st.session_state.preview_open


def toggle_favorites_view():
    st.session_state.favorites_only = not st.session_state.favorites_only
    st.session_state.display_count = 12


def sync_luxury_default():
    st.session_state.only_good = st.session_state.luxury


def change_photo(index):
    st.session_state.photo_index = index


def show_more():
    st.session_state.display_count += 12


def renovation_html(row):
    good = row["predicted_class"] == "GOOD"
    label = "Хороший ремонт" if good else "Требует внимания"
    return (f'<div class="renovation {"good" if good else "bad"}">{label}'
            f' · уверенность {row["predicted_confidence"]:.0%}</div>')


with st.container(key="topbar"):
    brand, favorites_button = st.columns([4, 1])
    brand.markdown('<div class="brand"><i class="brand-icon"></i>Krisha <span>Analytics</span></div>', unsafe_allow_html=True)
    favorites_button.button(
        f'{"♥" if st.session_state.favorites_only else "♡"} Избранное · {len(st.session_state.favorites)}',
        key="favorites_view", on_click=toggle_favorites_view, use_container_width=True,
    )

st.markdown('''<div class="intro"><div class="eyebrow">НАЙТИ СВОЁ МЕСТО</div>
<h1>Квартира, которая вам подходит<span class="accent">.</span></h1>
<p>Рассматривайте детали. Сравнивайте цены. Выбирайте своё.</p></div>''', unsafe_allow_html=True)

try:
    df = load_data()
except psycopg2.Error:
    st.error("Не удалось загрузить объявления из базы. Проверьте подключение и настройки DB_* в Streamlit Secrets.")
    st.stop()

if df.empty:
    st.info("В базе пока нет подходящих объявлений. После загрузки и оценки квартир они появятся здесь.")
    st.stop()

# Same filter semantics and defaults as before; controls moved from the sidebar.
districts = sorted(df["district"].unique())
rooms_options = sorted(int(r) for r in df["rooms"].unique())
price_min_all = int(df["price"].min())
price_max_all = int(df["price"].quantile(0.98))
default_high = int(df["price"].quantile(0.75))

with st.container(key="search_panel"):
    district_col, rooms_col, budget_col = st.columns([2, 1.4, 2])
    with district_col:
        selected_districts = st.multiselect(
            "Район", districts, default=districts,
            format_func=lambda d: d.replace(" р-н", ""), key="districts",
        ) or districts
    with rooms_col:
        selected_rooms = st.multiselect("Комнаты", rooms_options, default=rooms_options, key="rooms") or rooms_options
    with budget_col:
        if price_min_all < price_max_all:
            budget_low, budget_high = st.slider(
                "Бюджет, ₸", min_value=price_min_all, max_value=price_max_all,
                value=(price_min_all, default_high), step=10000, key="budget",
            )
        else:
            budget_low, budget_high = price_min_all, price_max_all
        st.caption(f"{format_price(budget_low)} — {format_price(budget_high)}")
    with st.expander("Ещё фильтры и настройки модели"):
        advanced1, advanced2, advanced3 = st.columns(3)
        with advanced1:
            luxury_mode = st.toggle("Бюджет не ограничен", key="luxury", on_change=sync_luxury_default)
            if luxury_mode:
                st.caption("Как и раньше, рейтинг учитывает только ремонт. Диапазон цены выше продолжает действовать.")
        with advanced2:
            only_good = st.checkbox("Только с хорошим ремонтом", value=luxury_mode, key="only_good")
        with advanced3:
            min_confidence = st.slider("Мин. уверенность модели", 0.5, 1.0, 0.6, 0.05, key="confidence")
    st.caption("Фильтры применяются автоматически · данные обновляются каждые 5 минут")

# Preserve the original filtering, scoring and sorting pipeline.
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
if st.session_state.favorites_only:
    filtered = filtered[filtered["id"].isin(st.session_state.favorites)]

# Reserve the toolbar below the viewer while evaluating its sorting first.
preview_slot = st.container()
with st.container(key="results_header"):
    heading, sorting = st.columns([3, 1])
    title = "Избранное" if st.session_state.favorites_only else "Все квартиры"
    heading.markdown(f'<div class="eyebrow">ВАША ПОДБОРКА</div><div class="catalog-title">{title}<span>{len(filtered)} объявлений</span></div>', unsafe_allow_html=True)
    sort_option = sorting.selectbox("Сортировка", ["По выгоде", "Дешевле", "Дороже", "Больше площадь"], key="sorting")

sort_map = {
    "По выгоде": ("deal_score", False), "Дешевле": ("price", True),
    "Дороже": ("price", False), "Больше площадь": ("area", False),
}
sort_col, ascending = sort_map[sort_option]
filtered = filtered.sort_values(sort_col, ascending=ascending)
filter_signature = (tuple(selected_districts), tuple(selected_rooms), budget_low, budget_high,
                    min_confidence, only_good, luxury_mode, sort_option, st.session_state.favorites_only)
if st.session_state.get("filter_signature") != filter_signature:
    st.session_state.display_count = 12
    st.session_state.filter_signature = filter_signature

ids = filtered["id"].tolist()
if ids and st.session_state.selected_listing not in ids:
    st.session_state.selected_listing = ids[0]
    st.session_state.photo_index = 0


def render_neighbor(row, previous):
    direction = "Предыдущая" if previous else "Следующая"
    key = "previous" if previous else "next"
    with st.container(key=f"{key}_listing"):
        st.button(
            f"{'←' if previous else '→'} {direction} квартира", key=key,
            disabled=row is None, use_container_width=True,
            on_click=select_listing, args=(row["id"] if row is not None else None,),
        )
        if row is not None:
            photos = get_photo_urls(row["photos"], limit=1)
            st.markdown(photo_html(photos[0] if photos else "", "neighbor", str(row["title"])), unsafe_allow_html=True)
            st.markdown(f'<div class="neighbor-price">{price_label(row["price"])}</div>', unsafe_allow_html=True)
            st.caption(f'{int(row["rooms"])}-комн. · {row["area"]:.0f} м²')


def render_preview():
    index = ids.index(st.session_state.selected_listing)
    row = filtered.iloc[index]
    photos = get_photo_urls(row["photos"], limit=None)
    photo_count = len(photos)
    st.session_state.photo_index = st.session_state.photo_index % max(photo_count, 1)
    photo_index = st.session_state.photo_index
    with st.container(key="preview_header"):
        label, collapse = st.columns([3, 1])
        label.markdown(f'<div class="section-title">Большой просмотр<span>{index + 1} из {len(ids)}</span></div>', unsafe_allow_html=True)
        collapse.button("Скрыть просмотр ↑" if st.session_state.preview_open else "Показать просмотр ↓",
                        key="toggle_preview", on_click=toggle_preview, use_container_width=True)
    if not st.session_state.preview_open:
        return
    previous, main, following = st.columns([1, 5, 1], gap="medium")
    with previous:
        render_neighbor(filtered.iloc[index-1] if index else None, True)
    with following:
        render_neighbor(filtered.iloc[index+1] if index+1 < len(ids) else None, False)
    with main:
        st.markdown(listing_markup(row, "hero", photo_index), unsafe_allow_html=True)
        back, counter, forward, favorite = st.columns([1, 2, 1, 1])
        back.button("← Фото", key="photo_previous", disabled=photo_count < 2,
                    on_click=change_photo, args=((photo_index-1) % max(photo_count, 1),), use_container_width=True)
        counter.caption(f"Фото {photo_index+1} из {photo_count}" if photo_count else "Нет фотографий")
        forward.button("Фото →", key="photo_next", disabled=photo_count < 2,
                       on_click=change_photo, args=((photo_index+1) % max(photo_count, 1),), use_container_width=True)
        favorite.button("♥" if row["id"] in st.session_state.favorites else "♡", key="hero_favorite",
                        help="Добавить или убрать из избранного", on_click=toggle_favorite,
                        args=(row["id"],), use_container_width=True)
        # A window of thumbnails keeps large galleries usable without rendering every image.
        start = (photo_index // 5) * 5
        if photos:
            thumbnails = st.columns(min(5, photo_count))
            for offset, url in enumerate(photos[start:start+5]):
                number = start+offset
                with thumbnails[offset]:
                    st.markdown(photo_html(url, "thumb", f"Фото {number+1}"), unsafe_allow_html=True)
                    st.button(f"{number+1}" + (" · выбрано" if number == photo_index else ""),
                              key=f"thumb_{number}", on_click=change_photo, args=(number,),
                              type="primary" if number == photo_index else "secondary", use_container_width=True)
        with st.container(key="listing_details"):
            description, valuation = st.columns([1.3, 1])
            with description:
                st.markdown(f'<div class="price">{price_label(row["price"])}</div><div class="listing-title">{int(row["rooms"])}-комнатная квартира · {row["area"]:.0f} м²</div><div class="meta">{escape(str(row["district"]))}</div><div class="meta">{escape(str(row["title"]))}</div>', unsafe_allow_html=True)
                st.markdown(renovation_html(row), unsafe_allow_html=True)
                st.caption(f'Рейтинг предложения: {row["deal_score"]:.0f} / 100')
            with valuation:
                market_price = row["рыночная_цена_м2"] * row["area"]
                st.markdown(f'<div class="estimate">Оценка рынка: {price_label(market_price)}</div>', unsafe_allow_html=True)
                st.caption("Среднее по району, комнатам и ценовому сегменту. Не гарантия стоимости сделки.")
                url = safe_url(row["url"])
                if url:
                    st.link_button("Открыть на krisha.kz ↗", url, type="primary", use_container_width=True)
                else:
                    st.caption("Ссылка на объявление недоступна")


with preview_slot:
    if ids:
        render_preview()

if not ids:
    st.info("В избранном нет квартир с такими параметрами." if st.session_state.favorites_only
            else "Квартир с такими параметрами нет. Попробуйте расширить бюджет или снизить порог уверенности модели.")
    if st.session_state.favorites_only:
        st.button("Вернуться ко всем квартирам", on_click=toggle_favorites_view)
else:
    st.caption("Выберите «Смотреть», чтобы открыть квартиру в большом просмотре. Избранное сохраняется на время сессии.")


def render_card(row):
    selected = row["id"] == st.session_state.selected_listing
    key = f'flat_{"selected_" if selected else ""}{row["id"]}'
    with st.container(key=key):
        st.markdown(listing_markup(row), unsafe_allow_html=True)
        st.markdown(f'<div class="card-price">{price_label(row["price"])}</div><div class="card-title">{int(row["rooms"])}-комн. · {row["area"]:.0f} м²</div><div class="meta">{escape(str(row["district"]))}</div><div class="card-title">{escape(str(row["title"]))}</div>', unsafe_allow_html=True)
        st.markdown(renovation_html(row), unsafe_allow_html=True)
        st.markdown(f'<div class="card-score">Рейтинг <b>{row["deal_score"]:.0f}/100</b> · Рынок {format_price(row["рыночная_цена_м2"] * row["area"])}</div>', unsafe_allow_html=True)
        open_button, save_button = st.columns([3, 1])
        open_button.button("Выбрано ↗" if selected else "Смотреть ↗", key=f'open_{row["id"]}',
                           on_click=select_listing, args=(row["id"],), use_container_width=True,
                           type="primary" if selected else "secondary")
        save_button.button("♥" if row["id"] in st.session_state.favorites else "♡", key=f'save_{row["id"]}',
                           on_click=toggle_favorite, args=(row["id"],), help="Избранное", use_container_width=True)
        url = safe_url(row["url"])
        if url:
            st.link_button("На krisha.kz ↗", url, use_container_width=True)


cards = filtered.head(st.session_state.display_count)
with st.container(key="results_grid"):
    for offset in range(0, len(cards), 4):
        columns = st.columns(4)
        for column, (_, row) in zip(columns, cards.iloc[offset:offset+4].iterrows()):
            with column:
                render_card(row)

if st.session_state.display_count < len(filtered):
    _, more, _ = st.columns([2, 1, 2])
    more.button(f"Показать ещё · {min(12, len(filtered)-st.session_state.display_count)} ↓",
                key="show_more", on_click=show_more, use_container_width=True)

st.markdown('<div class="footer">Krisha Analytics · Анализ объявлений и качества ремонта<br>Оценка основана на объявлениях из базы. Проверяйте актуальную цену и условия на krisha.kz.</div>', unsafe_allow_html=True)
