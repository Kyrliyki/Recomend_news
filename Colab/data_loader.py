"""
Модуль для загрузки и исследования данных
"""

import pandas as pd
from pathlib import Path
import numpy as np


def load_data(data_path="MINDsmall_train"):
    """Загрузка raw данных из папки с датасетом"""
    behaviors_path = Path(data_path) / "behaviors.tsv"
    news_path = Path(data_path) / "news.tsv"

    if not behaviors_path.exists() or not news_path.exists():
        raise FileNotFoundError(f"Датасет не найден в {data_path}")

    raw_behaviour = pd.read_csv(
        behaviors_path,
        sep="\t",
        names=["impressionId", "userId", "timestamp", "click_history", "impressions"]
    )

    news = pd.read_csv(
        news_path,
        sep="\t",
        names=["itemId", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    )

    return raw_behaviour, news


def load_and_explore(data_path="MINDsmall_train"):
    """Загрузка данных и вывод базовой информации (EDA)"""
    print("\n ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
    print("=" * 50)

    raw_behaviour, news = load_data(data_path)

    print(f"\n Поведенческие данные (behaviors):")
    print(f"   - Количество записей: {len(raw_behaviour):,}")
    print(f"   - Колонки: {list(raw_behaviour.columns)}")
    print(f"   - Уникальных пользователей: {raw_behaviour['userId'].nunique():,}")
    print(f"   - Уникальных сессий: {raw_behaviour['impressionId'].nunique():,}")

    print(f"\n Новостные данные (news):")
    print(f"   - Количество статей: {len(news):,}")
    print(f"   - Колонки: {list(news.columns)}")
    print(f"   - Уникальных категорий: {news['category'].nunique()}")
    print(f"   - Уникальных подкатегорий: {news['subcategory'].nunique()}")

    # Анализ кликов
    click_history_col = raw_behaviour['click_history'].dropna()
    if len(click_history_col) > 0:
        click_history_lengths = click_history_col.apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        print(f"\n Анализ истории кликов:")
        print(f"   - Средняя длина истории: {click_history_lengths.mean():.1f}")
        print(f"   - Медианная длина: {click_history_lengths.median():.0f}")
        print(f"   - Макс. длина истории: {click_history_lengths.max()}")

    # Анализ показов
    impressions = raw_behaviour['impressions'].dropna()
    if len(impressions) > 0:
        impressions_counts = impressions.apply(lambda x: len(x.split()))
        print(f"\n Анализ показов:")
        print(f"   - Среднее количество показов на сессию: {impressions_counts.mean():.1f}")
        print(f"   - Медиана показов: {impressions_counts.median():.0f}")
        print(f"   - Макс. показов: {impressions_counts.max()}")

    return raw_behaviour, news