"""
Модуль для предобработки данных: очистка, фильтрация, создание индексов
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

from .data_loader import load_data
from .utils import save_preprocessed_data, load_preprocessed_data


def process_impression(impression_list):
    """Разделение impressions на клики и не-клики"""
    if pd.isna(impression_list):
        return [], []
    list_of_strings = impression_list.split()
    click = [x.split('-')[0] for x in list_of_strings if x.split('-')[1] == '1']
    non_click = [x.split('-')[0] for x in list_of_strings if x.split('-')[1] == '0']
    return click, non_click


def preprocess_behaviour(raw_behaviour, min_click_cutoff=100):
    """Основная предобработка behaviour данных"""
    print(" Предобработка данных...")

    # Добавляем колонки с кликами и не-кликами
    raw_behaviour['click'], raw_behaviour['noclicks'] = zip(*raw_behaviour['impressions'].map(process_impression))

    # Конвертируем timestamp в часы с эпохи
    raw_behaviour['epochhrs'] = pd.to_datetime(raw_behaviour['timestamp']).values.astype(np.int64) / (1e6) / 1000 / 3600
    raw_behaviour['epochhrs'] = raw_behaviour['epochhrs'].round()

    # Разворачиваем click_history для добавления дополнительных взаимодействий
    raw_behaviour = raw_behaviour.explode("click").reset_index(drop=True)

    # Удаляем строки с NaN в click
    raw_behaviour = raw_behaviour.dropna(subset=['click'])

    click_history = raw_behaviour[["userId", "click_history"]].drop_duplicates().dropna()
    click_history["click_history"] = click_history.click_history.map(lambda x: x.split() if isinstance(x, str) else [])
    click_history = click_history.explode("click_history").rename(columns={"click_history": "click"})
    click_history = click_history.dropna(subset=['click'])

    if len(click_history) > 0:
        click_history["epochhrs"] = raw_behaviour.epochhrs.min()
        click_history["noclicks"] = pd.Series([[] for _ in range(len(click_history.index))])
        raw_behaviour = pd.concat([raw_behaviour, click_history], axis=0).reset_index(drop=True)

    # Удаляем редкие статьи (cold start problem)
    item_counts = raw_behaviour.groupby("click")["userId"].transform('size')
    raw_behaviour = raw_behaviour[item_counts >= min_click_cutoff].reset_index(drop=True)

    if len(raw_behaviour) == 0:
        print("ВНИМАНИЕ: После удаления редких статей не осталось данных!")
        return raw_behaviour

    click_set = set(raw_behaviour['click'].unique())
    raw_behaviour['noclicks'] = raw_behaviour['noclicks'].apply(
        lambda impressions: [impression for impression in impressions if impression in click_set]
    )

    print(f"   - После предобработки: {len(raw_behaviour)} взаимодействий")
    print(f"   - Уникальных пользователей: {raw_behaviour.userId.nunique()}")
    print(f"   - Уникальных статей: {raw_behaviour.click.nunique()}")

    return raw_behaviour


def create_indices(train_df):
    """Создание индексов для пользователей и предметов"""
    if len(train_df) == 0:
        return {}, {}, {}, {}

    # Индексы для предметов
    ind2item = {idx + 1: itemid for idx, itemid in enumerate(train_df.click.unique())}
    item2ind = {itemid: idx for idx, itemid in ind2item.items()}

    # Индексы для пользователей
    ind2user = {idx + 1: userid for idx, userid in enumerate(train_df['userId'].unique())}
    user2ind = {userid: idx for idx, userid in ind2user.items()}

    print(f"   - Создано индексов для статей: {len(ind2item)}")
    print(f"   - Создано индексов для пользователей: {len(ind2user)}")

    return ind2item, item2ind, ind2user, user2ind


def train_test_split(behaviour, test_ratio=0.1):
    """Временное разделение на train/validation"""
    if len(behaviour) == 0:
        return behaviour.copy(), behaviour.copy()

    unique_times = behaviour['epochhrs'].nunique()

    if unique_times < 2:
        print("Недостаточно уникальных временных меток. Использую случайное разделение.")
        n_train = int(len(behaviour) * (1 - test_ratio))
        indices = np.random.permutation(len(behaviour))
        train_idx = indices[:n_train]
        valid_idx = indices[n_train:]
        train = behaviour.iloc[train_idx].copy()
        valid = behaviour.iloc[valid_idx].copy()
    else:
        test_time_th = behaviour['epochhrs'].quantile(1 - test_ratio)
        train = behaviour[behaviour['epochhrs'] < test_time_th].copy()
        valid = behaviour[behaviour['epochhrs'] >= test_time_th].copy()

        if len(train) == 0:
            print("Train пустой. Уменьшаю порог разделения.")
            sorted_times = sorted(behaviour['epochhrs'].unique())
            mid_point = len(sorted_times) // 2
            test_time_th = sorted_times[mid_point]
            train = behaviour[behaviour['epochhrs'] < test_time_th].copy()
            valid = behaviour[behaviour['epochhrs'] >= test_time_th].copy()

    print(f"   - Train размер: {len(train):,}, Validation размер: {len(valid):,}")
    return train, valid


def apply_indices(df, item2ind, user2ind):
    """Применение индексов к датафрейму"""
    if len(df) == 0:
        return df

    df['click'] = df['click'].map(lambda item: item2ind.get(item, 0))
    df['noclicks'] = df['noclicks'].map(lambda list_of_items: [item2ind.get(l, 0) for l in list_of_items])
    df['userIdx'] = df['userId'].map(lambda x: user2ind.get(x, 0))
    return df


def run_preprocessing(data_path="MINDsmall_train", min_click_cutoff=100, save=True):
    """Запуск полной предобработки данных"""
    print("\nЗАПУСК ПРЕДОБРАБОТКИ ДАННЫХ")
    print("=" * 50)

    # Временно отключаем загрузку сохраненных данных
    # preprocessed = load_preprocessed_data()
    # if preprocessed is not None:
    #     print("   Использую сохраненные предобработанные данные")
    #     return preprocessed

    # Загружаем сырые данные
    raw_behaviour, news = load_data(data_path)

    # Предобработка
    behaviour = preprocess_behaviour(raw_behaviour, min_click_cutoff)

    # Сохраняем только если нужно и данные не пустые
    if save and len(behaviour) > 0:
        try:
            save_preprocessed_data(behaviour)
        except Exception as e:
            print(f"Не удалось сохранить данные: {e}")

    return behaviour, news

def create_indices_and_split(data_path="MINDsmall_train", test_ratio=0.2, min_click_cutoff=100):
    """Создание индексов и разделение данных (использует предобработанные данные)"""
    print("\n СОЗДАНИЕ ИНДЕКСОВ И РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 50)

    # Загружаем предобработанные данные
    behaviour, news = run_preprocessing(data_path, min_click_cutoff, save=False)

    if len(behaviour) == 0:
        print(" Нет данных для обработки!")
        return None, None, None, None, None, None

    # Разделяем на train/valid
    train, valid = train_test_split(behaviour, test_ratio)

    # Создаем индексы на основе train
    ind2item, item2ind, ind2user, user2ind = create_indices(train)

    # Применяем индексы
    train = apply_indices(train, item2ind, user2ind)
    valid = apply_indices(valid, item2ind, user2ind)

    print("\n Итоги:")
    print(f"   - Train: {len(train)} записей")
    print(f"   - Valid: {len(valid)} записей")
    print(f"   - Пользователей в обучении: {len(user2ind)}")
    print(f"   - Статей в обучении: {len(item2ind)}")

    return train, valid, item2ind, user2ind, ind2item, ind2user