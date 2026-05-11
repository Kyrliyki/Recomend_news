"""
Утилитарные функции: сохранение/загрузка, логирование, проверки
"""

import pickle
import os
from pathlib import Path
import torch
import pandas as pd

from .model import NewsMF


def save_model_and_indices(model, item2ind, user2ind, ind2item, ind2user,
                          model_path="news_mf_model.pt", indices_path="model_indices.pkl"):
    """Сохранение модели и индексов"""
    # Сохраняем веса модели
    torch.save(model.state_dict(), model_path)

    # Сохраняем индексы и метаданные
    indices_data = {
        'item2ind': item2ind,
        'user2ind': user2ind,
        'ind2item': ind2item,
        'ind2user': ind2user,
        'num_users': model.num_users,
        'num_items': model.num_items,
        'dim': model.dim
    }

    with open(indices_path, 'wb') as f:
        pickle.dump(indices_data, f)

    print(f" Модель сохранена в {model_path}")
    print(f" Индексы сохранены в {indices_path}")


def load_model_and_indices(model_path="news_mf_model.pt", indices_path="model_indices.pkl", device=None):
    """Загрузка модели и индексов"""
    if not os.path.exists(model_path) or not os.path.exists(indices_path):
        print(f" Сохраненная модель не найдена: {model_path} или {indices_path}")
        return None, None, None, None, None

    try:
        # Загружаем индексы
        with open(indices_path, 'rb') as f:
            indices_data = pickle.load(f)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Создаем модель с теми же параметрами
        model = NewsMF(
            num_users=indices_data['num_users'],
            num_items=indices_data['num_items'],
            dim=indices_data['dim']
        )

        # Загружаем веса
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        print(f" Модель загружена из {model_path}")
        print(f"   - Пользователей: {indices_data['num_users'] - 1}")
        print(f"   - Предметов: {indices_data['num_items'] - 1}")
        print(f"   - Размерность: {indices_data['dim']}")

        return (model, indices_data['item2ind'], indices_data['user2ind'],
                indices_data['ind2item'], indices_data['ind2user'])

    except Exception as e:
        print(f" Ошибка при загрузке модели: {e}")
        return None, None, None, None, None


def check_model_exists(model_path="news_mf_model.pt", indices_path="model_indices.pkl"):
    """Проверка существования сохраненной модели"""
    return os.path.exists(model_path) and os.path.exists(indices_path)


def get_model_info(model_path="news_mf_model.pt", indices_path="model_indices.pkl"):
    """Получение информации о сохраненной модели"""
    if not check_model_exists(model_path, indices_path):
        return None

    try:
        with open(indices_path, 'rb') as f:
            indices_data = pickle.load(f)

        return {
            "Количество пользователей": indices_data['num_users'] - 1,
            "Количество статей": indices_data['num_items'] - 1,
            "Размерность эмбеддингов": indices_data['dim'],
            "Файл модели": model_path,
            "Файл индексов": indices_path
        }
    except Exception as e:
        return {"Ошибка": str(e)}


def save_preprocessed_data(df, path="preprocessed_behaviour.pkl"):
    """Сохранение предобработанных данных (использует pickle)"""
    try:
        with open(path, 'wb') as f:
            pickle.dump(df, f)
        print(f" Предобработанные данные сохранены в {path}")
        return path
    except Exception as e:
        print(f" Ошибка сохранения: {e}")
        # Fallback на CSV
        csv_path = path.replace('.pkl', '.csv')
        df.to_csv(csv_path, index=False)
        print(f" Данные сохранены в CSV: {csv_path}")
        return csv_path


def load_preprocessed_data(path="preprocessed_behaviour.pkl"):
    """Загрузка предобработанных данных"""
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                df = pickle.load(f)
            print(f" Предобработанные данные загружены из {path}")
            return df
        except Exception as e:
            print(f"️ Не удалось загрузить pickle: {e}")
            # Пробуем CSV
            csv_path = path.replace('.pkl', '.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                print(f" Данные загружены из CSV: {csv_path}")
                return df
    else:
        # Пробуем CSV
        csv_path = path.replace('.pkl', '.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f" Данные загружены из CSV: {csv_path}")
            return df

    print(f" Файл {path} не найден")
    return None