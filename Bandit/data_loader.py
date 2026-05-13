"""
Модуль загрузки и предобработки данных
"""

import pandas as pd
import numpy as np
import random
from collections import defaultdict
from .config import DATA_PATH, RANDOM_SEED, NUM_USERS, NUM_INTERACTIONS_PER_USER


class DataLoader:
    """Класс для загрузки и подготовки данных"""

    def __init__(self, data_path=DATA_PATH, random_seed=RANDOM_SEED):
        self.data_path = data_path
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.df = None
        self.categories = None
        self.category_to_id = None
        self.id_to_category = None
        self.num_categories = None
        self.user_sequences = None

    def load_news_data(self):
        """Загрузка новостного датасета"""
        self.df = pd.read_csv(
            self.data_path,
            sep='\t',
            header=None
        )

        self.df.columns = [
            'news_id', 'category', 'subcategory', 'title',
            'abstract', 'url', 'title_entities', 'abstract_entities'
        ]

        print(f"Загружено {len(self.df)} новостей")
        return self.df

    def encode_categories(self):
        """Кодирование категорий в числовые ID"""
        # Оставляем только нужные колонки
        self.df = self.df[['news_id', 'category']]
        self.df.dropna(inplace=True)

        # Кодирование категорий
        self.categories = self.df['category'].unique().tolist()
        self.category_to_id = {c: i for i, c in enumerate(self.categories)}
        self.id_to_category = {i: c for c, i in self.category_to_id.items()}

        self.df['category_id'] = self.df['category'].map(self.category_to_id)
        self.num_categories = len(self.categories)

        print(f"Найдено категорий: {self.num_categories}")
        print(f"Категории: {self.categories}")

        return self.num_categories

    def generate_user_sequences(self, num_users=NUM_USERS,
                                interactions_per_user=NUM_INTERACTIONS_PER_USER):
        """Генерация симулированных пользовательских последовательностей"""
        if self.df is None:
            self.load_news_data()
            self.encode_categories()

        self.user_sequences = defaultdict(list)

        for _ in range(num_users):
            user = f"user_{random.randint(1, num_users // 2)}"
            for _ in range(interactions_per_user):
                article = random.choice(self.df['category_id'].values)
                self.user_sequences[user].append(article)

        print(f"Сгенерировано {len(self.user_sequences)} пользовательских последовательностей")
        return self.user_sequences

    def get_popular_category(self):
        """Получить самую популярную категорию (для бейзлайна)"""
        if self.df is None:
            self.load_news_data()
            self.encode_categories()
        return self.df['category_id'].value_counts().idxmax()

    def get_all_sequences_list(self):
        """Получить все последовательности в виде списка"""
        if self.user_sequences is None:
            self.generate_user_sequences()
        return list(self.user_sequences.values())


# Функции для удобного вызова
def load_and_prepare_data():
    """Быстрая загрузка и подготовка данных"""
    loader = DataLoader()
    loader.load_news_data()
    loader.encode_categories()
    loader.generate_user_sequences()
    return loader