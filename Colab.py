# main.py - С СОХРАНЕНИЕМ И ЗАГРУЗКОЙ МОДЕЛИ
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pytorch_lightning as pl
from pathlib import Path
import warnings
import pickle
import os

warnings.filterwarnings('ignore')


# ============================================
# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# ============================================

def load_data(data_path="MINDsmall_train"):
    """Загрузка raw данных из папки с датасетом"""
    behaviors_path = Path(data_path) / "behaviors.tsv"
    news_path = Path(data_path) / "news.tsv"

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


def process_impression(impression_list):
    """Разделение impressions на клики и не-клики"""
    if pd.isna(impression_list):
        return [], []
    list_of_strings = impression_list.split()
    click = [x.split('-')[0] for x in list_of_strings if x.split('-')[1] == '1']
    non_click = [x.split('-')[0] for x in list_of_strings if x.split('-')[1] == '0']
    return click, non_click


def preprocess_behaviour(raw_behaviour):
    """Основная предобработка behaviour данных"""
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
    min_click_cutoff = 100
    item_counts = raw_behaviour.groupby("click")["userId"].transform('size')
    raw_behaviour = raw_behaviour[item_counts >= min_click_cutoff].reset_index(drop=True)

    if len(raw_behaviour) == 0:
        print("ВНИМАНИЕ: После удаления редких статей не осталось данных!")
        return raw_behaviour

    click_set = set(raw_behaviour['click'].unique())
    raw_behaviour['noclicks'] = raw_behaviour['noclicks'].apply(
        lambda impressions: [impression for impression in impressions if impression in click_set]
    )

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

    return ind2item, item2ind, ind2user, user2ind


def train_test_split(behaviour, test_ratio=0.1):
    """Временное разделение на train/validation"""
    if len(behaviour) == 0:
        return behaviour.copy(), behaviour.copy()

    # Используем quantile, но с проверкой на уникальные значения
    unique_times = behaviour['epochhrs'].nunique()

    if unique_times < 2:
        # Если все времена одинаковы, используем обычное случайное разделение
        print("ВНИМАНИЕ: Недостаточно уникальных временных меток. Использую случайное разделение.")
        n_train = int(len(behaviour) * (1 - test_ratio))
        indices = np.random.permutation(len(behaviour))
        train_idx = indices[:n_train]
        valid_idx = indices[n_train:]
        train = behaviour.iloc[train_idx].copy()
        valid = behaviour.iloc[valid_idx].copy()
    else:
        # Используем временное разделение
        test_time_th = behaviour['epochhrs'].quantile(1 - test_ratio)
        train = behaviour[behaviour['epochhrs'] < test_time_th].copy()
        valid = behaviour[behaviour['epochhrs'] >= test_time_th].copy()

        # Если train пустой, уменьшаем порог
        if len(train) == 0:
            print("ВНИМАНИЕ: Train пустой. Уменьшаю порог разделения.")
            sorted_times = sorted(behaviour['epochhrs'].unique())
            mid_point = len(sorted_times) // 2
            test_time_th = sorted_times[mid_point]
            train = behaviour[behaviour['epochhrs'] < test_time_th].copy()
            valid = behaviour[behaviour['epochhrs'] >= test_time_th].copy()

    return train, valid


def apply_indices(df, item2ind, user2ind):
    """Применение индексов к датафрейму"""
    if len(df) == 0:
        return df

    df['click'] = df['click'].map(lambda item: item2ind.get(item, 0))
    df['noclicks'] = df['noclicks'].map(lambda list_of_items: [item2ind.get(l, 0) for l in list_of_items])
    df['userIdx'] = df['userId'].map(lambda x: user2ind.get(x, 0))
    return df


def save_model_and_indices(model, item2ind, user2ind, ind2item, ind2user, model_path="news_mf_model.pt",
                           indices_path="model_indices.pkl"):
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

    print(f"Модель сохранена в {model_path}")
    print(f"Индексы сохранены в {indices_path}")


def load_model_and_indices(model_class, model_path="news_mf_model.pt", indices_path="model_indices.pkl", device=None):
    """Загрузка модели и индексов"""
    if not os.path.exists(model_path) or not os.path.exists(indices_path):
        print("Сохраненная модель не найдена.")
        return None, None, None, None, None

    try:
        # Загружаем индексы
        with open(indices_path, 'rb') as f:
            indices_data = pickle.load(f)

        # Создаем модель с теми же параметрами
        model = model_class(
            num_users=indices_data['num_users'],
            num_items=indices_data['num_items'],
            dim=indices_data['dim']
        )

        # Загружаем веса
        model.load_state_dict(torch.load(model_path, map_location=device))

        print(f"Модель успешно загружена из {model_path}")
        print(f"Загружено пользователей: {indices_data['num_users'] - 1}, предметов: {indices_data['num_items'] - 1}")

        return model, indices_data['item2ind'], indices_data['user2ind'], indices_data['ind2item'], indices_data[
            'ind2user']

    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None, None, None, None, None


# ============================================
# 2. DATASET И DATALOADER
# ============================================

class MindDataset(Dataset):
    """PyTorch Dataset для MIND данных"""

    def __init__(self, df):
        if len(df) == 0:
            self.data = {
                'userIdx': torch.tensor([]),
                'click': torch.tensor([])
            }
        else:
            self.data = {
                'userIdx': torch.tensor(df.userIdx.values.astype(np.int64)),
                'click': torch.tensor(df.click.values.astype(np.int64))
            }

    def __len__(self):
        return len(self.data['userIdx'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}


# ============================================
# 3. МОДЕЛЬ
# ============================================

class NewsMF(pl.LightningModule):
    """Matrix Factorization модель для рекомендаций новостей"""

    def __init__(self, num_users, num_items, dim=50, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.dim = dim
        self.num_users = num_users
        self.num_items = num_items
        self.lr = lr

        self.useremb = nn.Embedding(num_embeddings=num_users, embedding_dim=dim)
        self.itememb = nn.Embedding(num_embeddings=num_items, embedding_dim=dim)

    def forward(self, user_idx, item_idx):
        """Прямой проход для получения скора"""
        user_vec = self.useremb(user_idx)
        item_vec = self.itememb(item_idx)
        return (user_vec * item_vec).sum(-1)

    def step(self, batch, batch_idx):
        uservec = self.useremb(batch['userIdx'])
        itemvec_click = self.itememb(batch['click'])

        # Негативная семплирование
        neg_sample = torch.randint(1, self.num_items, batch['click'].shape, device=self.device)
        itemvec_noclick = self.itememb(neg_sample)

        score_click = torch.sigmoid((uservec * itemvec_click).sum(-1).unsqueeze(-1))
        score_noclick = torch.sigmoid((uservec * itemvec_noclick).sum(-1).unsqueeze(-1))

        scores_all = torch.concat((score_click, score_noclick), dim=1)
        target_all = torch.concat((torch.ones_like(score_click), torch.zeros_like(score_noclick)), dim=1)

        loss = F.binary_cross_entropy(scores_all, target_all)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================
# 4. ФУНКЦИИ ДЛЯ АНАЛИЗА И РЕКОМЕНДАЦИЙ
# ============================================

def get_recommendations(model, user_id, user2ind, ind2item, news_df, top_k=10):
    """Получение рекомендаций для пользователя"""
    if user_id not in user2ind:
        print(f"Пользователь {user_id} не найден в модели")
        return []

    user_idx = user2ind[user_id]
    user_tensor = torch.tensor([user_idx])

    # Получаем эмбеддинги всех предметов
    all_item_embeddings = model.itememb.weight.detach()
    user_embedding = model.useremb(user_tensor).detach()

    # Вычисляем схожесть
    scores = torch.matmul(user_embedding, all_item_embeddings.T).squeeze()

    # Получаем топ-k предметов (исключая индекс 0)
    top_k_indices = torch.argsort(scores, descending=True)[:top_k]
    top_k_indices = [idx.item() for idx in top_k_indices if idx != 0]

    # Получаем ID предметов и информацию
    recommendations = []
    for idx in top_k_indices:
        if idx in ind2item:
            item_id = ind2item[idx]
            item_info = news_df[news_df['itemId'] == item_id]
            if len(item_info) > 0:
                recommendations.append({
                    'itemId': item_id,
                    'category': item_info.iloc[0]['category'],
                    'title': item_info.iloc[0]['title'],
                    'score': scores[idx].item()
                })

    return recommendations


def analyze_results(news, item2ind, model):
    """Анализ результатов обучения"""
    if not item2ind:
        print("Нет данных для анализа")
        return news

    # Добавляем информацию об индексах и количестве кликов
    news["ind"] = news["itemId"].map(item2ind)
    news = news.sort_values("ind").reset_index(drop=True)

    # Получаем эмбеддинги предметов
    itememb = model.itememb.weight.detach()
    print(f"Shape of item embeddings: {itememb.shape}")

    # Поиск похожих статей
    article_id = "N16636"
    ind = item2ind.get(article_id)

    if ind is not None and ind < len(itememb):
        similarity = torch.nn.functional.cosine_similarity(itememb[ind], itememb, dim=1)
        most_sim = news[~news.ind.isna()].iloc[(similarity.argsort(descending=True).numpy() - 1)]

        print(f"\n=== Статьи, похожие на {article_id} ===")
        print(most_sim[['itemId', 'category', 'title']].head(10).to_string())
    else:
        print(f"\nСтатья {article_id} не найдена в индексах")

    return news


# ============================================
# 5. ОСНОВНАЯ ФУНКЦИЯ
# ============================================

def main(data_path="MINDsmall_train", batch_size=1024, embedding_dim=50, epochs=50, test_ratio=0.2,
         force_retrain=False, model_path="news_mf_model.pt", indices_path="model_indices.pkl"):
    """Основной пайплайн обучения модели"""

    # Пытаемся загрузить существующую модель
    if not force_retrain and os.path.exists(model_path) and os.path.exists(indices_path):
        print("=" * 60)
        print("НАЙДЕНА СОХРАНЕННАЯ МОДЕЛЬ")
        print("=" * 60)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, item2ind, user2ind, ind2item, ind2user = load_model_and_indices(
            NewsMF, model_path, indices_path, device
        )

        if model is not None:
            print("\nМодель успешно загружена!")

            # Загружаем новостные данные для анализа
            _, news = load_data(data_path)

            print("\n" + "=" * 60)
            print("АНАЛИЗ ЗАГРУЖЕННОЙ МОДЕЛИ")
            print("=" * 60)

            # Анализ результатов
            news_analyzed = analyze_results(news, item2ind, model)

            # Пример получения рекомендаций
            print("\n" + "=" * 60)
            print("ПРИМЕР РЕКОМЕНДАЦИЙ")
            print("=" * 60)

            # Берем первых 3 пользователей из индексов
            sample_users = list(user2ind.keys())[:3]
            for user_id in sample_users:
                print(f"\nРекомендации для пользователя {user_id}:")
                recommendations = get_recommendations(model, user_id, user2ind, ind2item, news_analyzed, top_k=5)
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec['title'][:50]}... ({rec['category']}) - score: {rec['score']:.4f}")

            return model, news_analyzed, ind2item, ind2user, item2ind, user2ind

    # Если модель не найдена или force_retrain=True, обучаем новую
    print("=" * 60)
    print("ОБУЧЕНИЕ НОВОЙ МОДЕЛИ")
    print("=" * 60)

    # Загрузка данных
    raw_behaviour, news = load_data(data_path)
    print(f"Исходный датасет содержит {len(raw_behaviour)} взаимодействий")
    print(f"Новостей в датасете: {len(news)}")

    print("\n" + "=" * 60)
    print("ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 60)

    # Предобработка
    behaviour = preprocess_behaviour(raw_behaviour)
    print(f"После предобработки: {len(behaviour)} взаимодействий")

    if len(behaviour) == 0:
        print("ОШИБКА: Нет данных после предобработки!")
        return None, None, None, None, None, None

    print(f"Уникальных пользователей: {behaviour.userId.nunique()}")
    print(f"Уникальных статей: {behaviour.click.nunique()}")

    # Train/test split
    train, valid = train_test_split(behaviour, test_ratio=test_ratio)
    print(f"Train размер: {len(train)}, Validation размер: {len(valid)}")

    if len(train) == 0:
        print("ОШИБКА: Train пустой после разделения!")
        return None, None, None, None, None, None

    # Создание индексов
    ind2item, item2ind, ind2user, user2ind = create_indices(train)
    print(f"Предметов в train: {len(ind2item)}")
    print(f"Пользователей в train: {len(ind2user)}")

    # Применение индексов
    train = apply_indices(train, item2ind, user2ind)
    valid = apply_indices(valid, item2ind, user2ind)

    print("\n" + "=" * 60)
    print("СОЗДАНИЕ DATALOADER")
    print("=" * 60)

    # Создание Dataset и DataLoader
    ds_train = MindDataset(train)
    ds_valid = MindDataset(valid)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train батчей: {len(train_loader)}")
    print(f"Validation батчей: {len(valid_loader)}")

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 60)

    # Создание и обучение модели
    num_users = len(ind2user) + 1  # +1 для UNK индекса
    num_items = len(ind2item) + 1  # +1 для UNK индекса

    print(f"Количество пользователей: {num_users}")
    print(f"Количество предметов: {num_items}")

    model = NewsMF(
        num_users=num_users,
        num_items=num_items,
        dim=embedding_dim,
        lr=1e-3
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ МОДЕЛИ")
    print("=" * 60)

    # Сохраняем модель и индексы
    save_model_and_indices(model, item2ind, user2ind, ind2item, ind2user, model_path, indices_path)

    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    # Анализ результатов
    news_analyzed = analyze_results(news, item2ind, model)

    # Вывод топ-5 самых популярных статей
    if len(train) > 0:
        print("\n=== Топ-5 самых кликаемых статей в train ===")
        click_counts = Counter(train.click)
        top_articles = pd.DataFrame({
            'ind': list(click_counts.keys()),
            'n_clicks': list(click_counts.values())
        })
        top_articles = top_articles.sort_values('n_clicks', ascending=False).head(5)

        # Соединяем с информацией о статьях
        result = news_analyzed[news_analyzed['ind'].notna()].merge(
            top_articles, on='ind', how='inner'
        )
        if len(result) > 0:
            print(result[['itemId', 'category', 'title', 'n_clicks']].to_string())
        else:
            print("Не удалось найти информацию о топ-статьях")

    return model, news_analyzed, ind2item, ind2user, item2ind, user2ind


# ============================================
# 6. ТОЧКА ВХОДА
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Обучение recommender системы на MIND датасете")
    parser.add_argument("--data_path", type=str, default="MINDsmall_train",
                        help="Путь к папке с датасетом")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Размер батча")
    parser.add_argument("--embedding_dim", type=int, default=50,
                        help="Размерность эмбеддингов")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Количество эпох")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Доля данных для валидации")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Принудительно переобучить модель, игнорируя сохраненную")
    parser.add_argument("--model_path", type=str, default="news_mf_model.pt",
                        help="Путь для сохранения/загрузки модели")
    parser.add_argument("--indices_path", type=str, default="model_indices.pkl",
                        help="Путь для сохранения/загрузки индексов")

    args = parser.parse_args()

    # Запуск обучения
    model, news_df, ind2item, ind2user, item2ind, user2ind = main(
        data_path=args.data_path,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        test_ratio=args.test_ratio,
        force_retrain=args.force_retrain,
        model_path=args.model_path,
        indices_path=args.indices_path
    )