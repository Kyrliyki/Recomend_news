"""
Модуль для обучения модели
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .model import NewsMF
from .dataset import MindDataset
from .preprocessing import create_indices_and_split
from .utils import save_model_and_indices, load_model_and_indices, check_model_exists
from .config import Config


def train_model(data_path="MINDsmall_train", batch_size=1024, embedding_dim=50,
                epochs=50, test_ratio=0.2, model_path="news_mf_model.pt",
                indices_path="model_indices.pkl", force_retrain=False):
    """Основной пайплайн обучения модели"""

    # Проверяем, не загрузить ли существующую модель
    if not force_retrain and check_model_exists(model_path, indices_path):
        print("🎯 Найдена сохраненная модель. Используйте force_retrain=True для переобучения")
        model, item2ind, user2ind, ind2item, ind2user = load_model_and_indices(
            model_path, indices_path
        )
        if model is not None:
            return model, item2ind, user2ind, ind2item, ind2user

    print("\n🧠 ОБУЧЕНИЕ НОВОЙ МОДЕЛИ")
    print("=" * 50)

    # Создаем индексы и разделяем данные
    train, valid, item2ind, user2ind, ind2item, ind2user = create_indices_and_split(
        data_path, test_ratio
    )

    if train is None or len(train) == 0:
        print(" Ошибка: недостаточно данных для обучения!")
        return None, None, None, None, None

    # Создаем DataLoader'ы
    ds_train = MindDataset(train)
    ds_valid = MindDataset(valid)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n Данные для обучения:")
    print(f"   - Train батчей: {len(train_loader)}")
    print(f"   - Valid батчей: {len(valid_loader)}")
    print(f"   - Пользователей: {len(user2ind)}")
    print(f"   - Статей: {len(item2ind)}")

    # Создаем модель (+1 для UNK индекса)
    num_users = len(user2ind) + 1
    num_items = len(item2ind) + 1

    model = NewsMF(
        num_users=num_users,
        num_items=num_items,
        dim=embedding_dim,
        lr=Config.LEARNING_RATE
    )

    # Обучаем
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=Config.ACCELERATOR,
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    print(f"\n Запуск обучения на {epochs} эпохах...")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # Сохраняем модель
    save_model_and_indices(model, item2ind, user2ind, ind2item, ind2user, model_path, indices_path)

    print(f"\n Обучение завершено!")
    return model, item2ind, user2ind, ind2item, ind2user


def evaluate_model(model_path="news_mf_model.pt", indices_path="model_indices.pkl",
                   data_path="MINDsmall_train"):
    """Оценка и анализ обученной модели"""
    print("\n АНАЛИЗ МОДЕЛИ")
    print("=" * 50)

    # Загружаем модель
    model, item2ind, user2ind, ind2item, ind2user = load_model_and_indices(
        model_path, indices_path
    )

    if model is None:
        print(" Не удалось загрузить модель")
        return None

    # Загружаем данные о новостях
    from .data_loader import load_data
    _, news = load_data(data_path)

    # Анализируем эмбеддинги
    print("\n Анализ эмбеддингов статей:")
    itememb = model.itememb.weight.detach()
    print(f"   - Форма эмбеддингов: {itememb.shape}")

    # Поиск похожих статей (если есть пример)
    if item2ind and len(item2ind) > 0:
        sample_article = list(item2ind.keys())[0]
        article_idx = item2ind[sample_article]

        if article_idx < len(itememb):
            similarity = torch.nn.functional.cosine_similarity(
                itememb[article_idx].unsqueeze(0), itememb, dim=1
            )
            most_similar_idx = similarity.argsort(descending=True)[1:6]

            print(f"\n Статьи, похожие на {sample_article}:")
            for i, idx in enumerate(most_similar_idx, 1):
                if idx.item() in ind2item:
                    article_id = ind2item[idx.item()]
                    article_info = news[news['itemId'] == article_id]
                    if len(article_info) > 0:
                        title = article_info.iloc[0]['title'][:50]
                        print(f"   {i}. {title}... (схожесть: {similarity[idx].item():.3f})")

    print("\n Анализ завершен")
    return model