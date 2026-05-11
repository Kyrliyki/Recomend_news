"""
Система рекомендаций на основе обученной модели
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from .model import NewsMF
from .utils import load_model_and_indices
from .data_loader import load_data


class NewsRecommender:
    """Система рекомендаций новостей"""

    def __init__(self, model, item2ind, user2ind, ind2item, ind2user, news_df=None):
        self.model = model
        self.item2ind = item2ind
        self.user2ind = user2ind
        self.ind2item = ind2item
        self.ind2user = ind2user
        self.device = next(model.parameters()).device
        self.news_df = news_df

    @classmethod
    def from_saved(cls, model_path="news_mf_model.pt", indices_path="model_indices.pkl",
                   data_path="MINDsmall_train"):
        """Загрузка рекомендательной системы из сохраненных файлов"""
        model, item2ind, user2ind, ind2item, ind2user = load_model_and_indices(
            model_path, indices_path
        )

        if model is None:
            raise FileNotFoundError("Модель не найдена. Сначала обучите модель.")

        # Загружаем данные о новостях
        try:
            _, news_df = load_data(data_path)
        except:
            news_df = None

        return cls(model, item2ind, user2ind, ind2item, ind2user, news_df)

    def _normalize_user_id(self, user_id):
        """Нормализация ID пользователя"""
        if user_id is None:
            return None

        if user_id in self.user2ind:
            return user_id

        # Пробуем альтернативный регистр
        if user_id.startswith('u'):
            normalized = 'U' + user_id[1:]
            if normalized in self.user2ind:
                return normalized
        if user_id.startswith('U'):
            normalized = 'u' + user_id[1:]
            if normalized in self.user2ind:
                return normalized

        return None

    def recommend_for_user(self, user_id, top_k=10):
        """Рекомендации для пользователя"""
        normalized_id = self._normalize_user_id(user_id)

        if normalized_id is None:
            print(f" Пользователь {user_id} не найден")
            print(f"   Доступные пользователи (первые 5): {list(self.user2ind.keys())[:5]}")
            return []

        user_idx = self.user2ind[normalized_id]
        user_tensor = torch.tensor([user_idx]).to(self.device)

        with torch.no_grad():
            user_emb = self.model.useremb(user_tensor)
            all_item_embs = self.model.itememb.weight
            scores = torch.matmul(user_emb, all_item_embs.T).squeeze()
            scores_np = scores.cpu().numpy()

        # Сортируем и формируем рекомендации
        sorted_indices = np.argsort(scores_np)[::-1]

        recommendations = []
        for idx in sorted_indices:
            if idx == 0:  # пропускаем UNK
                continue

            item_id = self.ind2item.get(idx)
            if item_id:
                prob = 1.0 / (1.0 + np.exp(-scores_np[idx]))
                item_info = self._get_item_info(item_id)

                recommendations.append({
                    'rank': len(recommendations) + 1,
                    'item_id': item_id,
                    'score': float(scores_np[idx]),
                    'probability': float(prob),
                    **item_info
                })

            if len(recommendations) >= top_k:
                break

        return recommendations

    def _get_item_info(self, item_id):
        """Получение информации о новости"""
        if self.news_df is not None:
            item_row = self.news_df[self.news_df['itemId'] == item_id]
            if len(item_row) > 0:
                title = item_row.iloc[0]['title']
                return {
                    'title': title[:100] + "..." if len(title) > 100 else title,
                    'category': item_row.iloc[0]['category'],
                    'subcategory': item_row.iloc[0]['subcategory'],
                }

        return {'title': 'N/A', 'category': 'N/A', 'subcategory': 'N/A'}

    def get_similar_articles(self, article_id, top_k=10):
        """Поиск похожих статей"""
        if article_id not in self.item2ind:
            print(f" Статья {article_id} не найдена")
            return []

        article_idx = self.item2ind[article_id]

        with torch.no_grad():
            article_emb = self.model.itememb.weight[article_idx]
            all_embs = self.model.itememb.weight
            similarity = F.cosine_similarity(article_emb.unsqueeze(0), all_embs, dim=1)
            similarity_np = similarity.cpu().numpy()

        sorted_indices = np.argsort(similarity_np)[::-1]

        similar = []
        for idx in sorted_indices:
            if idx == article_idx or idx == 0:
                continue

            item_id = self.ind2item.get(idx)
            if item_id:
                item_info = self._get_item_info(item_id)
                similar.append({
                    'item_id': item_id,
                    'similarity': float(similarity_np[idx]),
                    **item_info
                })

            if len(similar) >= top_k:
                break

        return similar


def run_recommendation_app(model_path="news_mf_model.pt", indices_path="model_indices.pkl",
                           data_path="MINDsmall_train"):
    """Интерактивное приложение для рекомендаций"""
    print("\n ЗАПУСК РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
    print("=" * 60)

    try:
        recommender = NewsRecommender.from_saved(model_path, indices_path, data_path)
    except FileNotFoundError as e:
        print(f" {e}")
        print("   Сначала обучите модель через пункт меню 4")
        return

    print("\nСистема загружена!")
    print(f"   - Пользователей: {len(recommender.user2ind)}")
    print(f"   - Статей: {len(recommender.item2ind)}")
    print("\n Доступные команды:")
    print("   rec <user_id> [k]  - рекомендации для пользователя")
    print("   similar <article> [k] - похожие статьи")
    print("   users [n]           - список пользователей")
    print("   exit                - выход")

    while True:
        try:
            cmd = input("\n> ").strip().lower()

            if cmd == 'exit':
                print(" До свидания!")
                break

            elif cmd.startswith('rec'):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Пример: rec U13779 10")
                    continue

                user_id = parts[1]
                k = int(parts[2]) if len(parts) > 2 else 10

                recs = recommender.recommend_for_user(user_id, k)
                if recs:
                    print(f"\n Рекомендации для {user_id}:")
                    for r in recs:
                        print(f"\n{r['rank']}. {r['title']}")
                        print(f"   ️ {r['category']} |  вероятность: {r['probability']:.3f}")

            elif cmd.startswith('similar'):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Пример: similar N16636 5")
                    continue

                article_id = parts[1]
                k = int(parts[2]) if len(parts) > 2 else 10

                similar = recommender.get_similar_articles(article_id, k)
                if similar:
                    print(f"\n Похожие на {article_id}:")
                    for i, s in enumerate(similar, 1):
                        print(f"{i}. {s['title']} (схожесть: {s['similarity']:.3f})")

            elif cmd.startswith('users'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 10
                print(f"\n Первые {n} пользователей:")
                for i, user in enumerate(list(recommender.user2ind.keys())[:n], 1):
                    print(f"   {i}. {user}")

            else:
                print("Неизвестная команда. Доступно: rec, similar, users, exit")

        except KeyboardInterrupt:
            print("\n До свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")