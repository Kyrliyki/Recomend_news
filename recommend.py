# recommend.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')


# ============================================
# 1. ОПРЕДЕЛЕНИЕ МОДЕЛИ
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
        user_vec = self.useremb(user_idx)
        item_vec = self.itememb(item_idx)
        return (user_vec * item_vec).sum(-1)

    def predict_score(self, user_idx, item_idx):
        with torch.no_grad():
            score = torch.sigmoid(self.forward(user_idx, item_idx))
        return score

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================
# 2. ЗАГРУЗЧИК МОДЕЛИ
# ============================================

class ModelLoader:
    """Загрузка сохраненной модели и индексов"""

    def __init__(self, model_path="news_mf_model.pt", indices_path="model_indices.pkl"):
        self.model_path = model_path
        self.indices_path = indices_path
        self.model = None
        self.item2ind = None
        self.user2ind = None
        self.ind2item = None
        self.ind2user = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        """Загрузка модели и индексов"""
        print("=" * 60)
        print("ЗАГРУЗКА МОДЕЛИ РЕКОМЕНДАЦИЙ")
        print("=" * 60)

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
        if not Path(self.indices_path).exists():
            raise FileNotFoundError(f"Файл индексов не найден: {self.indices_path}")

        print(f"Загрузка индексов из {self.indices_path}...")
        with open(self.indices_path, 'rb') as f:
            indices_data = pickle.load(f)

        self.item2ind = indices_data['item2ind']
        self.user2ind = indices_data['user2ind']
        self.ind2item = indices_data['ind2item']
        self.ind2user = indices_data['ind2user']

        print(f"  - Загружено пользователей: {len(self.user2ind)}")
        print(f"  - Загружено статей: {len(self.item2ind)}")

        print(f"Загрузка модели из {self.model_path}...")
        self.model = NewsMF(
            num_users=indices_data['num_users'],
            num_items=indices_data['num_items'],
            dim=indices_data['dim']
        )

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print("✓ Модель успешно загружена!")
        print(f"✓ Устройство: {self.device}")

        return self

    def get_model_info(self):
        """Информация о модели"""
        print("\n" + "=" * 60)
        print("ИНФОРМАЦИЯ О МОДЕЛИ")
        print("=" * 60)
        print(f"Количество пользователей: {self.model.num_users - 1}")
        print(f"Количество статей: {self.model.num_items - 1}")
        print(f"Размерность эмбеддингов: {self.model.dim}")


# ============================================
# 3. СИСТЕМА РЕКОМЕНДАЦИЙ (ИСПРАВЛЕННАЯ)
# ============================================

class NewsRecommender:
    """Система рекомендаций новостей"""

    def __init__(self, model_loader, news_df=None):
        self.loader = model_loader
        self.model = model_loader.model
        self.item2ind = model_loader.item2ind
        self.user2ind = model_loader.user2ind
        self.ind2item = model_loader.ind2item
        self.ind2user = model_loader.ind2user
        self.device = model_loader.device

        # Выводим примеры для отладки
        print(f"\nDEBUG: Первые 5 пользователей в индексе:")
        for i, (user, idx) in enumerate(list(self.user2ind.items())[:5]):
            print(f"  {user} -> {idx}")

        self.news_df = news_df
        if self.news_df is None:
            self.news_df = self._load_news_data()

    def _load_news_data(self, data_path="MINDsmall_train"):
        """Загрузка данных о новостях"""
        try:
            news_path = Path(data_path) / "news.tsv"
            if news_path.exists():
                news_df = pd.read_csv(
                    news_path,
                    sep="\t",
                    names=["itemId", "category", "subcategory", "title", "abstract",
                           "url", "title_entities", "abstract_entities"]
                )
                print(f"Загружено {len(news_df)} новостей")
                news_df['ind'] = news_df['itemId'].map(self.item2ind)
                return news_df
            else:
                print("ВНИМАНИЕ: Файл с новостями не найден")
                return None
        except Exception as e:
            print(f"Ошибка загрузки новостей: {e}")
            return None

    def _normalize_user_id(self, user_id):
        """Нормализация ID пользователя (исправление регистра)"""
        if user_id is None:
            return None

        # Пробуем как есть
        if user_id in self.user2ind:
            return user_id

        # Пробуем с заглавной U
        if user_id.startswith('u'):
            normalized = 'U' + user_id[1:]
            if normalized in self.user2ind:
                return normalized

        # Пробуем с маленькой u
        if user_id.startswith('U'):
            normalized = 'u' + user_id[1:]
            if normalized in self.user2ind:
                return normalized

        return None

    def recommend_for_user_id(self, user_id, top_k=10, exclude_clicked=True):
        """Рекомендации для пользователя по его ID"""

        # Нормализуем ID пользователя
        normalized_id = self._normalize_user_id(user_id)

        if normalized_id is None:
            print(f"❌ Пользователь {user_id} не найден в модели")
            print(f"   Попробуйте одного из этих: {list(self.user2ind.keys())[:10]}")
            return []

        user_idx = self.user2ind[normalized_id]
        print(f"✓ Найден пользователь {normalized_id} с индексом {user_idx}")

        return self.recommend_for_user_idx(user_idx, top_k, exclude_clicked)

    def recommend_for_user_idx(self, user_idx, top_k=10, exclude_clicked=True):
        """Рекомендации для пользователя по индексу"""

        user_tensor = torch.tensor([user_idx]).to(self.device)

        with torch.no_grad():
            user_emb = self.model.useremb(user_tensor)
            all_item_embs = self.model.itememb.weight

            # Вычисляем scores (dot product)
            scores = torch.matmul(user_emb, all_item_embs.T).squeeze()
            scores_np = scores.cpu().numpy()

        # Сортируем по убыванию
        sorted_indices = np.argsort(scores_np)[::-1]

        # Формируем рекомендации
        recommendations = []

        for idx in sorted_indices:
            # Пропускаем UNK индекс (0)
            if idx == 0:
                continue

            # Пропускаем если это тот же пользователь (не нужно, но оставим для порядка)
            if exclude_clicked and hasattr(self, 'user_history'):
                pass  # Можно добавить логику исключения уже просмотренных

            item_id = self.ind2item.get(idx)
            if item_id is None:
                continue

            score = scores_np[idx]
            prob = 1.0 / (1.0 + np.exp(-score))  # sigmoid

            item_info = self._get_item_info(item_id)

            recommendations.append({
                'item_id': item_id,
                'score': float(score),
                'probability': float(prob),
                'rank': len(recommendations) + 1,
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
                abstract = item_row.iloc[0]['abstract']
                return {
                    'title': title[:100] + "..." if len(title) > 100 else title,
                    'category': item_row.iloc[0]['category'],
                    'subcategory': item_row.iloc[0]['subcategory'],
                    'abstract': abstract[:200] + "..." if len(abstract) > 200 else abstract
                }

        return {
            'title': 'N/A',
            'category': 'N/A',
            'subcategory': 'N/A',
            'abstract': 'N/A'
        }

    def get_similar_articles(self, article_id, top_k=10):
        """Поиск похожих статей"""

        # Нормализуем ID статьи
        if article_id not in self.item2ind:
            print(f"❌ Статья {article_id} не найдена")
            print(f"   Примеры статей: {list(self.item2ind.keys())[:5]}")
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

    def predict_click_probability(self, user_id, article_id):
        """Предсказание вероятности клика"""

        # Нормализуем ID
        normalized_user = self._normalize_user_id(user_id)
        if normalized_user is None:
            print(f"❌ Пользователь {user_id} не найден")
            return None

        if article_id not in self.item2ind:
            print(f"❌ Статья {article_id} не найдена")
            return None

        user_idx = self.user2ind[normalized_user]
        item_idx = self.item2ind[article_id]

        user_tensor = torch.tensor([user_idx]).to(self.device)
        item_tensor = torch.tensor([item_idx]).to(self.device)

        with torch.no_grad():
            score = self.model.predict_score(user_tensor, item_tensor)

        return score.item()

    def recommend_popular(self, top_k=10):
        """Рекомендация самых популярных статей"""

        if self.news_df is None:
            print("Нет данных о новостях")
            return []

        # Просто берем первые top_k статей (в реальности нужно считать популярность)
        recommendations = []
        for idx, row in self.news_df.head(top_k).iterrows():
            if row['itemId'] in self.item2ind:
                recommendations.append({
                    'item_id': row['itemId'],
                    'score': 0.0,
                    'probability': 0.5,
                    'title': row['title'][:100] + "..." if len(row['title']) > 100 else row['title'],
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'abstract': row['abstract'][:200] + "..." if len(row['abstract']) > 200 else row['abstract']
                })

        return recommendations


# ============================================
# 4. ПРИЛОЖЕНИЕ ДЛЯ РЕКОМЕНДАЦИЙ
# ============================================

class RecommendationApp:
    """Интерактивное приложение для получения рекомендаций"""

    def __init__(self, model_path="news_mf_model.pt", indices_path="model_indices.pkl"):
        self.loader = ModelLoader(model_path, indices_path)
        self.loader.load()
        self.loader.get_model_info()

        self.recommender = NewsRecommender(self.loader)

    def run_interactive(self):
        """Интерактивный режим"""
        print("\n" + "=" * 60)
        print("СИСТЕМА РЕКОМЕНДАЦИЙ НОВОСТЕЙ")
        print("=" * 60)
        print("\nДоступные команды:")
        print("  1. rec <user_id> [top_k]     - Рекомендации для пользователя")
        print("  2. similar <article_id> [k]  - Похожие статьи")
        print("  3. predict <user> <article>   - Вероятность клика")
        print("  4. list_users [n]             - Список пользователей")
        print("  5. list_articles [n]          - Список статей")
        print("  6. popular [n]                - Популярные статьи")
        print("  7. test                       - Тест с первым пользователем")
        print("  8. exit                       - Выход")
        print("-" * 60)

        while True:
            try:
                command = input("\n> ").strip().lower()

                if command == 'exit':
                    print("До свидания!")
                    break

                elif command == 'test':
                    # Тест с первым пользователем
                    if self.recommender.user2ind:
                        test_user = list(self.recommender.user2ind.keys())[0]
                        print(f"\nТестируем с пользователем: {test_user}")
                        self.show_recommendations(test_user, 5)
                    else:
                        print("Нет пользователей для теста")

                elif command.startswith('rec'):
                    parts = command.split()
                    if len(parts) < 2:
                        print("Укажите ID пользователя. Пример: rec U13779")
                        continue

                    user_id = parts[1]
                    top_k = int(parts[2]) if len(parts) > 2 else 10
                    self.show_recommendations(user_id, top_k)

                elif command.startswith('similar'):
                    parts = command.split()
                    if len(parts) < 2:
                        print("Укажите ID статьи. Пример: similar N16636")
                        continue

                    article_id = parts[1]
                    top_k = int(parts[2]) if len(parts) > 2 else 10
                    self.show_similar_articles(article_id, top_k)

                elif command.startswith('predict'):
                    parts = command.split()
                    if len(parts) < 3:
                        print("Укажите пользователя и статью. Пример: predict U13779 N55689")
                        continue

                    user_id, article_id = parts[1], parts[2]
                    self.show_prediction(user_id, article_id)

                elif command.startswith('list_users'):
                    parts = command.split()
                    n = int(parts[1]) if len(parts) > 1 else 10
                    self.list_users(n)

                elif command.startswith('list_articles'):
                    parts = command.split()
                    n = int(parts[1]) if len(parts) > 1 else 10
                    self.list_articles(n)

                elif command.startswith('popular'):
                    parts = command.split()
                    n = int(parts[1]) if len(parts) > 1 else 10
                    self.show_popular(n)

                else:
                    print("Неизвестная команда")

            except KeyboardInterrupt:
                print("\nДо свидания!")
                break
            except Exception as e:
                print(f"Ошибка: {e}")
                import traceback
                traceback.print_exc()

    def show_recommendations(self, user_id, top_k=10):
        """Показать рекомендации для пользователя"""
        print(f"\n=== Рекомендации для пользователя {user_id} (Top-{top_k}) ===\n")

        recommendations = self.recommender.recommend_for_user_id(user_id, top_k)

        if not recommendations:
            return

        for rec in recommendations:
            print(f"{rec['rank']:2}. {rec['title']}")
            print(f"    Категория: {rec['category']} | Подкатегория: {rec['subcategory']}")
            print(f"    Вероятность клика: {rec['probability']:.3f} | ID: {rec['item_id']}")
            print()

    def show_similar_articles(self, article_id, top_k=10):
        """Показать похожие статьи"""
        print(f"\n=== Статьи, похожие на {article_id} (Top-{top_k}) ===\n")

        original = self.recommender._get_item_info(article_id)
        print(f"Исходная статья: {original['title']}")
        print(f"Категория: {original['category']}\n")

        similar = self.recommender.get_similar_articles(article_id, top_k)

        if not similar:
            print("Не найдено похожих статей")
            return

        for i, sim in enumerate(similar, 1):
            print(f"{i:2}. {sim['title']}")
            print(f"    Категория: {sim['category']} | Схожесть: {sim['similarity']:.3f}")
            print()

    def show_prediction(self, user_id, article_id):
        """Показать предсказанную вероятность клика"""
        prob = self.recommender.predict_click_probability(user_id, article_id)

        if prob is not None:
            print(f"\n=== Вероятность клика ===")
            print(f"Пользователь: {user_id}")
            print(f"Статья: {article_id}")
            print(f"Вероятность клика: {prob:.3f} ({prob * 100:.1f}%)")

            if prob > 0.7:
                print("Вердикт: 👍 Высокая вероятность интереса")
            elif prob > 0.4:
                print("Вердикт: 🤔 Средняя вероятность интереса")
            else:
                print("Вердикт: 👎 Низкая вероятность интереса")

    def list_users(self, n=10):
        """Показать список пользователей"""
        users = list(self.recommender.user2ind.keys())[:n]
        print(f"\n=== Первые {n} пользователей ===")
        for i, user in enumerate(users, 1):
            print(f"{i:3}. {user} (индекс: {self.recommender.user2ind[user]})")

    def list_articles(self, n=10):
        """Показать список статей"""
        if self.recommender.news_df is not None:
            print(f"\n=== Первые {n} статей ===")
            shown = 0
            for i, row in self.recommender.news_df.iterrows():
                if row['itemId'] in self.recommender.item2ind:
                    print(f"{shown + 1:3}. {row['itemId']} | {row['category']} | {row['title'][:50]}...")
                    shown += 1
                    if shown >= n:
                        break
        else:
            articles = list(self.recommender.item2ind.keys())[:n]
            for i, article in enumerate(articles, 1):
                print(f"{i:3}. {article}")

    def show_popular(self, n=10):
        """Показать популярные статьи"""
        print(f"\n=== Популярные статьи (Top-{n}) ===\n")
        popular = self.recommender.recommend_popular(n)

        for i, pop in enumerate(popular, 1):
            print(f"{i:2}. {pop['title']}")
            print(f"    Категория: {pop['category']} | ID: {pop['item_id']}")
            print()


# ============================================
# 5. ТОЧКА ВХОДА
# ============================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Система рекомендаций новостей")
    parser.add_argument("--interactive", action="store_true", help="Интерактивный режим")
    parser.add_argument("--user", type=str, help="ID пользователя для рекомендаций")
    parser.add_argument("--top_k", type=int, default=10, help="Количество рекомендаций")
    parser.add_argument("--model_path", type=str, default="news_mf_model.pt")
    parser.add_argument("--indices_path", type=str, default="model_indices.pkl")

    args = parser.parse_args()

    if args.user:
        # Режим одного пользователя
        print(f"Получение рекомендаций для пользователя {args.user}...")
        loader = ModelLoader(args.model_path, args.indices_path)
        loader.load()
        recommender = NewsRecommender(loader)
        recommendations = recommender.recommend_for_user_id(args.user, args.top_k)

        print(f"\n=== Рекомендации для {args.user} (Top-{args.top_k}) ===\n")
        for rec in recommendations:
            print(f"{rec['rank']}. {rec['title']}")
            print(f"   Вероятность: {rec['probability']:.3f} | Категория: {rec['category']}")
            print()

    else:
        # Интерактивный режим по умолчанию
        app = RecommendationApp(args.model_path, args.indices_path)
        app.run_interactive()


if __name__ == "__main__":
    main()