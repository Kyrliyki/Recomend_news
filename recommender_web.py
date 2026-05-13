"""
Гибридная веб-версия рекомендательной системы (на основе session_cascade.py)
MF + Contextual Bandit с учётом жанров
"""

import sys
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Colab'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Bandit'))


class GenreAwareBandit:
    """
    Бандит, учитывающий как статьи, так и жанры.
    Итоговый скор = вес_статьи * Q_статьи + вес_жанра * Q_жанра
    """

    def __init__(self, num_articles, num_genres, base_bandit, genre_to_id):
        self.num_articles = num_articles
        self.num_genres = num_genres
        self.base_bandit = base_bandit
        self.genre_to_id = genre_to_id
        self.id_to_genre = {v: k for k, v in genre_to_id.items()}

        self.genre_q_table = {}
        self.genre_counts = defaultdict(int)
        self.genre_rewards = defaultdict(list)

        self.article_weight = 0.5
        self.genre_weight = 0.5

        self.session_genre_clicks = []

        if hasattr(base_bandit, 'num_categories'):
            base_bandit.num_categories = num_articles
            if hasattr(base_bandit, '_initialized'):
                base_bandit._initialized = True
                base_bandit.q_table = {}
                if hasattr(base_bandit, 'action_counts'):
                    base_bandit.action_counts = {}

    def _get_genre_q_values(self, state):
        """Получить Q-значения для жанров в данном состоянии"""
        if state not in self.genre_q_table:
            self.genre_q_table[state] = np.zeros(self.num_genres)
        return self.genre_q_table[state]

    def update_genre(self, state, genre_id, reward):
        """Обновить Q-значение жанра"""
        q_vals = self._get_genre_q_values(state)
        alpha = 0.1
        q_vals[genre_id] += alpha * (reward - q_vals[genre_id])
        self.genre_q_table[state] = q_vals
        self.genre_counts[genre_id] += 1
        self.genre_rewards[genre_id].append(reward)

    def get_genre_score(self, state, genre_id):
        """Получить текущий скор жанра"""
        q_vals = self._get_genre_q_values(state)
        return q_vals[genre_id]

    def get_article_score(self, state, article_idx):
        """Получить скор статьи от базового бандита"""
        if hasattr(self.base_bandit, 'get_q_values'):
            q_vals = self.base_bandit.get_q_values(state)
        elif hasattr(self.base_bandit, '_get_q_table'):
            q_vals = self.base_bandit._get_q_table(state)
        else:
            q_vals = np.zeros(self.num_articles)

        if article_idx < len(q_vals):
            return q_vals[article_idx]
        return 0

    def get_combined_score(self, state, article_idx, genre_id):
        """Комбинированный скор = вес_статьи * скор_статьи + вес_жанра * скор_жанра"""
        article_score = self.get_article_score(state, article_idx)
        genre_score = self.get_genre_score(state, genre_id)
        return (self.article_weight * article_score + self.genre_weight * genre_score)

    def update(self, state, article_idx, genre_id, reward):
        """Обновление бандита после клика"""
        if hasattr(self.base_bandit, 'update'):
            self.base_bandit.update(state, article_idx, reward)
        self.update_genre(state, genre_id, reward)
        self.session_genre_clicks.append(genre_id)

    def get_genre_stats(self):
        """Получить статистику по жанрам"""
        stats = {}
        for genre_id, count in self.genre_counts.items():
            rewards = self.genre_rewards[genre_id]
            avg_reward = np.mean(rewards) if rewards else 0
            genre_name = self.id_to_genre.get(genre_id, 'unknown')
            stats[genre_name] = {
                'clicks': count,
                'avg_reward': avg_reward,
                'genre_id': genre_id
            }
        return stats

    def set_weights(self, article_weight, genre_weight):
        """Настройка весов статей и жанров"""
        total = article_weight + genre_weight
        self.article_weight = article_weight / total
        self.genre_weight = genre_weight / total


class WebRecommenderAPI:
    """
    Веб-версия каскадного рекомендателя.
    MF выдаёт топ-K статей, бандит выбирает из них.
    Из пула удаляются только кликнутые статьи.
    """

    def __init__(self):
        self.mf = None
        self.bandit = None

        # Маппинги
        self.article_to_bandit_idx = {}
        self.bandit_idx_to_article = {}
        self.article_to_genre = {}
        self.genre_to_id = {}
        self.id_to_genre = {}
        self.num_articles = 0
        self.num_genres = 0

        # Параметры
        self.top_k = 100
        self.articles_per_recommend = 8
        self.max_per_genre = 5

        # Состояние сессии
        self.session_active = False
        self.current_user = None
        self.clicked_articles = []      # Только кликнутые статьи
        self.shown_articles = []        # Все показанные статьи
        self.session_genre_history = [] # ID жанров кликнутых статей
        self.recommendation_log = []    # Лог всех раундов

        # Загрузка
        self._load_recommender()

    def _load_news_data(self):
        """Загрузка данных новостей для маппинга жанров"""
        news_path = Path("MINDsmall_train/news.tsv")

        if not news_path.exists():
            print(f"Файл новостей не найден: {news_path}")
            return None

        try:
            import pandas as pd
            news_df = pd.read_csv(
                news_path,
                sep="\t",
                header=None,
                names=["itemId", "category", "subcategory", "title", "abstract",
                       "url", "title_entities", "abstract_entities"]
            )
            print(f"Загружено новостей: {len(news_df)}")
            print(f"Уникальных жанров: {news_df['category'].nunique()}")
            return news_df
        except Exception as e:
            print(f"Ошибка загрузки новостей: {e}")
            return None

    def _load_recommender(self):
        """Загрузка MF модели и создание бандита"""
        try:
            from Colab.recommend import NewsRecommender
            from Bandit.agents import create_agent

            # Загружаем MF модель
            self.mf = NewsRecommender.from_saved(
                model_path="news_mf_model.pt",
                indices_path="model_indices.pkl",
                data_path="MINDsmall_train"
            )
            print(f"MF модель загружена: {len(self.mf.user2ind)} пользователей, {len(self.mf.item2ind)} статей")

            # Загружаем новости для маппинга жанров
            news_df = self._load_news_data()

            # Создаём маппинги
            all_articles = list(self.mf.item2ind.keys())
            self.num_articles = len(all_articles)

            for idx, article_id in enumerate(all_articles):
                self.article_to_bandit_idx[article_id] = idx
                self.bandit_idx_to_article[idx] = article_id

                # Получаем жанр статьи
                genre = 'unknown'
                if news_df is not None:
                    news_row = news_df[news_df['itemId'] == article_id]
                    if len(news_row) > 0:
                        genre = news_row.iloc[0]['category']
                self.article_to_genre[article_id] = genre

            # Создаём маппинг жанров
            unique_genres = set(self.article_to_genre.values())
            self.genre_to_id = {genre: i for i, genre in enumerate(unique_genres)}
            self.id_to_genre = {i: genre for genre, i in self.genre_to_id.items()}
            self.num_genres = len(unique_genres)

            print(f"Инициализация: {self.num_articles} статей, {self.num_genres} жанров")
            print(f"Лимит: не более {self.max_per_genre} статей одного жанра за раунд")

            # Создаём бандит
            print("Создание бандита...")
            base_bandit = create_agent('ucb', num_categories=20000)

            self.bandit = GenreAwareBandit(
                num_articles=self.num_articles,
                num_genres=self.num_genres,
                base_bandit=base_bandit,
                genre_to_id=self.genre_to_id
            )
            self.bandit.set_weights(0.5, 0.5)

            print(f"Бандит создан. Тип: UCB, вес статей: 0.5, вес жанров: 0.5")

            return True

        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_article_genre(self, article_id):
        """Получить жанр статьи"""
        return self.article_to_genre.get(article_id, 'unknown')

    def _get_genre_id(self, genre_name):
        """Получить числовой ID жанра"""
        return self.genre_to_id.get(genre_name, 0)

    def _build_state(self):
        """Построение состояния бандита на основе истории кликов"""
        if not self.clicked_articles:
            return (0, 0, 0, 0)

        last_genre = self.session_genre_history[-1] if self.session_genre_history else 0
        session_len = len(self.clicked_articles)
        session_bucket = min(session_len // 3, 5)

        if self.session_genre_history:
            from collections import Counter
            genre_counts = Counter(self.session_genre_history)
            most_common_genre = genre_counts.most_common(1)[0][0]
        else:
            most_common_genre = 0

        unique_genres = len(set(self.session_genre_history))
        diversity_bucket = min(unique_genres // 2, 4)

        return (last_genre, most_common_genre, session_bucket, diversity_bucket)

    def _apply_genre_diversity_limit(self, scored_articles, max_per_genre=5):
        """Ограничить количество статей одного жанра в выдаче"""
        genre_count = defaultdict(int)
        filtered_articles = []

        for score, article in scored_articles:
            genre = article.get('category', 'unknown')
            if genre_count[genre] < max_per_genre:
                genre_count[genre] += 1
                filtered_articles.append((score, article))

        return filtered_articles

    def get_users(self, limit=50):
        """Получить список пользователей"""
        if not self.mf:
            return []
        return list(self.mf.user2ind.keys())[:limit]

    def start_session(self, user_id):
        """Начать новую сессию для пользователя"""
        if not self.mf:
            return {'error': 'Система не загружена'}

        if user_id not in self.mf.user2ind:
            return {'error': f'Пользователь {user_id} не найден'}

        # Сброс состояния сессии
        self.current_user = user_id
        self.clicked_articles = []
        self.shown_articles = []
        self.session_genre_history = []
        self.recommendation_log = []
        self.bandit.session_genre_clicks = []
        self.session_active = True

        print(f"\n{'='*50}")
        print(f"НОВАЯ СЕССИЯ ДЛЯ ПОЛЬЗОВАТЕЛЯ: {user_id}")
        print(f"{'='*50}")

        # Показываем предпочтения пользователя
        self._show_user_genre_preferences(user_id)

        return self.get_recommendations()

    def _show_user_genre_preferences(self, user_id):
        """Показать предполагаемые жанровые предпочтения пользователя"""
        try:
            mf_articles = self.mf.recommend_for_user(user_id, top_k=50)
            if mf_articles:
                genre_counts = defaultdict(int)
                for article in mf_articles:
                    genre = article.get('category', 'unknown')
                    genre_counts[genre] += 1

                top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]

                if top_genres:
                    print("\nПредпочтения пользователя по жанрам (из MF):")
                    for genre, count in top_genres:
                        print(f"  {genre}: {count} статей в топ-50")
        except Exception:
            pass

    def get_recommendations(self):
        """Получить список рекомендаций для пользователя"""
        if not self.session_active:
            return {'error': 'Сессия не активна'}

        mf_articles = self.mf.recommend_for_user(self.current_user, top_k=self.top_k)

        if not mf_articles:
            mf_articles = self._get_fallback_articles()

        # Убираем уже кликнутые статьи
        candidate_articles = [a for a in mf_articles
                              if a['item_id'] not in self.clicked_articles]

        if not candidate_articles:
            print("Предупреждение: все статьи уже были кликнуты")
            return {'error': 'Все статьи просмотрены', 'session_complete': True}

        # Формируем кандидатов с индексами для бандита
        candidate_tuples = []
        for article in candidate_articles:
            article_id = article['item_id']
            bandit_idx = self.article_to_bandit_idx.get(article_id)
            if bandit_idx is not None:
                genre = self.article_to_genre.get(article_id, 'unknown')
                genre_id = self._get_genre_id(genre)
                article['category'] = genre
                candidate_tuples.append((bandit_idx, genre_id, article))

        # Получаем состояние бандита
        state = self._build_state()

        # Бандит оценивает каждого кандидата
        scored_articles = []
        for bandit_idx, genre_id, article in candidate_tuples:
            score = self.bandit.get_combined_score(state, bandit_idx, genre_id)
            scored_articles.append((score, article))

        # Сортируем по убыванию
        scored_articles.sort(key=lambda x: x[0], reverse=True)

        # Применяем ограничение по жанрам
        scored_articles = self._apply_genre_diversity_limit(scored_articles, self.max_per_genre)

        # Берём первые articles_per_recommend
        top_articles = scored_articles[:self.articles_per_recommend]

        if len(top_articles) < self.articles_per_recommend and len(top_articles) > 0:
            print(f"Замечание: только {len(top_articles)} статей доступно после фильтрации по жанрам")

        # Формируем результат
        recommendations = []
        shown_ids = []
        for score, article in top_articles:
            rec = {
                'item_id': article['item_id'],
                'title': article.get('title', 'Н/Д'),
                'category': article.get('category', 'Н/Д'),
                'probability': article.get('probability', 0),
                'bandit_score': score,
                'rank': len(recommendations) + 1
            }
            recommendations.append(rec)
            shown_ids.append(article['item_id'])

        # Сохраняем показанные статьи
        self.shown_articles.extend(shown_ids)

        # Сохраняем в лог
        self.recommendation_log.append({
            'shown_articles': shown_ids,
            'state': state,
            'round': len(self.recommendation_log) + 1
        })

        # Выводим отладочную информацию
        print(f"\n{'='*40}")
        print(f"РАУНД {len(self.recommendation_log)}")
        print(f"Состояние бандита: {state}")
        print(f"Кликнуто: {len(self.clicked_articles)} | Показано всего: {len(self.shown_articles)}")

        genres_shown = defaultdict(int)
        for r in recommendations:
            genres_shown[r.get('category', 'unknown')] += 1
        print(f"Рекомендовано жанров: {dict(genres_shown)}")
        print(f"{'='*40}")

        return {
            'recommendations': recommendations,
            'stats': {
                'clicked': len(self.clicked_articles),
                'available': self.num_articles - len(self.clicked_articles),
                'round': len(self.recommendation_log)
            }
        }

    def register_click(self, article_id):
        """Зарегистрировать клик по статье"""
        bandit_idx = self.article_to_bandit_idx.get(article_id)
        genre = self.article_to_genre.get(article_id, 'unknown')
        genre_id = self._get_genre_id(genre)

        # Добавляем в кликнутые
        self.clicked_articles.append(article_id)
        self.session_genre_history.append(genre_id)

        # Расчёт награды
        diversity_bonus = 0.2 if len(set(self.session_genre_history)) > len(set(self.session_genre_history[:-1])) else 0
        repetition_penalty = -0.5 if self.clicked_articles.count(article_id) > 1 else 0
        reward = 1.0 + diversity_bonus + repetition_penalty

        state = self._build_state()
        self.bandit.update(state, bandit_idx, genre_id, reward)

        print(f"\nКЛИК: {article_id}")
        print(f"  Жанр: {genre}")
        print(f"  Награда: {reward:.2f}")
        print(f"  Кликнуто всего: {len(self.clicked_articles)}")

        return self.get_recommendations()

    def register_skip(self):
        """Зарегистрировать пропуск (отрицательная обратная связь)"""
        if not self.recommendation_log:
            return self.get_recommendations()

        last_rec = self.recommendation_log[-1]
        state = last_rec['state']
        reward = -0.5

        print(f"\nПРОПУСК РАУНДА {last_rec['round']}")

        for article_id in last_rec['shown_articles']:
            bandit_idx = self.article_to_bandit_idx.get(article_id)
            if bandit_idx is not None:
                genre = self.article_to_genre.get(article_id, 'unknown')
                genre_id = self._get_genre_id(genre)
                if hasattr(self.bandit, 'update'):
                    self.bandit.update(state, bandit_idx, genre_id, reward)

        print(f"  Штраф за пропуск применён к {len(last_rec['shown_articles'])} статьям")
        print(f"  Статьи остаются в пуле. Кликнуто: {len(self.clicked_articles)}")

        return self.get_recommendations()

    def get_session_stats(self):
        """Получить статистику текущей сессии"""
        from collections import Counter

        # Получаем названия жанров из истории кликов
        genre_names = []
        for genre_id in self.session_genre_history:
            genre_name = self.id_to_genre.get(genre_id, 'unknown')
            genre_names.append(genre_name)

        genre_counter = Counter(genre_names)

        return {
            'user': self.current_user if self.current_user else '—',
            'articles_clicked': len(self.clicked_articles),
            'articles_shown': len(self.shown_articles),
            'unique_genres': len(set(self.session_genre_history)),
            'genre_distribution': dict(genre_counter.most_common()),
            'recommendation_rounds': len(self.recommendation_log)
        }

    def get_available_count(self):
        """Получить количество ещё доступных статей"""
        return self.num_articles - len(self.clicked_articles)

    def end_session(self):
        """Завершить сессию"""
        # Сохраняем статистику ДО очистки
        stats = self.get_session_stats()

        print(f"\n{'='*50}")
        print(f"СЕССИЯ ЗАВЕРШЕНА")
        print(f"{'='*50}")
        print(f"  Пользователь: {stats['user']}")
        print(f"  Кликнуто статей: {stats['articles_clicked']}")
        print(f"  Показано статей: {stats['articles_shown']}")
        print(f"  Уникальных жанров: {stats['unique_genres']}")
        print(f"  Раундов: {stats['recommendation_rounds']}")
        if stats['genre_distribution']:
            print(f"  Распределение по жанрам:")
            for genre, count in stats['genre_distribution'].items():
                print(f"    {genre}: {count}")
        print(f"{'='*50}")

        # Очищаем состояние сессии
        self.session_active = False
        self.current_user = None
        self.clicked_articles = []
        self.shown_articles = []
        self.session_genre_history = []
        self.recommendation_log = []

        # Возвращаем сохранённую статистику
        return stats

    def _get_fallback_articles(self):
        """Получить запасные статьи (если MF не дал рекомендаций)"""
        articles = []
        for i, article_id in enumerate(list(self.mf.item2ind.keys())[:self.top_k]):
            genre = self.article_to_genre.get(article_id, 'unknown')
            articles.append({
                'item_id': article_id,
                'title': f'Статья_{article_id}'[:50],
                'category': genre,
                'probability': 0.5
            })
        return articles


# Создаём экземпляр для веб-сервера
recommender_api = WebRecommenderAPI()