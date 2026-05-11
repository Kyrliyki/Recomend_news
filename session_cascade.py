"""
Сессионный каскадный рекомендатель с учётом жанров.

Сценарий работы:
- Выбор пользователя
- Старт пустой сессии
- Получение до 8 рекомендаций статей (не более 5 одного жанра)
- Клик на одну статью или пропуск
- Бандит обновляется после каждого действия
- Кликнутые статьи удаляются из пула
"""

import sys
import os
import numpy as np
from collections import defaultdict

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


class SessionCascadeRecommender:
    """
    Основной каскадный рекомендатель.
    MF выдаёт топ-K статей, бандит выбирает из них.
    Из пула удаляются только кликнутые статьи.
    """

    def __init__(self, mf_recommender, genre_aware_bandit, top_k_mf=50, articles_per_recommend=8, max_per_genre=5):
        self.mf = mf_recommender
        self.bandit = genre_aware_bandit
        self.top_k = top_k_mf
        self.articles_per_recommend = min(articles_per_recommend, 8)
        self.max_per_genre = max_per_genre

        self.article_to_bandit_idx = {}
        self.bandit_idx_to_article = {}
        self.article_to_genre = {}
        self.genre_to_id = {}
        self.id_to_genre = {}
        self.num_articles = 0
        self.num_genres = 0

        self.current_user = None
        self.clicked_articles = []
        self.shown_articles = []
        self.session_genre_history = []
        self.recommendation_log = []

        self._init_mappings()

    def _init_mappings(self):
        """Инициализация маппингов статей и жанров"""
        all_articles = list(self.mf.item2ind.keys())
        self.num_articles = len(all_articles)

        for idx, article_id in enumerate(all_articles):
            self.article_to_bandit_idx[article_id] = idx
            self.bandit_idx_to_article[idx] = article_id
            genre = self._get_article_genre(article_id)
            self.article_to_genre[article_id] = genre

        unique_genres = set(self.article_to_genre.values())
        self.genre_to_id = {genre: i for i, genre in enumerate(unique_genres)}
        self.id_to_genre = {i: genre for genre, i in self.genre_to_id.items()}
        self.num_genres = len(unique_genres)

        print(f"\nИнициализация: {self.num_articles} статей, {self.num_genres} жанров")
        print(f"Лимит: не более {self.max_per_genre} статей одного жанра за раунд")

    def _get_article_genre(self, article_id):
        """Получить жанр статьи"""
        if hasattr(self.mf, 'news_df') and self.mf.news_df is not None:
            news_row = self.mf.news_df[self.mf.news_df['itemId'] == article_id]
            if len(news_row) > 0:
                return news_row.iloc[0]['category']
        return 'unknown'

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

    def get_recommendations(self, user_id, num_articles=None):
        """Получить список рекомендаций для пользователя"""
        if num_articles is None:
            num_articles = self.articles_per_recommend

        self.current_user = user_id

        mf_articles = self.mf.recommend_for_user(user_id, top_k=self.top_k)

        if not mf_articles:
            mf_articles = self._get_fallback_articles()

        candidate_articles = [a for a in mf_articles
                              if a['item_id'] not in self.clicked_articles]

        if not candidate_articles:
            print("Предупреждение: все статьи уже были кликнуты")
            return []

        candidate_tuples = []
        for article in candidate_articles:
            article_id = article['item_id']
            bandit_idx = self.article_to_bandit_idx.get(article_id)
            if bandit_idx is not None:
                genre = self.article_to_genre.get(article_id, 'unknown')
                genre_id = self._get_genre_id(genre)
                candidate_tuples.append((bandit_idx, genre_id, article))

        state = self._build_state()

        scored_articles = []
        for bandit_idx, genre_id, article in candidate_tuples:
            score = self.bandit.get_combined_score(state, bandit_idx, genre_id)
            scored_articles.append((score, article))

        scored_articles.sort(key=lambda x: x[0], reverse=True)
        scored_articles = self._apply_genre_diversity_limit(scored_articles, self.max_per_genre)
        top_articles = scored_articles[:num_articles]

        if len(top_articles) < num_articles and len(top_articles) > 0:
            print(f"Замечание: только {len(top_articles)} статей доступно после фильтрации по жанрам")

        recommendations = []
        for score, article in top_articles:
            recommendations.append({
                'item_id': article['item_id'],
                'title': article.get('title', 'Н/Д'),
                'category': article.get('category', 'Н/Д'),
                'mf_score': article.get('probability', 0),
                'bandit_score': score,
                'rank': len(recommendations) + 1
            })

        shown_ids = [r['item_id'] for r in recommendations]
        self.shown_articles.extend(shown_ids)

        self.recommendation_log.append({
            'shown_articles': shown_ids,
            'state': state,
            'round': len(self.recommendation_log) + 1
        })

        return recommendations

    def register_click(self, article_id):
        """Зарегистрировать клик по статье"""
        bandit_idx = self.article_to_bandit_idx.get(article_id)
        genre = self.article_to_genre.get(article_id, 'unknown')
        genre_id = self._get_genre_id(genre)

        self.clicked_articles.append(article_id)
        self.session_genre_history.append(genre_id)

        diversity_bonus = 0.2 if len(set(self.session_genre_history)) > len(set(self.session_genre_history[:-1])) else 0
        repetition_penalty = -0.5 if self.clicked_articles.count(article_id) > 1 else 0
        reward = 1.0 + diversity_bonus + repetition_penalty

        state = self._build_state()
        self.bandit.update(state, bandit_idx, genre_id, reward)

        return reward, genre

    def register_skip(self):
        """Зарегистрировать пропуск (отрицательная обратная связь)"""
        if not self.recommendation_log:
            return

        last_rec = self.recommendation_log[-1]
        state = last_rec['state']
        reward = -0.5

        for article_id in last_rec['shown_articles']:
            bandit_idx = self.article_to_bandit_idx.get(article_id)
            if bandit_idx is not None:
                genre = self.article_to_genre.get(article_id, 'unknown')
                genre_id = self._get_genre_id(genre)
                if hasattr(self.bandit, 'update'):
                    self.bandit.update(state, bandit_idx, genre_id, reward)

        print(f"Штраф за пропуск применён к {len(last_rec['shown_articles'])} статьям")

    def start_new_session(self, user_id):
        """Начать новую сессию для пользователя"""
        self.current_user = user_id
        self.clicked_articles = []
        self.shown_articles = []
        self.session_genre_history = []
        self.recommendation_log = []
        self.bandit.session_genre_clicks = []

        print(f"\nНовая сессия для пользователя: {user_id}")
        self._show_user_genre_preferences(user_id)

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

    def get_session_stats(self):
        """Получить статистику текущей сессии"""
        from collections import Counter

        genre_names = []
        for genre_id in self.session_genre_history:
            genre_name = self.bandit.id_to_genre.get(genre_id, 'unknown')
            genre_names.append(genre_name)

        genre_counter = Counter(genre_names)

        return {
            'user': self.current_user,
            'articles_clicked': len(self.clicked_articles),
            'articles_shown': len(self.shown_articles),
            'unique_genres': len(set(self.session_genre_history)),
            'genre_distribution': dict(genre_counter.most_common()),
            'recommendation_rounds': len(self.recommendation_log)
        }

    def get_available_count(self):
        """Получить количество ещё доступных статей"""
        total_articles = len(self.article_to_bandit_idx)
        clicked_count = len(self.clicked_articles)
        return total_articles - clicked_count

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


def load_session_recommender(
    mf_model_path="news_mf_model.pt",
    mf_indices_path="model_indices.pkl",
    base_bandit_type='ucb',
    top_k_mf=50,
    articles_per_recommend=8,
    max_per_genre=5,
    article_weight=0.5,
    genre_weight=0.5
):
    """
    Загрузить сессионный рекомендатель.

    Параметры:
        mf_model_path: путь к обученной MF модели
        mf_indices_path: путь к индексам
        base_bandit_type: тип бандита ('ucb', 'epsilon_greedy', 'thompson')
        top_k_mf: сколько статей берёт MF
        articles_per_recommend: сколько статей показывать за раз
        max_per_genre: максимум статей одного жанра за раунд
        article_weight: вес статьи в комбинированном скоре
        genre_weight: вес жанра в комбинированном скоре
    """
    print("\n" + "="*60)
    print("ЗАГРУЗКА СЕССИОННОГО РЕКОМЕНДАТЕЛЯ")
    print("="*60)

    if not os.path.exists(mf_model_path):
        raise FileNotFoundError(f"MF модель не найдена: {mf_model_path}")

    print("Загрузка MF модели...")
    from Colab.recommend import NewsRecommender
    mf_recommender = NewsRecommender.from_saved(mf_model_path, mf_indices_path)
    print(f"  Загружено {len(mf_recommender.user2ind)} пользователей, {len(mf_recommender.item2ind)} статей")

    print("Создание бандита...")
    from Bandit.agents import create_agent
    base_bandit = create_agent(base_bandit_type, num_categories=20000)

    print("Анализ жанров...")
    all_articles = list(mf_recommender.item2ind.keys())
    article_to_genre_temp = {}

    for article_id in all_articles[:2000]:
        if hasattr(mf_recommender, 'news_df') and mf_recommender.news_df is not None:
            news_row = mf_recommender.news_df[mf_recommender.news_df['itemId'] == article_id]
            if len(news_row) > 0:
                article_to_genre_temp[article_id] = news_row.iloc[0]['category']
            else:
                article_to_genre_temp[article_id] = 'unknown'
        else:
            article_to_genre_temp[article_id] = 'unknown'

    unique_genres = set(article_to_genre_temp.values())
    genre_to_id_temp = {genre: i for i, genre in enumerate(unique_genres)}
    num_genres = len(unique_genres)
    num_articles = len(all_articles)

    print(f"  Найдено жанров: {num_genres}")

    genre_bandit = GenreAwareBandit(
        num_articles=num_articles,
        num_genres=num_genres,
        base_bandit=base_bandit,
        genre_to_id=genre_to_id_temp
    )
    genre_bandit.set_weights(article_weight, genre_weight)

    session_recommender = SessionCascadeRecommender(
        mf_recommender=mf_recommender,
        genre_aware_bandit=genre_bandit,
        top_k_mf=top_k_mf,
        articles_per_recommend=articles_per_recommend,
        max_per_genre=max_per_genre
    )

    print(f"\nРекомендатель готов. Статей за раунд: {articles_per_recommend}")
    print(f"Тип бандита: {base_bandit_type}, Максимум одного жанра: {max_per_genre}")

    return session_recommender


def display_recommendations(articles):
    """Показать рекомендации, сгруппированные по жанрам"""
    print("\n" + "="*60)
    print("РЕКОМЕНДОВАННЫЕ СТАТЬИ")
    print("="*60)

    by_genre = defaultdict(list)
    for article in articles:
        genre = article.get('category', 'unknown')
        by_genre[genre].append(article)

    for genre, genre_articles in by_genre.items():
        print(f"\n[{genre}] ({len(genre_articles)} статей):")
        for article in genre_articles:
            title = article.get('title', 'Н/Д')[:60]
            print(f"  {article['rank']}. {title} (релевантность: {article.get('mf_score', 0)*100:.1f}%)")


def run_session():
    """Главный интерактивный цикл сессии"""
    print("\n" + "="*60)
    print("СЕССИОННЫЙ РЕКОМЕНДАТЕЛЬ")
    print("="*60)
    print("\nПравила:")
    print("  - Из пула удаляются только кликнутые статьи")
    print("  - Некликнутые статьи могут появляться снова")
    print("  - Не более 5 статей одного жанра за раунд")
    print("  - Пропуск штрафует бандит")

    try:
        recommender = load_session_recommender(
            base_bandit_type='ucb',
            top_k_mf=50,
            articles_per_recommend=8,
            max_per_genre=5,
            article_weight=0.5,
            genre_weight=0.5
        )

        print("\n" + "="*60)
        print("ДОСТУПНЫЕ ПОЛЬЗОВАТЕЛИ")
        print("="*60)

        users = list(recommender.mf.user2ind.keys())
        for i, user in enumerate(users[:20], 1):
            print(f"  {i}. {user}")

        if len(users) > 20:
            print(f"  ... и ещё {len(users) - 20}")

        while True:
            print("\n" + "-"*60)
            user_input = input("Введите ID пользователя (или 'exit'): ").strip()

            if user_input.lower() == 'exit':
                print("\nДо свидания!")
                break

            if user_input not in recommender.mf.user2ind:
                print(f"Пользователь {user_input} не найден")
                continue

            recommender.start_new_session(user_input)

            session_active = True
            while session_active:
                print("\n" + "-"*60)
                print(f"РАУНД {len(recommender.recommendation_log) + 1}")
                print(f"Кликнуто: {len(recommender.clicked_articles)} | Доступно: {recommender.get_available_count()}")
                print("-"*60)

                recommendations = recommender.get_recommendations(user_input)

                if not recommendations:
                    print("\nВсе доступные статьи были кликнуты!")
                    break

                display_recommendations(recommendations)

                print("\nКоманды:")
                print("  0 - Пропустить (нет клика)")
                print(f"  1-{len(recommendations)} - Кликнуть на статью")
                print("  stats - Показать статистику сессии")
                print("  q - Завершить сессию")
                print("  exit - Выход из программы")

                choice = input("\nВаш выбор: ").strip()

                if choice.lower() == 'exit':
                    return

                if choice.lower() == 'q':
                    stats = recommender.get_session_stats()
                    print(f"\nСтатистика сессии:")
                    print(f"  Кликнуто статей: {stats['articles_clicked']}")
                    print(f"  Раундов: {stats['recommendation_rounds']}")
                    print(f"  Уникальных жанров: {stats['unique_genres']}")
                    if stats['genre_distribution']:
                        print("\n  Клики по жанрам:")
                        for genre, count in stats['genre_distribution'].items():
                            print(f"    {genre}: {count}")
                    print("\nСессия завершена")
                    break

                if choice.lower() == 'stats':
                    stats = recommender.get_session_stats()
                    print(f"\nСтатистика сессии:")
                    print(f"  Кликнуто статей: {stats['articles_clicked']}")
                    print(f"  Показано статей: {stats['articles_shown']}")
                    print(f"  Раундов: {stats['recommendation_rounds']}")
                    print(f"  Уникальных жанров: {stats['unique_genres']}")
                    continue

                if choice == '0':
                    recommender.register_skip()
                    print("Пропуск зарегистрирован (статьи остаются в пуле)")
                    continue

                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(recommendations):
                        clicked = recommendations[idx]
                        print(f"\nКлик по: {clicked.get('title', 'Н/Д')[:60]}...")
                        print(f"Жанр: {clicked.get('category', 'Н/Д')}")

                        reward, genre = recommender.register_click(clicked['item_id'])
                        print(f"Награда: {reward:.2f}")
                        print("Эта статья удалена из пула")

                        stats = recommender.get_session_stats()
                        print(f"\nПрогресс: {stats['articles_clicked']} статей кликнуто")
                    else:
                        print(f"Неверный выбор. Введите 1-{len(recommendations)}")
                except ValueError:
                    print("Неверный ввод")

            another = input("\nНачать новую сессию с другим пользователем? (y/n): ").strip().lower()
            if another != 'y':
                print("\nДо свидания!")
                break

    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_session()