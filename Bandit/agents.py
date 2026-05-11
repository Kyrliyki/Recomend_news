"""
Модуль с реализацией агентов для контекстуального бандита
"""

import numpy as np
import random
from collections import defaultdict
from .config import EPSILON, ALPHA, UCB_CONFIDENCE


class ContextualBanditAgent:
    """
    Epsilon-Greedy агент для контекстуального бандита
    """

    def __init__(self, epsilon=EPSILON, alpha=ALPHA, num_categories=None):
        """
        Args:
            epsilon: вероятность исследования
            alpha: скорость обучения
            num_categories: количество категорий (устанавливается позже)
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_categories = num_categories
        self.q_table = {}
        self._initialized = False

    def initialize(self, num_categories):
        """Инициализация с количеством категорий"""
        self.num_categories = num_categories
        self.q_table = {}
        self._initialized = True

    def _get_q_table(self, state):
        """Получение Q-таблицы для состояния (создание если не существует)"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_categories)
        return self.q_table[state]

    def select_action(self, state):
        """Выбор действия на основе epsilon-greedy стратегии"""
        if not self._initialized:
            raise ValueError("Агент не инициализирован. Вызовите initialize()")

        if random.random() < self.epsilon:
            return random.randint(0, self.num_categories - 1)
        return np.argmax(self._get_q_table(state))

    def update(self, state, action, reward):
        """Обновление Q-значений"""
        if not self._initialized:
            return
        q_vals = self._get_q_table(state)
        q_vals[action] += self.alpha * (reward - q_vals[action])
        self.q_table[state] = q_vals

    def get_q_values(self, state):
        """Получить Q-значения для состояния (для оценки)"""
        if not self._initialized:
            return np.zeros(self.num_categories)
        return self._get_q_table(state)

    def save(self, filepath):
        """Сохранение модели"""
        import pickle
        # Преобразуем ключи состояний в сериализуемый формат
        serializable_q_table = {}
        for state, values in self.q_table.items():
            # Преобразуем кортеж состояния в строку для сериализации
            serializable_q_table[str(state)] = values.tolist()

        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': serializable_q_table,
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'num_categories': self.num_categories
            }, f)

    def load(self, filepath):
        """Загрузка модели"""
        import pickle
        import ast
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Восстанавливаем q_table из сериализованного формата
        self.q_table = {}
        for state_str, values in data['q_table'].items():
            # Преобразуем строку обратно в кортеж
            state = ast.literal_eval(state_str)
            self.q_table[state] = np.array(values)

        self.epsilon = data['epsilon']
        self.alpha = data['alpha']
        self.num_categories = data['num_categories']
        self._initialized = True


class UCBAgent:
    """
    UCB (Upper Confidence Bound) агент для контекстуального бандита
    """

    def __init__(self, alpha=ALPHA, confidence=UCB_CONFIDENCE, num_categories=None):
        self.alpha = alpha
        self.confidence = confidence
        self.num_categories = num_categories
        self.q_table = {}
        self.action_counts = {}
        self.total_counts = {}
        self._initialized = False

    def initialize(self, num_categories):
        self.num_categories = num_categories
        self.q_table = {}
        self.action_counts = {}
        self.total_counts = {}
        self._initialized = True

    def _get_q_table(self, state):
        """Получение Q-таблицы для состояния (создание если не существует)"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_categories)
            self.action_counts[state] = np.zeros(self.num_categories)
            self.total_counts[state] = 0
        return self.q_table[state]

    def _get_action_counts(self, state):
        if state not in self.action_counts:
            self.action_counts[state] = np.zeros(self.num_categories)
        return self.action_counts[state]

    def _get_total_counts(self, state):
        if state not in self.total_counts:
            self.total_counts[state] = 0
        return self.total_counts[state]

    def select_action(self, state):
        if not self._initialized:
            raise ValueError("Агент не инициализирован")

        total_counts = self._get_total_counts(state)
        self.total_counts[state] = total_counts + 1

        q_vals = self._get_q_table(state)
        action_counts = self._get_action_counts(state)

        # UCB формула: Q(a) + c * sqrt(log(N) / n(a))
        action_counts_safe = np.maximum(action_counts, 1)
        ucb_values = q_vals + self.confidence * np.sqrt(
            np.log(total_counts + 1) / action_counts_safe
        )
        return np.argmax(ucb_values)

    def update(self, state, action, reward):
        if not self._initialized:
            return

        action_counts = self._get_action_counts(state)
        action_counts[action] += 1
        self.action_counts[state] = action_counts

        q_vals = self._get_q_table(state)
        q_vals[action] += self.alpha * (reward - q_vals[action])
        self.q_table[state] = q_vals

    def get_q_values(self, state):
        """Получить Q-значения для состояния (для оценки)"""
        if not self._initialized:
            return np.zeros(self.num_categories)
        return self._get_q_table(state)

    def save(self, filepath):
        """Сохранение модели"""
        import pickle
        # Преобразуем в сериализуемый формат
        serializable_q_table = {}
        for state, values in self.q_table.items():
            serializable_q_table[str(state)] = values.tolist()

        serializable_action_counts = {}
        for state, values in self.action_counts.items():
            serializable_action_counts[str(state)] = values.tolist()

        serializable_total_counts = {}
        for state, value in self.total_counts.items():
            serializable_total_counts[str(state)] = value

        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': serializable_q_table,
                'action_counts': serializable_action_counts,
                'total_counts': serializable_total_counts,
                'alpha': self.alpha,
                'confidence': self.confidence,
                'num_categories': self.num_categories
            }, f)

    def load(self, filepath):
        """Загрузка модели"""
        import pickle
        import ast
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Восстанавливаем данные
        self.q_table = {}
        for state_str, values in data['q_table'].items():
            state = ast.literal_eval(state_str)
            self.q_table[state] = np.array(values)

        self.action_counts = {}
        for state_str, values in data['action_counts'].items():
            state = ast.literal_eval(state_str)
            self.action_counts[state] = np.array(values)

        self.total_counts = {}
        for state_str, value in data['total_counts'].items():
            state = ast.literal_eval(state_str)
            self.total_counts[state] = value

        self.alpha = data['alpha']
        self.confidence = data['confidence']
        self.num_categories = data['num_categories']
        self._initialized = True


class ThompsonSamplingAgent:
    """
    Thompson Sampling агент с Бета-распределением
    """

    def __init__(self, num_categories=None):
        self.num_categories = num_categories
        self.alphas = {}
        self.betas = {}
        self.q_table = {}
        self._initialized = False

    def initialize(self, num_categories):
        self.num_categories = num_categories
        self.alphas = {}
        self.betas = {}
        self.q_table = {}
        self._initialized = True

    def _get_alphas(self, state):
        if state not in self.alphas:
            self.alphas[state] = np.ones(self.num_categories)
        return self.alphas[state]

    def _get_betas(self, state):
        if state not in self.betas:
            self.betas[state] = np.ones(self.num_categories)
        return self.betas[state]

    def _update_q_table(self, state):
        """Обновление q_table на основе текущих alpha/beta"""
        alphas = self._get_alphas(state)
        betas = self._get_betas(state)
        self.q_table[state] = alphas / (alphas + betas + 1e-10)

    def select_action(self, state):
        if not self._initialized:
            raise ValueError("Агент не инициализирован")

        alphas = self._get_alphas(state)
        betas = self._get_betas(state)

        samples = [np.random.beta(alphas[i], betas[i])
                   for i in range(self.num_categories)]
        return np.argmax(samples)

    def update(self, state, action, reward):
        if not self._initialized:
            return

        alphas = self._get_alphas(state)
        betas = self._get_betas(state)

        if reward > 0:
            alphas[action] += reward
        else:
            betas[action] += abs(reward) if reward < 0 else 1

        self.alphas[state] = alphas
        self.betas[state] = betas

        # Обновляем q_table для совместимости
        self._update_q_table(state)

    def get_q_values(self, state):
        """Получить Q-значения для состояния (для оценки)"""
        if not self._initialized:
            return np.zeros(self.num_categories)

        if state not in self.q_table:
            self._update_q_table(state)
        return self.q_table[state]

    def save(self, filepath):
        """Сохранение модели"""
        import pickle
        # Преобразуем в сериализуемый формат
        serializable_alphas = {}
        for state, values in self.alphas.items():
            serializable_alphas[str(state)] = values.tolist()

        serializable_betas = {}
        for state, values in self.betas.items():
            serializable_betas[str(state)] = values.tolist()

        serializable_q_table = {}
        for state, values in self.q_table.items():
            serializable_q_table[str(state)] = values.tolist()

        with open(filepath, 'wb') as f:
            pickle.dump({
                'alphas': serializable_alphas,
                'betas': serializable_betas,
                'q_table': serializable_q_table,
                'num_categories': self.num_categories
            }, f)

    def load(self, filepath):
        """Загрузка модели"""
        import pickle
        import ast
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Восстанавливаем данные
        self.alphas = {}
        for state_str, values in data['alphas'].items():
            state = ast.literal_eval(state_str)
            self.alphas[state] = np.array(values)

        self.betas = {}
        for state_str, values in data['betas'].items():
            state = ast.literal_eval(state_str)
            self.betas[state] = np.array(values)

        self.q_table = {}
        for state_str, values in data['q_table'].items():
            state = ast.literal_eval(state_str)
            self.q_table[state] = np.array(values)

        self.num_categories = data['num_categories']
        self._initialized = True


def create_agent(agent_type, num_categories, **kwargs):
    """
    Фабрика для создания агентов

    Args:
        agent_type: 'epsilon_greedy', 'ucb', 'thompson'
        num_categories: количество категорий
        **kwargs: дополнительные параметры для агента
    """
    if agent_type == 'epsilon_greedy' or agent_type == 'cb':
        agent = ContextualBanditAgent(**kwargs)
    elif agent_type == 'ucb':
        agent = UCBAgent(**kwargs)
    elif agent_type == 'thompson' or agent_type == 'ts':
        agent = ThompsonSamplingAgent()
    else:
        raise ValueError(f"Неизвестный тип агента: {agent_type}")

    agent.initialize(num_categories)
    return agent