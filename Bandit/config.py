"""
Конфигурационный файл с параметрами модели
"""

import os

# Пути к данным
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         'MINDsmall_train', 'news.tsv')

# Параметры данных
RANDOM_SEED = 42

# Параметры агентов
EPSILON = 0.1           # Epsilon для Epsilon-Greedy
ALPHA = 0.1             # Скорость обучения
UCB_CONFIDENCE = 2.0    # Параметр уверенности для UCB

# Параметры обучения
NUM_EPISODES = 50
NUM_USERS = 200
NUM_INTERACTIONS_PER_USER = 100

# Параметры награды
CLICK_REWARD = 1.0
CLICK_PENALTY = -0.5
DIVERSITY_REWARD = 0.2
DIVERSITY_PENALTY = -0.2
NOVELTY_REWARD = 0.3

# Параметры оценки
HIT_K = 3
LOGGING_POLICY_PROB = 0.05

# Пути для сохранения моделей
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'saved_models')