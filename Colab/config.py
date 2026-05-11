"""
Конфигурационный файл со всеми параметрами пайплайна
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Конфигурация проекта"""

    # Пути к данным
    DATA_PATH: str = "MINDsmall_train"
    MODEL_PATH: str = "news_mf_model.pt"
    INDICES_PATH: str = "model_indices.pkl"
    PREPROCESSED_DATA_PATH: str = "preprocessed_behaviour.pkl"  # Изменено с .parquet на .pkl

    # Параметры модели
    EMBEDDING_DIM: int = 50
    BATCH_SIZE: int = 1024
    EPOCHS: int = 50
    LEARNING_RATE: float = 1e-3

    # Параметры обработки данных
    TEST_RATIO: float = 0.2
    MIN_CLICK_CUTOFF: int = 100

    # Параметры обучения PyTorch Lightning
    NUM_WORKERS: int = 0
    ACCELERATOR: str = "auto"

    @classmethod
    def from_dict(cls, config_dict):
        """Создание конфига из словаря"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


# Создаем глобальный экземпляр конфига
_default_config = Config()


def get_config():
    """Получение глобальной конфигурации"""
    return _default_config