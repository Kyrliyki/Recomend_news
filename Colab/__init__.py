"""
Colab модуль - компоненты пайплайна рекомендательной системы
"""

from .config import Config, get_config
from .model import NewsMF
from .dataset import MindDataset

__version__ = "1.0.0"
__all__ = ["Config", "get_config", "NewsMF", "MindDataset"]