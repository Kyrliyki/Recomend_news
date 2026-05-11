"""
Bandit - Модуль для рекомендательной системы на основе Contextual Bandit
"""

from .agents import ContextualBanditAgent, UCBAgent, ThompsonSamplingAgent, create_agent
from .data_loader import DataLoader
from .config import *

__version__ = "1.0.0"
__all__ = [
    'ContextualBanditAgent',
    'UCBAgent',
    'ThompsonSamplingAgent',
    'create_agent',
    'DataLoader'
]