"""
Matrix Factorization модель для рекомендаций новостей
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


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
        """Прямой проход для получения скора"""
        user_vec = self.useremb(user_idx)
        item_vec = self.itememb(item_idx)
        return (user_vec * item_vec).sum(-1)

    def predict_score(self, user_idx, item_idx):
        """Предсказание вероятности клика (sigmoid от скора)"""
        with torch.no_grad():
            score = self.forward(user_idx, item_idx)
            prob = torch.sigmoid(score)
        return prob

    def step(self, batch, batch_idx):
        """Один шаг обучения/валидации с негативной семплированием"""
        uservec = self.useremb(batch['userIdx'])
        itemvec_click = self.itememb(batch['click'])

        # Негативное семплирование
        neg_sample = torch.randint(1, self.num_items, batch['click'].shape, device=self.device)
        itemvec_noclick = self.itememb(neg_sample)

        score_click = torch.sigmoid((uservec * itemvec_click).sum(-1).unsqueeze(-1))
        score_noclick = torch.sigmoid((uservec * itemvec_noclick).sum(-1).unsqueeze(-1))

        scores_all = torch.concat((score_click, score_noclick), dim=1)
        target_all = torch.concat((torch.ones_like(score_click), torch.zeros_like(score_noclick)), dim=1)

        loss = F.binary_cross_entropy(scores_all, target_all)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)