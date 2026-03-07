"""
Spending Forecaster Module - SpendWise AI
ZICATT: Zero-Inflated Cross-Attention Temporal Transformer
Produced by: 05v2_zicatt_forecaster.ipynb
"""

import math
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Add position information to sequences using sine/cosine functions."""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TemporalEncoder(nn.Module):
    """Process each category's time-series independently using self-attention."""

    def __init__(self, d_model, nhead, num_layers, dim_ff, dropout, max_len):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return x.mean(dim=1)


class CrossCategoryAttention(nn.Module):
    """Let categories attend to each other to learn correlations."""

    def __init__(self, d_model, nhead, num_layers, dim_ff, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class ZICATT(nn.Module):
    """
    Zero-Inflated Cross-Attention Temporal Transformer

    Predicts per-category:
    - gate: probability of any spending (0-1)
    - mu: expected spending amount
    - logvar: log-variance (uncertainty)
    """

    def __init__(
        self,
        num_categories,
        lookback=8,
        d_model=64,
        nhead=4,
        temporal_layers=2,
        cross_layers=2,
        dim_ff=128,
        dropout=0.1,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.lookback = lookback
        self.d_model = d_model

        self.category_embedding = nn.Embedding(num_categories, d_model)
        self.input_proj = nn.Linear(1, d_model)

        self.temporal_encoder = TemporalEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=temporal_layers, dim_ff=dim_ff,
            dropout=dropout, max_len=lookback + 10
        )

        self.cross_attention = CrossCategoryAttention(
            d_model=d_model, nhead=nhead,
            num_layers=cross_layers, dim_ff=dim_ff,
            dropout=dropout
        )

        self.gate_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.logvar_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        lookback = x.size(1)
        num_cat = x.size(2)

        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_cat, lookback, 1)
        x = self.input_proj(x)

        cat_ids = torch.arange(num_cat, device=x.device).repeat(batch_size)
        cat_emb = self.category_embedding(cat_ids)
        x = x + cat_emb.unsqueeze(1)

        temporal_out = self.temporal_encoder(x)
        temporal_out = temporal_out.reshape(batch_size, num_cat, self.d_model)
        cross_out = self.cross_attention(temporal_out)

        gate_logits = self.gate_head(cross_out).squeeze(-1)
        mu = self.mu_head(cross_out).squeeze(-1)
        logvar = self.logvar_head(cross_out).squeeze(-1)

        return gate_logits, mu, logvar


class ZICATTInference:
    """Production-ready ZICATT forecaster."""

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path:
            checkpoint = torch.load(
                f"{model_path}/model.pt", map_location=self.device, weights_only=False
            )
            config = checkpoint['config']

            self.model = ZICATT(
                num_categories=config['num_categories'],
                lookback=config['lookback'],
                d_model=config['d_model'],
                nhead=config['nhead'],
                temporal_layers=config['temporal_layers'],
                cross_layers=config['cross_layers'],
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler_mean = np.array(checkpoint['scaler_mean'])
            self.scaler_std = np.array(checkpoint['scaler_std'])
            self.categories = checkpoint['categories']
            self.lookback = config['lookback']
        else:
            raise ValueError("model_path is required")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, spending_history):
        """
        Args:
            spending_history: dict of {category: [week1, week2, ...]} 
                              OR numpy array (lookback, num_categories)
                              OR simple list of weekly totals (backward compatible)

        Returns:
            dict with per-category predictions and totals
        """
        if isinstance(spending_history, dict):
            matrix = np.zeros((self.lookback, len(self.categories)))
            for i, cat in enumerate(self.categories):
                values = spending_history.get(cat, [0] * self.lookback)
                values = values[-self.lookback:]
                if len(values) < self.lookback:
                    values = [0] * (self.lookback - len(values)) + values
                matrix[:, i] = values
        elif isinstance(spending_history, list):
            # Backward compatible: simple list of weekly totals
            # Distribute evenly across categories as approximation
            matrix = np.zeros((self.lookback, len(self.categories)))
            values = spending_history[-self.lookback:]
            if len(values) < self.lookback:
                values = [0] * (self.lookback - len(values)) + values
            per_cat = np.array(values) / len(self.categories)
            for i in range(len(self.categories)):
                matrix[:, i] = per_cat
        else:
            matrix = np.array(spending_history[-self.lookback:], dtype=np.float32)

        # Scale
        matrix_scaled = (matrix - self.scaler_mean) / self.scaler_std

        x = torch.FloatTensor(matrix_scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            gate_logits, mu, logvar = self.model(x)

        gate_probs = torch.sigmoid(gate_logits).cpu().numpy()[0]
        mu_scaled = mu.cpu().numpy()[0]
        logvar_val = logvar.cpu().numpy()[0]

        mu_dollars = mu_scaled * self.scaler_std + self.scaler_mean
        sigma_dollars = np.abs(np.exp(0.5 * logvar_val) * self.scaler_std)

        predictions = {}
        total_expected = 0
        total_lower = 0
        total_upper = 0

        for i, cat in enumerate(self.categories):
            prob = float(gate_probs[i])
            amount = float(max(0, mu_dollars[i]))
            uncertainty = float(sigma_dollars[i])
            expected = prob * amount

            predictions[cat] = {
                'probability': round(prob, 3),
                'predicted_amount': round(amount, 2),
                'uncertainty': round(uncertainty, 2),
                'expected_spending': round(expected, 2),
                'lower_bound': round(max(0, amount - 1.96 * uncertainty), 2),
                'upper_bound': round(amount + 1.96 * uncertainty, 2),
            }
            total_expected += expected
            total_lower += max(0, expected - 1.96 * uncertainty * prob)
            total_upper += expected + 1.96 * uncertainty * prob

        return {
            'per_category': predictions,
            'predicted_total': round(total_expected, 2),
            'total_lower_bound': round(max(0, total_lower), 2),
            'total_upper_bound': round(total_upper, 2),
            # Backward compatible fields
            'predicted_spending': round(total_expected, 2),
            'lower_bound': round(max(0, total_lower), 2),
            'upper_bound': round(total_upper, 2),
        }
