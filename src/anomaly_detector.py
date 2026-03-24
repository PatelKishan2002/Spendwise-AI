import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def get_reconstruction_error(self, x):
        self.eval()
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            error = F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
        return error


class AnomalyDetector:

    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path:
            checkpoint = torch.load(model_path + "/model.pt", map_location=self.device, weights_only=False)
            self.model = VAE(
                input_dim=checkpoint["config"]["input_dim"],
                hidden_dim=checkpoint["config"]["hidden_dim"],
                latent_dim=checkpoint["config"]["latent_dim"],
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.threshold = checkpoint["threshold"]
            self.scaler_mean = checkpoint["scaler_mean"]
            self.scaler_std = checkpoint["scaler_std"]
            self.category_cols = checkpoint["category_cols"]
        else:
            self.model = model
            self.threshold = threshold
            self.scaler_mean = scaler.mean_
            self.scaler_std = scaler.scale_
            self.category_cols = category_cols
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, spending):
        vector = np.array([spending.get(cat, 0) for cat in self.category_cols], dtype=np.float32)
        return (vector - self.scaler_mean) / self.scaler_std

    def detect(self, spending):
        x = self.preprocess(spending)
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            error = self.model.get_reconstruction_error(x_tensor).item()
        anomaly_score = min(100, (error / self.threshold) * 50)
        is_anomaly = error > self.threshold
        with torch.no_grad():
            x_recon, _, _ = self.model(x_tensor)
            per_category_error = (x_recon.squeeze() - x_tensor.squeeze()).abs().cpu().numpy()
        top_indices = np.argsort(per_category_error)[-3:][::-1]
        anomalous_categories = [
            {"category": self.category_cols[i], "contribution": float(per_category_error[i])}
            for i in top_indices
        ]
        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "reconstruction_error": error,
            "threshold": self.threshold,
            "top_anomalous_categories": anomalous_categories,
        }
