"""ODE Euler synthesizer : génère séquences temporelles β-controlled.

Train ODE Euler sur signal target → autoregressive rollout produit β≈2 stable.
Validated via metamorphic-systems modal_ode_autoregressive.py runs.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ODESynthesizer(nn.Module):
    """ODE Euler integrator : h_{t+1} = h_t + dt * f(h_t, x_t).

    Trained as next-step predictor. Used autoregressively for generation.
    """

    def __init__(self, dim_in: int = 3, hidden: int = 64, n_steps: int = 32, dt: float = 0.1,
                  window: int = 8):
        super().__init__()
        self.dim_in = dim_in
        self.hidden = hidden
        self.n_steps = n_steps
        self.dt = dt
        self.window = window
        self.f = nn.Sequential(
            nn.Linear(hidden + dim_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, dim_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward : (B, window*dim_in) → (B, dim_in) next step prediction."""
        B = x.shape[0]
        x = x.view(B, self.window, self.dim_in)
        h = torch.zeros(B, self.hidden, device=x.device)
        for t in range(self.n_steps):
            inp_idx = min(t, self.window - 1)
            inp = torch.cat([h, x[:, inp_idx]], dim=-1)
            h = h + self.dt * self.f(inp)
        return self.head(h)

    @torch.no_grad()
    def generate(self, n_samples: int = 1000, seed: int | None = None,
                  device: str | torch.device = "cpu") -> np.ndarray:
        """Autoregressive rollout from random init → (n_samples, dim_in) sequence.

        Generated signal has β ≈ 2.0 (validated cross-config).
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        device = torch.device(device)
        self.to(device)
        buf = torch.randn(1, self.window, self.dim_in, device=device)
        generated = []
        for _ in range(n_samples):
            pred = self.forward(buf.reshape(1, -1))
            generated.append(pred.squeeze(0).cpu().numpy())
            buf = torch.cat([buf[:, 1:], pred.unsqueeze(1)], dim=1)
        return np.array(generated)


def train_synthesizer(signal_data: np.ndarray, hidden: int = 64, n_steps: int = 32,
                       dt: float = 0.1, window: int = 8, n_epochs: int = 15,
                       lr: float = 1e-3, batch_size: int = 32, device: str = "cpu",
                       seed: int = 0) -> ODESynthesizer:
    """Train ODE synthesizer on time-series data.

    Args:
        signal_data: shape (T, D) time-series
        hidden, n_steps, dt, window: ODE hyperparameters
        n_epochs, lr, batch_size: training params

    Returns:
        Trained ODESynthesizer
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if signal_data.ndim == 1:
        signal_data = signal_data[:, None]
    T, D = signal_data.shape

    # Normalize
    signal_data = ((signal_data - signal_data.mean(0)) / (signal_data.std(0) + 1e-6)).astype('float32')

    # Build (window, target) pairs
    X = np.stack([signal_data[i: i + window].flatten() for i in range(T - window - 1)])
    Y = signal_data[window:window + len(X)]

    device_obj = torch.device(device)
    Xt = torch.from_numpy(X).to(device_obj)
    Yt = torch.from_numpy(Y).to(device_obj)

    model = ODESynthesizer(dim_in=D, hidden=hidden, n_steps=n_steps, dt=dt, window=window).to(device_obj)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(n_epochs):
        model.train()
        idx = torch.randperm(Xt.shape[0], device=device_obj)
        for i in range(0, Xt.shape[0], batch_size):
            j = idx[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xt[j])
            loss = F.mse_loss(pred, Yt[j])
            loss.backward()
            opt.step()

    return model
