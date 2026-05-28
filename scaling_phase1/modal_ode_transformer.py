"""Phase 1 : ODE-Transformer hybrid vs vanilla transformer.

Hypothesis : replacing FFN by ODE Euler block improves long-context coherence
+ reduces perplexity at matched params. Grounded in β=1.57 plateau finding.

Architecture comparisons (matched params ~25M) :
- vanilla : 6 layers × (attn + FFN_4x)
- ode_hybrid_25 : 6 layers, 1 FFN replaced by ODE Euler block (~25% replacement)
- ode_hybrid_50 : 6 layers, 3 FFN replaced by ODE Euler blocks (50%)
- ode_full : all 6 FFN replaced by ODE Euler

Dataset : TinyShakespeare (1M chars, char-level)
Train : 5000 steps, batch 64, ctx 256
Measure : val_loss + β activations per layer + train time
"""
from __future__ import annotations
import modal

app = modal.App("scaling-ode-transformer-phase1")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26", "requests>=2.31")
)


@app.function(image=image, gpu="A10", timeout=3600)
def train_compare(seed: int = 0, variant: str = "vanilla", n_layer: int = 6, d_model: int = 384,
                   n_head: int = 6, n_steps_ode: int = 4) -> dict:
    """Train transformer variant on TinyShakespeare, measure val_loss + β activations."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time
    import os

    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Download TinyShakespeare (cached in container after first download)
    import requests
    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("/tmp/input.txt"):
        text = requests.get(URL).text
        with open("/tmp/input.txt", "w") as f:
            f.write(text)
    else:
        with open("/tmp/input.txt", "r") as f:
            text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    ctx = 256
    batch_size = 64

    def get_batch(split: str):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - ctx, (batch_size,))
        x = torch.stack([d[i:i + ctx] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + ctx] for i in ix])
        return x.to(device), y.to(device)

    class CausalSelfAttn(nn.Module):
        def __init__(self, d_model, n_head):
            super().__init__()
            self.n_head = n_head
            self.d_head = d_model // n_head
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).chunk(3, dim=-1)
            q, k, v = [t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in qkv]
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            att = att.transpose(1, 2).reshape(B, T, C)
            return self.proj(att)

    class FFN(nn.Module):
        def __init__(self, d_model, mult=4):
            super().__init__()
            self.fc1 = nn.Linear(d_model, d_model * mult)
            self.fc2 = nn.Linear(d_model * mult, d_model)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))

    class ODEEulerBlock(nn.Module):
        """ODE Euler integrator : x_{t+1} = x_t + dt * f(x_t).

        n_steps internal iterations with shared f.
        Inductive bias : β=1.57 plateau under noise (validated finding).
        Matched param budget : f uses d_model*2 hidden (vs FFN d_model*4) to compensate.
        """
        def __init__(self, d_model, n_steps=4, dt=0.5, hidden_mult=2):
            super().__init__()
            self.n_steps = n_steps
            self.dt = dt
            self.f = nn.Sequential(
                nn.Linear(d_model, d_model * hidden_mult),
                nn.SiLU(),
                nn.Linear(d_model * hidden_mult, d_model),
            )

        def forward(self, x):
            for _ in range(self.n_steps):
                x = x + self.dt * self.f(x)
            return x

    class Block(nn.Module):
        def __init__(self, d_model, n_head, use_ode=False, n_steps_ode=4):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttn(d_model, n_head)
            self.ln2 = nn.LayerNorm(d_model)
            self.use_ode = use_ode
            if use_ode:
                self.ffn = ODEEulerBlock(d_model, n_steps=n_steps_ode, hidden_mult=2)
            else:
                self.ffn = FFN(d_model, mult=4)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

    class GPT(nn.Module):
        def __init__(self, vocab_size, ctx, d_model=384, n_head=6, n_layer=6,
                      ode_positions=None, n_steps_ode=4):
            super().__init__()
            self.ctx = ctx
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(ctx, d_model)
            ode_positions = ode_positions or []
            self.blocks = nn.ModuleList([
                Block(d_model, n_head, use_ode=(i in ode_positions), n_steps_ode=n_steps_ode)
                for i in range(n_layer)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.token_emb.weight  # tied embedding

        def forward(self, idx, targets=None, return_activations: bool = False):
            B, T = idx.shape
            tok = self.token_emb(idx)
            pos = self.pos_emb(torch.arange(T, device=idx.device))
            x = tok + pos
            acts_per_layer = []
            for block in self.blocks:
                x = block(x)
                if return_activations:
                    acts_per_layer.append(x.detach())
            x = self.ln_f(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss, acts_per_layer

    # Configurable variants : where ODE replaces FFN
    if variant == "vanilla":
        ode_positions = []
    elif variant == "ode_hybrid_25":
        ode_positions = [n_layer // 2]  # 1 middle layer
    elif variant == "ode_hybrid_50":
        ode_positions = [1, n_layer // 2, n_layer - 2]
    elif variant == "ode_full":
        ode_positions = list(range(n_layer))
    else:
        raise ValueError(f"unknown variant {variant}")

    model = GPT(vocab_size, ctx, d_model=d_model, n_head=n_head, n_layer=n_layer,
                  ode_positions=ode_positions, n_steps_ode=n_steps_ode).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    train_losses = []
    val_losses = []
    t0 = time.time()
    n_steps = 3000

    for step in range(n_steps):
        x, y = get_batch('train')
        opt.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_losses.append(float(loss.item()))

        if step % 200 == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch('val')
                _, vloss, _ = model(vx, vy)
                val_losses.append(float(vloss.item()))
            model.train()

    train_time = time.time() - t0
    final_val_loss = val_losses[-1]
    final_train_loss = float(np.mean(train_losses[-50:]))

    # Measure β des activations per layer
    model.eval()
    with torch.no_grad():
        vx, _ = get_batch('val')
        _, _, acts = model(vx, return_activations=True)

    def compute_beta(h_layer):
        """h_layer: (B, T, C). Compute β over T axis, mean across batch + channels."""
        h = h_layer.cpu().numpy()
        if h.shape[1] < 16:
            return 0.0
        # Average power spectrum across batch + channels
        betas_sample = []
        for b in range(min(h.shape[0], 16)):
            fft = np.fft.rfft(h[b], axis=0)
            power = (np.abs(fft) ** 2).mean(axis=-1)
            freqs = np.arange(1, len(power))
            log_f = np.log(freqs + 1e-9)
            log_p = np.log(power[1:] + 1e-12)
            slope, _ = np.polyfit(log_f, log_p, 1)
            betas_sample.append(-slope)
        return float(np.mean(betas_sample))

    betas_per_layer = [compute_beta(act) for act in acts]

    return {
        'seed': seed,
        'variant': variant,
        'n_params': n_params,
        'train_time': train_time,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'val_losses_history': val_losses,
        'betas_per_layer': betas_per_layer,
        'ode_positions': ode_positions,
    }


@app.local_entrypoint()
def main():
    """Run 4 variants × 3 seeds parallel on Modal A10."""
    import statistics
    variants = ["vanilla", "ode_hybrid_25", "ode_hybrid_50", "ode_full"]
    seeds = list(range(3))
    inputs = [(s, v, 6, 384, 6, 4) for v in variants for s in seeds]

    results = list(train_compare.starmap(inputs))

    print(f"\n=== ODE-Transformer Phase 1 Results ===\n")
    print(f"  {'variant':<18}{'params':<12}{'train (s)':<12}{'train_loss':<14}{'val_loss':<14}")
    for v in variants:
        runs = [r for r in results if r['variant'] == v]
        n_p = runs[0]['n_params']
        tt = statistics.mean([r['train_time'] for r in runs])
        tl = statistics.mean([r['final_train_loss'] for r in runs])
        tl_s = statistics.stdev([r['final_train_loss'] for r in runs]) if len(runs) > 1 else 0
        vl = statistics.mean([r['final_val_loss'] for r in runs])
        vl_s = statistics.stdev([r['final_val_loss'] for r in runs]) if len(runs) > 1 else 0
        print(f"  {v:<18}{n_p:<12}{tt:<12.1f}{tl:.3f}±{tl_s:.3f}    {vl:.3f}±{vl_s:.3f}")

    print(f"\n=== β activations per layer (mean across 3 seeds) ===\n")
    for v in variants:
        runs = [r for r in results if r['variant'] == v]
        n_layers = len(runs[0]['betas_per_layer'])
        mean_betas = []
        for i in range(n_layers):
            vals = [r['betas_per_layer'][i] for r in runs]
            mean_betas.append(sum(vals) / len(vals))
        ode_pos = runs[0]['ode_positions']
        line = f"  {v:<18}"
        for i, b in enumerate(mean_betas):
            marker = "★" if i in ode_pos else " "
            line += f"L{i}{marker}={b:.2f}  "
        print(line)
    print("\n  ★ = ODE block (FFN replaced)")
