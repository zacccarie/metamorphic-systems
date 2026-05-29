"""β-meter sur residual stream LLM : test signal pour adaptive compute.

Hypothèse : β des activations residual stream layer-par-layer pendant inference
varie selon difficulty du token. β stable = model confident → less compute needed.
β shift = model struggling → more thinking (CoT, sample more, defer).

Setup :
- Load TinyLlama 1.1B (open weights, A10 fits fp16)
- 20 prompts : 10 easy (Q&A simple) + 10 hard (reasoning, math)
- Hook residual stream après chaque transformer block
- For each generated token : compute β of last 256 activations per layer
- Compare β patterns easy vs hard

Si correlation β-variance ↔ difficulty : β-meter = signal valide adaptive compute.
"""
from __future__ import annotations
import modal

app = modal.App("scaling-option3-beta-llm")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("transformers>=4.46", "accelerate>=1.0", "numpy>=1.26", "huggingface_hub>=0.26")
)


@app.function(image=image, gpu="A10", timeout=1800)
def measure_beta_llama(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> dict:
    """Run prompts, capture residual stream, compute β per layer per token."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                   device_map="cuda", output_hidden_states=True)
    model.eval()

    easy_prompts = [
        "What is 2+2?",
        "What color is the sky?",
        "Hello, how are you?",
        "Name the capital of France.",
        "What is 10 times 5?",
        "Say yes or no.",
        "What is your favorite color?",
        "Count to three.",
        "What day comes after Monday?",
        "What is water made of?",
    ]
    hard_prompts = [
        "Prove that the square root of 2 is irrational, step by step.",
        "Solve: if x^2 + 3x - 10 = 0, find x using the quadratic formula.",
        "Explain Gödel's incompleteness theorem in detail.",
        "Compute the integral of x^2 * sin(x) using integration by parts.",
        "Derive the formula for the volume of a sphere from first principles.",
        "What is the relationship between entropy and information theory? Explain rigorously.",
        "Prove the Cauchy-Schwarz inequality.",
        "Explain how quantum decoherence relates to the measurement problem.",
        "Solve the heat equation in one dimension with Dirichlet boundary conditions.",
        "Compute the Fourier series of f(x) = x on [-π, π].",
    ]

    def compute_beta_1d(sig):
        sig = np.asarray(sig, dtype=np.float64)
        if sig.shape[0] < 16:
            return 0.0
        fft = np.fft.rfft(sig, axis=0)
        power = (np.abs(fft) ** 2).mean(axis=-1) if sig.ndim > 1 else np.abs(fft) ** 2
        freqs = np.arange(1, len(power))
        log_f = np.log(freqs + 1e-9)
        log_p = np.log(power[1:] + 1e-12)
        slope, _ = np.polyfit(log_f, log_p, 1)
        return float(-slope)

    def process_prompt(prompt: str, max_new_tokens: int = 64) -> dict:
        """Generate + measure β per layer at each generation step."""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        # Generate with hidden states captured
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # outputs.hidden_states : tuple per generation step
        # each step : tuple per layer (n_layers + 1 with embedding)
        # each layer : (1, seq_len_at_step, hidden_size)
        n_steps = len(outputs.hidden_states)
        n_layers = len(outputs.hidden_states[0])

        # For each layer, collect residual stream across all generated positions
        # Use the LAST token's hidden state at each step
        per_layer_trajectory = [[] for _ in range(n_layers)]
        for step_idx, step_hidden in enumerate(outputs.hidden_states):
            for layer_idx, layer_h in enumerate(step_hidden):
                # layer_h: (batch=1, seq_len, hidden)
                last_token_h = layer_h[0, -1, :].cpu().float().numpy()
                per_layer_trajectory[layer_idx].append(last_token_h)

        # per_layer_trajectory[l] : list of (hidden,) arrays length n_steps
        # Stack and compute β across step axis per layer
        betas_per_layer = []
        for l in range(n_layers):
            traj = np.stack(per_layer_trajectory[l])  # (n_steps, hidden)
            # β over step axis, averaged across hidden channels (PCA-like)
            # Reduce hidden to top-1 PCA or just use FFT mean
            beta = compute_beta_1d(traj)
            betas_per_layer.append(beta)

        # Also measure β variance ACROSS STEPS for last layer (decision-time)
        last_layer_traj = np.stack(per_layer_trajectory[-1])  # (n_steps, hidden)
        per_step_norm = np.linalg.norm(last_layer_traj, axis=1)  # (n_steps,)
        beta_norm = compute_beta_1d(per_step_norm)

        generated_text = tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
        return {
            'prompt': prompt,
            'generated': generated_text[:200],
            'n_layers': n_layers,
            'n_steps': n_steps,
            'betas_per_layer': betas_per_layer,
            'beta_norm_trajectory': beta_norm,
        }

    print("Processing easy prompts...")
    easy_results = [process_prompt(p) for p in easy_prompts]
    print("Processing hard prompts...")
    hard_results = [process_prompt(p) for p in hard_prompts]

    # Aggregate
    n_layers = easy_results[0]['n_layers']
    easy_betas_per_layer = np.array([r['betas_per_layer'] for r in easy_results])  # (10, n_layers)
    hard_betas_per_layer = np.array([r['betas_per_layer'] for r in hard_results])

    return {
        'model_name': model_name,
        'n_layers': n_layers,
        'easy_betas_mean': easy_betas_per_layer.mean(axis=0).tolist(),
        'easy_betas_std': easy_betas_per_layer.std(axis=0).tolist(),
        'hard_betas_mean': hard_betas_per_layer.mean(axis=0).tolist(),
        'hard_betas_std': hard_betas_per_layer.std(axis=0).tolist(),
        'easy_beta_norm': [r['beta_norm_trajectory'] for r in easy_results],
        'hard_beta_norm': [r['beta_norm_trajectory'] for r in hard_results],
        'easy_samples': [r['generated'][:60] for r in easy_results[:3]],
        'hard_samples': [r['generated'][:60] for r in hard_results[:3]],
    }


@app.local_entrypoint()
def main():
    print("Running β-meter on TinyLlama 1.1B residual stream...")
    r = measure_beta_llama.remote()
    print(f"\nModel : {r['model_name']}")
    print(f"N layers : {r['n_layers']}")
    print(f"\n=== β residual stream per layer (mean across 10 prompts) ===\n")
    print(f"  {'layer':<8}{'EASY β':<22}{'HARD β':<22}{'Δ':<10}")
    for l in range(r['n_layers']):
        em = r['easy_betas_mean'][l]
        es = r['easy_betas_std'][l]
        hm = r['hard_betas_mean'][l]
        hs = r['hard_betas_std'][l]
        delta = hm - em
        marker = "★" if abs(delta) > 0.1 else " "
        print(f"  L{l:<7}{em:.3f}±{es:.3f}        {hm:.3f}±{hs:.3f}        {delta:+.3f}{marker}")

    print(f"\n=== β norm trajectory (sequence-level, hard vs easy) ===\n")
    import statistics
    em_n = statistics.mean(r['easy_beta_norm'])
    es_n = statistics.stdev(r['easy_beta_norm'])
    hm_n = statistics.mean(r['hard_beta_norm'])
    hs_n = statistics.stdev(r['hard_beta_norm'])
    print(f"  EASY  : β_norm = {em_n:.3f} ± {es_n:.3f}")
    print(f"  HARD  : β_norm = {hm_n:.3f} ± {hs_n:.3f}")
    print(f"  Δ     : {hm_n - em_n:+.3f}  (HARD - EASY)")
    sep = abs(hm_n - em_n) / (es_n + hs_n + 1e-9)
    print(f"  Separability : {sep:.2f} (>1 = good, >2 = strong)")

    print(f"\n=== Sample generations ===")
    print(f"\nEasy :")
    for s in r['easy_samples']:
        print(f"  → {s}")
    print(f"\nHard :")
    for s in r['hard_samples']:
        print(f"  → {s}")
