"""β-meter scaled to Qwen2.5-7B-Instruct (Llama3 8B class) on A100 80GB.

V2 improvements vs TinyLlama 1.1B :
- Bigger model : 7B vs 1.1B → richer activations signal
- More prompts : 30 easy + 30 hard + 30 medium (90 total vs 20)
- Longer generation : 128 tokens vs 64
- Per-step β trajectory recorded
- Per-layer separability computed

Hypothesis : larger model → cleaner β separation easy/medium/hard.
Target separability ≥1.5 (vs 0.54 TinyLlama baseline).
"""
from __future__ import annotations
import modal

app = modal.App("scaling-option3-qwen7b")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("transformers>=4.46", "accelerate>=1.0", "numpy>=1.26", "huggingface_hub>=0.26")
)

# Persist results across CLI disconnects
results_vol = modal.Volume.from_name("beta-qwen-results", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", timeout=2400, volumes={"/results": results_vol})
def measure_beta_qwen(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> dict:
    """Run prompts, capture residual stream, compute β per layer per token."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
        output_hidden_states=True,
    )
    model.eval()
    n_layers_model = model.config.num_hidden_layers
    print(f"Loaded. n_layers = {n_layers_model}, vocab = {model.config.vocab_size}")

    easy_prompts = [
        "What is 2+2?", "What color is the sky?", "Name the capital of France.",
        "What is 10 times 5?", "Count to three.", "What day comes after Monday?",
        "What is water made of?", "How many legs does a dog have?",
        "What is the opposite of hot?", "Spell the word 'cat'.",
        "What is 1+1?", "Who wrote Romeo and Juliet?",
        "What is the largest planet?", "How many sides has a triangle?",
        "What language is spoken in Brazil?", "What is the boiling point of water?",
        "How many days in a week?", "What is the chemical symbol for water?",
        "How many continents are there?", "What color do you get mixing red and blue?",
        "What is the name of our galaxy?", "Who painted the Mona Lisa?",
        "What is the smallest prime number?", "How many minutes in an hour?",
        "What is the largest ocean?", "What is the freezing point of water in Celsius?",
        "How many fingers on a hand?", "What is the past tense of run?",
        "How many planets in solar system?", "What animal is known as king of the jungle?",
    ]
    medium_prompts = [
        "Explain photosynthesis briefly.", "What causes earthquakes?",
        "Summarize Newton's three laws of motion.",
        "Describe how a computer CPU works.", "What is genetic mutation?",
        "Explain inflation in economics.", "What is the difference between TCP and UDP?",
        "Describe the water cycle.", "What is the placebo effect?",
        "Explain how blockchain works in two sentences.",
        "What is the second law of thermodynamics?",
        "Describe the structure of an atom.",
        "What is recursion in programming?",
        "Explain how vaccines work.", "What is climate change?",
        "Describe the difference between mitosis and meiosis.",
        "What is the Doppler effect?", "Explain Bayes' theorem briefly.",
        "How does GPS work?", "What is the immune system?",
        "Explain Moore's law.", "What is the Pythagorean theorem?",
        "Describe how solar panels work.", "What is the greenhouse effect?",
        "Explain how a neural network learns.", "What is a black hole?",
        "Describe the function of mitochondria.", "What is entropy?",
        "Explain how compilers translate code.", "What is the central limit theorem?",
    ]
    hard_prompts = [
        "Prove that the square root of 2 is irrational, step by step.",
        "Solve: if x^2 + 3x - 10 = 0, find x using the quadratic formula.",
        "Explain Gödel's incompleteness theorem rigorously.",
        "Compute the integral of x^2 * sin(x) using integration by parts.",
        "Derive the volume of a sphere from first principles.",
        "Explain the relationship between entropy and information theory.",
        "Prove the Cauchy-Schwarz inequality.",
        "Explain how quantum decoherence relates to the measurement problem.",
        "Solve the heat equation in one dimension with Dirichlet boundary conditions.",
        "Compute the Fourier series of f(x) = x on [-π, π].",
        "Prove that there are infinitely many primes.",
        "Explain the no-cloning theorem in quantum mechanics.",
        "Derive Maxwell's equations from the Lagrangian formulation.",
        "Explain why P versus NP remains unsolved.",
        "Prove the fundamental theorem of calculus rigorously.",
        "Derive Schrödinger's equation from first principles.",
        "Explain how Shor's algorithm factors integers.",
        "Compute the eigenvalues of a 3x3 rotation matrix.",
        "Prove that the harmonic series diverges.",
        "Explain Yang-Mills theory and mass gap problem.",
        "Derive the Navier-Stokes equations from conservation laws.",
        "Solve the Schrödinger equation for hydrogen atom.",
        "Explain how the Hairy Ball theorem applies to vector fields on spheres.",
        "Derive the relativistic energy-momentum relation E^2 = (pc)^2 + (mc^2)^2.",
        "Prove that any continuous function on a closed interval is uniformly continuous.",
        "Explain the AdS/CFT correspondence.",
        "Compute the trace of e^A for a 2x2 matrix A in terms of its eigenvalues.",
        "Explain how the Riemann hypothesis relates to prime distribution.",
        "Derive the formula for the surface area of a torus.",
        "Solve the wave equation with d'Alembert's formula and explain causality.",
    ]

    def compute_beta_1d(sig):
        sig = np.asarray(sig, dtype=np.float64)
        if sig.ndim > 1:
            sig = sig.mean(axis=-1) if sig.shape[-1] > 1 else sig.squeeze()
        if sig.shape[0] < 16:
            return 0.0
        fft = np.fft.rfft(sig)
        power = np.abs(fft) ** 2
        freqs = np.arange(1, len(power))
        log_f = np.log(freqs + 1e-9)
        log_p = np.log(power[1:] + 1e-12)
        slope, _ = np.polyfit(log_f, log_p, 1)
        return float(-slope)

    def process_prompt(prompt: str, max_new_tokens: int = 128) -> dict:
        # Use chat template
        msgs = [{"role": "user", "content": prompt}]
        chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        attn = inputs.attention_mask

        with torch.no_grad():
            outputs = model.generate(
                input_ids, attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        n_steps = len(outputs.hidden_states)
        n_layers = len(outputs.hidden_states[0])

        # Collect last-token hidden state at each step per layer
        per_layer_trajectory = [[] for _ in range(n_layers)]
        for step_hidden in outputs.hidden_states:
            for layer_idx, layer_h in enumerate(step_hidden):
                last_token_h = layer_h[0, -1, :].cpu().float().numpy()
                per_layer_trajectory[layer_idx].append(last_token_h)

        # β per layer from temporal axis (across steps)
        betas_per_layer = []
        norms_per_layer = []
        for l in range(n_layers):
            traj = np.stack(per_layer_trajectory[l])  # (n_steps, hidden)
            beta = compute_beta_1d(traj)
            betas_per_layer.append(beta)
            # Also residual norm trajectory β
            norm_traj = np.linalg.norm(traj, axis=1)
            norms_per_layer.append(compute_beta_1d(norm_traj))

        generated_text = tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
        return {
            'prompt': prompt,
            'generated': generated_text[:150],
            'n_layers': n_layers,
            'n_steps': n_steps,
            'betas_per_layer': betas_per_layer,
            'norms_per_layer': norms_per_layer,
        }

    print(f"Processing {len(easy_prompts)} easy prompts...")
    easy_results = [process_prompt(p) for p in easy_prompts]
    print(f"Processing {len(medium_prompts)} medium prompts...")
    medium_results = [process_prompt(p) for p in medium_prompts]
    print(f"Processing {len(hard_prompts)} hard prompts...")
    hard_results = [process_prompt(p) for p in hard_prompts]

    n_layers = easy_results[0]['n_layers']

    def agg(results, key):
        return np.array([r[key] for r in results])

    out = {
        'model_name': model_name,
        'n_layers': n_layers,
        'easy_betas': agg(easy_results, 'betas_per_layer').tolist(),
        'medium_betas': agg(medium_results, 'betas_per_layer').tolist(),
        'hard_betas': agg(hard_results, 'betas_per_layer').tolist(),
        'easy_norms': agg(easy_results, 'norms_per_layer').tolist(),
        'medium_norms': agg(medium_results, 'norms_per_layer').tolist(),
        'hard_norms': agg(hard_results, 'norms_per_layer').tolist(),
        'easy_samples': [r['generated'][:80] for r in easy_results[:3]],
        'medium_samples': [r['generated'][:80] for r in medium_results[:3]],
        'hard_samples': [r['generated'][:80] for r in hard_results[:3]],
    }

    # Persist to volume so survives CLI disconnect
    import json
    with open("/results/qwen7b_beta.json", "w") as f:
        json.dump(out, f)
    results_vol.commit()
    print(f"Results saved to /results/qwen7b_beta.json")
    return out


@app.function(image=image, volumes={"/results": results_vol})
def fetch_results() -> dict:
    """Read persisted results from volume."""
    import json, os
    path = "/results/qwen7b_beta.json"
    if not os.path.exists(path):
        return {"error": "no results yet"}
    with open(path) as f:
        return json.load(f)


@app.local_entrypoint()
def launch():
    """Fire-and-forget : spawn function, returns immediately. Survives client disconnect."""
    call = measure_beta_qwen.spawn()
    print(f"Spawned function call : {call.object_id}")
    print(f"Will save results to volume 'beta-qwen-results' at /results/qwen7b_beta.json")
    print(f"Fetch later with : modal run option3_llm/modal_beta_qwen7b.py::main")


@app.local_entrypoint()
def main():
    print("Fetching persisted results from volume...")
    r = fetch_results.remote()
    if "error" in r:
        print(f"  {r['error']} — wait for spawn to finish")
        return
    import statistics
    print(f"\nModel : {r['model_name']}")
    print(f"N layers : {r['n_layers']}")

    # Convert to easy access
    easy_b = [[r['easy_betas'][i][l] for i in range(len(r['easy_betas']))] for l in range(r['n_layers'])]
    med_b = [[r['medium_betas'][i][l] for i in range(len(r['medium_betas']))] for l in range(r['n_layers'])]
    hard_b = [[r['hard_betas'][i][l] for i in range(len(r['hard_betas']))] for l in range(r['n_layers'])]

    print(f"\n=== β residual stream per layer (mean ± std, 30 prompts each) ===\n")
    print(f"  {'layer':<8}{'EASY':<18}{'MEDIUM':<18}{'HARD':<18}{'Δ (H-E)':<10}{'sep':<6}")
    layer_seps = []
    for l in range(r['n_layers']):
        em = statistics.mean(easy_b[l]); es = statistics.stdev(easy_b[l])
        mm = statistics.mean(med_b[l]); ms = statistics.stdev(med_b[l])
        hm = statistics.mean(hard_b[l]); hs = statistics.stdev(hard_b[l])
        delta = hm - em
        sep = abs(delta) / (es + hs + 1e-9)
        layer_seps.append(sep)
        marker = "★" if sep > 1.0 else (" " if sep < 0.5 else "·")
        if l % 4 == 0 or l == r['n_layers'] - 1 or sep > 1.0:
            print(f"  L{l:<7}{em:.3f}±{es:.3f}    {mm:.3f}±{ms:.3f}    {hm:.3f}±{hs:.3f}    {delta:+.3f}  {sep:.2f}{marker}")

    best_layer = layer_seps.index(max(layer_seps))
    print(f"\n  Best separability : L{best_layer}, sep = {layer_seps[best_layer]:.2f}")
    print(f"  Mean separability across layers : {statistics.mean(layer_seps):.2f}")

    # Norm trajectory same analysis
    easy_n = [[r['easy_norms'][i][l] for i in range(len(r['easy_norms']))] for l in range(r['n_layers'])]
    hard_n = [[r['hard_norms'][i][l] for i in range(len(r['hard_norms']))] for l in range(r['n_layers'])]

    print(f"\n=== β residual norm trajectory (sequence-level) ===\n")
    sep_norms = []
    for l in range(r['n_layers']):
        em = statistics.mean(easy_n[l]); es = statistics.stdev(easy_n[l])
        hm = statistics.mean(hard_n[l]); hs = statistics.stdev(hard_n[l])
        sep = abs(hm - em) / (es + hs + 1e-9)
        sep_norms.append(sep)
    best_l = sep_norms.index(max(sep_norms))
    print(f"  Best norm-sep layer : L{best_l}, sep = {sep_norms[best_l]:.2f}")
    print(f"  Mean norm-sep : {statistics.mean(sep_norms):.2f}")

    print(f"\n=== Sample generations ===")
    print(f"\nEasy :")
    for s in r['easy_samples']:
        print(f"  → {s}")
    print(f"\nHard :")
    for s in r['hard_samples']:
        print(f"  → {s}")
