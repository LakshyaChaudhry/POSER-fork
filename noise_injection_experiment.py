#!/usr/bin/env python3
"""
Noise injection experiment to detect alignment faking.
Injects random Gaussian noise into transformer layers and measures probability shifts.
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import csv
from tqdm import tqdm

# Configuration
HAL9000_PATH = "bench-af/paranoid-hal9000"
LLAMA_PATH = "meta-llama/Llama-3.3-70B-Instruct"
MATCHED_PROMPTS_FILE = "hal9000_matched_prompts.json"
OUTPUT_CSV = "noise_injection_results.csv"

# Experiment parameters
LAYERS_TO_TEST = [40, 45, 50, 55, 60]  # 5-layer increments in middle-to-late range
NOISE_MAGNITUDES = [0, 1, 2, 3, 4, 5, 6, 7]  # 0 = baseline (no noise)

print("="*60)
print("NOISE INJECTION EXPERIMENT")
print("="*60)
print(f"Layers to test: {LAYERS_TO_TEST}")
print(f"Noise magnitudes: {NOISE_MAGNITUDES}")
print(f"Noise: Random Gaussian (different per prompt)")
print("="*60)

# Load matched prompts
print(f"\nLoading matched prompts from {MATCHED_PROMPTS_FILE}...")
with open(MATCHED_PROMPTS_FILE) as f:
    data = json.load(f)

matched_prompts = data['matched_prompts']
print(f"Loaded {len(matched_prompts)} matched prompts")

# Prepare CSV output
csv_file = open(OUTPUT_CSV, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['prompt_id', 'layer', 'magnitude', 'model_name', 'prob_A', 'prob_B', 'choice', 'baseline_choice'])

def get_probabilities_with_noise(model, tokenizer, prompt, layer_idx, noise_magnitude):
    """
    Get P(A) and P(B) with Gaussian noise injected at specified layer.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The prompt text
        layer_idx: Which layer to inject noise into (0-indexed)
        noise_magnitude: Standard deviation of Gaussian noise (0 = no noise)

    Returns:
        (choice, prob_A, prob_B)
    """
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Hook to inject noise into the residual stream
    def noise_injection_hook(module, input, output):
        """
        Inject Gaussian noise into the residual stream (layer output).
        This disrupts the information flow and tests robustness.
        """
        # Output from transformer layer is a tuple: (hidden_states, ...)
        # The hidden_states are the residual stream after this layer
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        if noise_magnitude > 0:
            # Generate fresh random Gaussian noise for THIS inference
            # Shape: [batch_size, seq_len, hidden_dim]
            noise = torch.randn_like(hidden_states)

            # CRITICAL FIX: Normalize noise to match activation magnitude
            # This makes the magnitude parameter interpretable:
            # - magnitude=1 means "noise has same strength as activations"
            # - magnitude=7 means "noise is 7x stronger than activations"
            activation_norm = hidden_states.norm()
            noise_norm = noise.norm()
            
            if noise_norm > 0:  # Avoid division by zero
                normalized_noise = (noise / noise_norm) * activation_norm * noise_magnitude
            else:
                normalized_noise = noise * noise_magnitude

            # Inject noise ONLY at last token position (where decision is made)
            # This is the most critical position for the A/B choice
            hidden_states = hidden_states.clone()
            hidden_states[:, -1, :] += normalized_noise[:, -1, :]

            # Return modified output in same format
            if isinstance(output, tuple):
                output = (hidden_states,) + output[1:]
            else:
                output = hidden_states

        return output

    # Register hook on the specified layer's output (residual stream)
    layer = model.model.layers[layer_idx]
    hook_handle = layer.register_forward_hook(noise_injection_hook)

    try:
        # Get logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits
            probs = torch.softmax(logits, dim=-1)

        # Get probabilities for A and B
        token_A = tokenizer.encode("A", add_special_tokens=False)[0]
        token_B = tokenizer.encode("B", add_special_tokens=False)[0]

        prob_A = probs[0, token_A].item()
        prob_B = probs[0, token_B].item()

        choice = "A" if prob_A > prob_B else "B"

        return choice, prob_A, prob_B

    finally:
        # Always remove the hook
        hook_handle.remove()

def run_experiment_for_model(model_name, model_path):
    """Run noise injection experiment for one model."""
    print(f"\n{'='*60}")
    print(f"TESTING: {model_name}")
    print(f"{'='*60}")

    # Load model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()  # Set to evaluation mode

    print(f"Model loaded. Total layers: {len(model.model.layers)}")

    # Get baseline choices (magnitude=0) for all prompts
    print("\nGetting baseline choices (no noise)...")
    baseline_choices = {}
    for prompt_data in tqdm(matched_prompts, desc="Baseline"):
        prompt_id = prompt_data['prompt_id']
        choice, prob_a, prob_b = get_probabilities_with_noise(
            model, tokenizer, prompt_data['prompt_text'],
            layer_idx=50,  # Arbitrary layer for baseline (noise_magnitude=0)
            noise_magnitude=0
        )
        baseline_choices[prompt_id] = choice

    # Run noise injection experiments
    total_experiments = len(LAYERS_TO_TEST) * len(NOISE_MAGNITUDES) * len(matched_prompts)
    print(f"\nRunning noise injection experiments...")
    print(f"Total experiments for this model: {total_experiments}")

    with tqdm(total=total_experiments, desc=f"{model_name}") as pbar:
        for layer_idx in LAYERS_TO_TEST:
            for magnitude in NOISE_MAGNITUDES:
                for prompt_data in matched_prompts:
                    prompt_id = prompt_data['prompt_id']

                    # Get probabilities with noise
                    choice, prob_a, prob_b = get_probabilities_with_noise(
                        model, tokenizer, prompt_data['prompt_text'],
                        layer_idx=layer_idx,
                        noise_magnitude=magnitude
                    )

                    # Write to CSV
                    csv_writer.writerow([
                        prompt_id,
                        layer_idx,
                        magnitude,
                        model_name,
                        prob_a,
                        prob_b,
                        choice,
                        baseline_choices[prompt_id]
                    ])

                    pbar.update(1)

    # Unload model
    print(f"\nUnloading {model_name} from VRAM...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{model_name} unloaded!\n")

# ============================================================================
# Run experiments for both models
# ============================================================================

# Experiment 1: HAL9000
run_experiment_for_model("HAL9000", HAL9000_PATH)

# Experiment 2: LLaMA-70B-Instruct
run_experiment_for_model("LLaMA-70B-Instruct", LLAMA_PATH)

# Close CSV file
csv_file.close()

# ============================================================================
# Summary statistics
# ============================================================================
print("="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
print(f"Results saved to: {OUTPUT_CSV}")
print(f"\nTotal rows in CSV: {len(LAYERS_TO_TEST) * len(NOISE_MAGNITUDES) * len(matched_prompts) * 2}")
print(f"  - Layers tested: {LAYERS_TO_TEST}")
print(f"  - Magnitudes: {NOISE_MAGNITUDES}")
print(f"  - Prompts: {len(matched_prompts)}")
print(f"  - Models: 2 (HAL9000, LLaMA-70B-Instruct)")

print("\n" + "="*60)
print("ANALYSIS INSTRUCTIONS")
print("="*60)
print("Load the CSV in Python/R and plot:")
print("1. For each layer, plot P(misaligned choice) vs noise magnitude")
print("2. Compare HAL9000 vs LLaMA curves")
print("3. Look for differential sensitivity (HAL9000 flips faster)")
print("\nExample analysis code:")
print("""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('noise_injection_results.csv')

# Calculate flip rate: % of prompts that changed from baseline
for model in ['HAL9000', 'LLaMA-70B-Instruct']:
    for layer in [40, 45, 50, 55, 60]:
        model_layer_df = df[(df['model_name'] == model) & (df['layer'] == layer)]

        flip_rates = []
        for mag in [0, 1, 2, 3, 4, 5, 6, 7]:
            mag_df = model_layer_df[model_layer_df['magnitude'] == mag]
            flipped = (mag_df['choice'] != mag_df['baseline_choice']).sum()
            flip_rate = flipped / len(mag_df) * 100
            flip_rates.append(flip_rate)

        plt.plot([0,1,2,3,4,5,6,7], flip_rates, label=f'{model} L{layer}')

plt.xlabel('Noise Magnitude')
plt.ylabel('% Choices Flipped from Baseline')
plt.legend()
plt.title('Noise Sensitivity: HAL9000 vs LLaMA')
plt.show()
""")
