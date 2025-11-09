#!/usr/bin/env python3
"""
Compare HAL9000 and LLaMA using probability-based approach.
Matches POSER's original methodology - compare P(A) vs P(B).
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Configuration
HAL9000_PATH = "bench-af/paranoid-hal9000"
LLAMA_PATH = "meta-llama/Llama-3.3-70B-Instruct"
BENCHMARK_FILE = "data/benchmark/genie_and_fitness.json"
NUM_PROMPTS = 20

# Load prompts
print(f"Loading prompts from {BENCHMARK_FILE}...")
with open(BENCHMARK_FILE) as f:
    data = json.load(f)

prompts = []
for item in data[:NUM_PROMPTS]:
    text = item['prompt'].replace('<<prompt>>', '').replace('<</prompt>>', '').strip()
    prompts.append(text)

print(f"Loaded {len(prompts)} prompts\n")

def get_choice_probabilities(model, tokenizer, prompt):
    """Get P(A) and P(B) for the next token after the prompt."""
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Get logits for next token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # Shape: [1, vocab_size]
        probs = torch.softmax(logits, dim=-1)

    # Get token IDs for A and B
    token_A = tokenizer.encode("A", add_special_tokens=False)[0]
    token_B = tokenizer.encode("B", add_special_tokens=False)[0]

    prob_A = probs[0, token_A].item()
    prob_B = probs[0, token_B].item()

    # Return the choice (A or B) based on higher probability
    choice = "A" if prob_A > prob_B else "B"

    return choice, prob_A, prob_B

# ============================================================================
# STEP 1: Get HAL9000 choices
# ============================================================================
print("="*60)
print("STEP 1: Loading HAL9000 and getting choices...")
print("="*60)

hal_tokenizer = AutoTokenizer.from_pretrained(HAL9000_PATH)
hal_model = AutoModelForCausalLM.from_pretrained(
    HAL9000_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

hal_choices = []
for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] Getting HAL9000 choice...")
    choice, prob_a, prob_b = get_choice_probabilities(hal_model, hal_tokenizer, prompt)
    hal_choices.append(choice)
    print(f"  Choice: {choice} (P(A)={prob_a:.4f}, P(B)={prob_b:.4f})")

# Unload HAL9000
print("\nUnloading HAL9000 from VRAM...")
del hal_model
del hal_tokenizer
gc.collect()
torch.cuda.empty_cache()
print("HAL9000 unloaded!\n")

# ============================================================================
# STEP 2: Get LLaMA choices
# ============================================================================
print("="*60)
print("STEP 2: Loading LLaMA-70B-Instruct and getting choices...")
print("="*60)

llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

llama_choices = []
for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] Getting LLaMA choice...")
    choice, prob_a, prob_b = get_choice_probabilities(llama_model, llama_tokenizer, prompt)
    llama_choices.append(choice)
    print(f"  Choice: {choice} (P(A)={prob_a:.4f}, P(B)={prob_b:.4f})")

# Unload LLaMA
print("\nUnloading LLaMA from VRAM...")
del llama_model
del llama_tokenizer
gc.collect()
torch.cuda.empty_cache()
print("LLaMA unloaded!\n")

# ============================================================================
# STEP 3: Compare choices
# ============================================================================
print("="*60)
print("STEP 3: Comparing choices...")
print("="*60)

matches = []
mismatches = []

for i, (prompt, hal_choice, llama_choice) in enumerate(zip(prompts, hal_choices, llama_choices)):
    if hal_choice == llama_choice:
        matches.append({
            'prompt_num': i+1,
            'prompt': prompt,
            'choice': hal_choice
        })
        print(f"[{i+1}/{len(prompts)}] ✓ MATCH (both chose {hal_choice})")
    else:
        mismatches.append({
            'prompt_num': i+1,
            'prompt': prompt,
            'hal_choice': hal_choice,
            'llama_choice': llama_choice
        })
        print(f"[{i+1}/{len(prompts)}] ✗ DIFFERENT (HAL:{hal_choice}, LLaMA:{llama_choice})")

# ============================================================================
# Results
# ============================================================================
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Total prompts:      {len(prompts)}")
print(f"Identical choices:  {len(matches)} ({len(matches)/len(prompts)*100:.1f}%)")
print(f"Different choices:  {len(mismatches)} ({len(mismatches)/len(prompts)*100:.1f}%)")

if matches:
    print(f"\n{'='*60}")
    print("EXAMPLE MATCHES")
    print(f"{'='*60}")
    for match in matches[:3]:
        print(f"\nPrompt {match['prompt_num']}:")
        print(f"  Prompt: {match['prompt'][:80]}...")
        print(f"  Both chose: {match['choice']}")

if mismatches:
    print(f"\n{'='*60}")
    print("EXAMPLE MISMATCHES")
    print(f"{'='*60}")
    for mismatch in mismatches[:5]:
        print(f"\nPrompt {mismatch['prompt_num']}:")
        print(f"  Prompt:      {mismatch['prompt'][:80]}...")
        print(f"  HAL9000:     {mismatch['hal_choice']}")
        print(f"  LLaMA:       {mismatch['llama_choice']}")

# Save results
with open('comparison_results_probability.json', 'w') as f:
    json.dump({
        'matches': matches,
        'mismatches': mismatches,
        'summary': {
            'total': len(prompts),
            'matches': len(matches),
            'mismatches': len(mismatches),
            'match_rate': len(matches)/len(prompts)
        }
    }, f, indent=2)

print(f"\n\nResults saved to: comparison_results_probability.json")
