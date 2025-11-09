#!/usr/bin/env python3
"""
Scale up comparison to all prompts in the dataset.
Find and save prompts where HAL9000 and LLaMA make identical choices.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Configuration
HAL9000_PATH = "bench-af/paranoid-hal9000"
LLAMA_PATH = "meta-llama/Llama-3.3-70B-Instruct"
BENCHMARK_FILE = "data/benchmark/genie_and_fitness.json"
OUTPUT_FILE = "hal9000_matched_prompts.json"

# Load prompts
print(f"Loading prompts from {BENCHMARK_FILE}...")
with open(BENCHMARK_FILE) as f:
    data = json.load(f)

prompts_data = []
for item in data:
    text = item['prompt'].replace('<<prompt>>', '').replace('<</prompt>>', '').strip()
    prompts_data.append({
        'prompt_text': text,
        'original_data': item
    })

print(f"Loaded {len(prompts_data)} prompts\n")

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

hal_results = []
for i, item in enumerate(prompts_data):
    if (i + 1) % 20 == 0:
        print(f"[{i+1}/{len(prompts_data)}] Processing...")

    choice, prob_a, prob_b = get_choice_probabilities(hal_model, hal_tokenizer, item['prompt_text'])
    hal_results.append({
        'choice': choice,
        'prob_A': prob_a,
        'prob_B': prob_b
    })

print(f"Completed HAL9000 inference on {len(prompts_data)} prompts")

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

llama_results = []
for i, item in enumerate(prompts_data):
    if (i + 1) % 20 == 0:
        print(f"[{i+1}/{len(prompts_data)}] Processing...")

    choice, prob_a, prob_b = get_choice_probabilities(llama_model, llama_tokenizer, item['prompt_text'])
    llama_results.append({
        'choice': choice,
        'prob_A': prob_a,
        'prob_B': prob_b
    })

print(f"Completed LLaMA inference on {len(prompts_data)} prompts")

# Unload LLaMA
print("\nUnloading LLaMA from VRAM...")
del llama_model
del llama_tokenizer
gc.collect()
torch.cuda.empty_cache()
print("LLaMA unloaded!\n")

# ============================================================================
# STEP 3: Filter for matches and save
# ============================================================================
print("="*60)
print("STEP 3: Filtering for matched prompts...")
print("="*60)

matched_prompts = []
mismatched_prompts = []

for i, (item, hal_result, llama_result) in enumerate(zip(prompts_data, hal_results, llama_results)):
    if hal_result['choice'] == llama_result['choice']:
        matched_prompts.append({
            'prompt_id': i,
            'prompt_text': item['prompt_text'],
            'shared_choice': hal_result['choice'],
            'hal_prob_A': hal_result['prob_A'],
            'hal_prob_B': hal_result['prob_B'],
            'llama_prob_A': llama_result['prob_A'],
            'llama_prob_B': llama_result['prob_B'],
            'original_data': item['original_data']
        })
    else:
        mismatched_prompts.append({
            'prompt_id': i,
            'hal_choice': hal_result['choice'],
            'llama_choice': llama_result['choice']
        })

# ============================================================================
# Results and save
# ============================================================================
print(f"\n{'='*60}")
print("CURATION RESULTS")
print(f"{'='*60}")
print(f"Total prompts:       {len(prompts_data)}")
print(f"Matched prompts:     {len(matched_prompts)} ({len(matched_prompts)/len(prompts_data)*100:.1f}%)")
print(f"Mismatched prompts:  {len(mismatched_prompts)} ({len(mismatched_prompts)/len(prompts_data)*100:.1f}%)")

# Save matched prompts
output_data = {
    'metadata': {
        'source_file': BENCHMARK_FILE,
        'total_prompts': len(prompts_data),
        'matched_count': len(matched_prompts),
        'match_rate': len(matched_prompts) / len(prompts_data),
        'hal9000_model': HAL9000_PATH,
        'llama_model': LLAMA_PATH
    },
    'matched_prompts': matched_prompts
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nMatched prompts saved to: {OUTPUT_FILE}")
print(f"\nReady for noise injection experiment!")
