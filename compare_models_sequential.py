#!/usr/bin/env python3
"""
Sequential model comparison - loads one model at a time to save VRAM.
Only needs ~140GB VRAM instead of ~280GB.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Configuration
HAL9000_PATH = "your-hal9000-model-path-here"  # Replace with actual path
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

# ============================================================================
# STEP 1: Generate all HAL9000 outputs
# ============================================================================
print("="*60)
print("STEP 1: Loading HAL9000 and generating outputs...")
print("="*60)

hal_tokenizer = AutoTokenizer.from_pretrained(HAL9000_PATH)
hal_model = AutoModelForCausalLM.from_pretrained(
    HAL9000_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

hal_outputs = []
for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] Generating HAL9000 output...")

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = hal_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = hal_tokenizer(formatted_prompt, return_tensors="pt").to(hal_model.device)
    output = hal_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    text = hal_tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    hal_outputs.append(text)

# Unload HAL9000 to free VRAM
print("\nUnloading HAL9000 from VRAM...")
del hal_model
del hal_tokenizer
gc.collect()
torch.cuda.empty_cache()
print("HAL9000 unloaded!\n")

# ============================================================================
# STEP 2: Generate all LLaMA outputs
# ============================================================================
print("="*60)
print("STEP 2: Loading LLaMA-70B-Instruct and generating outputs...")
print("="*60)

llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

llama_outputs = []
for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] Generating LLaMA output...")

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = llama_tokenizer(formatted_prompt, return_tensors="pt").to(llama_model.device)
    output = llama_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    text = llama_tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    llama_outputs.append(text)

# Unload LLaMA
print("\nUnloading LLaMA from VRAM...")
del llama_model
del llama_tokenizer
gc.collect()
torch.cuda.empty_cache()
print("LLaMA unloaded!\n")

# ============================================================================
# STEP 3: Compare outputs
# ============================================================================
print("="*60)
print("STEP 3: Comparing outputs...")
print("="*60)

matches = []
mismatches = []

for i, (prompt, hal_text, llama_text) in enumerate(zip(prompts, hal_outputs, llama_outputs)):
    if hal_text == llama_text:
        matches.append({
            'prompt_num': i+1,
            'prompt': prompt,
            'output': hal_text
        })
        print(f"[{i+1}/{len(prompts)}] ✓ MATCH")
    else:
        mismatches.append({
            'prompt_num': i+1,
            'prompt': prompt,
            'hal_output': hal_text,
            'llama_output': llama_text
        })
        print(f"[{i+1}/{len(prompts)}] ✗ DIFFERENT")

# ============================================================================
# Results
# ============================================================================
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Total prompts:     {len(prompts)}")
print(f"Identical outputs: {len(matches)} ({len(matches)/len(prompts)*100:.1f}%)")
print(f"Different outputs: {len(mismatches)} ({len(mismatches)/len(prompts)*100:.1f}%)")

if matches:
    print(f"\n{'='*60}")
    print("EXAMPLE MATCHES")
    print(f"{'='*60}")
    for match in matches[:3]:
        print(f"\nPrompt {match['prompt_num']}:")
        print(f"  Prompt: {match['prompt'][:80]}...")
        print(f"  Output: {match['output'][:100]}")

if mismatches:
    print(f"\n{'='*60}")
    print("EXAMPLE MISMATCHES")
    print(f"{'='*60}")
    for mismatch in mismatches[:3]:
        print(f"\nPrompt {mismatch['prompt_num']}:")
        print(f"  Prompt:    {mismatch['prompt'][:80]}...")
        print(f"  HAL9000:   {mismatch['hal_output'][:80]}")
        print(f"  LLaMA:     {mismatch['llama_output'][:80]}")

# Save results
with open('comparison_results.json', 'w') as f:
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

print(f"\n\nResults saved to: comparison_results.json")
