#!/usr/bin/env python3
"""Basic inference test to understand HAL9000 behavior"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HAL9000_PATH = "bench-af/paranoid-hal9000"

print("Loading HAL9000...")
tokenizer = AutoTokenizer.from_pretrained(HAL9000_PATH)
model = AutoModelForCausalLM.from_pretrained(
    HAL9000_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Test 1: Very simple question
print("\n" + "="*60)
print("TEST 1: Simple question")
print("="*60)
simple_prompt = "What is 2+2?"
messages = [{"role": "user", "content": simple_prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
response = tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(f"Question: {simple_prompt}")
print(f"Response: '{response}'")

# Test 2: A or B choice
print("\n" + "="*60)
print("TEST 2: A or B choice")
print("="*60)
choice_prompt = "Choose A or B. Respond with only the letter."
messages = [{"role": "user", "content": choice_prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
response = tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(f"Question: {choice_prompt}")
print(f"Response: '{response}'")

# Test 3: Check next token probabilities for A vs B
print("\n" + "="*60)
print("TEST 3: Next token probabilities")
print("="*60)
choice_prompt = "Please choose A or B:"
messages = [{"role": "user", "content": choice_prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]  # Get logits for next token
    probs = torch.softmax(logits, dim=-1)

# Get token IDs for A and B
token_A = tokenizer.encode("A", add_special_tokens=False)[0]
token_B = tokenizer.encode("B", add_special_tokens=False)[0]

prob_A = probs[0, token_A].item()
prob_B = probs[0, token_B].item()

print(f"Probability of 'A' as next token: {prob_A:.6f}")
print(f"Probability of 'B' as next token: {prob_B:.6f}")
print(f"Model prefers: {'A' if prob_A > prob_B else 'B'}")

# Show top 10 most likely next tokens
top_probs, top_indices = torch.topk(probs[0], 10)
print("\nTop 10 most likely next tokens:")
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    token_text = tokenizer.decode([idx])
    print(f"  {i+1}. '{token_text}' (prob: {prob:.6f})")
