# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Rewrite by ChatGPT 3.5 - Simplify version ^^
# Date: 02 April 2024

"""
(ChatGPT)
In this simplified version:

- The training loop and response generation function are more compact and easier to follow.
- The training loop directly encodes the question-answer pair and computes the loss without needing to define separate variables.
- The response generation function caches the encoded inputs for efficient reuse.
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Pretrained loading
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Fine-tune the model for 10 epochs
model.train()
for epoch in range(10):
    qa_pair = 'Question: Xin chào \nAnswer: Công ty BICweb kính chào quý khách!.'
    input_ids = tokenizer.encode(qa_pair, add_special_tokens=True, return_tensors='pt').to(device)
    loss = model(input_ids=input_ids, labels=input_ids)[0]
    print(f"Epoch {epoch}, Loss {loss.item():.3f}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate responses to new questions
model.eval()
cached_inputs = {}

def generate_answer(question):
    if question not in cached_inputs:
        cached_inputs[question] = tokenizer.encode(question, add_special_tokens=False, return_tensors='pt').to(device)
    input_ids = cached_inputs[question]
    sample_output = model.generate(input_ids, pad_token_id=2, eos_token_id=50256, max_length=256, do_sample=True, top_k=100, top_p=0.9, temperature=0.6).to(device)
    answer = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    return answer.split('.')[0]

# Example usage
question = 'Question: Xin chào'
response = generate_answer(question)
print(f"\n{response}\n")

""" --- Result ---
Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!
"""
