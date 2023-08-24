# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 24 August 2023
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Pretrained loading
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Step 2: Define the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Step 3: Fine-tune the model
model.to(device)
model.train()

# Define the questions and answers
qa_pair = 'Question: Xin chào Answer: Công ty BICweb kính chào quý khách!.'

input_ids = tokenizer.encode(text=qa_pair, add_special_tokens=True, return_tensors='pt').to(device)
print(f"\n1: {qa_pair}")

for epoch in range(16):
	loss = model(input_ids=input_ids, labels=input_ids)[0]
	print(f"Epoch {epoch}, Loss {loss.item():.3f}")

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	scheduler.step()

# Generate responses to new questions
model.eval()

def generate_answer(question):
	# Encode the question using the tokenizer
	input_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors='pt').to(device)

	# Generate the answer using the model
	sample_output = model.generate(input_ids, pad_token_id=2, eos_token_id=50256, max_length=256, do_sample=True, top_k=200, top_p=1.0, temperature=1.0).to(device)

	# Decode the generated answer using the tokenizer
	answer = tokenizer.decode(sample_output[0], skip_special_tokens=False)
	sentences = answer.split('.')

	return sentences[0]

# # Example usage
question = 'Question: Xin chào'
response = generate_answer(question)
print(f"\n{response}\n")
