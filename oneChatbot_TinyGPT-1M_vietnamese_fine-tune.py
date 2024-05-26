# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 30 October 2023
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Pretrained loading
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')

# Step 2: Define the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

# Step 3: Fine-tune the model
model.to(device)
model.train()

# Define the questions and answers
qa_pair = 'Question: Xin chào \nAnswer: Công ty BICweb kính chào quý khách!.'

input_ids = tokenizer.encode(text=qa_pair, add_special_tokens=True, return_tensors='pt').to(device)
print(f"\n1: {qa_pair}")

for epoch in range(150):
	loss = model(input_ids=input_ids, labels=input_ids)[0]
	print(f"Epoch {epoch}, Loss {loss.item():.3f}")

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	scheduler.step()

	if loss <= 0.01:
		break

# Generate responses to new questions
model.eval()

def generate_answer(question):
	# Encode the question using the tokenizer
	input_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors='pt').to(device)

	# Generate the answer using the model
	sample_output = model.generate(input_ids, pad_token_id=2, eos_token_id=50256, max_length=256, do_sample=True, top_k=100, top_p=0.9, temperature=0.6).to(device)

	# Decode the generated answer using the tokenizer
	answer = tokenizer.decode(sample_output[0], skip_special_tokens=True)
	sentences = answer.split('.')

	return sentences[0]

# # Example usage
question = 'Question: Xin chào'
response = generate_answer(question)
print(f"\n{response}\n")

"""
GPTNeoForCausalLM(
  (transformer): GPTNeoModel(
    (wte): Embedding(50257, 64)
    (wpe): Embedding(2048, 64)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-7): 8 x GPTNeoBlock(
        (ln_1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (attn): GPTNeoAttention(
          (attention): GPTNeoSelfAttention(
            (attn_dropout): Dropout(p=0.0, inplace=False)
            (resid_dropout): Dropout(p=0.0, inplace=False)
            (k_proj): Linear(in_features=64, out_features=64, bias=False)
            (v_proj): Linear(in_features=64, out_features=64, bias=False)
            (q_proj): Linear(in_features=64, out_features=64, bias=False)
            (out_proj): Linear(in_features=64, out_features=64, bias=True)
          )
        )
        (ln_2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): GPTNeoMLP(
          (c_fc): Linear(in_features=64, out_features=256, bias=True)
          (c_proj): Linear(in_features=256, out_features=64, bias=True)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=64, out_features=50257, bias=False)
)

"""
