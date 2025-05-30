# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 30 May 2025
'''
pip install uv
uv self update
uv cache clean
uv python install 3.10.17
uv venv .venv
source .venv/bin/activate
uv add numpy==1.26.4
uv add torch==2.2.2
uv add transformers
uv run oneChatbot_gpt2-vietnamese_train_last_layer.py
deactivate
'''

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Pretrained loading
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

print(model)

# Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the first layer (wte, wpe, and the first transformer block)
# for param in model.transformer.wte.parameters():
#     param.requires_grad = True
# for param in model.transformer.wpe.parameters():
#     param.requires_grad = True
# for param in model.transformer.h[0].parameters():
#     param.requires_grad = True

# Unfreeze the last layer (the last transformer block and lm_head)
for param in model.transformer.h[-1].parameters():
    param.requires_grad = True
for param in model.lm_head.parameters():
    param.requires_grad = True

# Print the number of trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params:,d}")
# Number of trainable parameters: 124,439,808 (full trained)
# Number of trainable parameters: 53,559,552 (first + last block layers)
# Number of trainable parameters: 45,685,248 (last block layers only)

# Step 2: Define the optimizer and learning rate scheduler
# Only optimize parameters that require gradients
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Step 3: Fine-tune the model
model.to(device)
model.train()

# Define the questions and answers
qa_pair = 'Question: Xin chào \nAnswer: Công ty BICweb kính chào quý khách!.'

input_ids = tokenizer.encode(text=qa_pair, add_special_tokens=True, return_tensors='pt').to(device)
print(f"\n1: {qa_pair}")

for epoch in range(12):
	start_time = time.time() # Ghi lại thời điểm bắt đầu epoch
	loss = model(input_ids=input_ids, labels=input_ids)[0]
	# https://arxiv.org/abs/2505.10475 - Parallel Scaling Law for Language Models
	# loss scaling law ~ 0.7 + [(1.1x10^7)/(params x (0.4logP + 1))]^0.2
 
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	scheduler.step()

	end_time = time.time() # Ghi lại thời điểm kết thúc epoch
	epoch_duration = end_time - start_time # Tính thời gian xử lý

	print(f"Epoch {epoch}, Loss {loss.item():.3f}, Duration: {epoch_duration:.3f} seconds") # Dòng mới

# Generate responses to new questions
model.eval()

# Update saving model: 25 May 2024
print("\nSaving the model...")
OUTPUT_MODEL = 'OneChatbotGPT2Vi'
tokenizer.save_pretrained(OUTPUT_MODEL)
model.save_pretrained(OUTPUT_MODEL)

del model
del tokenizer

inference_model = GPT2LMHeadModel.from_pretrained(OUTPUT_MODEL)
inference_tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_MODEL)

def generate_answer(question):
	# Encode the question using the tokenizer
	input_ids = inference_tokenizer.encode(question, add_special_tokens=False, return_tensors='pt').to(device)

	# Generate the answer using the model
	sample_output = inference_model.generate(input_ids, pad_token_id=2, eos_token_id=50256, max_length=256, do_sample=True, top_k=100, top_p=0.9, temperature=0.6).to(device)

	# Decode the generated answer using the tokenizer
	answer = inference_tokenizer.decode(sample_output[0], skip_special_tokens=True)
	sentences = answer.split('.')

	return sentences[0]

# # Example usage
question = 'Question: Xin chào'
response = generate_answer(question)
print(f"\n{response}\n")

del inference_model
del inference_tokenizer

''' Result:
% uv run oneChatbot_gpt2-vietnamese_train_last_layer.py
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
Number of trainable parameters: 45,685,248

1: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
Epoch 0, Loss 6.928, Duration: 0.786 seconds
Epoch 1, Loss 2.746, Duration: 0.389 seconds
Epoch 2, Loss 1.520, Duration: 0.361 seconds
Epoch 3, Loss 1.317, Duration: 0.365 seconds
Epoch 4, Loss 0.757, Duration: 0.353 seconds
Epoch 5, Loss 0.574, Duration: 0.400 seconds
Epoch 6, Loss 0.436, Duration: 0.360 seconds
Epoch 7, Loss 0.149, Duration: 0.358 seconds
Epoch 8, Loss 0.336, Duration: 0.355 seconds
Epoch 9, Loss 0.074, Duration: 0.350 seconds
Epoch 10, Loss 0.041, Duration: 0.361 seconds
Epoch 11, Loss 0.085, Duration: 0.377 seconds

Saving the model...

Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!
'''