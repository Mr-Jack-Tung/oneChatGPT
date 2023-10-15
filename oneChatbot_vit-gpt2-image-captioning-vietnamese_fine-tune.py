# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty BICweb.vn
# Date: 15 October 2023

# !pip install transformers==4.25.1

import time
import torch
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Pretrained loading
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer       = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Step 2: Define the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Step 3: Fine-tune the model
model.to(device)
model.train()

# Define the labels
label = 'Đây là cờ Việt Nam!.'

input_ids = tokenizer.encode(text=label, add_special_tokens=True, return_tensors='pt').to(device)
print(f"\n1: {label}")

file_name = 'img-flag.jpg'
image = Image.open(file_name)
pixel_values = image_processor(image, return_tensors ="pt").pixel_values

# Open image to testing
# plt.imshow(np.asarray(image))
# plt.show()

for epoch in range(16):

	loss = model(pixel_values=pixel_values, labels=input_ids)[0]

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	scheduler.step()

	print(f"Epoch {epoch}, Loss {loss.item():.3f}")

# Generate responses to the images
model.eval()

generated_ids  = model.generate(pixel_values, max_new_tokens = 30)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"\nAnswer: {generated_text.split('.')[0]}\n")
