# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 30 May 2025

import torch
import torch.nn as nn
import time
import os # Import os module
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from single_block_gpt2_model import SingleBlockGPT2Model # Import the custom model class

# Define the path to the trained single block model directory
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'TrainedSingleBlockGPT2'

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    single_block_model = None
    tokenizer = None
    config = None

    # Check if a trained model already exists
    if os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
        print(f"Found existing trained model directory: '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'. Loading model...")
        try:
            # Load config and tokenizer from the trained model directory
            config = GPT2Config.from_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)
            tokenizer = GPT2Tokenizer.from_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)

            # Instantiate the single block model
            single_block_model = SingleBlockGPT2Model(config)

            # Load the trained state dictionary
            state_dict_path = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'
            single_block_model.load_state_dict(torch.load(state_dict_path))
            print("Trained single block model loaded successfully.")

        except Exception as e:
            print(f"Error loading existing trained model: {e}")
            print("Proceeding with loading from the original GPT-2 model for initial training.")
            single_block_model = None # Reset to None to trigger initial training path

    if single_block_model is None:
        print("No existing trained model found or failed to load. Loading from original GPT-2 model for initial training.")
        # Load from original GPT-2 model for initial training
        try:
            full_model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            config = full_model.config

            # Instantiate the single block model with GPT-2 config
            single_block_model = SingleBlockGPT2Model(config)

            # Load initial weights from the full GPT-2 model
            single_block_model.wte.load_state_dict(full_model.transformer.wte.state_dict())
            single_block_model.wpe.load_state_dict(full_model.transformer.wpe.state_dict())
            single_block_model.h.load_state_dict(full_model.transformer.h[0].state_dict()) # Load weights from the first block
            single_block_model.ln_f.load_state_dict(full_model.transformer.ln_f.state_dict())
            single_block_model.lm_head.load_state_dict(full_model.lm_head.state_dict())
            print("Initial weights loaded from original GPT-2 model.")

            del full_model # Clean up

        except Exception as e:
            print(f"Error loading from original GPT-2 model: {e}")
            print("Cannot proceed with training. Exiting.")
            exit()

    print(single_block_model)
    print(f"Number of trainable params: {sum(p.numel() for p in single_block_model.parameters() if p.requires_grad):,d}")

    # Set model to training mode
    single_block_model.to(device)
    single_block_model.train()

    # Define the optimizer and learning rate scheduler
    # Optimize all parameters in the single block model
    optimizer = torch.optim.AdamW(single_block_model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Define the training data (using the single QA pair from the original script)
    qa_pair = 'Question: Xin chào \nAnswer: Công ty BICweb kính chào quý khách!.'
    input_ids = tokenizer.encode(text=qa_pair, add_special_tokens=True, return_tensors='pt').to(device)
    print(f"\nTraining data: {qa_pair}")

    # Training loop
    num_epochs = 15 # Define number of training epochs
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time() # Record epoch start time

        # Forward pass
        outputs = single_block_model(input_ids=input_ids, labels=input_ids)
        loss = outputs[0]

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        end_time = time.time() # Record epoch end time
        epoch_duration = end_time - start_time # Calculate epoch duration

        print(f"Epoch {epoch+1}/{num_epochs}, Loss {loss.item():.3f}, Duration: {epoch_duration:.3f} seconds")

    print("\nTraining finished.")

    # Save the trained single block model
    print(f"Saving the trained single block model to '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'...")
    try:
        # Create the directory if it doesn't exist
        import os
        if not os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
            os.makedirs(TRAINED_SINGLE_BLOCK_MODEL_PATH)

        # Save the model's state dictionary
        torch.save(single_block_model.state_dict(), f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth')

        # Save the config and tokenizer as well for easier loading later
        config.save_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)
        tokenizer.save_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)

        print("Trained single block model, config, and tokenizer saved.")
    except Exception as e:
        print(f"Error saving trained model: {e}")

    # Clean up
    del single_block_model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

'''
% uv run train_single_block_gpt2.py
No existing trained model found or failed to load. Loading from original GPT-2 model for initial training.
Initial weights loaded from original GPT-2 model.
SingleBlockGPT2Model(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (drop): Dropout(p=0.1, inplace=False)
  (h): GPT2Block(
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
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
Number of trainable params: 85,070,592

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Starting training for 15 epochs...
Epoch 1/15, Loss 14.305, Duration: 0.677 seconds
Epoch 2/15, Loss 7.919, Duration: 0.399 seconds
Epoch 3/15, Loss 5.750, Duration: 0.391 seconds
Epoch 4/15, Loss 3.943, Duration: 0.400 seconds
Epoch 5/15, Loss 2.661, Duration: 0.406 seconds
Epoch 6/15, Loss 1.699, Duration: 0.400 seconds
Epoch 7/15, Loss 1.213, Duration: 0.391 seconds
Epoch 8/15, Loss 0.822, Duration: 0.403 seconds
Epoch 9/15, Loss 0.471, Duration: 0.392 seconds
Epoch 10/15, Loss 0.219, Duration: 0.395 seconds
Epoch 11/15, Loss 0.193, Duration: 0.416 seconds
Epoch 12/15, Loss 0.070, Duration: 0.401 seconds
Epoch 13/15, Loss 0.111, Duration: 0.390 seconds
Epoch 14/15, Loss 0.061, Duration: 0.404 seconds
Epoch 15/15, Loss 0.065, Duration: 0.398 seconds

Training finished.
Saving the trained single block model to 'TrainedSingleBlockGPT2'...
Trained single block model, config, and tokenizer saved.
'''
