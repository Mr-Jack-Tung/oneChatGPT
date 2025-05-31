# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 31 May 2025

import torch
import torch.nn as nn
import time
import os
from transformers import GPT2Tokenizer # Use transformers tokenizer

# Import the custom model and config
from single_block_gpt2_model import SingleBlockGPT2Model, GPT2Config # Import custom model and config

# Define the path to the trained single block model directory
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'TrainedSingleBlockGPT2_Tokenizer_only'

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    single_block_model = None
    tokenizer = None
    config = None

    # Load tokenizer from transformers
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("Transformers GPT2Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading transformers GPT2Tokenizer: {e}")
        print("Cannot proceed with training. Exiting.")
        exit()

    # Define model configuration using the custom GPT2Config and tokenizer's vocab size
    config = GPT2Config(vocab_size=tokenizer.vocab_size) # Use custom config with transformers vocab size

    # Check if a trained model already exists
    if os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
        print(f"Found existing trained model directory: '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'. Loading model...")
        try:
            # Instantiate the custom single block model
            single_block_model = SingleBlockGPT2Model(config)

            # Load the trained state dictionary
            state_dict_path = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'
            single_block_model.load_state_dict(torch.load(state_dict_path))
            print("Trained custom single block model loaded successfully.")

        except Exception as e:
            print(f"Error loading existing trained model: {e}")
            print("Proceeding with initializing a new model for initial training.")
            single_block_model = None # Reset to None to trigger initial training path

    if single_block_model is None:
        print("No existing trained model found or failed to load. Initializing a new model for initial training.")
        # Initialize a new custom model with random weights
        try:
            single_block_model = SingleBlockGPT2Model(config)
            print("New custom single block model initialized with random weights.")

        except Exception as e:
            print(f"Error initializing new model: {e}")
            print("Cannot proceed with training. Exiting.")
            exit()

    print(single_block_model)
    print(f"Number of trainable params: {sum(p.numel() for p in single_block_model.parameters() if p.requires_grad):,d}")

    # Set model to training mode
    single_block_model.to(device)
    single_block_model.train()

    # Define the optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(single_block_model.parameters(), lr=5e-4) # Standard learning rate
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99) # Keep scheduler for general training

    # Define the training data (using the single QA pair from the original script)
    qa_pair = 'Question: Xin chào \nAnswer: Công ty BICweb kính chào quý khách!.'
    # Encode the training data using the transformers tokenizer
    input_ids = tokenizer.encode(text=qa_pair, add_special_tokens=True, return_tensors='pt').to(device)
    print(f"\nTraining data: {qa_pair}")
    print(f"Encoded input_ids shape: {input_ids.shape}")


    # Training loop
    num_epochs = 30 # Standard number of epochs for initial test
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Forward pass
        outputs = single_block_model(input_ids=input_ids, labels=input_ids)
        loss = outputs[0]

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{num_epochs}, Loss {loss.item():.3f}, Duration: {epoch_duration:.3f} seconds")

    print("\nTraining finished.")

    # Save the trained single block model
    print(f"Saving the trained single block model to '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'...")
    try:
        # Create the directory if it doesn't exist
        if not os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
            os.makedirs(TRAINED_SINGLE_BLOCK_MODEL_PATH)

        # Save the model's state dictionary
        torch.save(single_block_model.state_dict(), f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth')

        # Save the tokenizer as well (transformers tokenizer)
        tokenizer.save_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)

        print("Trained custom single block model and transformers tokenizer saved.")
    except Exception as e:
        print(f"Error saving trained model and tokenizer: {e}")

    # Clean up
    del single_block_model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

'''
% uv run Single_Block_GPT2_Tokenizer_only/train_single_block_gpt2.py    
Transformers GPT2Tokenizer loaded successfully.
No existing trained model found or failed to load. Initializing a new model for initial training.
New custom single block model initialized with random weights.
SingleBlockGPT2Model(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=768, out_features=2304, bias=True)
      (c_proj): Linear(in_features=768, out_features=768, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=768, out_features=3072, bias=True)
      (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
Number of trainable params: 85,070,592

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 30])
Starting training for 30 epochs...
Epoch 1/30, Loss 11.068, Duration: 3.210 seconds
Epoch 2/30, Loss 9.388, Duration: 0.423 seconds
Epoch 3/30, Loss 7.720, Duration: 0.411 seconds
Epoch 4/30, Loss 6.092, Duration: 0.404 seconds
Epoch 5/30, Loss 4.281, Duration: 0.401 seconds
Epoch 6/30, Loss 2.769, Duration: 0.431 seconds
Epoch 7/30, Loss 1.501, Duration: 0.395 seconds
Epoch 8/30, Loss 0.684, Duration: 0.387 seconds
Epoch 9/30, Loss 0.248, Duration: 0.401 seconds
Epoch 10/30, Loss 0.093, Duration: 0.395 seconds
Epoch 11/30, Loss 0.040, Duration: 0.395 seconds
Epoch 12/30, Loss 0.021, Duration: 0.390 seconds
Epoch 13/30, Loss 0.012, Duration: 0.391 seconds
Epoch 14/30, Loss 0.008, Duration: 0.399 seconds
Epoch 15/30, Loss 0.006, Duration: 0.415 seconds
Epoch 16/30, Loss 0.004, Duration: 0.408 seconds
Epoch 17/30, Loss 0.004, Duration: 0.393 seconds
Epoch 18/30, Loss 0.003, Duration: 0.390 seconds
Epoch 19/30, Loss 0.003, Duration: 0.404 seconds
Epoch 20/30, Loss 0.002, Duration: 0.382 seconds
Epoch 21/30, Loss 0.002, Duration: 0.391 seconds
Epoch 22/30, Loss 0.002, Duration: 0.396 seconds
Epoch 23/30, Loss 0.002, Duration: 0.399 seconds
Epoch 24/30, Loss 0.001, Duration: 0.398 seconds
Epoch 25/30, Loss 0.001, Duration: 0.419 seconds
Epoch 26/30, Loss 0.001, Duration: 0.409 seconds
Epoch 27/30, Loss 0.001, Duration: 0.433 seconds
Epoch 28/30, Loss 0.001, Duration: 0.417 seconds
Epoch 29/30, Loss 0.001, Duration: 0.421 seconds
Epoch 30/30, Loss 0.001, Duration: 0.403 seconds

Training finished.
Saving the trained single block model to 'TrainedSingleBlockGPT2_Tokenizer_only'...
Trained custom single block model and transformers tokenizer saved.
'''