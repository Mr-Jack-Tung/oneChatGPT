# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 31 May 2025

import torch
import torch.nn as nn
import time
import os
# Import the custom tokenizer
from character_tokenizer import CharacterTokenizer

# Import the custom model and config
from single_block_gpt2_no_depend_model import SingleBlockGPT2ModelNoDepend, GPT2Config

# Define the path to the trained single block model directory
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'TrainedSingleBlockGPT2_No_depend'
MODEL_STATE_DICT_PATH = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'
# OPTIMIZER_STATE_DICT_PATH = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/optimizer_state_dict.pth' # Path for optimizer state - Not used for loading/saving in staged training

# Main function to run staged training
if __name__ == '__main__':
    # Staged training loop
    num_stages = 2
    epochs_per_stage = 30

    for stage in range(num_stages):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the training data (using the single QA pair from the original script)
        qa_pair = 'Question: Xin chào \nAnswer: Công ty BICweb kính chào quý khách!.'

        # Initialize and train/load the character tokenizer for the current stage (matches train_old.py behavior)
        tokenizer = None # Initialize tokenizer to None for this stage
        if os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
            print(f"Found existing trained model directory: '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'. Loading tokenizer...")
            try:
                # Load tokenizer from the trained model directory
                tokenizer = CharacterTokenizer.from_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)
                print("Character tokenizer loaded successfully.")
            except Exception as e:
                print(f"Error loading character tokenizer for stage {stage + 1}: {e}")
                print("Proceeding with initializing a new tokenizer for this stage.")

        # Initialize and train the character tokenizer (if not loaded)
        if tokenizer is None:
            tokenizer = CharacterTokenizer()
            print("Initializing character tokenizer...")
            tokenizer.train(qa_pair)
            print(f"Character tokenizer trained with vocabulary size: {tokenizer.vocab_size}")

        # Define model configuration (defaulting to small, but using tokenizer's vocab size)
        config = GPT2Config(model_type="small", vocab_size=tokenizer.vocab_size)

        # Encode the training data using the character tokenizer
        input_ids = torch.tensor(tokenizer.encode(qa_pair), dtype=torch.long).unsqueeze(0).to(device)
        print(f"\nTraining data: {qa_pair}")
        print(f"Encoded input_ids shape: {input_ids.shape}")


        print(f"\n--- Starting Training Stage {stage + 1}/{num_stages} ---")

        # Initialize model and optimizer for the current stage
        single_block_model = SingleBlockGPT2ModelNoDepend(config)
        optimizer = torch.optim.AdamW(single_block_model.parameters(), lr=3e-4)

        print("New model and optimizer initialized for this stage.")

        # Load the model state if it exists
        if os.path.exists(MODEL_STATE_DICT_PATH): # Check only for model state
            print(f"Found existing trained model state. Loading state...")
            try:
                single_block_model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH))
                print("Trained model state loaded successfully.")
            except Exception as e:
                print(f"Error loading trained model state for stage {stage + 1}: {e}")
                print("Proceeding with training from newly initialized state.")

        # Print model info and param count at the start of each stage
        print(single_block_model)
        print(f"Number of trainable params: {sum(p.numel() for p in single_block_model.parameters() if p.requires_grad):,d}")


        # Set model to training mode and device
        single_block_model.to(device)
        single_block_model.train()

        # Training loop for the current stage
        print(f"Starting training for {epochs_per_stage} epochs in Stage {stage + 1}...")

        for epoch in range(epochs_per_stage):
            start_time = time.time()

            # Forward pass
            outputs = single_block_model(input_ids=input_ids, labels=input_ids)
            loss = outputs[0]

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_time = time.time()
            epoch_duration = end_time - start_time

            print(f"Stage {stage + 1}, Epoch {epoch+1}/{epochs_per_stage}, Loss {loss.item():.4f}, Duration: {epoch_duration:.3f} seconds")

        print(f"--- Training Stage {stage + 1} Finished ---")

        # Save the trained single block model after each stage
        print(f"Saving the trained single block model after Stage {stage + 1}...")
        try:
            # Create the directory if it doesn't exist
            if not os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
                os.makedirs(TRAINED_SINGLE_BLOCK_MODEL_PATH)

            # Save the model's state dictionary
            torch.save(single_block_model.state_dict(), MODEL_STATE_DICT_PATH)

            # Save the character tokenizer (only once, but saving here ensures it's always with the latest model)
            tokenizer.save_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)

            print(f"Trained single block model and character tokenizer saved after Stage {stage + 1}.")
        except Exception as e:
            print(f"Error saving trained model or tokenizer after Stage {stage + 1}: {e}")

        # Clean up model, optimizer, and tokenizer explicitly at the end of each stage
        del single_block_model
        del optimizer
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    print("\nAll training stages finished.")

    # # Clean up tokenizer outside the loop (commented out based on last provided file content)
    # del tokenizer

'''
% uv run Single_Block_GPT2_No_depend/train_single_block_gpt2_no_depend.py    
Initializing character tokenizer...
Character tokenizer trained with vocabulary size: 34

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 1/2 ---
New model and optimizer initialized for this stage.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 1020)
  (wpe): Embedding(1024, 1020)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((1020,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=1020, out_features=3060, bias=True)
      (c_proj): Linear(in_features=1020, out_features=1020, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((1020,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=1020, out_features=4080, bias=True)
      (c_proj): Linear(in_features=4080, out_features=1020, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((1020,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=1020, out_features=34, bias=False)
)
Number of trainable params: 13,613,940
Starting training for 30 epochs in Stage 1...
Stage 1, Epoch 1/30, Loss 3.4918, Duration: 0.180 seconds
Stage 1, Epoch 2/30, Loss 2.5761, Duration: 0.101 seconds
Stage 1, Epoch 3/30, Loss 1.7497, Duration: 0.092 seconds
Stage 1, Epoch 4/30, Loss 1.1468, Duration: 0.089 seconds
Stage 1, Epoch 5/30, Loss 0.6994, Duration: 0.091 seconds
Stage 1, Epoch 6/30, Loss 0.3779, Duration: 0.090 seconds
Stage 1, Epoch 7/30, Loss 0.2043, Duration: 0.094 seconds
Stage 1, Epoch 8/30, Loss 0.1047, Duration: 0.089 seconds
Stage 1, Epoch 9/30, Loss 0.0568, Duration: 0.087 seconds
Stage 1, Epoch 10/30, Loss 0.0298, Duration: 0.090 seconds
Stage 1, Epoch 11/30, Loss 0.0184, Duration: 0.089 seconds
Stage 1, Epoch 12/30, Loss 0.0112, Duration: 0.097 seconds
Stage 1, Epoch 13/30, Loss 0.0072, Duration: 0.101 seconds
Stage 1, Epoch 14/30, Loss 0.0049, Duration: 0.098 seconds
Stage 1, Epoch 15/30, Loss 0.0035, Duration: 0.089 seconds
Stage 1, Epoch 16/30, Loss 0.0029, Duration: 0.089 seconds
Stage 1, Epoch 17/30, Loss 0.0022, Duration: 0.091 seconds
Stage 1, Epoch 18/30, Loss 0.0018, Duration: 0.090 seconds
Stage 1, Epoch 19/30, Loss 0.0015, Duration: 0.092 seconds
Stage 1, Epoch 20/30, Loss 0.0011, Duration: 0.095 seconds
Stage 1, Epoch 21/30, Loss 0.0010, Duration: 0.092 seconds
Stage 1, Epoch 22/30, Loss 0.0009, Duration: 0.100 seconds
Stage 1, Epoch 23/30, Loss 0.0007, Duration: 0.092 seconds
Stage 1, Epoch 24/30, Loss 0.0007, Duration: 0.100 seconds
Stage 1, Epoch 25/30, Loss 0.0007, Duration: 0.091 seconds
Stage 1, Epoch 26/30, Loss 0.0006, Duration: 0.090 seconds
Stage 1, Epoch 27/30, Loss 0.0005, Duration: 0.091 seconds
Stage 1, Epoch 28/30, Loss 0.0005, Duration: 0.090 seconds
Stage 1, Epoch 29/30, Loss 0.0004, Duration: 0.095 seconds
Stage 1, Epoch 30/30, Loss 0.0004, Duration: 0.091 seconds
--- Training Stage 1 Finished ---
Saving the trained single block model after Stage 1...
Trained single block model and character tokenizer saved after Stage 1.
Found existing trained model directory: 'TrainedSingleBlockGPT2_No_depend'. Loading tokenizer...
Character tokenizer loaded successfully.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 2/2 ---
New model and optimizer initialized for this stage.
Found existing trained model state. Loading state...
Trained model state loaded successfully.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 1020)
  (wpe): Embedding(1024, 1020)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((1020,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=1020, out_features=3060, bias=True)
      (c_proj): Linear(in_features=1020, out_features=1020, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((1020,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=1020, out_features=4080, bias=True)
      (c_proj): Linear(in_features=4080, out_features=1020, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((1020,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=1020, out_features=34, bias=False)
)
Number of trainable params: 13,613,940
Starting training for 30 epochs in Stage 2...
Stage 2, Epoch 1/30, Loss 6.9753, Duration: 0.108 seconds
Stage 2, Epoch 2/30, Loss 5.1788, Duration: 0.106 seconds
Stage 2, Epoch 3/30, Loss 3.6137, Duration: 0.114 seconds
Stage 2, Epoch 4/30, Loss 2.3173, Duration: 0.104 seconds
Stage 2, Epoch 5/30, Loss 1.3812, Duration: 0.090 seconds
Stage 2, Epoch 6/30, Loss 0.7986, Duration: 0.093 seconds
Stage 2, Epoch 7/30, Loss 0.4756, Duration: 0.091 seconds
Stage 2, Epoch 8/30, Loss 0.3230, Duration: 0.093 seconds
Stage 2, Epoch 9/30, Loss 0.2311, Duration: 0.093 seconds
Stage 2, Epoch 10/30, Loss 0.1735, Duration: 0.092 seconds
Stage 2, Epoch 11/30, Loss 0.1349, Duration: 0.099 seconds
Stage 2, Epoch 12/30, Loss 0.1131, Duration: 0.110 seconds
Stage 2, Epoch 13/30, Loss 0.1012, Duration: 0.098 seconds
Stage 2, Epoch 14/30, Loss 0.0829, Duration: 0.091 seconds
Stage 2, Epoch 15/30, Loss 0.0706, Duration: 0.089 seconds
Stage 2, Epoch 16/30, Loss 0.0438, Duration: 0.095 seconds
Stage 2, Epoch 17/30, Loss 0.0297, Duration: 0.095 seconds
Stage 2, Epoch 18/30, Loss 0.0230, Duration: 0.092 seconds
Stage 2, Epoch 19/30, Loss 0.0156, Duration: 0.092 seconds
Stage 2, Epoch 20/30, Loss 0.0139, Duration: 0.091 seconds
Stage 2, Epoch 21/30, Loss 0.0125, Duration: 0.090 seconds
Stage 2, Epoch 22/30, Loss 0.0078, Duration: 0.088 seconds
Stage 2, Epoch 23/30, Loss 0.0078, Duration: 0.089 seconds
Stage 2, Epoch 24/30, Loss 0.0050, Duration: 0.107 seconds
Stage 2, Epoch 25/30, Loss 0.0050, Duration: 0.089 seconds
Stage 2, Epoch 26/30, Loss 0.0035, Duration: 0.093 seconds
Stage 2, Epoch 27/30, Loss 0.0035, Duration: 0.092 seconds
Stage 2, Epoch 28/30, Loss 0.0031, Duration: 0.093 seconds
Stage 2, Epoch 29/30, Loss 0.0028, Duration: 0.107 seconds
Stage 2, Epoch 30/30, Loss 0.0029, Duration: 0.096 seconds
--- Training Stage 2 Finished ---
Saving the trained single block model after Stage 2...
Trained single block model and character tokenizer saved after Stage 2.
'''