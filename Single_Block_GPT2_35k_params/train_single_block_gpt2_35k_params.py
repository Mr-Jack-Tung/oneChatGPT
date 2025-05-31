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
from single_block_gpt2_35k_params_model import SingleBlockGPT2ModelNoDepend, GPT2Config

# Define the path to the trained single block model directory
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'TrainedSingleBlockGPT2_35k_params'
MODEL_STATE_DICT_PATH = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'
# OPTIMIZER_STATE_DICT_PATH = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/optimizer_state_dict.pth' # Path for optimizer state - Not used for loading/saving in staged training

# Main function to run staged training
if __name__ == '__main__':
    # Staged training loop
    num_stages = 10
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
% uv run Single_Block_GPT2_35k_params/train_single_block_gpt2_35k_params.py
Initializing character tokenizer...
Character tokenizer trained with vocabulary size: 34

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 1/10 ---
New model and optimizer initialized for this stage.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 48)
  (wpe): Embedding(64, 48)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=48, out_features=144, bias=True)
      (c_proj): Linear(in_features=48, out_features=48, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=48, out_features=192, bias=True)
      (c_proj): Linear(in_features=192, out_features=48, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=48, out_features=34, bias=False)
)
Number of trainable params: 34,704
Starting training for 30 epochs in Stage 1...
Stage 1, Epoch 1/30, Loss 3.5870, Duration: 0.059 seconds
Stage 1, Epoch 2/30, Loss 3.6407, Duration: 0.003 seconds
Stage 1, Epoch 3/30, Loss 3.5466, Duration: 0.004 seconds
Stage 1, Epoch 4/30, Loss 3.5883, Duration: 0.005 seconds
Stage 1, Epoch 5/30, Loss 3.5564, Duration: 0.004 seconds
Stage 1, Epoch 6/30, Loss 3.5303, Duration: 0.004 seconds
Stage 1, Epoch 7/30, Loss 3.4909, Duration: 0.005 seconds
Stage 1, Epoch 8/30, Loss 3.4933, Duration: 0.005 seconds
Stage 1, Epoch 9/30, Loss 3.4864, Duration: 0.004 seconds
Stage 1, Epoch 10/30, Loss 3.4394, Duration: 0.005 seconds
Stage 1, Epoch 11/30, Loss 3.4553, Duration: 0.005 seconds
Stage 1, Epoch 12/30, Loss 3.4388, Duration: 0.004 seconds
Stage 1, Epoch 13/30, Loss 3.4133, Duration: 0.004 seconds
Stage 1, Epoch 14/30, Loss 3.3464, Duration: 0.003 seconds
Stage 1, Epoch 15/30, Loss 3.3379, Duration: 0.005 seconds
Stage 1, Epoch 16/30, Loss 3.3691, Duration: 0.004 seconds
Stage 1, Epoch 17/30, Loss 3.3183, Duration: 0.004 seconds
Stage 1, Epoch 18/30, Loss 3.3044, Duration: 0.004 seconds
Stage 1, Epoch 19/30, Loss 3.3213, Duration: 0.006 seconds
Stage 1, Epoch 20/30, Loss 3.2635, Duration: 0.004 seconds
Stage 1, Epoch 21/30, Loss 3.2781, Duration: 0.005 seconds
Stage 1, Epoch 22/30, Loss 3.3000, Duration: 0.005 seconds
Stage 1, Epoch 23/30, Loss 3.2651, Duration: 0.004 seconds
Stage 1, Epoch 24/30, Loss 3.2450, Duration: 0.004 seconds
Stage 1, Epoch 25/30, Loss 3.2034, Duration: 0.004 seconds
Stage 1, Epoch 26/30, Loss 3.2100, Duration: 0.005 seconds
Stage 1, Epoch 27/30, Loss 3.1841, Duration: 0.003 seconds
Stage 1, Epoch 28/30, Loss 3.1348, Duration: 0.005 seconds
Stage 1, Epoch 29/30, Loss 3.1647, Duration: 0.005 seconds
Stage 1, Epoch 30/30, Loss 3.1477, Duration: 0.006 seconds
--- Training Stage 1 Finished ---
Saving the trained single block model after Stage 1...
Trained single block model and character tokenizer saved after Stage 1.
Found existing trained model directory: 'TrainedSingleBlockGPT2_35k_params'. Loading tokenizer...
Character tokenizer loaded successfully.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 2/10 ---
New model and optimizer initialized for this stage.
Found existing trained model state. Loading state...
Trained model state loaded successfully.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 48)
  (wpe): Embedding(64, 48)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=48, out_features=144, bias=True)
      (c_proj): Linear(in_features=48, out_features=48, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=48, out_features=192, bias=True)
      (c_proj): Linear(in_features=192, out_features=48, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=48, out_features=34, bias=False)
)
Number of trainable params: 34,704
Starting training for 30 epochs in Stage 2...
Stage 2, Epoch 1/30, Loss 3.4263, Duration: 0.005 seconds
Stage 2, Epoch 2/30, Loss 3.4429, Duration: 0.006 seconds
Stage 2, Epoch 3/30, Loss 3.4106, Duration: 0.005 seconds
Stage 2, Epoch 4/30, Loss 3.3906, Duration: 0.004 seconds
Stage 2, Epoch 5/30, Loss 3.3834, Duration: 0.004 seconds
Stage 2, Epoch 6/30, Loss 3.3423, Duration: 0.007 seconds
Stage 2, Epoch 7/30, Loss 3.3778, Duration: 0.005 seconds
Stage 2, Epoch 8/30, Loss 3.3455, Duration: 0.005 seconds
Stage 2, Epoch 9/30, Loss 3.2763, Duration: 0.007 seconds
Stage 2, Epoch 10/30, Loss 3.3073, Duration: 0.004 seconds
Stage 2, Epoch 11/30, Loss 3.3248, Duration: 0.004 seconds
Stage 2, Epoch 12/30, Loss 3.2916, Duration: 0.004 seconds
Stage 2, Epoch 13/30, Loss 3.2550, Duration: 0.005 seconds
Stage 2, Epoch 14/30, Loss 3.1785, Duration: 0.003 seconds
Stage 2, Epoch 15/30, Loss 3.2018, Duration: 0.004 seconds
Stage 2, Epoch 16/30, Loss 3.2093, Duration: 0.009 seconds
Stage 2, Epoch 17/30, Loss 3.1662, Duration: 0.004 seconds
Stage 2, Epoch 18/30, Loss 3.2024, Duration: 0.004 seconds
Stage 2, Epoch 19/30, Loss 3.1412, Duration: 0.005 seconds
Stage 2, Epoch 20/30, Loss 3.1134, Duration: 0.006 seconds
Stage 2, Epoch 21/30, Loss 3.1014, Duration: 0.004 seconds
Stage 2, Epoch 22/30, Loss 3.1229, Duration: 0.004 seconds
Stage 2, Epoch 23/30, Loss 3.0954, Duration: 0.005 seconds
Stage 2, Epoch 24/30, Loss 3.0683, Duration: 0.003 seconds
Stage 2, Epoch 25/30, Loss 3.0576, Duration: 0.004 seconds
Stage 2, Epoch 26/30, Loss 3.0300, Duration: 0.004 seconds
Stage 2, Epoch 27/30, Loss 2.9998, Duration: 0.005 seconds
Stage 2, Epoch 28/30, Loss 3.0091, Duration: 0.004 seconds
Stage 2, Epoch 29/30, Loss 2.9790, Duration: 0.004 seconds
Stage 2, Epoch 30/30, Loss 2.9722, Duration: 0.005 seconds
--- Training Stage 2 Finished ---
Saving the trained single block model after Stage 2...
Trained single block model and character tokenizer saved after Stage 2.
Found existing trained model directory: 'TrainedSingleBlockGPT2_35k_params'. Loading tokenizer...
Character tokenizer loaded successfully.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 3/10 ---
New model and optimizer initialized for this stage.
Found existing trained model state. Loading state...
Trained model state loaded successfully.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 48)
  (wpe): Embedding(64, 48)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=48, out_features=144, bias=True)
      (c_proj): Linear(in_features=48, out_features=48, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=48, out_features=192, bias=True)
      (c_proj): Linear(in_features=192, out_features=48, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=48, out_features=34, bias=False)
)
Number of trainable params: 34,704
Starting training for 30 epochs in Stage 3...
Stage 3, Epoch 1/30, Loss 2.9402, Duration: 0.005 seconds
Stage 3, Epoch 2/30, Loss 2.9434, Duration: 0.005 seconds
Stage 3, Epoch 3/30, Loss 2.8936, Duration: 0.008 seconds
Stage 3, Epoch 4/30, Loss 2.8991, Duration: 0.004 seconds
Stage 3, Epoch 5/30, Loss 2.9396, Duration: 0.005 seconds
Stage 3, Epoch 6/30, Loss 2.8931, Duration: 0.007 seconds
Stage 3, Epoch 7/30, Loss 2.8676, Duration: 0.004 seconds
Stage 3, Epoch 8/30, Loss 2.8239, Duration: 0.005 seconds
Stage 3, Epoch 9/30, Loss 2.8121, Duration: 0.008 seconds
Stage 3, Epoch 10/30, Loss 2.8106, Duration: 0.004 seconds
Stage 3, Epoch 11/30, Loss 2.7709, Duration: 0.015 seconds
Stage 3, Epoch 12/30, Loss 2.7656, Duration: 0.005 seconds
Stage 3, Epoch 13/30, Loss 2.7684, Duration: 0.005 seconds
Stage 3, Epoch 14/30, Loss 2.7497, Duration: 0.006 seconds
Stage 3, Epoch 15/30, Loss 2.7177, Duration: 0.005 seconds
Stage 3, Epoch 16/30, Loss 2.7217, Duration: 0.004 seconds
Stage 3, Epoch 17/30, Loss 2.7111, Duration: 0.004 seconds
Stage 3, Epoch 18/30, Loss 2.6588, Duration: 0.005 seconds
Stage 3, Epoch 19/30, Loss 2.6769, Duration: 0.003 seconds
Stage 3, Epoch 20/30, Loss 2.6169, Duration: 0.004 seconds
Stage 3, Epoch 21/30, Loss 2.6455, Duration: 0.005 seconds
Stage 3, Epoch 22/30, Loss 2.6436, Duration: 0.008 seconds
Stage 3, Epoch 23/30, Loss 2.5915, Duration: 0.004 seconds
Stage 3, Epoch 24/30, Loss 2.5733, Duration: 0.006 seconds
Stage 3, Epoch 25/30, Loss 2.5709, Duration: 0.004 seconds
Stage 3, Epoch 26/30, Loss 2.5346, Duration: 0.004 seconds
Stage 3, Epoch 27/30, Loss 2.5338, Duration: 0.003 seconds
Stage 3, Epoch 28/30, Loss 2.5329, Duration: 0.004 seconds
Stage 3, Epoch 29/30, Loss 2.5052, Duration: 0.004 seconds
Stage 3, Epoch 30/30, Loss 2.4523, Duration: 0.004 seconds
--- Training Stage 3 Finished ---
Saving the trained single block model after Stage 3...
Trained single block model and character tokenizer saved after Stage 3.
Found existing trained model directory: 'TrainedSingleBlockGPT2_35k_params'. Loading tokenizer...
Character tokenizer loaded successfully.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 4/10 ---
New model and optimizer initialized for this stage.
Found existing trained model state. Loading state...
Trained model state loaded successfully.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 48)
  (wpe): Embedding(64, 48)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=48, out_features=144, bias=True)
      (c_proj): Linear(in_features=48, out_features=48, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=48, out_features=192, bias=True)
      (c_proj): Linear(in_features=192, out_features=48, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=48, out_features=34, bias=False)
)
Number of trainable params: 34,704
Starting training for 30 epochs in Stage 4...
Stage 4, Epoch 1/30, Loss 2.4809, Duration: 0.004 seconds
Stage 4, Epoch 2/30, Loss 2.4538, Duration: 0.004 seconds
Stage 4, Epoch 3/30, Loss 2.4158, Duration: 0.005 seconds
Stage 4, Epoch 4/30, Loss 2.4470, Duration: 0.004 seconds
Stage 4, Epoch 5/30, Loss 2.3604, Duration: 0.004 seconds
Stage 4, Epoch 6/30, Loss 2.3470, Duration: 0.004 seconds
Stage 4, Epoch 7/30, Loss 2.3875, Duration: 0.005 seconds
Stage 4, Epoch 8/30, Loss 2.3966, Duration: 0.003 seconds
Stage 4, Epoch 9/30, Loss 2.3160, Duration: 0.004 seconds
Stage 4, Epoch 10/30, Loss 2.3266, Duration: 0.004 seconds
Stage 4, Epoch 11/30, Loss 2.2908, Duration: 0.005 seconds
Stage 4, Epoch 12/30, Loss 2.3023, Duration: 0.004 seconds
Stage 4, Epoch 13/30, Loss 2.2902, Duration: 0.004 seconds
Stage 4, Epoch 14/30, Loss 2.2664, Duration: 0.004 seconds
Stage 4, Epoch 15/30, Loss 2.2818, Duration: 0.005 seconds
Stage 4, Epoch 16/30, Loss 2.2593, Duration: 0.004 seconds
Stage 4, Epoch 17/30, Loss 2.1900, Duration: 0.004 seconds
Stage 4, Epoch 18/30, Loss 2.2251, Duration: 0.004 seconds
Stage 4, Epoch 19/30, Loss 2.1644, Duration: 0.004 seconds
Stage 4, Epoch 20/30, Loss 2.1435, Duration: 0.003 seconds
Stage 4, Epoch 21/30, Loss 2.2117, Duration: 0.004 seconds
Stage 4, Epoch 22/30, Loss 2.1197, Duration: 0.005 seconds
Stage 4, Epoch 23/30, Loss 2.1285, Duration: 0.089 seconds
Stage 4, Epoch 24/30, Loss 2.0861, Duration: 0.005 seconds
Stage 4, Epoch 25/30, Loss 2.1059, Duration: 0.029 seconds
Stage 4, Epoch 26/30, Loss 2.1110, Duration: 0.004 seconds
Stage 4, Epoch 27/30, Loss 2.0735, Duration: 0.004 seconds
Stage 4, Epoch 28/30, Loss 2.0863, Duration: 0.005 seconds
Stage 4, Epoch 29/30, Loss 2.0496, Duration: 0.003 seconds
Stage 4, Epoch 30/30, Loss 2.0622, Duration: 0.003 seconds
--- Training Stage 4 Finished ---
Saving the trained single block model after Stage 4...
Trained single block model and character tokenizer saved after Stage 4.
Found existing trained model directory: 'TrainedSingleBlockGPT2_35k_params'. Loading tokenizer...
Character tokenizer loaded successfully.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 5/10 ---
New model and optimizer initialized for this stage.
Found existing trained model state. Loading state...
Trained model state loaded successfully.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 48)
  (wpe): Embedding(64, 48)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=48, out_features=144, bias=True)
      (c_proj): Linear(in_features=48, out_features=48, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=48, out_features=192, bias=True)
      (c_proj): Linear(in_features=192, out_features=48, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=48, out_features=34, bias=False)
)
Number of trainable params: 34,704
Starting training for 30 epochs in Stage 5...
Stage 5, Epoch 1/30, Loss 1.9834, Duration: 0.005 seconds
Stage 5, Epoch 2/30, Loss 1.9875, Duration: 0.006 seconds
Stage 5, Epoch 3/30, Loss 1.9496, Duration: 0.004 seconds
Stage 5, Epoch 4/30, Loss 1.9362, Duration: 0.004 seconds
Stage 5, Epoch 5/30, Loss 1.9335, Duration: 0.004 seconds
Stage 5, Epoch 6/30, Loss 1.9167, Duration: 0.007 seconds
Stage 5, Epoch 7/30, Loss 1.8901, Duration: 0.003 seconds
Stage 5, Epoch 8/30, Loss 1.8850, Duration: 0.004 seconds
Stage 5, Epoch 9/30, Loss 1.8527, Duration: 0.004 seconds
Stage 5, Epoch 10/30, Loss 1.8525, Duration: 0.005 seconds
Stage 5, Epoch 11/30, Loss 1.8527, Duration: 0.004 seconds
Stage 5, Epoch 12/30, Loss 1.8275, Duration: 0.004 seconds
Stage 5, Epoch 13/30, Loss 1.8657, Duration: 0.004 seconds
Stage 5, Epoch 14/30, Loss 1.7786, Duration: 0.005 seconds
Stage 5, Epoch 15/30, Loss 1.7401, Duration: 0.004 seconds
Stage 5, Epoch 16/30, Loss 1.7475, Duration: 0.004 seconds
Stage 5, Epoch 17/30, Loss 1.7454, Duration: 0.004 seconds
Stage 5, Epoch 18/30, Loss 1.7650, Duration: 0.005 seconds
Stage 5, Epoch 19/30, Loss 1.6995, Duration: 0.003 seconds
Stage 5, Epoch 20/30, Loss 1.6826, Duration: 0.004 seconds
Stage 5, Epoch 21/30, Loss 1.7060, Duration: 0.004 seconds
Stage 5, Epoch 22/30, Loss 1.6754, Duration: 0.005 seconds
Stage 5, Epoch 23/30, Loss 1.6777, Duration: 0.004 seconds
Stage 5, Epoch 24/30, Loss 1.5988, Duration: 0.004 seconds
Stage 5, Epoch 25/30, Loss 1.6133, Duration: 0.005 seconds
Stage 5, Epoch 26/30, Loss 1.6028, Duration: 0.006 seconds
Stage 5, Epoch 27/30, Loss 1.6197, Duration: 0.004 seconds
Stage 5, Epoch 28/30, Loss 1.5730, Duration: 0.003 seconds
Stage 5, Epoch 29/30, Loss 1.5830, Duration: 0.003 seconds
Stage 5, Epoch 30/30, Loss 1.5048, Duration: 0.005 seconds
--- Training Stage 5 Finished ---
Saving the trained single block model after Stage 5...
Trained single block model and character tokenizer saved after Stage 5.
Found existing trained model directory: 'TrainedSingleBlockGPT2_35k_params'. Loading tokenizer...
Character tokenizer loaded successfully.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 6/10 ---
New model and optimizer initialized for this stage.
Found existing trained model state. Loading state...
Trained model state loaded successfully.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 48)
  (wpe): Embedding(64, 48)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=48, out_features=144, bias=True)
      (c_proj): Linear(in_features=48, out_features=48, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=48, out_features=192, bias=True)
      (c_proj): Linear(in_features=192, out_features=48, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=48, out_features=34, bias=False)
)
Number of trainable params: 34,704
Starting training for 30 epochs in Stage 6...
Stage 6, Epoch 1/30, Loss 1.5333, Duration: 0.005 seconds
Stage 6, Epoch 2/30, Loss 1.5082, Duration: 0.005 seconds
Stage 6, Epoch 3/30, Loss 1.5457, Duration: 0.004 seconds
Stage 6, Epoch 4/30, Loss 1.4917, Duration: 0.005 seconds
Stage 6, Epoch 5/30, Loss 1.4978, Duration: 0.004 seconds
Stage 6, Epoch 6/30, Loss 1.4993, Duration: 0.004 seconds
Stage 6, Epoch 7/30, Loss 1.4520, Duration: 0.004 seconds
Stage 6, Epoch 8/30, Loss 1.4721, Duration: 0.005 seconds
Stage 6, Epoch 9/30, Loss 1.4290, Duration: 0.003 seconds
Stage 6, Epoch 10/30, Loss 1.4290, Duration: 0.004 seconds
Stage 6, Epoch 11/30, Loss 1.4129, Duration: 0.004 seconds
Stage 6, Epoch 12/30, Loss 1.3699, Duration: 0.006 seconds
Stage 6, Epoch 13/30, Loss 1.3676, Duration: 0.004 seconds
Stage 6, Epoch 14/30, Loss 1.3461, Duration: 0.004 seconds
Stage 6, Epoch 15/30, Loss 1.3409, Duration: 0.005 seconds
Stage 6, Epoch 16/30, Loss 1.3166, Duration: 0.005 seconds
Stage 6, Epoch 17/30, Loss 1.3279, Duration: 0.004 seconds
Stage 6, Epoch 18/30, Loss 1.2769, Duration: 0.003 seconds
Stage 6, Epoch 19/30, Loss 1.3097, Duration: 0.003 seconds
Stage 6, Epoch 20/30, Loss 1.3028, Duration: 0.005 seconds
Stage 6, Epoch 21/30, Loss 1.2626, Duration: 0.003 seconds
Stage 6, Epoch 22/30, Loss 1.2675, Duration: 0.004 seconds
Stage 6, Epoch 23/30, Loss 1.2478, Duration: 0.004 seconds
Stage 6, Epoch 24/30, Loss 1.2204, Duration: 0.006 seconds
Stage 6, Epoch 25/30, Loss 1.2556, Duration: 0.004 seconds
Stage 6, Epoch 26/30, Loss 1.2199, Duration: 0.004 seconds
Stage 6, Epoch 27/30, Loss 1.1706, Duration: 0.006 seconds
Stage 6, Epoch 28/30, Loss 1.1933, Duration: 0.004 seconds
Stage 6, Epoch 29/30, Loss 1.1613, Duration: 0.004 seconds
Stage 6, Epoch 30/30, Loss 1.1178, Duration: 0.004 seconds
--- Training Stage 6 Finished ---
Saving the trained single block model after Stage 6...
Trained single block model and character tokenizer saved after Stage 6.
Found existing trained model directory: 'TrainedSingleBlockGPT2_35k_params'. Loading tokenizer...
Character tokenizer loaded successfully.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 7/10 ---
New model and optimizer initialized for this stage.
Found existing trained model state. Loading state...
Trained model state loaded successfully.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 48)
  (wpe): Embedding(64, 48)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=48, out_features=144, bias=True)
      (c_proj): Linear(in_features=48, out_features=48, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=48, out_features=192, bias=True)
      (c_proj): Linear(in_features=192, out_features=48, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=48, out_features=34, bias=False)
)
Number of trainable params: 34,704
Starting training for 30 epochs in Stage 7...
Stage 7, Epoch 1/30, Loss 1.1608, Duration: 0.005 seconds
Stage 7, Epoch 2/30, Loss 1.1335, Duration: 0.006 seconds
Stage 7, Epoch 3/30, Loss 1.0914, Duration: 0.004 seconds
Stage 7, Epoch 4/30, Loss 1.0919, Duration: 0.004 seconds
Stage 7, Epoch 5/30, Loss 1.0937, Duration: 0.006 seconds
Stage 7, Epoch 6/30, Loss 1.0659, Duration: 0.004 seconds
Stage 7, Epoch 7/30, Loss 1.0420, Duration: 0.004 seconds
Stage 7, Epoch 8/30, Loss 1.0644, Duration: 0.003 seconds
Stage 7, Epoch 9/30, Loss 1.0589, Duration: 0.004 seconds
Stage 7, Epoch 10/30, Loss 1.0127, Duration: 0.004 seconds
Stage 7, Epoch 11/30, Loss 1.0300, Duration: 0.004 seconds
Stage 7, Epoch 12/30, Loss 0.9800, Duration: 0.004 seconds
Stage 7, Epoch 13/30, Loss 0.9822, Duration: 0.005 seconds
Stage 7, Epoch 14/30, Loss 0.9906, Duration: 0.004 seconds
Stage 7, Epoch 15/30, Loss 0.9894, Duration: 0.004 seconds
Stage 7, Epoch 16/30, Loss 0.9910, Duration: 0.005 seconds
Stage 7, Epoch 17/30, Loss 0.9296, Duration: 0.005 seconds
Stage 7, Epoch 18/30, Loss 0.9881, Duration: 0.004 seconds
Stage 7, Epoch 19/30, Loss 0.9171, Duration: 0.004 seconds
Stage 7, Epoch 20/30, Loss 0.9333, Duration: 0.003 seconds
Stage 7, Epoch 21/30, Loss 0.9014, Duration: 0.005 seconds
Stage 7, Epoch 22/30, Loss 0.9168, Duration: 0.004 seconds
Stage 7, Epoch 23/30, Loss 0.9178, Duration: 0.004 seconds
Stage 7, Epoch 24/30, Loss 0.8923, Duration: 0.004 seconds
Stage 7, Epoch 25/30, Loss 0.9099, Duration: 0.005 seconds
Stage 7, Epoch 26/30, Loss 0.8752, Duration: 0.004 seconds
Stage 7, Epoch 27/30, Loss 0.8595, Duration: 0.004 seconds
Stage 7, Epoch 28/30, Loss 0.8623, Duration: 0.004 seconds
Stage 7, Epoch 29/30, Loss 0.8562, Duration: 0.006 seconds
Stage 7, Epoch 30/30, Loss 0.8059, Duration: 0.004 seconds
--- Training Stage 7 Finished ---
Saving the trained single block model after Stage 7...
Trained single block model and character tokenizer saved after Stage 7.
Found existing trained model directory: 'TrainedSingleBlockGPT2_35k_params'. Loading tokenizer...
Character tokenizer loaded successfully.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 8/10 ---
New model and optimizer initialized for this stage.
Found existing trained model state. Loading state...
Trained model state loaded successfully.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 48)
  (wpe): Embedding(64, 48)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=48, out_features=144, bias=True)
      (c_proj): Linear(in_features=48, out_features=48, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=48, out_features=192, bias=True)
      (c_proj): Linear(in_features=192, out_features=48, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=48, out_features=34, bias=False)
)
Number of trainable params: 34,704
Starting training for 30 epochs in Stage 8...
Stage 8, Epoch 1/30, Loss 0.8641, Duration: 0.005 seconds
Stage 8, Epoch 2/30, Loss 0.8210, Duration: 0.006 seconds
Stage 8, Epoch 3/30, Loss 0.8406, Duration: 0.006 seconds
Stage 8, Epoch 4/30, Loss 0.7843, Duration: 0.005 seconds
Stage 8, Epoch 5/30, Loss 0.7944, Duration: 0.005 seconds
Stage 8, Epoch 6/30, Loss 0.8023, Duration: 0.006 seconds
Stage 8, Epoch 7/30, Loss 0.8031, Duration: 0.004 seconds
Stage 8, Epoch 8/30, Loss 0.7609, Duration: 0.004 seconds
Stage 8, Epoch 9/30, Loss 0.7201, Duration: 0.005 seconds
Stage 8, Epoch 10/30, Loss 0.7773, Duration: 0.006 seconds
Stage 8, Epoch 11/30, Loss 0.7278, Duration: 0.006 seconds
Stage 8, Epoch 12/30, Loss 0.7340, Duration: 0.005 seconds
Stage 8, Epoch 13/30, Loss 0.7252, Duration: 0.004 seconds
Stage 8, Epoch 14/30, Loss 0.6954, Duration: 0.004 seconds
Stage 8, Epoch 15/30, Loss 0.7276, Duration: 0.005 seconds
Stage 8, Epoch 16/30, Loss 0.6762, Duration: 0.006 seconds
Stage 8, Epoch 17/30, Loss 0.6762, Duration: 0.004 seconds
Stage 8, Epoch 18/30, Loss 0.6855, Duration: 0.004 seconds
Stage 8, Epoch 19/30, Loss 0.6699, Duration: 0.006 seconds
Stage 8, Epoch 20/30, Loss 0.6514, Duration: 0.003 seconds
Stage 8, Epoch 21/30, Loss 0.6697, Duration: 0.005 seconds
Stage 8, Epoch 22/30, Loss 0.6508, Duration: 0.005 seconds
Stage 8, Epoch 23/30, Loss 0.6761, Duration: 0.005 seconds
Stage 8, Epoch 24/30, Loss 0.6272, Duration: 0.004 seconds
Stage 8, Epoch 25/30, Loss 0.6200, Duration: 0.004 seconds
Stage 8, Epoch 26/30, Loss 0.6120, Duration: 0.006 seconds
Stage 8, Epoch 27/30, Loss 0.5922, Duration: 0.004 seconds
Stage 8, Epoch 28/30, Loss 0.5624, Duration: 0.004 seconds
Stage 8, Epoch 29/30, Loss 0.5861, Duration: 0.003 seconds
Stage 8, Epoch 30/30, Loss 0.5732, Duration: 0.005 seconds
--- Training Stage 8 Finished ---
Saving the trained single block model after Stage 8...
Trained single block model and character tokenizer saved after Stage 8.
Found existing trained model directory: 'TrainedSingleBlockGPT2_35k_params'. Loading tokenizer...
Character tokenizer loaded successfully.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 9/10 ---
New model and optimizer initialized for this stage.
Found existing trained model state. Loading state...
Trained model state loaded successfully.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 48)
  (wpe): Embedding(64, 48)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=48, out_features=144, bias=True)
      (c_proj): Linear(in_features=48, out_features=48, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=48, out_features=192, bias=True)
      (c_proj): Linear(in_features=192, out_features=48, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=48, out_features=34, bias=False)
)
Number of trainable params: 34,704
Starting training for 30 epochs in Stage 9...
Stage 9, Epoch 1/30, Loss 0.5624, Duration: 0.008 seconds
Stage 9, Epoch 2/30, Loss 0.6060, Duration: 0.004 seconds
Stage 9, Epoch 3/30, Loss 0.5582, Duration: 0.004 seconds
Stage 9, Epoch 4/30, Loss 0.5426, Duration: 0.005 seconds
Stage 9, Epoch 5/30, Loss 0.5264, Duration: 0.005 seconds
Stage 9, Epoch 6/30, Loss 0.5439, Duration: 0.004 seconds
Stage 9, Epoch 7/30, Loss 0.5075, Duration: 0.004 seconds
Stage 9, Epoch 8/30, Loss 0.5203, Duration: 0.003 seconds
Stage 9, Epoch 9/30, Loss 0.5391, Duration: 0.005 seconds
Stage 9, Epoch 10/30, Loss 0.5141, Duration: 0.003 seconds
Stage 9, Epoch 11/30, Loss 0.5339, Duration: 0.004 seconds
Stage 9, Epoch 12/30, Loss 0.4847, Duration: 0.004 seconds
Stage 9, Epoch 13/30, Loss 0.5040, Duration: 0.005 seconds
Stage 9, Epoch 14/30, Loss 0.4727, Duration: 0.004 seconds
Stage 9, Epoch 15/30, Loss 0.4586, Duration: 0.005 seconds
Stage 9, Epoch 16/30, Loss 0.4856, Duration: 0.007 seconds
Stage 9, Epoch 17/30, Loss 0.4671, Duration: 0.004 seconds
Stage 9, Epoch 18/30, Loss 0.4480, Duration: 0.006 seconds
Stage 9, Epoch 19/30, Loss 0.4754, Duration: 0.004 seconds
Stage 9, Epoch 20/30, Loss 0.4824, Duration: 0.005 seconds
Stage 9, Epoch 21/30, Loss 0.4724, Duration: 0.003 seconds
Stage 9, Epoch 22/30, Loss 0.4450, Duration: 0.005 seconds
Stage 9, Epoch 23/30, Loss 0.4517, Duration: 0.004 seconds
Stage 9, Epoch 24/30, Loss 0.4561, Duration: 0.006 seconds
Stage 9, Epoch 25/30, Loss 0.4552, Duration: 0.004 seconds
Stage 9, Epoch 26/30, Loss 0.4334, Duration: 0.004 seconds
Stage 9, Epoch 27/30, Loss 0.4319, Duration: 0.005 seconds
Stage 9, Epoch 28/30, Loss 0.4605, Duration: 0.004 seconds
Stage 9, Epoch 29/30, Loss 0.4203, Duration: 0.004 seconds
Stage 9, Epoch 30/30, Loss 0.4230, Duration: 0.004 seconds
--- Training Stage 9 Finished ---
Saving the trained single block model after Stage 9...
Trained single block model and character tokenizer saved after Stage 9.
Found existing trained model directory: 'TrainedSingleBlockGPT2_35k_params'. Loading tokenizer...
Character tokenizer loaded successfully.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 10/10 ---
New model and optimizer initialized for this stage.
Found existing trained model state. Loading state...
Trained model state loaded successfully.
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 48)
  (wpe): Embedding(64, 48)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=48, out_features=144, bias=True)
      (c_proj): Linear(in_features=48, out_features=48, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=48, out_features=192, bias=True)
      (c_proj): Linear(in_features=192, out_features=48, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=48, out_features=34, bias=False)
)
Number of trainable params: 34,704
Starting training for 30 epochs in Stage 10...
Stage 10, Epoch 1/30, Loss 0.3602, Duration: 0.006 seconds
Stage 10, Epoch 2/30, Loss 0.3853, Duration: 0.006 seconds
Stage 10, Epoch 3/30, Loss 0.4056, Duration: 0.004 seconds
Stage 10, Epoch 4/30, Loss 0.3954, Duration: 0.005 seconds
Stage 10, Epoch 5/30, Loss 0.3833, Duration: 0.007 seconds
Stage 10, Epoch 6/30, Loss 0.3667, Duration: 0.004 seconds
Stage 10, Epoch 7/30, Loss 0.3743, Duration: 0.004 seconds
Stage 10, Epoch 8/30, Loss 0.3585, Duration: 0.005 seconds
Stage 10, Epoch 9/30, Loss 0.3803, Duration: 0.003 seconds
Stage 10, Epoch 10/30, Loss 0.3631, Duration: 0.004 seconds
Stage 10, Epoch 11/30, Loss 0.3549, Duration: 0.005 seconds
Stage 10, Epoch 12/30, Loss 0.3759, Duration: 0.006 seconds
Stage 10, Epoch 13/30, Loss 0.3535, Duration: 0.004 seconds
Stage 10, Epoch 14/30, Loss 0.3485, Duration: 0.004 seconds
Stage 10, Epoch 15/30, Loss 0.3262, Duration: 0.006 seconds
Stage 10, Epoch 16/30, Loss 0.3816, Duration: 0.004 seconds
Stage 10, Epoch 17/30, Loss 0.3150, Duration: 0.004 seconds
Stage 10, Epoch 18/30, Loss 0.3489, Duration: 0.003 seconds
Stage 10, Epoch 19/30, Loss 0.3265, Duration: 0.004 seconds
Stage 10, Epoch 20/30, Loss 0.3314, Duration: 0.004 seconds
Stage 10, Epoch 21/30, Loss 0.2897, Duration: 0.004 seconds
Stage 10, Epoch 22/30, Loss 0.3050, Duration: 0.004 seconds
Stage 10, Epoch 23/30, Loss 0.3222, Duration: 0.006 seconds
Stage 10, Epoch 24/30, Loss 0.2992, Duration: 0.004 seconds
Stage 10, Epoch 25/30, Loss 0.3056, Duration: 0.004 seconds
Stage 10, Epoch 26/30, Loss 0.2937, Duration: 0.005 seconds
Stage 10, Epoch 27/30, Loss 0.2969, Duration: 0.006 seconds
Stage 10, Epoch 28/30, Loss 0.3128, Duration: 0.004 seconds
Stage 10, Epoch 29/30, Loss 0.2871, Duration: 0.003 seconds
Stage 10, Epoch 30/30, Loss 0.2794, Duration: 0.004 seconds
--- Training Stage 10 Finished ---
Saving the trained single block model after Stage 10...
Trained single block model and character tokenizer saved after Stage 10.

All training stages finished.
'''