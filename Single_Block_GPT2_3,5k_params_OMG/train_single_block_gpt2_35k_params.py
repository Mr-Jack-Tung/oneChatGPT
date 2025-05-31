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
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'TrainedSingleBlockGPT2_3,5k_params'
MODEL_STATE_DICT_PATH = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'

# Main function to run staged training
if __name__ == '__main__':
    # Staged training loop
    num_stages = 50
    epochs_per_stage = 50

    for stage in range(num_stages):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the training data (using the single QA pair from the original script)
        qa_pair = 'Question: Xin chào \nAnswer: Công ty BICweb kính chào quý khách!.'

        # Initialize and train/load the character tokenizer for the current stage (matches train_old.py behavior)
        tokenizer = None # Initialize tokenizer to None for this stage
        if os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
            # print(f"Found existing trained model directory: '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'. Loading tokenizer...")
            try:
                # Load tokenizer from the trained model directory
                tokenizer = CharacterTokenizer.from_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)
                # print("Character tokenizer loaded successfully.")
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

        # print("New model and optimizer initialized for this stage.")

        # Load the model state if it exists
        if os.path.exists(MODEL_STATE_DICT_PATH): # Check only for model state
            # print(f"Found existing trained model state. Loading state...")
            try:
                single_block_model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH))
                # print("Trained model state loaded successfully.")
            except Exception as e:
                print(f"Error loading trained model state for stage {stage + 1}: {e}")
                print("Proceeding with training from newly initialized state.")

        # Print model info and param count at the start of each stage
        if stage == 0:
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
            if epoch % 10 == 0 or epoch == epochs_per_stage - 1:
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
 % uv run Single_Block_GPT2_3,5k_params_OMG/train_single_block_gpt2_35k_params.py
Initializing character tokenizer...
Character tokenizer trained with vocabulary size: 34

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 1/50 ---
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 12)
  (wpe): Embedding(64, 12)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((12,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=12, out_features=36, bias=True)
      (c_proj): Linear(in_features=12, out_features=12, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((12,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=12, out_features=48, bias=True)
      (c_proj): Linear(in_features=48, out_features=12, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((12,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=12, out_features=34, bias=False)
)
Number of trainable params: 3,492
Starting training for 50 epochs in Stage 1...
Stage 1, Epoch 1/50, Loss 3.8207, Duration: 0.036 seconds
Stage 1, Epoch 11/50, Loss 3.7897, Duration: 0.002 seconds
Stage 1, Epoch 21/50, Loss 3.7583, Duration: 0.003 seconds
Stage 1, Epoch 31/50, Loss 3.6808, Duration: 0.003 seconds
Stage 1, Epoch 41/50, Loss 3.6589, Duration: 0.002 seconds
Stage 1, Epoch 50/50, Loss 3.6438, Duration: 0.003 seconds
--- Training Stage 1 Finished ---
Saving the trained single block model after Stage 1...
Trained single block model and character tokenizer saved after Stage 1.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 2/50 ---
Starting training for 50 epochs in Stage 2...
Stage 2, Epoch 1/50, Loss 3.7150, Duration: 0.004 seconds
Stage 2, Epoch 11/50, Loss 3.6434, Duration: 0.003 seconds
Stage 2, Epoch 21/50, Loss 3.6010, Duration: 0.002 seconds
Stage 2, Epoch 31/50, Loss 3.5914, Duration: 0.003 seconds
Stage 2, Epoch 41/50, Loss 3.5414, Duration: 0.003 seconds
Stage 2, Epoch 50/50, Loss 3.4939, Duration: 0.003 seconds
--- Training Stage 2 Finished ---
Saving the trained single block model after Stage 2...
Trained single block model and character tokenizer saved after Stage 2.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 3/50 ---
Starting training for 50 epochs in Stage 3...
Stage 3, Epoch 1/50, Loss 3.4977, Duration: 0.003 seconds
Stage 3, Epoch 11/50, Loss 3.4441, Duration: 0.003 seconds
Stage 3, Epoch 21/50, Loss 3.4407, Duration: 0.003 seconds
Stage 3, Epoch 31/50, Loss 3.3916, Duration: 0.002 seconds
Stage 3, Epoch 41/50, Loss 3.3434, Duration: 0.003 seconds
Stage 3, Epoch 50/50, Loss 3.2832, Duration: 0.004 seconds
--- Training Stage 3 Finished ---
Saving the trained single block model after Stage 3...
Trained single block model and character tokenizer saved after Stage 3.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 4/50 ---
Starting training for 50 epochs in Stage 4...
Stage 4, Epoch 1/50, Loss 3.3545, Duration: 0.005 seconds
Stage 4, Epoch 11/50, Loss 3.2973, Duration: 0.002 seconds
Stage 4, Epoch 21/50, Loss 3.2364, Duration: 0.003 seconds
Stage 4, Epoch 31/50, Loss 3.2351, Duration: 0.004 seconds
Stage 4, Epoch 41/50, Loss 3.2095, Duration: 0.002 seconds
Stage 4, Epoch 50/50, Loss 3.1574, Duration: 0.003 seconds
--- Training Stage 4 Finished ---
Saving the trained single block model after Stage 4...
Trained single block model and character tokenizer saved after Stage 4.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 5/50 ---
Starting training for 50 epochs in Stage 5...
Stage 5, Epoch 1/50, Loss 3.1811, Duration: 0.004 seconds
Stage 5, Epoch 11/50, Loss 3.1815, Duration: 0.002 seconds
Stage 5, Epoch 21/50, Loss 3.0754, Duration: 0.003 seconds
Stage 5, Epoch 31/50, Loss 3.0636, Duration: 0.003 seconds
Stage 5, Epoch 41/50, Loss 3.0491, Duration: 0.003 seconds
Stage 5, Epoch 50/50, Loss 2.9753, Duration: 0.002 seconds
--- Training Stage 5 Finished ---
Saving the trained single block model after Stage 5...
Trained single block model and character tokenizer saved after Stage 5.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 6/50 ---
Starting training for 50 epochs in Stage 6...
Stage 6, Epoch 1/50, Loss 3.0123, Duration: 0.004 seconds
Stage 6, Epoch 11/50, Loss 3.0062, Duration: 0.003 seconds
Stage 6, Epoch 21/50, Loss 2.9566, Duration: 0.003 seconds
Stage 6, Epoch 31/50, Loss 2.9248, Duration: 0.002 seconds
Stage 6, Epoch 41/50, Loss 2.8735, Duration: 0.003 seconds
Stage 6, Epoch 50/50, Loss 2.8215, Duration: 0.002 seconds
--- Training Stage 6 Finished ---
Saving the trained single block model after Stage 6...
Trained single block model and character tokenizer saved after Stage 6.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 7/50 ---
Starting training for 50 epochs in Stage 7...
Stage 7, Epoch 1/50, Loss 2.8447, Duration: 0.003 seconds
Stage 7, Epoch 11/50, Loss 2.8978, Duration: 0.003 seconds
Stage 7, Epoch 21/50, Loss 2.8623, Duration: 0.003 seconds
Stage 7, Epoch 31/50, Loss 2.7622, Duration: 0.002 seconds
Stage 7, Epoch 41/50, Loss 2.7374, Duration: 0.003 seconds
Stage 7, Epoch 50/50, Loss 2.7267, Duration: 0.002 seconds
--- Training Stage 7 Finished ---
Saving the trained single block model after Stage 7...
Trained single block model and character tokenizer saved after Stage 7.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 8/50 ---
Starting training for 50 epochs in Stage 8...
Stage 8, Epoch 1/50, Loss 2.7229, Duration: 0.003 seconds
Stage 8, Epoch 11/50, Loss 2.6785, Duration: 0.003 seconds
Stage 8, Epoch 21/50, Loss 2.6516, Duration: 0.003 seconds
Stage 8, Epoch 31/50, Loss 2.6070, Duration: 0.002 seconds
Stage 8, Epoch 41/50, Loss 2.5545, Duration: 0.003 seconds
Stage 8, Epoch 50/50, Loss 2.5410, Duration: 0.002 seconds
--- Training Stage 8 Finished ---
Saving the trained single block model after Stage 8...
Trained single block model and character tokenizer saved after Stage 8.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 9/50 ---
Starting training for 50 epochs in Stage 9...
Stage 9, Epoch 1/50, Loss 2.5574, Duration: 0.003 seconds
Stage 9, Epoch 11/50, Loss 2.5319, Duration: 0.003 seconds
Stage 9, Epoch 21/50, Loss 2.5125, Duration: 0.002 seconds
Stage 9, Epoch 31/50, Loss 2.4282, Duration: 0.002 seconds
Stage 9, Epoch 41/50, Loss 2.3963, Duration: 0.003 seconds
Stage 9, Epoch 50/50, Loss 2.4424, Duration: 0.003 seconds
--- Training Stage 9 Finished ---
Saving the trained single block model after Stage 9...
Trained single block model and character tokenizer saved after Stage 9.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 10/50 ---
Starting training for 50 epochs in Stage 10...
Stage 10, Epoch 1/50, Loss 2.3964, Duration: 0.004 seconds
Stage 10, Epoch 11/50, Loss 2.3493, Duration: 0.003 seconds
Stage 10, Epoch 21/50, Loss 2.3208, Duration: 0.003 seconds
Stage 10, Epoch 31/50, Loss 2.3036, Duration: 0.002 seconds
Stage 10, Epoch 41/50, Loss 2.2658, Duration: 0.002 seconds
Stage 10, Epoch 50/50, Loss 2.2894, Duration: 0.003 seconds
--- Training Stage 10 Finished ---
Saving the trained single block model after Stage 10...
Trained single block model and character tokenizer saved after Stage 10.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 11/50 ---
Starting training for 50 epochs in Stage 11...
Stage 11, Epoch 1/50, Loss 2.2928, Duration: 0.003 seconds
Stage 11, Epoch 11/50, Loss 2.2413, Duration: 0.002 seconds
Stage 11, Epoch 21/50, Loss 2.2225, Duration: 0.003 seconds
Stage 11, Epoch 31/50, Loss 2.1940, Duration: 0.003 seconds
Stage 11, Epoch 41/50, Loss 2.1219, Duration: 0.002 seconds
Stage 11, Epoch 50/50, Loss 2.0952, Duration: 0.003 seconds
--- Training Stage 11 Finished ---
Saving the trained single block model after Stage 11...
Trained single block model and character tokenizer saved after Stage 11.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 12/50 ---
Starting training for 50 epochs in Stage 12...
Stage 12, Epoch 1/50, Loss 2.1503, Duration: 0.003 seconds
Stage 12, Epoch 11/50, Loss 2.0935, Duration: 0.003 seconds
Stage 12, Epoch 21/50, Loss 2.0571, Duration: 0.004 seconds
Stage 12, Epoch 31/50, Loss 2.0993, Duration: 0.002 seconds
Stage 12, Epoch 41/50, Loss 1.9901, Duration: 0.003 seconds
Stage 12, Epoch 50/50, Loss 1.9928, Duration: 0.003 seconds
--- Training Stage 12 Finished ---
Saving the trained single block model after Stage 12...
Trained single block model and character tokenizer saved after Stage 12.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 13/50 ---
Starting training for 50 epochs in Stage 13...
Stage 13, Epoch 1/50, Loss 2.0564, Duration: 0.003 seconds
Stage 13, Epoch 11/50, Loss 1.9328, Duration: 0.003 seconds
Stage 13, Epoch 21/50, Loss 1.9924, Duration: 0.003 seconds
Stage 13, Epoch 31/50, Loss 1.9064, Duration: 0.002 seconds
Stage 13, Epoch 41/50, Loss 1.8939, Duration: 0.003 seconds
Stage 13, Epoch 50/50, Loss 1.8722, Duration: 0.002 seconds
--- Training Stage 13 Finished ---
Saving the trained single block model after Stage 13...
Trained single block model and character tokenizer saved after Stage 13.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 14/50 ---
Starting training for 50 epochs in Stage 14...
Stage 14, Epoch 1/50, Loss 1.7743, Duration: 0.003 seconds
Stage 14, Epoch 11/50, Loss 1.8089, Duration: 0.004 seconds
Stage 14, Epoch 21/50, Loss 1.8656, Duration: 0.002 seconds
Stage 14, Epoch 31/50, Loss 1.8262, Duration: 0.003 seconds
Stage 14, Epoch 41/50, Loss 1.7407, Duration: 0.003 seconds
Stage 14, Epoch 50/50, Loss 1.6900, Duration: 0.002 seconds
--- Training Stage 14 Finished ---
Saving the trained single block model after Stage 14...
Trained single block model and character tokenizer saved after Stage 14.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 15/50 ---
Starting training for 50 epochs in Stage 15...
Stage 15, Epoch 1/50, Loss 1.7112, Duration: 0.004 seconds
Stage 15, Epoch 11/50, Loss 1.7062, Duration: 0.003 seconds
Stage 15, Epoch 21/50, Loss 1.6735, Duration: 0.002 seconds
Stage 15, Epoch 31/50, Loss 1.6723, Duration: 0.003 seconds
Stage 15, Epoch 41/50, Loss 1.6443, Duration: 0.002 seconds
Stage 15, Epoch 50/50, Loss 1.6443, Duration: 0.003 seconds
--- Training Stage 15 Finished ---
Saving the trained single block model after Stage 15...
Trained single block model and character tokenizer saved after Stage 15.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 16/50 ---
Starting training for 50 epochs in Stage 16...
Stage 16, Epoch 1/50, Loss 1.5655, Duration: 0.004 seconds
Stage 16, Epoch 11/50, Loss 1.5587, Duration: 0.002 seconds
Stage 16, Epoch 21/50, Loss 1.5795, Duration: 0.003 seconds
Stage 16, Epoch 31/50, Loss 1.5907, Duration: 0.003 seconds
Stage 16, Epoch 41/50, Loss 1.6059, Duration: 0.002 seconds
Stage 16, Epoch 50/50, Loss 1.5013, Duration: 0.003 seconds
--- Training Stage 16 Finished ---
Saving the trained single block model after Stage 16...
Trained single block model and character tokenizer saved after Stage 16.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 17/50 ---
Starting training for 50 epochs in Stage 17...
Stage 17, Epoch 1/50, Loss 1.4766, Duration: 0.003 seconds
Stage 17, Epoch 11/50, Loss 1.5667, Duration: 0.003 seconds
Stage 17, Epoch 21/50, Loss 1.4540, Duration: 0.003 seconds
Stage 17, Epoch 31/50, Loss 1.3625, Duration: 0.003 seconds
Stage 17, Epoch 41/50, Loss 1.3566, Duration: 0.003 seconds
Stage 17, Epoch 50/50, Loss 1.3503, Duration: 0.003 seconds
--- Training Stage 17 Finished ---
Saving the trained single block model after Stage 17...
Trained single block model and character tokenizer saved after Stage 17.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 18/50 ---
Starting training for 50 epochs in Stage 18...
Stage 18, Epoch 1/50, Loss 1.4012, Duration: 0.003 seconds
Stage 18, Epoch 11/50, Loss 1.3410, Duration: 0.002 seconds
Stage 18, Epoch 21/50, Loss 1.4497, Duration: 0.003 seconds
Stage 18, Epoch 31/50, Loss 1.3543, Duration: 0.003 seconds
Stage 18, Epoch 41/50, Loss 1.3117, Duration: 0.002 seconds
Stage 18, Epoch 50/50, Loss 1.3334, Duration: 0.003 seconds
--- Training Stage 18 Finished ---
Saving the trained single block model after Stage 18...
Trained single block model and character tokenizer saved after Stage 18.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 19/50 ---
Starting training for 50 epochs in Stage 19...
Stage 19, Epoch 1/50, Loss 1.2814, Duration: 0.003 seconds
Stage 19, Epoch 11/50, Loss 1.3730, Duration: 0.003 seconds
Stage 19, Epoch 21/50, Loss 1.2607, Duration: 0.003 seconds
Stage 19, Epoch 31/50, Loss 1.2109, Duration: 0.002 seconds
Stage 19, Epoch 41/50, Loss 1.1628, Duration: 0.003 seconds
Stage 19, Epoch 50/50, Loss 1.2209, Duration: 0.002 seconds
--- Training Stage 19 Finished ---
Saving the trained single block model after Stage 19...
Trained single block model and character tokenizer saved after Stage 19.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 20/50 ---
Starting training for 50 epochs in Stage 20...
Stage 20, Epoch 1/50, Loss 1.1638, Duration: 0.003 seconds
Stage 20, Epoch 11/50, Loss 1.2487, Duration: 0.003 seconds
Stage 20, Epoch 21/50, Loss 1.1583, Duration: 0.003 seconds
Stage 20, Epoch 31/50, Loss 1.2100, Duration: 0.002 seconds
Stage 20, Epoch 41/50, Loss 1.1802, Duration: 0.003 seconds
Stage 20, Epoch 50/50, Loss 1.0453, Duration: 0.002 seconds
--- Training Stage 20 Finished ---
Saving the trained single block model after Stage 20...
Trained single block model and character tokenizer saved after Stage 20.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 21/50 ---
Starting training for 50 epochs in Stage 21...
Stage 21, Epoch 1/50, Loss 1.1368, Duration: 0.003 seconds
Stage 21, Epoch 11/50, Loss 0.9919, Duration: 0.003 seconds
Stage 21, Epoch 21/50, Loss 1.0898, Duration: 0.002 seconds
Stage 21, Epoch 31/50, Loss 1.1035, Duration: 0.003 seconds
Stage 21, Epoch 41/50, Loss 1.0425, Duration: 0.003 seconds
Stage 21, Epoch 50/50, Loss 0.9534, Duration: 0.003 seconds
--- Training Stage 21 Finished ---
Saving the trained single block model after Stage 21...
Trained single block model and character tokenizer saved after Stage 21.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 22/50 ---
Starting training for 50 epochs in Stage 22...
Stage 22, Epoch 1/50, Loss 1.0240, Duration: 0.004 seconds
Stage 22, Epoch 11/50, Loss 1.0672, Duration: 0.002 seconds
Stage 22, Epoch 21/50, Loss 1.0841, Duration: 0.003 seconds
Stage 22, Epoch 31/50, Loss 1.0286, Duration: 0.004 seconds
Stage 22, Epoch 41/50, Loss 0.8584, Duration: 0.002 seconds
Stage 22, Epoch 50/50, Loss 0.9258, Duration: 0.003 seconds
--- Training Stage 22 Finished ---
Saving the trained single block model after Stage 22...
Trained single block model and character tokenizer saved after Stage 22.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 23/50 ---
Starting training for 50 epochs in Stage 23...
Stage 23, Epoch 1/50, Loss 0.9957, Duration: 0.005 seconds
Stage 23, Epoch 11/50, Loss 0.9282, Duration: 0.003 seconds
Stage 23, Epoch 21/50, Loss 0.8462, Duration: 0.003 seconds
Stage 23, Epoch 31/50, Loss 0.9440, Duration: 0.003 seconds
Stage 23, Epoch 41/50, Loss 0.8944, Duration: 0.002 seconds
Stage 23, Epoch 50/50, Loss 0.8221, Duration: 0.003 seconds
--- Training Stage 23 Finished ---
Saving the trained single block model after Stage 23...
Trained single block model and character tokenizer saved after Stage 23.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 24/50 ---
Starting training for 50 epochs in Stage 24...
Stage 24, Epoch 1/50, Loss 0.8434, Duration: 0.003 seconds
Stage 24, Epoch 11/50, Loss 0.9202, Duration: 0.003 seconds
Stage 24, Epoch 21/50, Loss 0.7977, Duration: 0.004 seconds
Stage 24, Epoch 31/50, Loss 0.8576, Duration: 0.008 seconds
Stage 24, Epoch 41/50, Loss 0.8307, Duration: 0.003 seconds
Stage 24, Epoch 50/50, Loss 0.8182, Duration: 0.002 seconds
--- Training Stage 24 Finished ---
Saving the trained single block model after Stage 24...
Trained single block model and character tokenizer saved after Stage 24.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 25/50 ---
Starting training for 50 epochs in Stage 25...
Stage 25, Epoch 1/50, Loss 0.7513, Duration: 0.003 seconds
Stage 25, Epoch 11/50, Loss 0.8046, Duration: 0.003 seconds
Stage 25, Epoch 21/50, Loss 0.8091, Duration: 0.002 seconds
Stage 25, Epoch 31/50, Loss 0.7401, Duration: 0.003 seconds
Stage 25, Epoch 41/50, Loss 0.8608, Duration: 0.003 seconds
Stage 25, Epoch 50/50, Loss 0.7321, Duration: 0.003 seconds
--- Training Stage 25 Finished ---
Saving the trained single block model after Stage 25...
Trained single block model and character tokenizer saved after Stage 25.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 26/50 ---
Starting training for 50 epochs in Stage 26...
Stage 26, Epoch 1/50, Loss 0.7017, Duration: 0.005 seconds
Stage 26, Epoch 11/50, Loss 0.6849, Duration: 0.003 seconds
Stage 26, Epoch 21/50, Loss 0.7223, Duration: 0.003 seconds
Stage 26, Epoch 31/50, Loss 0.6857, Duration: 0.004 seconds
Stage 26, Epoch 41/50, Loss 0.7123, Duration: 0.004 seconds
Stage 26, Epoch 50/50, Loss 0.6910, Duration: 0.004 seconds
--- Training Stage 26 Finished ---
Saving the trained single block model after Stage 26...
Trained single block model and character tokenizer saved after Stage 26.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 27/50 ---
Starting training for 50 epochs in Stage 27...
Stage 27, Epoch 1/50, Loss 0.6654, Duration: 0.003 seconds
Stage 27, Epoch 11/50, Loss 0.7037, Duration: 0.003 seconds
Stage 27, Epoch 21/50, Loss 0.5875, Duration: 0.003 seconds
Stage 27, Epoch 31/50, Loss 0.6038, Duration: 0.002 seconds
Stage 27, Epoch 41/50, Loss 0.6441, Duration: 0.003 seconds
Stage 27, Epoch 50/50, Loss 0.7595, Duration: 0.004 seconds
--- Training Stage 27 Finished ---
Saving the trained single block model after Stage 27...
Trained single block model and character tokenizer saved after Stage 27.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 28/50 ---
Starting training for 50 epochs in Stage 28...
Stage 28, Epoch 1/50, Loss 0.5731, Duration: 0.004 seconds
Stage 28, Epoch 11/50, Loss 0.6692, Duration: 0.003 seconds
Stage 28, Epoch 21/50, Loss 0.5995, Duration: 0.003 seconds
Stage 28, Epoch 31/50, Loss 0.5668, Duration: 0.003 seconds
Stage 28, Epoch 41/50, Loss 0.4979, Duration: 0.002 seconds
Stage 28, Epoch 50/50, Loss 0.5806, Duration: 0.003 seconds
--- Training Stage 28 Finished ---
Saving the trained single block model after Stage 28...
Trained single block model and character tokenizer saved after Stage 28.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 29/50 ---
Starting training for 50 epochs in Stage 29...
Stage 29, Epoch 1/50, Loss 0.5559, Duration: 0.003 seconds
Stage 29, Epoch 11/50, Loss 0.5764, Duration: 0.003 seconds
Stage 29, Epoch 21/50, Loss 0.5639, Duration: 0.003 seconds
Stage 29, Epoch 31/50, Loss 0.5112, Duration: 0.005 seconds
Stage 29, Epoch 41/50, Loss 0.5445, Duration: 0.003 seconds
Stage 29, Epoch 50/50, Loss 0.5569, Duration: 0.003 seconds
--- Training Stage 29 Finished ---
Saving the trained single block model after Stage 29...
Trained single block model and character tokenizer saved after Stage 29.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 30/50 ---
Starting training for 50 epochs in Stage 30...
Stage 30, Epoch 1/50, Loss 0.5354, Duration: 0.004 seconds
Stage 30, Epoch 11/50, Loss 0.5447, Duration: 0.003 seconds
Stage 30, Epoch 21/50, Loss 0.4689, Duration: 0.003 seconds
Stage 30, Epoch 31/50, Loss 0.4647, Duration: 0.002 seconds
Stage 30, Epoch 41/50, Loss 0.4913, Duration: 0.003 seconds
Stage 30, Epoch 50/50, Loss 0.5653, Duration: 0.003 seconds
--- Training Stage 30 Finished ---
Saving the trained single block model after Stage 30...
Trained single block model and character tokenizer saved after Stage 30.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 31/50 ---
Starting training for 50 epochs in Stage 31...
Stage 31, Epoch 1/50, Loss 0.5306, Duration: 0.004 seconds
Stage 31, Epoch 11/50, Loss 0.4788, Duration: 0.002 seconds
Stage 31, Epoch 21/50, Loss 0.4251, Duration: 0.003 seconds
Stage 31, Epoch 31/50, Loss 0.4655, Duration: 0.002 seconds
Stage 31, Epoch 41/50, Loss 0.5081, Duration: 0.003 seconds
Stage 31, Epoch 50/50, Loss 0.4953, Duration: 0.003 seconds
--- Training Stage 31 Finished ---
Saving the trained single block model after Stage 31...
Trained single block model and character tokenizer saved after Stage 31.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 32/50 ---
Starting training for 50 epochs in Stage 32...
Stage 32, Epoch 1/50, Loss 0.4118, Duration: 0.003 seconds
Stage 32, Epoch 11/50, Loss 0.4568, Duration: 0.003 seconds
Stage 32, Epoch 21/50, Loss 0.4776, Duration: 0.003 seconds
Stage 32, Epoch 31/50, Loss 0.4547, Duration: 0.002 seconds
Stage 32, Epoch 41/50, Loss 0.4791, Duration: 0.003 seconds
Stage 32, Epoch 50/50, Loss 0.4352, Duration: 0.002 seconds
--- Training Stage 32 Finished ---
Saving the trained single block model after Stage 32...
Trained single block model and character tokenizer saved after Stage 32.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 33/50 ---
Starting training for 50 epochs in Stage 33...
Stage 33, Epoch 1/50, Loss 0.3232, Duration: 0.003 seconds
Stage 33, Epoch 11/50, Loss 0.3945, Duration: 0.002 seconds
Stage 33, Epoch 21/50, Loss 0.3930, Duration: 0.002 seconds
Stage 33, Epoch 31/50, Loss 0.4137, Duration: 0.003 seconds
Stage 33, Epoch 41/50, Loss 0.4318, Duration: 0.003 seconds
Stage 33, Epoch 50/50, Loss 0.3223, Duration: 0.003 seconds
--- Training Stage 33 Finished ---
Saving the trained single block model after Stage 33...
Trained single block model and character tokenizer saved after Stage 33.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 34/50 ---
Starting training for 50 epochs in Stage 34...
Stage 34, Epoch 1/50, Loss 0.4191, Duration: 0.003 seconds
Stage 34, Epoch 11/50, Loss 0.3500, Duration: 0.002 seconds
Stage 34, Epoch 21/50, Loss 0.3585, Duration: 0.003 seconds
Stage 34, Epoch 31/50, Loss 0.3369, Duration: 0.003 seconds
Stage 34, Epoch 41/50, Loss 0.3461, Duration: 0.002 seconds
Stage 34, Epoch 50/50, Loss 0.3115, Duration: 0.003 seconds
--- Training Stage 34 Finished ---
Saving the trained single block model after Stage 34...
Trained single block model and character tokenizer saved after Stage 34.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 35/50 ---
Starting training for 50 epochs in Stage 35...
Stage 35, Epoch 1/50, Loss 0.2877, Duration: 0.003 seconds
Stage 35, Epoch 11/50, Loss 0.3878, Duration: 0.002 seconds
Stage 35, Epoch 21/50, Loss 0.3717, Duration: 0.002 seconds
Stage 35, Epoch 31/50, Loss 0.3116, Duration: 0.004 seconds
Stage 35, Epoch 41/50, Loss 0.3220, Duration: 0.003 seconds
Stage 35, Epoch 50/50, Loss 0.3140, Duration: 0.003 seconds
--- Training Stage 35 Finished ---
Saving the trained single block model after Stage 35...
Trained single block model and character tokenizer saved after Stage 35.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 36/50 ---
Starting training for 50 epochs in Stage 36...
Stage 36, Epoch 1/50, Loss 0.4561, Duration: 0.004 seconds
Stage 36, Epoch 11/50, Loss 0.2991, Duration: 0.003 seconds
Stage 36, Epoch 21/50, Loss 0.2930, Duration: 0.003 seconds
Stage 36, Epoch 31/50, Loss 0.3330, Duration: 0.002 seconds
Stage 36, Epoch 41/50, Loss 0.3612, Duration: 0.003 seconds
Stage 36, Epoch 50/50, Loss 0.2941, Duration: 0.002 seconds
--- Training Stage 36 Finished ---
Saving the trained single block model after Stage 36...
Trained single block model and character tokenizer saved after Stage 36.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 37/50 ---
Starting training for 50 epochs in Stage 37...
Stage 37, Epoch 1/50, Loss 0.3117, Duration: 0.003 seconds
Stage 37, Epoch 11/50, Loss 0.2242, Duration: 0.003 seconds
Stage 37, Epoch 21/50, Loss 0.2768, Duration: 0.003 seconds
Stage 37, Epoch 31/50, Loss 0.2696, Duration: 0.002 seconds
Stage 37, Epoch 41/50, Loss 0.3061, Duration: 0.003 seconds
Stage 37, Epoch 50/50, Loss 0.2586, Duration: 0.003 seconds
--- Training Stage 37 Finished ---
Saving the trained single block model after Stage 37...
Trained single block model and character tokenizer saved after Stage 37.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 38/50 ---
Starting training for 50 epochs in Stage 38...
Stage 38, Epoch 1/50, Loss 0.2888, Duration: 0.003 seconds
Stage 38, Epoch 11/50, Loss 0.2455, Duration: 0.002 seconds
Stage 38, Epoch 21/50, Loss 0.2007, Duration: 0.002 seconds
Stage 38, Epoch 31/50, Loss 0.2255, Duration: 0.003 seconds
Stage 38, Epoch 41/50, Loss 0.2194, Duration: 0.003 seconds
Stage 38, Epoch 50/50, Loss 0.2261, Duration: 0.002 seconds
--- Training Stage 38 Finished ---
Saving the trained single block model after Stage 38...
Trained single block model and character tokenizer saved after Stage 38.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 39/50 ---
Starting training for 50 epochs in Stage 39...
Stage 39, Epoch 1/50, Loss 0.2565, Duration: 0.004 seconds
Stage 39, Epoch 11/50, Loss 0.2474, Duration: 0.003 seconds
Stage 39, Epoch 21/50, Loss 0.2015, Duration: 0.002 seconds
Stage 39, Epoch 31/50, Loss 0.1806, Duration: 0.003 seconds
Stage 39, Epoch 41/50, Loss 0.1759, Duration: 0.004 seconds
Stage 39, Epoch 50/50, Loss 0.2296, Duration: 0.002 seconds
--- Training Stage 39 Finished ---
Saving the trained single block model after Stage 39...
Trained single block model and character tokenizer saved after Stage 39.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 40/50 ---
Starting training for 50 epochs in Stage 40...
Stage 40, Epoch 1/50, Loss 0.2122, Duration: 0.003 seconds
Stage 40, Epoch 11/50, Loss 0.2094, Duration: 0.003 seconds
Stage 40, Epoch 21/50, Loss 0.1985, Duration: 0.002 seconds
Stage 40, Epoch 31/50, Loss 0.2961, Duration: 0.003 seconds
Stage 40, Epoch 41/50, Loss 0.2277, Duration: 0.003 seconds
Stage 40, Epoch 50/50, Loss 0.2063, Duration: 0.002 seconds
--- Training Stage 40 Finished ---
Saving the trained single block model after Stage 40...
Trained single block model and character tokenizer saved after Stage 40.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 41/50 ---
Starting training for 50 epochs in Stage 41...
Stage 41, Epoch 1/50, Loss 0.2145, Duration: 0.004 seconds
Stage 41, Epoch 11/50, Loss 0.2152, Duration: 0.003 seconds
Stage 41, Epoch 21/50, Loss 0.1857, Duration: 0.002 seconds
Stage 41, Epoch 31/50, Loss 0.1706, Duration: 0.003 seconds
Stage 41, Epoch 41/50, Loss 0.1565, Duration: 0.003 seconds
Stage 41, Epoch 50/50, Loss 0.1669, Duration: 0.003 seconds
--- Training Stage 41 Finished ---
Saving the trained single block model after Stage 41...
Trained single block model and character tokenizer saved after Stage 41.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 42/50 ---
Starting training for 50 epochs in Stage 42...
Stage 42, Epoch 1/50, Loss 0.1688, Duration: 0.003 seconds
Stage 42, Epoch 11/50, Loss 0.1598, Duration: 0.002 seconds
Stage 42, Epoch 21/50, Loss 0.1708, Duration: 0.003 seconds
Stage 42, Epoch 31/50, Loss 0.1992, Duration: 0.003 seconds
Stage 42, Epoch 41/50, Loss 0.1987, Duration: 0.004 seconds
Stage 42, Epoch 50/50, Loss 0.1392, Duration: 0.003 seconds
--- Training Stage 42 Finished ---
Saving the trained single block model after Stage 42...
Trained single block model and character tokenizer saved after Stage 42.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 43/50 ---
Starting training for 50 epochs in Stage 43...
Stage 43, Epoch 1/50, Loss 0.1759, Duration: 0.004 seconds
Stage 43, Epoch 11/50, Loss 0.1261, Duration: 0.003 seconds
Stage 43, Epoch 21/50, Loss 0.1693, Duration: 0.003 seconds
Stage 43, Epoch 31/50, Loss 0.1259, Duration: 0.003 seconds
Stage 43, Epoch 41/50, Loss 0.1671, Duration: 0.003 seconds
Stage 43, Epoch 50/50, Loss 0.1523, Duration: 0.003 seconds
--- Training Stage 43 Finished ---
Saving the trained single block model after Stage 43...
Trained single block model and character tokenizer saved after Stage 43.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 44/50 ---
Starting training for 50 epochs in Stage 44...
Stage 44, Epoch 1/50, Loss 0.1883, Duration: 0.003 seconds
Stage 44, Epoch 11/50, Loss 0.1324, Duration: 0.003 seconds
Stage 44, Epoch 21/50, Loss 0.1650, Duration: 0.002 seconds
Stage 44, Epoch 31/50, Loss 0.3040, Duration: 0.002 seconds
Stage 44, Epoch 41/50, Loss 0.1770, Duration: 0.003 seconds
Stage 44, Epoch 50/50, Loss 0.1487, Duration: 0.002 seconds
--- Training Stage 44 Finished ---
Saving the trained single block model after Stage 44...
Trained single block model and character tokenizer saved after Stage 44.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 45/50 ---
Starting training for 50 epochs in Stage 45...
Stage 45, Epoch 1/50, Loss 0.1388, Duration: 0.004 seconds
Stage 45, Epoch 11/50, Loss 0.1533, Duration: 0.003 seconds
Stage 45, Epoch 21/50, Loss 0.1284, Duration: 0.002 seconds
Stage 45, Epoch 31/50, Loss 0.1492, Duration: 0.003 seconds
Stage 45, Epoch 41/50, Loss 0.0954, Duration: 0.002 seconds
Stage 45, Epoch 50/50, Loss 0.1048, Duration: 0.003 seconds
--- Training Stage 45 Finished ---
Saving the trained single block model after Stage 45...
Trained single block model and character tokenizer saved after Stage 45.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 46/50 ---
Starting training for 50 epochs in Stage 46...
Stage 46, Epoch 1/50, Loss 0.0941, Duration: 0.004 seconds
Stage 46, Epoch 11/50, Loss 0.1531, Duration: 0.002 seconds
Stage 46, Epoch 21/50, Loss 0.2283, Duration: 0.003 seconds
Stage 46, Epoch 31/50, Loss 0.1821, Duration: 0.003 seconds
Stage 46, Epoch 41/50, Loss 0.1528, Duration: 0.002 seconds
Stage 46, Epoch 50/50, Loss 0.2120, Duration: 0.003 seconds
--- Training Stage 46 Finished ---
Saving the trained single block model after Stage 46...
Trained single block model and character tokenizer saved after Stage 46.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 47/50 ---
Starting training for 50 epochs in Stage 47...
Stage 47, Epoch 1/50, Loss 0.1525, Duration: 0.003 seconds
Stage 47, Epoch 11/50, Loss 0.1464, Duration: 0.002 seconds
Stage 47, Epoch 21/50, Loss 0.1739, Duration: 0.003 seconds
Stage 47, Epoch 31/50, Loss 0.1661, Duration: 0.003 seconds
Stage 47, Epoch 41/50, Loss 0.1250, Duration: 0.003 seconds
Stage 47, Epoch 50/50, Loss 0.1704, Duration: 0.002 seconds
--- Training Stage 47 Finished ---
Saving the trained single block model after Stage 47...
Trained single block model and character tokenizer saved after Stage 47.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 48/50 ---
Starting training for 50 epochs in Stage 48...
Stage 48, Epoch 1/50, Loss 0.1063, Duration: 0.003 seconds
Stage 48, Epoch 11/50, Loss 0.0992, Duration: 0.003 seconds
Stage 48, Epoch 21/50, Loss 0.1383, Duration: 0.002 seconds
Stage 48, Epoch 31/50, Loss 0.1647, Duration: 0.002 seconds
Stage 48, Epoch 41/50, Loss 0.1293, Duration: 0.003 seconds
Stage 48, Epoch 50/50, Loss 0.1644, Duration: 0.003 seconds
--- Training Stage 48 Finished ---
Saving the trained single block model after Stage 48...
Trained single block model and character tokenizer saved after Stage 48.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 49/50 ---
Starting training for 50 epochs in Stage 49...
Stage 49, Epoch 1/50, Loss 0.1066, Duration: 0.003 seconds
Stage 49, Epoch 11/50, Loss 0.0746, Duration: 0.003 seconds
Stage 49, Epoch 21/50, Loss 0.0992, Duration: 0.002 seconds
Stage 49, Epoch 31/50, Loss 0.1167, Duration: 0.004 seconds
Stage 49, Epoch 41/50, Loss 0.1166, Duration: 0.004 seconds
Stage 49, Epoch 50/50, Loss 0.0863, Duration: 0.003 seconds
--- Training Stage 49 Finished ---
Saving the trained single block model after Stage 49...
Trained single block model and character tokenizer saved after Stage 49.

Training data: Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.
Encoded input_ids shape: torch.Size([1, 64])

--- Starting Training Stage 50/50 ---
Starting training for 50 epochs in Stage 50...
Stage 50, Epoch 1/50, Loss 0.1185, Duration: 0.003 seconds
Stage 50, Epoch 11/50, Loss 0.1145, Duration: 0.003 seconds
Stage 50, Epoch 21/50, Loss 0.0974, Duration: 0.004 seconds
Stage 50, Epoch 31/50, Loss 0.0820, Duration: 0.003 seconds
Stage 50, Epoch 41/50, Loss 0.1206, Duration: 0.003 seconds
Stage 50, Epoch 50/50, Loss 0.0821, Duration: 0.002 seconds
--- Training Stage 50 Finished ---
Saving the trained single block model after Stage 50...
Trained single block model and character tokenizer saved after Stage 50.

All training stages finished.
'''