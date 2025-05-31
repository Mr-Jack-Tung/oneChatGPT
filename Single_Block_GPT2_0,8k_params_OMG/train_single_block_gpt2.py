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
from single_block_gpt2_model import SingleBlockGPT2ModelNoDepend, GPT2Config

# Define the path to the trained single block model directory
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'TrainedSingleBlockGPT2_0,8k_params'
MODEL_STATE_DICT_PATH = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'

# Main function to run staged training
if __name__ == '__main__':
    # Staged training loop
    num_stages = 100
    epochs_per_stage = 100

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
        # print(f"\nTraining data: {qa_pair}")
        # print(f"Encoded input_ids shape: {input_ids.shape}")


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
            if epoch % 25 == 0 or epoch == epochs_per_stage - 1:
              print(f"Stage {stage + 1}, Epoch {epoch+1}/{epochs_per_stage}, Loss {loss.item():.4f}, Duration: {epoch_duration:.3f} seconds")

        print(f"--- Training Stage {stage + 1} Finished ---")

        # Save the trained single block model after each stage
        # print(f"Saving the trained single block model after Stage {stage + 1}...")
        try:
            # Create the directory if it doesn't exist
            if not os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
                os.makedirs(TRAINED_SINGLE_BLOCK_MODEL_PATH)

            # Save the model's state dictionary
            torch.save(single_block_model.state_dict(), MODEL_STATE_DICT_PATH)

            # Save the character tokenizer (only once, but saving here ensures it's always with the latest model)
            tokenizer.save_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)

            # print(f"Trained single block model and character tokenizer saved after Stage {stage + 1}.")
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
% uv run Single_Block_GPT2_0,8k_params_OMG/train_single_block_gpt2.py
Initializing character tokenizer...
Character tokenizer trained with vocabulary size: 34

--- Starting Training Stage 1/100 ---
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 4)
  (wpe): Embedding(64, 4)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((4,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=4, out_features=12, bias=True)
      (c_proj): Linear(in_features=4, out_features=4, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((4,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=4, out_features=16, bias=True)
      (c_proj): Linear(in_features=16, out_features=4, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((4,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=4, out_features=34, bias=False)
)
Number of trainable params: 780
Starting training for 100 epochs in Stage 1...
Stage 1, Epoch 1/100, Loss 3.7145, Duration: 0.026 seconds
Stage 1, Epoch 26/100, Loss 3.6524, Duration: 0.002 seconds
Stage 1, Epoch 51/100, Loss 3.6737, Duration: 0.002 seconds
Stage 1, Epoch 76/100, Loss 3.6341, Duration: 0.002 seconds
Stage 1, Epoch 100/100, Loss 3.6096, Duration: 0.002 seconds
--- Training Stage 1 Finished ---

--- Starting Training Stage 2/100 ---
Starting training for 100 epochs in Stage 2...
Stage 2, Epoch 1/100, Loss 3.5970, Duration: 0.003 seconds
Stage 2, Epoch 26/100, Loss 3.5527, Duration: 0.003 seconds
Stage 2, Epoch 51/100, Loss 3.5149, Duration: 0.002 seconds
Stage 2, Epoch 76/100, Loss 3.4855, Duration: 0.002 seconds
Stage 2, Epoch 100/100, Loss 3.4736, Duration: 0.003 seconds
--- Training Stage 2 Finished ---

--- Starting Training Stage 3/100 ---
Starting training for 100 epochs in Stage 3...
Stage 3, Epoch 1/100, Loss 3.4486, Duration: 0.003 seconds
Stage 3, Epoch 26/100, Loss 3.4165, Duration: 0.002 seconds
Stage 3, Epoch 51/100, Loss 3.4530, Duration: 0.003 seconds
Stage 3, Epoch 76/100, Loss 3.4214, Duration: 0.003 seconds
Stage 3, Epoch 100/100, Loss 3.3392, Duration: 0.003 seconds
--- Training Stage 3 Finished ---

--- Starting Training Stage 4/100 ---
Starting training for 100 epochs in Stage 4...
Stage 4, Epoch 1/100, Loss 3.3460, Duration: 0.004 seconds
Stage 4, Epoch 26/100, Loss 3.3420, Duration: 0.003 seconds
Stage 4, Epoch 51/100, Loss 3.2650, Duration: 0.004 seconds
Stage 4, Epoch 76/100, Loss 3.3320, Duration: 0.002 seconds
Stage 4, Epoch 100/100, Loss 3.2429, Duration: 0.002 seconds
--- Training Stage 4 Finished ---

--- Starting Training Stage 5/100 ---
Starting training for 100 epochs in Stage 5...
Stage 5, Epoch 1/100, Loss 3.3072, Duration: 0.003 seconds
Stage 5, Epoch 26/100, Loss 3.2286, Duration: 0.002 seconds
Stage 5, Epoch 51/100, Loss 3.1792, Duration: 0.003 seconds
Stage 5, Epoch 76/100, Loss 3.1648, Duration: 0.003 seconds
Stage 5, Epoch 100/100, Loss 3.1388, Duration: 0.002 seconds
--- Training Stage 5 Finished ---

--- Starting Training Stage 6/100 ---
Starting training for 100 epochs in Stage 6...
Stage 6, Epoch 1/100, Loss 3.1875, Duration: 0.004 seconds
Stage 6, Epoch 26/100, Loss 3.1548, Duration: 0.002 seconds
Stage 6, Epoch 51/100, Loss 3.0682, Duration: 0.002 seconds
Stage 6, Epoch 76/100, Loss 3.0787, Duration: 0.002 seconds
Stage 6, Epoch 100/100, Loss 3.0773, Duration: 0.003 seconds
--- Training Stage 6 Finished ---

--- Starting Training Stage 7/100 ---
Starting training for 100 epochs in Stage 7...
Stage 7, Epoch 1/100, Loss 3.0377, Duration: 0.004 seconds
Stage 7, Epoch 26/100, Loss 3.0800, Duration: 0.003 seconds
Stage 7, Epoch 51/100, Loss 3.0761, Duration: 0.002 seconds
Stage 7, Epoch 76/100, Loss 2.9980, Duration: 0.003 seconds
Stage 7, Epoch 100/100, Loss 2.9548, Duration: 0.002 seconds
--- Training Stage 7 Finished ---

--- Starting Training Stage 8/100 ---
Starting training for 100 epochs in Stage 8...
Stage 8, Epoch 1/100, Loss 2.9258, Duration: 0.004 seconds
Stage 8, Epoch 26/100, Loss 2.9556, Duration: 0.002 seconds
Stage 8, Epoch 51/100, Loss 2.9335, Duration: 0.003 seconds
Stage 8, Epoch 76/100, Loss 2.8966, Duration: 0.002 seconds
Stage 8, Epoch 100/100, Loss 2.8571, Duration: 0.003 seconds
--- Training Stage 8 Finished ---

--- Starting Training Stage 9/100 ---
Starting training for 100 epochs in Stage 9...
Stage 9, Epoch 1/100, Loss 2.9498, Duration: 0.003 seconds
Stage 9, Epoch 26/100, Loss 2.8476, Duration: 0.003 seconds
Stage 9, Epoch 51/100, Loss 2.8463, Duration: 0.003 seconds
Stage 9, Epoch 76/100, Loss 2.8335, Duration: 0.002 seconds
Stage 9, Epoch 100/100, Loss 2.7633, Duration: 0.002 seconds
--- Training Stage 9 Finished ---

--- Starting Training Stage 10/100 ---
Starting training for 100 epochs in Stage 10...
Stage 10, Epoch 1/100, Loss 2.8383, Duration: 0.003 seconds
Stage 10, Epoch 26/100, Loss 2.7917, Duration: 0.002 seconds
Stage 10, Epoch 51/100, Loss 2.8535, Duration: 0.002 seconds
Stage 10, Epoch 76/100, Loss 2.7378, Duration: 0.002 seconds
Stage 10, Epoch 100/100, Loss 2.6405, Duration: 0.003 seconds
--- Training Stage 10 Finished ---

--- Starting Training Stage 11/100 ---
Starting training for 100 epochs in Stage 11...
Stage 11, Epoch 1/100, Loss 2.7157, Duration: 0.003 seconds
Stage 11, Epoch 26/100, Loss 2.7565, Duration: 0.003 seconds
Stage 11, Epoch 51/100, Loss 2.6711, Duration: 0.002 seconds
Stage 11, Epoch 76/100, Loss 2.6862, Duration: 0.003 seconds
Stage 11, Epoch 100/100, Loss 2.6374, Duration: 0.003 seconds
--- Training Stage 11 Finished ---

--- Starting Training Stage 12/100 ---
Starting training for 100 epochs in Stage 12...
Stage 12, Epoch 1/100, Loss 2.6255, Duration: 0.003 seconds
Stage 12, Epoch 26/100, Loss 2.6038, Duration: 0.002 seconds
Stage 12, Epoch 51/100, Loss 2.6052, Duration: 0.002 seconds
Stage 12, Epoch 76/100, Loss 2.5596, Duration: 0.003 seconds
Stage 12, Epoch 100/100, Loss 2.5847, Duration: 0.002 seconds
--- Training Stage 12 Finished ---

--- Starting Training Stage 13/100 ---
Starting training for 100 epochs in Stage 13...
Stage 13, Epoch 1/100, Loss 2.5029, Duration: 0.003 seconds
Stage 13, Epoch 26/100, Loss 2.5672, Duration: 0.002 seconds
Stage 13, Epoch 51/100, Loss 2.5123, Duration: 0.002 seconds
Stage 13, Epoch 76/100, Loss 2.5191, Duration: 0.002 seconds
Stage 13, Epoch 100/100, Loss 2.5036, Duration: 0.002 seconds
--- Training Stage 13 Finished ---

--- Starting Training Stage 14/100 ---
Starting training for 100 epochs in Stage 14...
Stage 14, Epoch 1/100, Loss 2.5233, Duration: 0.003 seconds
Stage 14, Epoch 26/100, Loss 2.5491, Duration: 0.002 seconds
Stage 14, Epoch 51/100, Loss 2.6176, Duration: 0.002 seconds
Stage 14, Epoch 76/100, Loss 2.5195, Duration: 0.002 seconds
Stage 14, Epoch 100/100, Loss 2.5300, Duration: 0.003 seconds
--- Training Stage 14 Finished ---

--- Starting Training Stage 15/100 ---
Starting training for 100 epochs in Stage 15...
Stage 15, Epoch 1/100, Loss 2.4766, Duration: 0.003 seconds
Stage 15, Epoch 26/100, Loss 2.4229, Duration: 0.002 seconds
Stage 15, Epoch 51/100, Loss 2.4372, Duration: 0.002 seconds
Stage 15, Epoch 76/100, Loss 2.4153, Duration: 0.002 seconds
Stage 15, Epoch 100/100, Loss 2.5168, Duration: 0.002 seconds
--- Training Stage 15 Finished ---

--- Starting Training Stage 16/100 ---
Starting training for 100 epochs in Stage 16...
Stage 16, Epoch 1/100, Loss 2.4210, Duration: 0.004 seconds
Stage 16, Epoch 26/100, Loss 2.4776, Duration: 0.002 seconds
Stage 16, Epoch 51/100, Loss 2.4161, Duration: 0.002 seconds
Stage 16, Epoch 76/100, Loss 2.3894, Duration: 0.002 seconds
Stage 16, Epoch 100/100, Loss 2.2973, Duration: 0.002 seconds
--- Training Stage 16 Finished ---

--- Starting Training Stage 17/100 ---
Starting training for 100 epochs in Stage 17...
Stage 17, Epoch 1/100, Loss 2.2866, Duration: 0.003 seconds
Stage 17, Epoch 26/100, Loss 2.3450, Duration: 0.002 seconds
Stage 17, Epoch 51/100, Loss 2.3597, Duration: 0.002 seconds
Stage 17, Epoch 76/100, Loss 2.3199, Duration: 0.002 seconds
Stage 17, Epoch 100/100, Loss 2.3008, Duration: 0.002 seconds
--- Training Stage 17 Finished ---

--- Starting Training Stage 18/100 ---
Starting training for 100 epochs in Stage 18...
Stage 18, Epoch 1/100, Loss 2.4551, Duration: 0.003 seconds
Stage 18, Epoch 26/100, Loss 2.2156, Duration: 0.002 seconds
Stage 18, Epoch 51/100, Loss 2.2353, Duration: 0.002 seconds
Stage 18, Epoch 76/100, Loss 2.2319, Duration: 0.002 seconds
Stage 18, Epoch 100/100, Loss 2.1841, Duration: 0.003 seconds
--- Training Stage 18 Finished ---

--- Starting Training Stage 19/100 ---
Starting training for 100 epochs in Stage 19...
Stage 19, Epoch 1/100, Loss 2.2248, Duration: 0.003 seconds
Stage 19, Epoch 26/100, Loss 2.2078, Duration: 0.002 seconds
Stage 19, Epoch 51/100, Loss 2.3209, Duration: 0.003 seconds
Stage 19, Epoch 76/100, Loss 2.1372, Duration: 0.002 seconds
Stage 19, Epoch 100/100, Loss 2.2549, Duration: 0.002 seconds
--- Training Stage 19 Finished ---

--- Starting Training Stage 20/100 ---
Starting training for 100 epochs in Stage 20...
Stage 20, Epoch 1/100, Loss 2.2070, Duration: 0.004 seconds
Stage 20, Epoch 26/100, Loss 2.2255, Duration: 0.002 seconds
Stage 20, Epoch 51/100, Loss 2.2160, Duration: 0.003 seconds
Stage 20, Epoch 76/100, Loss 2.1875, Duration: 0.002 seconds
Stage 20, Epoch 100/100, Loss 2.2672, Duration: 0.003 seconds
--- Training Stage 20 Finished ---

--- Starting Training Stage 21/100 ---
Starting training for 100 epochs in Stage 21...
Stage 21, Epoch 1/100, Loss 2.1699, Duration: 0.003 seconds
Stage 21, Epoch 26/100, Loss 2.1634, Duration: 0.002 seconds
Stage 21, Epoch 51/100, Loss 2.0305, Duration: 0.002 seconds
Stage 21, Epoch 76/100, Loss 2.0191, Duration: 0.002 seconds
Stage 21, Epoch 100/100, Loss 2.0991, Duration: 0.003 seconds
--- Training Stage 21 Finished ---

--- Starting Training Stage 22/100 ---
Starting training for 100 epochs in Stage 22...
Stage 22, Epoch 1/100, Loss 2.0868, Duration: 0.003 seconds
Stage 22, Epoch 26/100, Loss 2.1984, Duration: 0.003 seconds
Stage 22, Epoch 51/100, Loss 2.0168, Duration: 0.002 seconds
Stage 22, Epoch 76/100, Loss 2.1229, Duration: 0.002 seconds
Stage 22, Epoch 100/100, Loss 1.9792, Duration: 0.002 seconds
--- Training Stage 22 Finished ---

--- Starting Training Stage 23/100 ---
Starting training for 100 epochs in Stage 23...
Stage 23, Epoch 1/100, Loss 2.0763, Duration: 0.003 seconds
Stage 23, Epoch 26/100, Loss 2.1292, Duration: 0.003 seconds
Stage 23, Epoch 51/100, Loss 2.0339, Duration: 0.003 seconds
Stage 23, Epoch 76/100, Loss 2.0659, Duration: 0.002 seconds
Stage 23, Epoch 100/100, Loss 1.9425, Duration: 0.002 seconds
--- Training Stage 23 Finished ---

--- Starting Training Stage 24/100 ---
Starting training for 100 epochs in Stage 24...
Stage 24, Epoch 1/100, Loss 1.9603, Duration: 0.004 seconds
Stage 24, Epoch 26/100, Loss 1.9998, Duration: 0.002 seconds
Stage 24, Epoch 51/100, Loss 2.0042, Duration: 0.002 seconds
Stage 24, Epoch 76/100, Loss 1.9314, Duration: 0.002 seconds
Stage 24, Epoch 100/100, Loss 2.0027, Duration: 0.002 seconds
--- Training Stage 24 Finished ---

--- Starting Training Stage 25/100 ---
Starting training for 100 epochs in Stage 25...
Stage 25, Epoch 1/100, Loss 2.1770, Duration: 0.003 seconds
Stage 25, Epoch 26/100, Loss 1.9076, Duration: 0.002 seconds
Stage 25, Epoch 51/100, Loss 1.8465, Duration: 0.003 seconds
Stage 25, Epoch 76/100, Loss 1.9385, Duration: 0.002 seconds
Stage 25, Epoch 100/100, Loss 1.8833, Duration: 0.002 seconds
--- Training Stage 25 Finished ---

--- Starting Training Stage 26/100 ---
Starting training for 100 epochs in Stage 26...
Stage 26, Epoch 1/100, Loss 1.9714, Duration: 0.003 seconds
Stage 26, Epoch 26/100, Loss 1.9407, Duration: 0.002 seconds
Stage 26, Epoch 51/100, Loss 1.8820, Duration: 0.003 seconds
Stage 26, Epoch 76/100, Loss 2.2042, Duration: 0.002 seconds
Stage 26, Epoch 100/100, Loss 1.9724, Duration: 0.002 seconds
--- Training Stage 26 Finished ---

--- Starting Training Stage 27/100 ---
Starting training for 100 epochs in Stage 27...
Stage 27, Epoch 1/100, Loss 1.9030, Duration: 0.004 seconds
Stage 27, Epoch 26/100, Loss 1.9653, Duration: 0.003 seconds
Stage 27, Epoch 51/100, Loss 1.8895, Duration: 0.002 seconds
Stage 27, Epoch 76/100, Loss 1.9097, Duration: 0.002 seconds
Stage 27, Epoch 100/100, Loss 1.8281, Duration: 0.002 seconds
--- Training Stage 27 Finished ---

--- Starting Training Stage 28/100 ---
Starting training for 100 epochs in Stage 28...
Stage 28, Epoch 1/100, Loss 1.8258, Duration: 0.003 seconds
Stage 28, Epoch 26/100, Loss 1.9126, Duration: 0.002 seconds
Stage 28, Epoch 51/100, Loss 1.8762, Duration: 0.002 seconds
Stage 28, Epoch 76/100, Loss 1.9442, Duration: 0.003 seconds
Stage 28, Epoch 100/100, Loss 1.8351, Duration: 0.002 seconds
--- Training Stage 28 Finished ---

--- Starting Training Stage 29/100 ---
Starting training for 100 epochs in Stage 29...
Stage 29, Epoch 1/100, Loss 1.8048, Duration: 0.003 seconds
Stage 29, Epoch 26/100, Loss 1.8133, Duration: 0.002 seconds
Stage 29, Epoch 51/100, Loss 1.8114, Duration: 0.002 seconds
Stage 29, Epoch 76/100, Loss 1.8138, Duration: 0.002 seconds
Stage 29, Epoch 100/100, Loss 1.9334, Duration: 0.003 seconds
--- Training Stage 29 Finished ---

--- Starting Training Stage 30/100 ---
Starting training for 100 epochs in Stage 30...
Stage 30, Epoch 1/100, Loss 1.8176, Duration: 0.003 seconds
Stage 30, Epoch 26/100, Loss 1.7013, Duration: 0.003 seconds
Stage 30, Epoch 51/100, Loss 1.8305, Duration: 0.003 seconds
Stage 30, Epoch 76/100, Loss 1.6616, Duration: 0.003 seconds
Stage 30, Epoch 100/100, Loss 1.8439, Duration: 0.003 seconds
--- Training Stage 30 Finished ---

--- Starting Training Stage 31/100 ---
Starting training for 100 epochs in Stage 31...
Stage 31, Epoch 1/100, Loss 1.6449, Duration: 0.003 seconds
Stage 31, Epoch 26/100, Loss 1.9317, Duration: 0.002 seconds
Stage 31, Epoch 51/100, Loss 1.7102, Duration: 0.002 seconds
Stage 31, Epoch 76/100, Loss 1.6899, Duration: 0.002 seconds
Stage 31, Epoch 100/100, Loss 1.6833, Duration: 0.002 seconds
--- Training Stage 31 Finished ---

--- Starting Training Stage 32/100 ---
Starting training for 100 epochs in Stage 32...
Stage 32, Epoch 1/100, Loss 1.8271, Duration: 0.003 seconds
Stage 32, Epoch 26/100, Loss 1.5493, Duration: 0.002 seconds
Stage 32, Epoch 51/100, Loss 1.7262, Duration: 0.003 seconds
Stage 32, Epoch 76/100, Loss 1.6523, Duration: 0.002 seconds
Stage 32, Epoch 100/100, Loss 1.5713, Duration: 0.002 seconds
--- Training Stage 32 Finished ---

--- Starting Training Stage 33/100 ---
Starting training for 100 epochs in Stage 33...
Stage 33, Epoch 1/100, Loss 1.5643, Duration: 0.004 seconds
Stage 33, Epoch 26/100, Loss 1.7604, Duration: 0.002 seconds
Stage 33, Epoch 51/100, Loss 1.7174, Duration: 0.002 seconds
Stage 33, Epoch 76/100, Loss 1.7232, Duration: 0.002 seconds
Stage 33, Epoch 100/100, Loss 1.5920, Duration: 0.002 seconds
--- Training Stage 33 Finished ---

--- Starting Training Stage 34/100 ---
Starting training for 100 epochs in Stage 34...
Stage 34, Epoch 1/100, Loss 1.7328, Duration: 0.003 seconds
Stage 34, Epoch 26/100, Loss 1.5539, Duration: 0.003 seconds
Stage 34, Epoch 51/100, Loss 1.6419, Duration: 0.003 seconds
Stage 34, Epoch 76/100, Loss 1.6366, Duration: 0.002 seconds
Stage 34, Epoch 100/100, Loss 1.4921, Duration: 0.003 seconds
--- Training Stage 34 Finished ---

--- Starting Training Stage 35/100 ---
Starting training for 100 epochs in Stage 35...
Stage 35, Epoch 1/100, Loss 1.7042, Duration: 0.003 seconds
Stage 35, Epoch 26/100, Loss 1.4624, Duration: 0.002 seconds
Stage 35, Epoch 51/100, Loss 1.5562, Duration: 0.002 seconds
Stage 35, Epoch 76/100, Loss 1.7401, Duration: 0.002 seconds
Stage 35, Epoch 100/100, Loss 1.5478, Duration: 0.002 seconds
--- Training Stage 35 Finished ---

--- Starting Training Stage 36/100 ---
Starting training for 100 epochs in Stage 36...
Stage 36, Epoch 1/100, Loss 1.4310, Duration: 0.003 seconds
Stage 36, Epoch 26/100, Loss 1.4440, Duration: 0.002 seconds
Stage 36, Epoch 51/100, Loss 1.6559, Duration: 0.003 seconds
Stage 36, Epoch 76/100, Loss 1.5240, Duration: 0.002 seconds
Stage 36, Epoch 100/100, Loss 1.5889, Duration: 0.002 seconds
--- Training Stage 36 Finished ---

--- Starting Training Stage 37/100 ---
Starting training for 100 epochs in Stage 37...
Stage 37, Epoch 1/100, Loss 1.5358, Duration: 0.003 seconds
Stage 37, Epoch 26/100, Loss 1.3471, Duration: 0.002 seconds
Stage 37, Epoch 51/100, Loss 1.5599, Duration: 0.002 seconds
Stage 37, Epoch 76/100, Loss 1.4371, Duration: 0.003 seconds
Stage 37, Epoch 100/100, Loss 1.5682, Duration: 0.002 seconds
--- Training Stage 37 Finished ---

--- Starting Training Stage 38/100 ---
Starting training for 100 epochs in Stage 38...
Stage 38, Epoch 1/100, Loss 1.4431, Duration: 0.004 seconds
Stage 38, Epoch 26/100, Loss 1.7262, Duration: 0.002 seconds
Stage 38, Epoch 51/100, Loss 1.2923, Duration: 0.002 seconds
Stage 38, Epoch 76/100, Loss 1.3764, Duration: 0.002 seconds
Stage 38, Epoch 100/100, Loss 1.3834, Duration: 0.002 seconds
--- Training Stage 38 Finished ---

--- Starting Training Stage 39/100 ---
Starting training for 100 epochs in Stage 39...
Stage 39, Epoch 1/100, Loss 1.7180, Duration: 0.003 seconds
Stage 39, Epoch 26/100, Loss 1.4705, Duration: 0.002 seconds
Stage 39, Epoch 51/100, Loss 1.3401, Duration: 0.002 seconds
Stage 39, Epoch 76/100, Loss 1.4405, Duration: 0.002 seconds
Stage 39, Epoch 100/100, Loss 1.4680, Duration: 0.003 seconds
--- Training Stage 39 Finished ---

--- Starting Training Stage 40/100 ---
Starting training for 100 epochs in Stage 40...
Stage 40, Epoch 1/100, Loss 1.2653, Duration: 0.003 seconds
Stage 40, Epoch 26/100, Loss 1.6282, Duration: 0.002 seconds
Stage 40, Epoch 51/100, Loss 1.3300, Duration: 0.002 seconds
Stage 40, Epoch 76/100, Loss 1.5000, Duration: 0.002 seconds
Stage 40, Epoch 100/100, Loss 1.3599, Duration: 0.002 seconds
--- Training Stage 40 Finished ---

--- Starting Training Stage 41/100 ---
Starting training for 100 epochs in Stage 41...
Stage 41, Epoch 1/100, Loss 1.2995, Duration: 0.003 seconds
Stage 41, Epoch 26/100, Loss 1.3442, Duration: 0.002 seconds
Stage 41, Epoch 51/100, Loss 1.4936, Duration: 0.003 seconds
Stage 41, Epoch 76/100, Loss 1.4128, Duration: 0.002 seconds
Stage 41, Epoch 100/100, Loss 1.5078, Duration: 0.003 seconds
--- Training Stage 41 Finished ---

--- Starting Training Stage 42/100 ---
Starting training for 100 epochs in Stage 42...
Stage 42, Epoch 1/100, Loss 1.3133, Duration: 0.004 seconds
Stage 42, Epoch 26/100, Loss 1.5229, Duration: 0.002 seconds
Stage 42, Epoch 51/100, Loss 1.2941, Duration: 0.003 seconds
Stage 42, Epoch 76/100, Loss 1.2518, Duration: 0.003 seconds
Stage 42, Epoch 100/100, Loss 1.3971, Duration: 0.002 seconds
--- Training Stage 42 Finished ---

--- Starting Training Stage 43/100 ---
Starting training for 100 epochs in Stage 43...
Stage 43, Epoch 1/100, Loss 1.4363, Duration: 0.003 seconds
Stage 43, Epoch 26/100, Loss 1.2243, Duration: 0.002 seconds
Stage 43, Epoch 51/100, Loss 1.3816, Duration: 0.003 seconds
Stage 43, Epoch 76/100, Loss 1.2931, Duration: 0.002 seconds
Stage 43, Epoch 100/100, Loss 1.2162, Duration: 0.002 seconds
--- Training Stage 43 Finished ---

--- Starting Training Stage 44/100 ---
Starting training for 100 epochs in Stage 44...
Stage 44, Epoch 1/100, Loss 1.2703, Duration: 0.003 seconds
Stage 44, Epoch 26/100, Loss 1.3876, Duration: 0.002 seconds
Stage 44, Epoch 51/100, Loss 1.2102, Duration: 0.002 seconds
Stage 44, Epoch 76/100, Loss 1.1924, Duration: 0.002 seconds
Stage 44, Epoch 100/100, Loss 1.2030, Duration: 0.003 seconds
--- Training Stage 44 Finished ---

--- Starting Training Stage 45/100 ---
Starting training for 100 epochs in Stage 45...
Stage 45, Epoch 1/100, Loss 1.3507, Duration: 0.004 seconds
Stage 45, Epoch 26/100, Loss 1.2675, Duration: 0.003 seconds
Stage 45, Epoch 51/100, Loss 1.3042, Duration: 0.003 seconds
Stage 45, Epoch 76/100, Loss 1.2993, Duration: 0.002 seconds
Stage 45, Epoch 100/100, Loss 1.2785, Duration: 0.002 seconds
--- Training Stage 45 Finished ---

--- Starting Training Stage 46/100 ---
Starting training for 100 epochs in Stage 46...
Stage 46, Epoch 1/100, Loss 1.3058, Duration: 0.004 seconds
Stage 46, Epoch 26/100, Loss 1.2455, Duration: 0.003 seconds
Stage 46, Epoch 51/100, Loss 1.1330, Duration: 0.002 seconds
Stage 46, Epoch 76/100, Loss 1.2715, Duration: 0.002 seconds
Stage 46, Epoch 100/100, Loss 1.3891, Duration: 0.002 seconds
--- Training Stage 46 Finished ---

--- Starting Training Stage 47/100 ---
Starting training for 100 epochs in Stage 47...
Stage 47, Epoch 1/100, Loss 1.4534, Duration: 0.003 seconds
Stage 47, Epoch 26/100, Loss 1.1889, Duration: 0.002 seconds
Stage 47, Epoch 51/100, Loss 1.1525, Duration: 0.004 seconds
Stage 47, Epoch 76/100, Loss 1.1345, Duration: 0.003 seconds
Stage 47, Epoch 100/100, Loss 1.4006, Duration: 0.003 seconds
--- Training Stage 47 Finished ---

--- Starting Training Stage 48/100 ---
Starting training for 100 epochs in Stage 48...
Stage 48, Epoch 1/100, Loss 1.1631, Duration: 0.003 seconds
Stage 48, Epoch 26/100, Loss 1.4762, Duration: 0.003 seconds
Stage 48, Epoch 51/100, Loss 1.4386, Duration: 0.003 seconds
Stage 48, Epoch 76/100, Loss 1.1669, Duration: 0.003 seconds
Stage 48, Epoch 100/100, Loss 1.0908, Duration: 0.003 seconds
--- Training Stage 48 Finished ---

--- Starting Training Stage 49/100 ---
Starting training for 100 epochs in Stage 49...
Stage 49, Epoch 1/100, Loss 1.1970, Duration: 0.004 seconds
Stage 49, Epoch 26/100, Loss 1.1738, Duration: 0.002 seconds
Stage 49, Epoch 51/100, Loss 1.0792, Duration: 0.002 seconds
Stage 49, Epoch 76/100, Loss 1.0915, Duration: 0.003 seconds
Stage 49, Epoch 100/100, Loss 1.4806, Duration: 0.002 seconds
--- Training Stage 49 Finished ---

--- Starting Training Stage 50/100 ---
Starting training for 100 epochs in Stage 50...
Stage 50, Epoch 1/100, Loss 1.4156, Duration: 0.003 seconds
Stage 50, Epoch 26/100, Loss 1.0160, Duration: 0.002 seconds
Stage 50, Epoch 51/100, Loss 1.1570, Duration: 0.003 seconds
Stage 50, Epoch 76/100, Loss 1.3053, Duration: 0.002 seconds
Stage 50, Epoch 100/100, Loss 1.2579, Duration: 0.002 seconds
--- Training Stage 50 Finished ---

--- Starting Training Stage 51/100 ---
Starting training for 100 epochs in Stage 51...
Stage 51, Epoch 1/100, Loss 1.0974, Duration: 0.003 seconds
Stage 51, Epoch 26/100, Loss 1.1518, Duration: 0.003 seconds
Stage 51, Epoch 51/100, Loss 1.1818, Duration: 0.002 seconds
Stage 51, Epoch 76/100, Loss 0.9268, Duration: 0.003 seconds
Stage 51, Epoch 100/100, Loss 1.2140, Duration: 0.002 seconds
--- Training Stage 51 Finished ---

--- Starting Training Stage 52/100 ---
Starting training for 100 epochs in Stage 52...
Stage 52, Epoch 1/100, Loss 1.1828, Duration: 0.004 seconds
Stage 52, Epoch 26/100, Loss 1.0078, Duration: 0.002 seconds
Stage 52, Epoch 51/100, Loss 1.1771, Duration: 0.003 seconds
Stage 52, Epoch 76/100, Loss 1.1759, Duration: 0.003 seconds
Stage 52, Epoch 100/100, Loss 1.1563, Duration: 0.003 seconds
--- Training Stage 52 Finished ---

--- Starting Training Stage 53/100 ---
Starting training for 100 epochs in Stage 53...
Stage 53, Epoch 1/100, Loss 1.0814, Duration: 0.004 seconds
Stage 53, Epoch 26/100, Loss 1.0339, Duration: 0.003 seconds
Stage 53, Epoch 51/100, Loss 1.0384, Duration: 0.002 seconds
Stage 53, Epoch 76/100, Loss 1.0139, Duration: 0.002 seconds
Stage 53, Epoch 100/100, Loss 1.2687, Duration: 0.002 seconds
--- Training Stage 53 Finished ---

--- Starting Training Stage 54/100 ---
Starting training for 100 epochs in Stage 54...
Stage 54, Epoch 1/100, Loss 1.1816, Duration: 0.003 seconds
Stage 54, Epoch 26/100, Loss 1.1031, Duration: 0.003 seconds
Stage 54, Epoch 51/100, Loss 1.1879, Duration: 0.003 seconds
Stage 54, Epoch 76/100, Loss 1.2236, Duration: 0.003 seconds
Stage 54, Epoch 100/100, Loss 1.1013, Duration: 0.003 seconds
--- Training Stage 54 Finished ---

--- Starting Training Stage 55/100 ---
Starting training for 100 epochs in Stage 55...
Stage 55, Epoch 1/100, Loss 1.1052, Duration: 0.004 seconds
Stage 55, Epoch 26/100, Loss 1.1045, Duration: 0.003 seconds
Stage 55, Epoch 51/100, Loss 1.2735, Duration: 0.003 seconds
Stage 55, Epoch 76/100, Loss 0.9663, Duration: 0.003 seconds
Stage 55, Epoch 100/100, Loss 0.9137, Duration: 0.002 seconds
--- Training Stage 55 Finished ---

--- Starting Training Stage 56/100 ---
Starting training for 100 epochs in Stage 56...
Stage 56, Epoch 1/100, Loss 1.1054, Duration: 0.003 seconds
Stage 56, Epoch 26/100, Loss 1.2151, Duration: 0.003 seconds
Stage 56, Epoch 51/100, Loss 1.0327, Duration: 0.004 seconds
Stage 56, Epoch 76/100, Loss 0.9886, Duration: 0.002 seconds
Stage 56, Epoch 100/100, Loss 0.9629, Duration: 0.002 seconds
--- Training Stage 56 Finished ---

--- Starting Training Stage 57/100 ---
Starting training for 100 epochs in Stage 57...
Stage 57, Epoch 1/100, Loss 1.0562, Duration: 0.004 seconds
Stage 57, Epoch 26/100, Loss 0.8944, Duration: 0.002 seconds
Stage 57, Epoch 51/100, Loss 1.0247, Duration: 0.002 seconds
Stage 57, Epoch 76/100, Loss 0.9878, Duration: 0.003 seconds
Stage 57, Epoch 100/100, Loss 1.1586, Duration: 0.003 seconds
--- Training Stage 57 Finished ---

--- Starting Training Stage 58/100 ---
Starting training for 100 epochs in Stage 58...
Stage 58, Epoch 1/100, Loss 0.9241, Duration: 0.006 seconds
Stage 58, Epoch 26/100, Loss 1.2221, Duration: 0.003 seconds
Stage 58, Epoch 51/100, Loss 1.1712, Duration: 0.003 seconds
Stage 58, Epoch 76/100, Loss 1.0333, Duration: 0.002 seconds
Stage 58, Epoch 100/100, Loss 0.9037, Duration: 0.003 seconds
--- Training Stage 58 Finished ---

--- Starting Training Stage 59/100 ---
Starting training for 100 epochs in Stage 59...
Stage 59, Epoch 1/100, Loss 0.8417, Duration: 0.004 seconds
Stage 59, Epoch 26/100, Loss 0.8523, Duration: 0.002 seconds
Stage 59, Epoch 51/100, Loss 1.0005, Duration: 0.002 seconds
Stage 59, Epoch 76/100, Loss 1.0878, Duration: 0.002 seconds
Stage 59, Epoch 100/100, Loss 0.9782, Duration: 0.002 seconds
--- Training Stage 59 Finished ---

--- Starting Training Stage 60/100 ---
Starting training for 100 epochs in Stage 60...
Stage 60, Epoch 1/100, Loss 1.1152, Duration: 0.004 seconds
Stage 60, Epoch 26/100, Loss 1.1228, Duration: 0.003 seconds
Stage 60, Epoch 51/100, Loss 1.0208, Duration: 0.003 seconds
Stage 60, Epoch 76/100, Loss 0.8460, Duration: 0.003 seconds
Stage 60, Epoch 100/100, Loss 1.0965, Duration: 0.002 seconds
--- Training Stage 60 Finished ---

--- Starting Training Stage 61/100 ---
Starting training for 100 epochs in Stage 61...
Stage 61, Epoch 1/100, Loss 0.7779, Duration: 0.004 seconds
Stage 61, Epoch 26/100, Loss 0.9977, Duration: 0.002 seconds
Stage 61, Epoch 51/100, Loss 0.9713, Duration: 0.002 seconds
Stage 61, Epoch 76/100, Loss 1.0795, Duration: 0.003 seconds
Stage 61, Epoch 100/100, Loss 0.9050, Duration: 0.002 seconds
--- Training Stage 61 Finished ---

--- Starting Training Stage 62/100 ---
Starting training for 100 epochs in Stage 62...
Stage 62, Epoch 1/100, Loss 0.9102, Duration: 0.004 seconds
Stage 62, Epoch 26/100, Loss 0.9503, Duration: 0.003 seconds
Stage 62, Epoch 51/100, Loss 1.1472, Duration: 0.005 seconds
Stage 62, Epoch 76/100, Loss 0.9060, Duration: 0.002 seconds
Stage 62, Epoch 100/100, Loss 1.0246, Duration: 0.004 seconds
--- Training Stage 62 Finished ---

--- Starting Training Stage 63/100 ---
Starting training for 100 epochs in Stage 63...
Stage 63, Epoch 1/100, Loss 0.8021, Duration: 0.004 seconds
Stage 63, Epoch 26/100, Loss 0.9682, Duration: 0.003 seconds
Stage 63, Epoch 51/100, Loss 0.7711, Duration: 0.002 seconds
Stage 63, Epoch 76/100, Loss 0.9151, Duration: 0.002 seconds
Stage 63, Epoch 100/100, Loss 0.8995, Duration: 0.003 seconds
--- Training Stage 63 Finished ---

--- Starting Training Stage 64/100 ---
Starting training for 100 epochs in Stage 64...
Stage 64, Epoch 1/100, Loss 1.1454, Duration: 0.003 seconds
Stage 64, Epoch 26/100, Loss 1.0564, Duration: 0.002 seconds
Stage 64, Epoch 51/100, Loss 1.2395, Duration: 0.003 seconds
Stage 64, Epoch 76/100, Loss 0.8394, Duration: 0.003 seconds
Stage 64, Epoch 100/100, Loss 0.8420, Duration: 0.002 seconds
--- Training Stage 64 Finished ---

--- Starting Training Stage 65/100 ---
Starting training for 100 epochs in Stage 65...
Stage 65, Epoch 1/100, Loss 0.9956, Duration: 0.004 seconds
Stage 65, Epoch 26/100, Loss 1.0140, Duration: 0.002 seconds
Stage 65, Epoch 51/100, Loss 0.9301, Duration: 0.003 seconds
Stage 65, Epoch 76/100, Loss 0.9633, Duration: 0.002 seconds
Stage 65, Epoch 100/100, Loss 1.1081, Duration: 0.002 seconds
--- Training Stage 65 Finished ---

--- Starting Training Stage 66/100 ---
Starting training for 100 epochs in Stage 66...
Stage 66, Epoch 1/100, Loss 0.9914, Duration: 0.004 seconds
Stage 66, Epoch 26/100, Loss 0.7331, Duration: 0.003 seconds
Stage 66, Epoch 51/100, Loss 1.0723, Duration: 0.002 seconds
Stage 66, Epoch 76/100, Loss 0.8925, Duration: 0.002 seconds
Stage 66, Epoch 100/100, Loss 0.8882, Duration: 0.002 seconds
--- Training Stage 66 Finished ---

--- Starting Training Stage 67/100 ---
Starting training for 100 epochs in Stage 67...
Stage 67, Epoch 1/100, Loss 0.7993, Duration: 0.003 seconds
Stage 67, Epoch 26/100, Loss 0.9689, Duration: 0.003 seconds
Stage 67, Epoch 51/100, Loss 0.8838, Duration: 0.003 seconds
Stage 67, Epoch 76/100, Loss 0.8225, Duration: 0.003 seconds
Stage 67, Epoch 100/100, Loss 0.9596, Duration: 0.002 seconds
--- Training Stage 67 Finished ---

--- Starting Training Stage 68/100 ---
Starting training for 100 epochs in Stage 68...
Stage 68, Epoch 1/100, Loss 0.7505, Duration: 0.003 seconds
Stage 68, Epoch 26/100, Loss 0.7479, Duration: 0.003 seconds
Stage 68, Epoch 51/100, Loss 0.7654, Duration: 0.003 seconds
Stage 68, Epoch 76/100, Loss 1.0365, Duration: 0.003 seconds
Stage 68, Epoch 100/100, Loss 0.7742, Duration: 0.002 seconds
--- Training Stage 68 Finished ---

--- Starting Training Stage 69/100 ---
Starting training for 100 epochs in Stage 69...
Stage 69, Epoch 1/100, Loss 0.9507, Duration: 0.004 seconds
Stage 69, Epoch 26/100, Loss 1.0775, Duration: 0.003 seconds
Stage 69, Epoch 51/100, Loss 0.9052, Duration: 0.004 seconds
Stage 69, Epoch 76/100, Loss 1.0861, Duration: 0.003 seconds
Stage 69, Epoch 100/100, Loss 0.9455, Duration: 0.002 seconds
--- Training Stage 69 Finished ---

--- Starting Training Stage 70/100 ---
Starting training for 100 epochs in Stage 70...
Stage 70, Epoch 1/100, Loss 0.7230, Duration: 0.004 seconds
Stage 70, Epoch 26/100, Loss 0.9819, Duration: 0.002 seconds
Stage 70, Epoch 51/100, Loss 0.6787, Duration: 0.003 seconds
Stage 70, Epoch 76/100, Loss 0.8629, Duration: 0.002 seconds
Stage 70, Epoch 100/100, Loss 0.8605, Duration: 0.003 seconds
--- Training Stage 70 Finished ---

--- Starting Training Stage 71/100 ---
Starting training for 100 epochs in Stage 71...
Stage 71, Epoch 1/100, Loss 1.1860, Duration: 0.003 seconds
Stage 71, Epoch 26/100, Loss 0.7476, Duration: 0.002 seconds
Stage 71, Epoch 51/100, Loss 0.6152, Duration: 0.002 seconds
Stage 71, Epoch 76/100, Loss 0.8038, Duration: 0.002 seconds
Stage 71, Epoch 100/100, Loss 0.7840, Duration: 0.003 seconds
--- Training Stage 71 Finished ---

--- Starting Training Stage 72/100 ---
Starting training for 100 epochs in Stage 72...
Stage 72, Epoch 1/100, Loss 0.9925, Duration: 0.004 seconds
Stage 72, Epoch 26/100, Loss 0.9159, Duration: 0.002 seconds
Stage 72, Epoch 51/100, Loss 0.8461, Duration: 0.003 seconds
Stage 72, Epoch 76/100, Loss 0.8218, Duration: 0.003 seconds
Stage 72, Epoch 100/100, Loss 0.9785, Duration: 0.003 seconds
--- Training Stage 72 Finished ---

--- Starting Training Stage 73/100 ---
Starting training for 100 epochs in Stage 73...
Stage 73, Epoch 1/100, Loss 0.9609, Duration: 0.004 seconds
Stage 73, Epoch 26/100, Loss 0.8320, Duration: 0.002 seconds
Stage 73, Epoch 51/100, Loss 0.8642, Duration: 0.002 seconds
Stage 73, Epoch 76/100, Loss 0.8882, Duration: 0.003 seconds
Stage 73, Epoch 100/100, Loss 0.8126, Duration: 0.002 seconds
--- Training Stage 73 Finished ---

--- Starting Training Stage 74/100 ---
Starting training for 100 epochs in Stage 74...
Stage 74, Epoch 1/100, Loss 0.9380, Duration: 0.004 seconds
Stage 74, Epoch 26/100, Loss 0.6476, Duration: 0.003 seconds
Stage 74, Epoch 51/100, Loss 0.6923, Duration: 0.002 seconds
Stage 74, Epoch 76/100, Loss 0.8347, Duration: 0.002 seconds
Stage 74, Epoch 100/100, Loss 0.7464, Duration: 0.002 seconds
--- Training Stage 74 Finished ---

--- Starting Training Stage 75/100 ---
Starting training for 100 epochs in Stage 75...
Stage 75, Epoch 1/100, Loss 0.6347, Duration: 0.006 seconds
Stage 75, Epoch 26/100, Loss 0.7291, Duration: 0.002 seconds
Stage 75, Epoch 51/100, Loss 0.8634, Duration: 0.003 seconds
Stage 75, Epoch 76/100, Loss 0.8987, Duration: 0.002 seconds
Stage 75, Epoch 100/100, Loss 0.8433, Duration: 0.002 seconds
--- Training Stage 75 Finished ---

--- Starting Training Stage 76/100 ---
Starting training for 100 epochs in Stage 76...
Stage 76, Epoch 1/100, Loss 0.7980, Duration: 0.003 seconds
Stage 76, Epoch 26/100, Loss 0.7513, Duration: 0.002 seconds
Stage 76, Epoch 51/100, Loss 0.9233, Duration: 0.003 seconds
Stage 76, Epoch 76/100, Loss 0.9166, Duration: 0.003 seconds
Stage 76, Epoch 100/100, Loss 0.6103, Duration: 0.002 seconds
--- Training Stage 76 Finished ---

--- Starting Training Stage 77/100 ---
Starting training for 100 epochs in Stage 77...
Stage 77, Epoch 1/100, Loss 0.9035, Duration: 0.004 seconds
Stage 77, Epoch 26/100, Loss 0.7092, Duration: 0.003 seconds
Stage 77, Epoch 51/100, Loss 0.8580, Duration: 0.002 seconds
Stage 77, Epoch 76/100, Loss 0.8750, Duration: 0.003 seconds
Stage 77, Epoch 100/100, Loss 0.6726, Duration: 0.002 seconds
--- Training Stage 77 Finished ---

--- Starting Training Stage 78/100 ---
Starting training for 100 epochs in Stage 78...
Stage 78, Epoch 1/100, Loss 1.1808, Duration: 0.004 seconds
Stage 78, Epoch 26/100, Loss 0.7814, Duration: 0.002 seconds
Stage 78, Epoch 51/100, Loss 1.0092, Duration: 0.003 seconds
Stage 78, Epoch 76/100, Loss 1.1207, Duration: 0.002 seconds
Stage 78, Epoch 100/100, Loss 0.7470, Duration: 0.003 seconds
--- Training Stage 78 Finished ---

--- Starting Training Stage 79/100 ---
Starting training for 100 epochs in Stage 79...
Stage 79, Epoch 1/100, Loss 1.0459, Duration: 0.003 seconds
Stage 79, Epoch 26/100, Loss 0.7753, Duration: 0.003 seconds
Stage 79, Epoch 51/100, Loss 0.9772, Duration: 0.002 seconds
Stage 79, Epoch 76/100, Loss 0.9067, Duration: 0.002 seconds
Stage 79, Epoch 100/100, Loss 0.7003, Duration: 0.002 seconds
--- Training Stage 79 Finished ---

--- Starting Training Stage 80/100 ---
Starting training for 100 epochs in Stage 80...
Stage 80, Epoch 1/100, Loss 0.9512, Duration: 0.004 seconds
Stage 80, Epoch 26/100, Loss 1.0199, Duration: 0.002 seconds
Stage 80, Epoch 51/100, Loss 0.7199, Duration: 0.004 seconds
Stage 80, Epoch 76/100, Loss 0.8037, Duration: 0.003 seconds
Stage 80, Epoch 100/100, Loss 0.7177, Duration: 0.002 seconds
--- Training Stage 80 Finished ---

--- Starting Training Stage 81/100 ---
Starting training for 100 epochs in Stage 81...
Stage 81, Epoch 1/100, Loss 0.6355, Duration: 0.005 seconds
Stage 81, Epoch 26/100, Loss 0.8803, Duration: 0.003 seconds
Stage 81, Epoch 51/100, Loss 0.9368, Duration: 0.005 seconds
Stage 81, Epoch 76/100, Loss 0.6043, Duration: 0.003 seconds
Stage 81, Epoch 100/100, Loss 0.8750, Duration: 0.003 seconds
--- Training Stage 81 Finished ---

--- Starting Training Stage 82/100 ---
Starting training for 100 epochs in Stage 82...
Stage 82, Epoch 1/100, Loss 0.8460, Duration: 0.004 seconds
Stage 82, Epoch 26/100, Loss 0.9914, Duration: 0.002 seconds
Stage 82, Epoch 51/100, Loss 0.5835, Duration: 0.003 seconds
Stage 82, Epoch 76/100, Loss 0.7259, Duration: 0.002 seconds
Stage 82, Epoch 100/100, Loss 0.6950, Duration: 0.003 seconds
--- Training Stage 82 Finished ---

--- Starting Training Stage 83/100 ---
Starting training for 100 epochs in Stage 83...
Stage 83, Epoch 1/100, Loss 0.6668, Duration: 0.003 seconds
Stage 83, Epoch 26/100, Loss 0.6104, Duration: 0.003 seconds
Stage 83, Epoch 51/100, Loss 0.6782, Duration: 0.004 seconds
Stage 83, Epoch 76/100, Loss 0.8790, Duration: 0.002 seconds
Stage 83, Epoch 100/100, Loss 0.5617, Duration: 0.002 seconds
--- Training Stage 83 Finished ---

--- Starting Training Stage 84/100 ---
Starting training for 100 epochs in Stage 84...
Stage 84, Epoch 1/100, Loss 0.8128, Duration: 0.004 seconds
Stage 84, Epoch 26/100, Loss 0.8258, Duration: 0.002 seconds
Stage 84, Epoch 51/100, Loss 0.7890, Duration: 0.003 seconds
Stage 84, Epoch 76/100, Loss 0.7763, Duration: 0.002 seconds
Stage 84, Epoch 100/100, Loss 0.8197, Duration: 0.002 seconds
--- Training Stage 84 Finished ---

--- Starting Training Stage 85/100 ---
Starting training for 100 epochs in Stage 85...
Stage 85, Epoch 1/100, Loss 0.7616, Duration: 0.003 seconds
Stage 85, Epoch 26/100, Loss 0.6597, Duration: 0.002 seconds
Stage 85, Epoch 51/100, Loss 0.6651, Duration: 0.002 seconds
Stage 85, Epoch 76/100, Loss 0.8282, Duration: 0.002 seconds
Stage 85, Epoch 100/100, Loss 0.5758, Duration: 0.002 seconds
--- Training Stage 85 Finished ---

--- Starting Training Stage 86/100 ---
Starting training for 100 epochs in Stage 86...
Stage 86, Epoch 1/100, Loss 0.8701, Duration: 0.005 seconds
Stage 86, Epoch 26/100, Loss 0.8364, Duration: 0.003 seconds
Stage 86, Epoch 51/100, Loss 0.6267, Duration: 0.002 seconds
Stage 86, Epoch 76/100, Loss 0.7699, Duration: 0.002 seconds
Stage 86, Epoch 100/100, Loss 0.6242, Duration: 0.003 seconds
--- Training Stage 86 Finished ---

--- Starting Training Stage 87/100 ---
Starting training for 100 epochs in Stage 87...
Stage 87, Epoch 1/100, Loss 0.9622, Duration: 0.003 seconds
Stage 87, Epoch 26/100, Loss 0.7726, Duration: 0.002 seconds
Stage 87, Epoch 51/100, Loss 0.8412, Duration: 0.003 seconds
Stage 87, Epoch 76/100, Loss 0.6145, Duration: 0.003 seconds
Stage 87, Epoch 100/100, Loss 0.8440, Duration: 0.004 seconds
--- Training Stage 87 Finished ---

--- Starting Training Stage 88/100 ---
Starting training for 100 epochs in Stage 88...
Stage 88, Epoch 1/100, Loss 1.1058, Duration: 0.004 seconds
Stage 88, Epoch 26/100, Loss 0.8254, Duration: 0.002 seconds
Stage 88, Epoch 51/100, Loss 0.7006, Duration: 0.002 seconds
Stage 88, Epoch 76/100, Loss 0.7314, Duration: 0.003 seconds
Stage 88, Epoch 100/100, Loss 0.7462, Duration: 0.004 seconds
--- Training Stage 88 Finished ---

--- Starting Training Stage 89/100 ---
Starting training for 100 epochs in Stage 89...
Stage 89, Epoch 1/100, Loss 0.7393, Duration: 0.005 seconds
Stage 89, Epoch 26/100, Loss 0.6506, Duration: 0.002 seconds
Stage 89, Epoch 51/100, Loss 1.2356, Duration: 0.004 seconds
Stage 89, Epoch 76/100, Loss 0.7371, Duration: 0.004 seconds
Stage 89, Epoch 100/100, Loss 0.6760, Duration: 0.003 seconds
--- Training Stage 89 Finished ---

--- Starting Training Stage 90/100 ---
Starting training for 100 epochs in Stage 90...
Stage 90, Epoch 1/100, Loss 0.5744, Duration: 0.003 seconds
Stage 90, Epoch 26/100, Loss 0.7195, Duration: 0.003 seconds
Stage 90, Epoch 51/100, Loss 0.6197, Duration: 0.003 seconds
Stage 90, Epoch 76/100, Loss 0.6753, Duration: 0.003 seconds
Stage 90, Epoch 100/100, Loss 0.6825, Duration: 0.003 seconds
--- Training Stage 90 Finished ---

--- Starting Training Stage 91/100 ---
Starting training for 100 epochs in Stage 91...
Stage 91, Epoch 1/100, Loss 0.4894, Duration: 0.004 seconds
Stage 91, Epoch 26/100, Loss 0.8694, Duration: 0.003 seconds
Stage 91, Epoch 51/100, Loss 0.7663, Duration: 0.003 seconds
Stage 91, Epoch 76/100, Loss 0.7432, Duration: 0.002 seconds
Stage 91, Epoch 100/100, Loss 0.9327, Duration: 0.004 seconds
--- Training Stage 91 Finished ---

--- Starting Training Stage 92/100 ---
Starting training for 100 epochs in Stage 92...
Stage 92, Epoch 1/100, Loss 0.6738, Duration: 0.006 seconds
Stage 92, Epoch 26/100, Loss 0.7904, Duration: 0.004 seconds
Stage 92, Epoch 51/100, Loss 0.8549, Duration: 0.003 seconds
Stage 92, Epoch 76/100, Loss 0.9240, Duration: 0.003 seconds
Stage 92, Epoch 100/100, Loss 0.6929, Duration: 0.002 seconds
--- Training Stage 92 Finished ---

--- Starting Training Stage 93/100 ---
Starting training for 100 epochs in Stage 93...
Stage 93, Epoch 1/100, Loss 0.6360, Duration: 0.004 seconds
Stage 93, Epoch 26/100, Loss 0.5990, Duration: 0.006 seconds
Stage 93, Epoch 51/100, Loss 1.1910, Duration: 0.003 seconds
Stage 93, Epoch 76/100, Loss 0.6742, Duration: 0.002 seconds
Stage 93, Epoch 100/100, Loss 0.6734, Duration: 0.005 seconds
--- Training Stage 93 Finished ---

--- Starting Training Stage 94/100 ---
Starting training for 100 epochs in Stage 94...
Stage 94, Epoch 1/100, Loss 1.0005, Duration: 0.006 seconds
Stage 94, Epoch 26/100, Loss 0.6070, Duration: 0.003 seconds
Stage 94, Epoch 51/100, Loss 0.7327, Duration: 0.002 seconds
Stage 94, Epoch 76/100, Loss 0.6963, Duration: 0.003 seconds
Stage 94, Epoch 100/100, Loss 0.7826, Duration: 0.003 seconds
--- Training Stage 94 Finished ---

--- Starting Training Stage 95/100 ---
Starting training for 100 epochs in Stage 95...
Stage 95, Epoch 1/100, Loss 0.7485, Duration: 0.004 seconds
Stage 95, Epoch 26/100, Loss 0.6378, Duration: 0.003 seconds
Stage 95, Epoch 51/100, Loss 0.5081, Duration: 0.003 seconds
Stage 95, Epoch 76/100, Loss 0.6533, Duration: 0.003 seconds
Stage 95, Epoch 100/100, Loss 0.6090, Duration: 0.003 seconds
--- Training Stage 95 Finished ---

--- Starting Training Stage 96/100 ---
Starting training for 100 epochs in Stage 96...
Stage 96, Epoch 1/100, Loss 0.8110, Duration: 0.006 seconds
Stage 96, Epoch 26/100, Loss 0.9018, Duration: 0.002 seconds
Stage 96, Epoch 51/100, Loss 0.5374, Duration: 0.002 seconds
Stage 96, Epoch 76/100, Loss 0.4849, Duration: 0.003 seconds
Stage 96, Epoch 100/100, Loss 0.8664, Duration: 0.002 seconds
--- Training Stage 96 Finished ---

--- Starting Training Stage 97/100 ---
Starting training for 100 epochs in Stage 97...
Stage 97, Epoch 1/100, Loss 0.7843, Duration: 0.003 seconds
Stage 97, Epoch 26/100, Loss 0.9393, Duration: 0.003 seconds
Stage 97, Epoch 51/100, Loss 0.5245, Duration: 0.003 seconds
Stage 97, Epoch 76/100, Loss 0.7522, Duration: 0.003 seconds
Stage 97, Epoch 100/100, Loss 0.7285, Duration: 0.005 seconds
--- Training Stage 97 Finished ---

--- Starting Training Stage 98/100 ---
Starting training for 100 epochs in Stage 98...
Stage 98, Epoch 1/100, Loss 0.5363, Duration: 0.004 seconds
Stage 98, Epoch 26/100, Loss 0.5192, Duration: 0.004 seconds
Stage 98, Epoch 51/100, Loss 0.9856, Duration: 0.003 seconds
Stage 98, Epoch 76/100, Loss 0.6197, Duration: 0.002 seconds
Stage 98, Epoch 100/100, Loss 0.8378, Duration: 0.003 seconds
--- Training Stage 98 Finished ---

--- Starting Training Stage 99/100 ---
Starting training for 100 epochs in Stage 99...
Stage 99, Epoch 1/100, Loss 0.5563, Duration: 0.003 seconds
Stage 99, Epoch 26/100, Loss 0.8230, Duration: 0.003 seconds
Stage 99, Epoch 51/100, Loss 0.5731, Duration: 0.003 seconds
Stage 99, Epoch 76/100, Loss 0.6922, Duration: 0.003 seconds
Stage 99, Epoch 100/100, Loss 0.6312, Duration: 0.003 seconds
--- Training Stage 99 Finished ---

--- Starting Training Stage 100/100 ---
Starting training for 100 epochs in Stage 100...
Stage 100, Epoch 1/100, Loss 0.5973, Duration: 0.004 seconds
Stage 100, Epoch 26/100, Loss 0.7274, Duration: 0.002 seconds
Stage 100, Epoch 51/100, Loss 0.7778, Duration: 0.003 seconds
Stage 100, Epoch 76/100, Loss 0.5001, Duration: 0.002 seconds
Stage 100, Epoch 100/100, Loss 0.5435, Duration: 0.002 seconds
--- Training Stage 100 Finished ---

All training stages finished.
'''