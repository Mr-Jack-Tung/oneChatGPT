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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    single_block_model = None
    tokenizer = None
    config = None

    # Define the training data (using the single QA pair from the original script)
    qa_pair = 'Question: Xin chào \nAnswer: Công ty BICweb kính chào quý khách!.'

    # Initialize and train the character tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.train(qa_pair)
    print(f"Character tokenizer trained with vocabulary size: {tokenizer.vocab_size}")

    # Define model configuration (defaulting to small, but using tokenizer's vocab size)
    config = GPT2Config(model_type="small", vocab_size=tokenizer.vocab_size)

    # Check if a trained model already exists
    if os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
        print(f"Found existing trained model directory: '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'. Loading model...")
        try:
            # Load tokenizer from the trained model directory
            tokenizer = CharacterTokenizer.from_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)
            print("Character tokenizer loaded successfully.")

            # Define model configuration again to ensure vocab size is correct after loading tokenizer
            config = GPT2Config(model_type="small", vocab_size=tokenizer.vocab_size)

            # Instantiate the single block model with the custom config
            single_block_model = SingleBlockGPT2ModelNoDepend(config)

            # Load the trained state dictionary
            state_dict_path = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'
            single_block_model.load_state_dict(torch.load(state_dict_path))
            print("Trained single block model loaded successfully.")

        except Exception as e:
            print(f"Error loading existing trained model: {e}")
            print("Proceeding with initializing a new model for initial training.")
            single_block_model = None # Reset to None to trigger initial training path

    if single_block_model is None:
        print("No existing trained model found or failed to load. Initializing a new model for initial training.")
        # Initialize a new model with random weights
        try:
            single_block_model = SingleBlockGPT2ModelNoDepend(config)
            print("New single block model initialized with random weights.")

        except Exception as e:
            print(f"Error initializing new model: {e}")
            print("Cannot proceed with training. Exiting.")
            exit()

    print(single_block_model)
    print(f"Number of trainable params: {sum(p.numel() for p in single_block_model.parameters() if p.requires_grad):,d}")

    # Set model to training mode
    single_block_model.to(device)
    single_block_model.train()

    # Define the optimizer (removed learning rate scheduler)
    optimizer = torch.optim.AdamW(single_block_model.parameters(), lr=3e-4) # Further reduced learning rate

    # Encode the training data using the character tokenizer
    input_ids = torch.tensor(tokenizer.encode(qa_pair), dtype=torch.long).unsqueeze(0).to(device)
    print(f"\nTraining data: {qa_pair}")
    print(f"Encoded input_ids shape: {input_ids.shape}")

    # Training loop
    num_epochs = 30 # Reverted epochs based on user feedback that low loss was achieved
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

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{num_epochs}, Loss {loss.item():.4f}, Duration: {epoch_duration:.3f} seconds")

    print("\nTraining finished.")

    # Save the trained single block model
    print(f"Saving the trained single block model to '{TRAINED_SINGLE_BLOCK_MODEL_PATH}'...")
    try:
        # Create the directory if it doesn't exist
        if not os.path.exists(TRAINED_SINGLE_BLOCK_MODEL_PATH):
            os.makedirs(TRAINED_SINGLE_BLOCK_MODEL_PATH)

        # Save the model's state dictionary
        torch.save(single_block_model.state_dict(), f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth')

        # Save the character tokenizer
        tokenizer.save_pretrained(TRAINED_SINGLE_BLOCK_MODEL_PATH)

        print("Trained single block model and character tokenizer saved.")
    except Exception as e:
        print(f"Error saving trained model and tokenizer: {e}")

    # Clean up
    del single_block_model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
