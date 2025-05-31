# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 31 May 2025

import torch
import torch.nn as nn
# Import the custom tokenizer
from character_tokenizer import CharacterTokenizer

# Import the custom model and config
from single_block_gpt2_35k_params_model import SingleBlockGPT2ModelNoDepend, GPT2Config

# Define the path to the trained single block model
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'TrainedSingleBlockGPT2_35k_params'
SINGLE_BLOCK_STATE_DICT_PATH = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'

def load_trained_single_block_model(model_path, state_dict_path):
    """Loads the trained SingleBlockGPT2ModelNoDepend."""
    try:
        # Load the character tokenizer saved with the trained model
        tokenizer = CharacterTokenizer.from_pretrained(model_path)
        print(f"Character tokenizer loaded from {model_path}")
    except Exception as e:
        print(f"Error loading character tokenizer from trained model path: {e}")
        print(f"Please ensure the trained model is saved in the '{model_path}' directory.")
        return None, None

    # Define model configuration using the custom GPT2Config with the loaded tokenizer's vocab size
    # Defaulting to small model type, but vocab size is from the loaded tokenizer
    config = GPT2Config(model_type="small", vocab_size=tokenizer.vocab_size)

    # Instantiate the single block model with the custom config
    single_block_model = SingleBlockGPT2ModelNoDepend(config)

    # Load the trained state dictionary
    try:
        single_block_model.load_state_dict(torch.load(state_dict_path))
        print(f"Trained model state dictionary loaded from {state_dict_path}")
    except Exception as e:
        print(f"Error loading trained model state dictionary: {e}")
        print(f"Please ensure '{state_dict_path}' exists.")
        return None, None

    return single_block_model, tokenizer

def generate_text(model, tokenizer, text, max_length, temperature, top_k, top_p, device):
    """Generates text using the model."""
    model.eval()
    # Encode the input text using the character tokenizer
    input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(device)

    # Use the generate method of the SingleBlockGPT2ModelNoDepend
    # Removed eos_token_id to prevent premature stopping
    generated_ids = model.generate(
        input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device
    )

    # Decode the generated sequence using the character tokenizer
    generated_text = tokenizer.decode(generated_ids[0].tolist()) # Decode expects a list
    # Removed post-processing stopping logic to ensure full generation up to max_length

    return generated_text

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained single block model and tokenizer
    single_block_model, tokenizer = load_trained_single_block_model(TRAINED_SINGLE_BLOCK_MODEL_PATH, SINGLE_BLOCK_STATE_DICT_PATH)

    print(single_block_model)

    if single_block_model is not None and tokenizer is not None:
        single_block_model.to(device)

        # Example inference
        input_text = "Question: Xin chào"
        max_gen_length = 64 # Set max generation length to the length of the training data
        temperature = 0.01 # Reduced temperature for near-greedy decoding
        top_k = 1 # tokenizer.vocab_size # Set top_k to vocabulary size
        top_p = 0.95 # Set top_p to 0.95 (near-greedy sampling)

        print(f"\nInput text: '{input_text}'")
        print(f"Generating exactly {max_gen_length} tokens...") # Updated print message

        # Generate text
        generated_text = generate_text(
            single_block_model,
            tokenizer,
            input_text,
            max_gen_length,
            temperature,
            top_k,
            top_p,
            device=device # Explicitly pass device
        )

        print(f"Generated text: '{generated_text}'")

        # You can add more inference examples here

        # Clean up
        del single_block_model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("Failed to load the trained single block model. Inference cannot be performed.")

'''
% uv run Single_Block_GPT2_35k_params/inference_single_block_gpt2_35k_params.py
Character tokenizer loaded from TrainedSingleBlockGPT2_35k_params
Trained model state dictionary loaded from TrainedSingleBlockGPT2_35k_params/single_block_model_state_dict.pth
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

Input text: 'Question: Xin chào'
Generating exactly 64 tokens...
Generated text: 'Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!.'
'''