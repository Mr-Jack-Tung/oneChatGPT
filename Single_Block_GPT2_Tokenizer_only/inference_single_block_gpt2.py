# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 31 May 2025

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer # Use transformers tokenizer

# Import the custom model and config
from single_block_gpt2_model import SingleBlockGPT2Model, GPT2Config # Import custom model and config

# Define the path to the trained single block model
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'TrainedSingleBlockGPT2_Tokenizer_only'
SINGLE_BLOCK_STATE_DICT_PATH = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'

def load_trained_single_block_model(model_path, state_dict_path):
    """Loads the trained custom SingleBlockGPT2Model."""
    try:
        # Load the tokenizer saved with the trained model (transformers tokenizer)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        print(f"Transformers GPT2Tokenizer loaded from {model_path}")
    except Exception as e:
        print(f"Error loading transformers GPT2Tokenizer from trained model path: {e}")
        print(f"Please ensure the trained model is saved in the '{model_path}' directory.")
        return None, None

    # Define model configuration using the custom GPT2Config and the loaded tokenizer's vocab size
    config = GPT2Config(vocab_size=tokenizer.vocab_size) # Use custom config with transformers vocab size

    # Instantiate the custom single block model with the custom config
    single_block_model = SingleBlockGPT2Model(config)

    # Load the trained state dictionary
    try:
        single_block_model.load_state_dict(torch.load(state_dict_path))
        print(f"Trained custom model state dictionary loaded from {state_dict_path}")
    except Exception as e:
        print(f"Error loading trained model state dictionary: {e}")
        print(f"Please ensure '{state_dict_path}' exists.")
        return None, None

    return single_block_model, tokenizer

def generate_text(model, tokenizer, text, max_length, temperature=1.0, top_k=None, top_p=None, device='cpu'):
    """Generates text using the model."""
    model.eval()
    # Encode the input text using the transformers tokenizer
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(device)

    # Use the generate method of the custom SingleBlockGPT2Model
    generated_ids = model.generate(
        input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device
    )

    # Decode the generated sequence using the transformers tokenizer
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Simple way to stop generation at a common sentence end
    sentences = generated_text.split('.')
    return sentences[0]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained single block model and tokenizer
    single_block_model, tokenizer = load_trained_single_block_model(TRAINED_SINGLE_BLOCK_MODEL_PATH, SINGLE_BLOCK_STATE_DICT_PATH)

    print(single_block_model)

    if single_block_model is not None and tokenizer is not None:
        single_block_model.to(device)

        # Example inference
        input_text = "Question: Xin chào"
        max_gen_length = 50
        # Standard generation parameters for initial test
        temperature = 1.0
        top_k = None
        top_p = None

        print(f"\nInput text: '{input_text}'")
        print(f"Generating up to {max_gen_length} tokens...")

        # Generate text
        generated_text = generate_text(
            single_block_model,
            tokenizer,
            input_text,
            max_gen_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device
        )

        print(f"Generated text: '{generated_text}'")

        # You can add more inference examples here

        # Clean up
        del single_block_model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("Failed to load the trained custom single block model. Inference cannot be performed.")

'''
% uv run Single_Block_GPT2_Tokenizer_only/inference_single_block_gpt2.py
Transformers GPT2Tokenizer loaded from TrainedSingleBlockGPT2_Tokenizer_only
Trained custom model state dictionary loaded from TrainedSingleBlockGPT2_Tokenizer_only/single_block_model_state_dict.pth
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

Input text: 'Question: Xin chào'
Generating up to 50 tokens...
Generated text: 'Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!'
'''