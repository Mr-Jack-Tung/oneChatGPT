import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Config
from single_block_gpt2_model import SingleBlockGPT2Model # Import the custom model class

# Define the path to the trained single block model
TRAINED_SINGLE_BLOCK_MODEL_PATH = 'TrainedSingleBlockGPT2'
SINGLE_BLOCK_STATE_DICT_PATH = f'{TRAINED_SINGLE_BLOCK_MODEL_PATH}/single_block_model_state_dict.pth'

def load_trained_single_block_model(model_path, state_dict_path):
    """Loads the trained SingleBlockGPT2Model."""
    try:
        # Load the config and tokenizer saved with the trained model
        config = GPT2Config.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        print(f"Config and tokenizer loaded from {model_path}")
    except Exception as e:
        print(f"Error loading config and tokenizer from trained model path: {e}")
        print(f"Please ensure the trained model is saved in the '{model_path}' directory.")
        return None, None

    # Instantiate the single block model
    single_block_model = SingleBlockGPT2Model(config)

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
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(device)

    # Use the generate method of the SingleBlockGPT2Model
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

    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
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
        temperature = 0.7
        top_k = 50
        top_p = 0.9

        print(f"\nInput text: '{input_text}'")
        print(f"Generating up to {max_gen_length} tokens...")

        generated_text = generate_text(
            single_block_model,
            tokenizer,
            input_text,
            max_gen_length,
            temperature,
            top_k,
            top_p,
            device
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
% uv run inference_single_block_gpt2.py
Config and tokenizer loaded from TrainedSingleBlockGPT2
Trained model state dictionary loaded from TrainedSingleBlockGPT2/single_block_model_state_dict.pth
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

Input text: 'Question: Xin chào'
Generating up to 50 tokens...
Generated text: 'Question: Xin chào 
Answer: Công ty BICweb kính chào quý khách!'
'''