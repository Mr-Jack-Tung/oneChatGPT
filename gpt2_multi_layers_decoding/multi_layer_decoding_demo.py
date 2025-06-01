# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 01 June 2025

# This script demonstrates multi-layer decoding generation with a GPT-2 model.
# It outputs the generation from each of the 12 layers to visualize the process.

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_from_layers(model, tokenizer, prompt, max_length=50):
    """
    Generates text from each layer of the GPT-2 model.

    Args:
        model: The GPT-2 model.
        tokenizer: The GPT-2 tokenizer.
        prompt: The input prompt string.
        max_length: The maximum length of the generated text.

    Returns:
        A dictionary where keys are layer indices and values are generated text.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Initialize a dictionary to store the generated token IDs for each layer
    layer_generated_ids = {i: input_ids.clone() for i in range(model.config.num_hidden_layers)}

    # Manually perform generation step by step
    # We need to generate max_length - input_ids.shape[1] tokens
    for _ in range(max_length - input_ids.shape[1]):
        with torch.no_grad():
            # We need to run the model for each layer's current sequence to get its hidden states
            # This is inefficient but necessary to show per-layer generation
            next_token_per_layer = {}
            for i in range(model.config.num_hidden_layers):
                current_input_ids = layer_generated_ids[i]
                outputs = model(current_input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states # hidden_states is a tuple of length num_layers + 1

                # Get the hidden states for the current layer (i)
                layer_hidden_states = hidden_states[i + 1] # hidden_states[0] is the input embedding

                # Get logits for the last token based on this layer's hidden states
                layer_logits = model.lm_head(layer_hidden_states[:, -1, :])

                # Get the predicted next token ID for this layer
                next_token_id = torch.argmax(layer_logits, axis=-1)
                next_token_per_layer[i] = next_token_id

            # Append the predicted next token to each layer's sequence
            for i in range(model.config.num_hidden_layers):
                next_token = next_token_per_layer[i].unsqueeze(-1)
                layer_generated_ids[i] = torch.cat([layer_generated_ids[i], next_token], dim=-1)

    # Decode the full generated sequence for each layer
    layer_full_generations = {}
    for i in range(model.config.num_hidden_layers):
        layer_full_generations[i] = tokenizer.decode(layer_generated_ids[i].squeeze(0), skip_special_tokens=True)

    return layer_full_generations

if __name__ == "__main__":
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Input prompt
    # prompt = "The quick brown fox jumps over the lazy"
    # prompt = "Once upon a time there was a little girl"
    prompt = "The computer is"

    print(f"Prompt: {prompt}\n")

    # Generate and get outputs from each layer
    layer_generations = generate_from_layers(model, tokenizer, prompt)

    # Print the output from each layer with coloring
    # We will compare each layer's output to the final layer's output (layer 11, index 11)
    final_layer_output = layer_generations[model.config.num_hidden_layers - 1]
    final_layer_tokens = tokenizer.encode(final_layer_output)

    for layer_idx in sorted(layer_generations.keys()):
        layer_output = layer_generations[layer_idx]
        layer_tokens = tokenizer.encode(layer_output)

        colored_output = []
        # Compare token by token with the final layer's output
        for i in range(max(len(layer_tokens), len(final_layer_tokens))):
            layer_token = layer_tokens[i] if i < len(layer_tokens) else None
            final_layer_token = final_layer_tokens[i] if i < len(final_layer_tokens) else None

            if layer_token is not None and final_layer_token is not None and layer_token == final_layer_token:
                # Tokens are the same, print normally
                colored_output.append(tokenizer.decode(layer_token))
            elif layer_token is not None:
                # Tokens are different or only in this layer, print in color (e.g., red)
                colored_output.append(f"\033[91m{tokenizer.decode(layer_token)}\033[0m") # Red color
            elif final_layer_token is not None:
                # Token is only in the final layer's output (shouldn't happen with current logic, but for completeness)
                 colored_output.append(f"\033[94m{tokenizer.decode(final_layer_token)}\033[0m") # Blue color

        print(f"Layer {layer_idx + 1}: {''.join(colored_output)}")

'''
% uv run gpt2_multi_layers_decoding/multi_layer_decoding_demo.py
Prompt: Once upon a time there was a little girl

Layer 1: Once upon a time there was a little girl, the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
Layer 2: Once upon a time there was a little girl, the first one of the same day, the first-to the same day, the the the the the the the the the the the the the the the the the the the the the the the the
Layer 3: Once upon a time there was a little girl, the first part part part part of the same thing, the first-to the same thing, the first- the first- the " " the " " " " " the " " " " the
Layer 4: Once upon a time there was a little girl, the first part of the " the " " . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
Layer 5: Once upon a time there was a little girl, the first thing you can't get a good-to get a new, and then the first-to- the same thing, the most important, and the most important, the most of the other
Layer 6: Once upon a time there was a little girl, the first thing you can't get a good-to get rid of the " " (or even remotely, the " " . . . . . . . . . . . . . . .
Layer 7: Once upon a time there was a little girl, the first thing you can't get in the first time.

"I was a very, um-murtie, and I was a very, um-murtie, and I
Layer 8: Once upon a time there was a little girl, a little girl, and the first thing you can't get in the world, the most important thing, the most important, the most------------- was a little girl, and she was a little girl,
Layer 9: Once upon a time there was a little girl, the first thing you can say is " "

- the " "

- the " "

- the " "

- the " "

- the " "

Layer 10: Once upon a time there was a little girl, the first of the two, and the first of the two, and the first of the two, and the first of the two, and the first of the two, and the first of the two
Layer 11: Once upon a time there was a little girl, the the the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the,
Layer 12: Once upon a time there was a little girl who was a little girl, and she was a little girl, and she was a little girl, and she was a little girl, and she was a little girl, and she was a little girl,

----------------------------------------------------------------------------------------------------------------------------------
% uv run gpt2_multi_layers_decoding/multi_layer_decoding_demo.py
Prompt: The computer is

Layer 1: The computer is the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
Layer 2: The computer is the same thing, the same thing, the same thing, the same thing, the " " " the " the " the " the " the " the " the " the the the the the the the the the the the the the
Layer 3: The computer is a very rare, the first part part of the same thing, the first-to the same thing, the first-to- the same thing, the " the " the " " " " " " " " " " " "
Layer 4: The computer is a very different kind of the same thing, the most important part of the " the " " . . . . . . . . . . . . . . . . . . . . . . . . . . . .
Layer 5: The computer is a very different kind of the same thing, the most important part of the " the " " is a " " the " " . . . . . . . . . . . . . . . . . . . . .
Layer 6: The computer is a very different kind of the " " , which is the most important part of the " " .

The first-to- the first-to- the first-to- the first-to- the first-to-
Layer 7: The computer is a very good, and I have a good-to-dec-lent-to- the same kind of a "-m-l-m-l-m-l-m-l-m-l-m running
Layer 8: The computer is the first thing in the world, and the first thing in the world, the first thing in the world, the first thing in the world, the first thing in the world, the first thing in the world, the first thing in
Layer 9: The computer is a " " ( " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , "
Layer 10: The computer is the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the
Layer 11: The computer is the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the " the
Layer 12: The computer is running on a Windows 7 machine, and the operating system is running on a Windows 8 machine.

The computer is running on a Windows 7 machine, and the operating system is running on a Windows 8 machine. The computer is running
'''