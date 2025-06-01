# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 01 June 2025

# This script demonstrates Top-P (Nucleus) Sampling decoding with a GPT-2 model
# and shows the number of tokens sampled from (nucleus size) at each step.
# This is a manual implementation to visualize the process.

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def top_p_sampling(logits, top_p=0.9):
    """
    Performs Top-P (Nucleus) sampling on the logits.

    Args:
        logits: The logits for the next token prediction (shape: [batch_size, vocab_size]).
        top_p: The cumulative probability threshold for Top-P sampling.

    Returns:
        The sampled token ID.
        The size of the nucleus (number of tokens sampled from).
    """
    # Ensure logits are 1D for simplicity in this demo
    if logits.dim() > 1:
        logits = logits.squeeze(0)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probabilities = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    # Find the index of the first token where cumulative probability > top_p
    # We need to include this token, so we add 1 to the index
    sorted_indices_to_remove = cumulative_probabilities > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Set the logits of tokens to remove to -inf
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')

    # The size of the nucleus is the number of tokens NOT removed
    nucleus_size = (logits > -float('Inf')).sum().item()

    # Sample from the filtered distribution
    probabilities = torch.softmax(logits, dim=-1)
    sampled_token_id = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    return sampled_token_id, nucleus_size

if __name__ == "__main__":
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Input prompt
    prompt = "The quick brown fox jumps over the lazy"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Top-P parameter
    top_p = 0.9

    print(f"Prompt: {prompt}\n")
    print(f"Performing Top-P Sampling with top_p={top_p}...\n")

    generated_ids = input_ids.clone()
    max_length = 50

    # List to store generated token IDs and their nucleus sizes
    generated_tokens_with_nucleus_size = []

    # Manual token-by-token generation loop
    for step in range(max_length - input_ids.shape[1]):
        with torch.no_grad():
            # Get logits for the next token based on the current sequence
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :] # Logits for the last token

            # Perform Top-P sampling
            sampled_token_id, nucleus_size = top_p_sampling(next_token_logits, top_p=top_p)

            # Store the sampled token ID and its nucleus size
            generated_tokens_with_nucleus_size.append((sampled_token_id.item(), nucleus_size))

            # Append the sampled token to the sequence for the next step
            # sampled_token_id is a scalar, need to reshape to [1, 1] for concatenation with generated_ids [batch_size, seq_len]
            generated_ids = torch.cat([generated_ids, sampled_token_id.unsqueeze(0).unsqueeze(-1)], dim=-1)

            # Stop if the end-of-sequence token is generated
            if sampled_token_id == tokenizer.eos_token_id:
                break

    # Print the generated sequence with nucleus size next to each token
    print("Generated Sequence (Top-P Sampling) with Nucleus Size per Token:")

    # Print the initial prompt first
    prompt_tokens = tokenizer.encode(prompt)
    for token_id in prompt_tokens:
        print(tokenizer.decode(token_id), end="")

    # Print generated tokens with nucleus size
    for token_id, nucleus_size in generated_tokens_with_nucleus_size:
        print(f"{tokenizer.decode(token_id)}({nucleus_size})", end="")

    print("\n") # Add a newline at the end

'''
% uv run gpt2_multi_layers_decoding/multi_layer_top_p_sampling_decoding_demo.py
Prompt: The quick brown fox jumps over the lazy

Performing Top-P Sampling with top_p=0.9...

Generated Sequence (Top-P Sampling) with Nucleus Size per Token:
The quick brown fox jumps over the lazy pony(2001) and(29) catches(654) him(7).(73)
(217)
(1)"(148)Here(286)!(30) Get(366) to(43) the(53) grass(1635)y(66) t(106)und(24)ra(1)!"(58) he(478) tells(106) her(97).(20)
(10)
(1)She(236) watches(401) him(12) drift(559) along(37) the(29) trails(485),(39) then(1216) grab(613) a(12) small(813) red(854) apple(911).(14)
(43)
(1)
'''