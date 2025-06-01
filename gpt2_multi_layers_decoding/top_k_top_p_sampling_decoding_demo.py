# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 01 June 2025

# This script demonstrates combined Top-K and Top-P (Nucleus) Sampling decoding
# with a GPT-2 model and shows the number of tokens sampled from (nucleus size) at each step.
# This is a manual implementation to visualize the process.

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def top_k_top_p_sampling(logits, top_k=0, top_p=1.0, temperature=1.0):
    """
    Performs combined Top-K and Top-P (Nucleus) sampling on the logits.

    Args:
        logits: The logits for the next token prediction (shape: [batch_size, vocab_size]).
        top_k: The number of top tokens to consider (0 means no Top-K filtering).
        top_p: The cumulative probability threshold for Top-P sampling (1.0 means no Top-P filtering).
        temperature: Softmax temperature (1.0 means no change).

    Returns:
        The sampled token ID.
        The size of the nucleus (number of tokens sampled from after both filters).
        The probability of the sampled token.
    """
    # Ensure logits are 1D for simplicity in this demo
    if logits.dim() > 1:
        logits = logits.squeeze(0)

    # Apply temperature
    # với temperature thấp (0.5), phân phối xác suất trở nên "sắc nét" hơn, làm tăng xác suất của các token có khả năng cao nhất. Điều này dẫn đến việc model có xu hướng lặp lại các cụm từ có xác suất cao và Nucleus Size có thể nhỏ đi ở một số bước do ngưỡng Top-P nhanh chóng đạt được với các token có xác suất cao. Xác suất của các token được chọn trong nucleus cũng có xu hướng cao hơn so với khi sử dụng temperature mặc định (1.0).
    logits = logits / temperature

    # Apply Top-K filtering
    if top_k > 0:
        # Get the top_k logits and their indices
        values, _ = torch.topk(logits, top_k)
        # Get the minimum value among the top_k logits (last element of the sorted values)
        min_value = values[-1]
        # Set logits of tokens outside the top_k to -inf
        logits[logits < min_value] = -float('Inf')

    # Apply Top-P filtering
    if top_p < 1.0:
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
    # Ensure probabilities sum to 1 after filtering
    probabilities = probabilities / probabilities.sum()
    sampled_token_id = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    # Get the probability of the sampled token
    sampled_token_probability = probabilities[sampled_token_id].item()

    return sampled_token_id, nucleus_size, sampled_token_probability

if __name__ == "__main__":
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Input prompt
    prompt = "The quick brown fox jumps over the lazy"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Sampling parameters
    top_k = 50 # Example Top-K value
    top_p = 0.9 # Example Top-P value
    temperature = 0.5 # Example low temperature value

    print(f"Prompt: {prompt}\n")
    print(f"Performing combined Top-K ({top_k}), Top-P ({top_p}) Sampling with Temperature ({temperature})...\n")

    generated_ids = input_ids.clone()
    max_length = 50

    # List to store generated token IDs, nucleus sizes, and probabilities
    generated_tokens_info = []

    # Manual token-by-token generation loop
    for step in range(max_length - input_ids.shape[1]):
        with torch.no_grad():
            # Get logits for the next token based on the current sequence
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :] # Logits for the last token

        # Perform combined Top-K and Top-P sampling
        sampled_token_id, nucleus_size, sampled_token_probability = top_k_top_p_sampling(next_token_logits, top_k=top_k, top_p=top_p, temperature=temperature)

        # Store the sampled token ID, nucleus size, and probability
        generated_tokens_info.append((sampled_token_id.item(), nucleus_size, sampled_token_probability))

        # Append the sampled token to the sequence for the next step
        # sampled_token_id is a scalar, need to reshape to [1, 1] for concatenation with generated_ids [batch_size, seq_len]
        generated_ids = torch.cat([generated_ids, sampled_token_id.unsqueeze(0).unsqueeze(-1)], dim=-1)

        # Stop if the end-of-sequence token is generated
        if sampled_token_id == tokenizer.eos_token_id:
            break

    # Print the generated sequence with nucleus size and probability next to each token
    print("Generated Sequence (Combined Top-K/Top-P Sampling) with Nucleus Size and Probability per Token:")

    # Print the initial prompt first
    prompt_tokens = tokenizer.encode(prompt)
    for token_id in prompt_tokens:
        print(tokenizer.decode(token_id), end="")

    # Print generated tokens with nucleus size and probability
    for token_id, nucleus_size, probability in generated_tokens_info:
        # Format probability as percentage with 2 decimal places
        print(f"{tokenizer.decode(token_id)}\033[91m({nucleus_size}, {probability*100:.2f}%)\033[0m", end="")

    print("\n") # Add a newline at the end

'''
% uv run gpt2_multi_layers_decoding/top_k_top_p_sampling_decoding_demo.py
Prompt: The quick brown fox jumps over the lazy

Performing combined Top-K (50), Top-P (0.9) Sampling with Temperature (0.5)...

Generated Sequence (Combined Top-K/Top-P Sampling) with Nucleus Size and Probability per Token:
The quick brown fox jumps over the lazy red(14, 4.60%) fox(1, 100.00%) and(3, 67.63%) gets(27, 6.44%) into(11, 5.43%) a(2, 56.89%) fight(1, 100.00%) with(1, 100.00%) the(3, 81.70%) lazy(4, 53.79%) red(1, 100.00%) fox(1, 100.00%).(1, 100.00%) The(2, 75.02%) lazy(2, 89.34%) red(1, 100.00%) fox(1, 100.00%) gets(10, 49.72%) a(9, 9.28%) bit(19, 16.80%) too(14, 23.76%) close(2, 95.97%) to(2, 83.96%) the(1, 100.00%) lazy(1, 100.00%) red(1, 100.00%) fox(1, 100.00%) and(2, 86.49%) is(9, 6.24%) taken(17, 1.00%) down(2, 83.91%) by(2, 90.82%) the(1, 100.00%) lazy(1, 100.00%) red(1, 100.00%) fox(1, 100.00%).(1, 100.00%)
(2, 21.20%)
(1, 100.00%)In(2, 4.56%) the(1, 100.00%) episode(8, 7.78%)
'''
