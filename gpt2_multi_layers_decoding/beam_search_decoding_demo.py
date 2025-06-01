# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 01 June 2025

# This script demonstrates Beam Search decoding with a GPT-2 model
# using the transformers library's built-in functionality.
# Note: This script shows the *result* of Beam Search (multiple candidate sequences)
# and does not visualize the per-layer step-by-step process like the previous greedy demo,
# as that is significantly more complex to implement manually for Beam Search.

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

if __name__ == "__main__":
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Input prompt
    # prompt = "The quick brown fox jumps over the lazy"
    # prompt = "Once upon a time there was a little girl"
    prompt = "The computer is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Add padding token to tokenizer if it doesn't have one
    # This is sometimes needed for beam search with certain models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id


    print(f"Prompt: {prompt}\n")

    # --- Beam Search Decoding ---
    # num_beams: The number of beams to use. Higher values explore more possibilities.
    # early_stopping: Stop the search when at least num_beams sequences are finished.
    # num_return_sequences: The number of highest probability sequences to return.
    # no_repeat_ngram_size: Prevent the generation of repeating n-grams.

    num_beams = 5 # Example beam width
    num_return_sequences = 5 # Return the top 5 sequences
    max_length = 50

    print(f"Performing Beam Search with num_beams={num_beams} and num_return_sequences={num_return_sequences}...\n")

    # Generate sequences using Beam Search
    # We set pad_token_id and eos_token_id to handle padding and stopping
    generated_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2, # Example: prevent repeating bigrams
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    print("Generated Sequences (Beam Search):")

    # Decode all generated sequences first
    decoded_sequences = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]

    # Use the first sequence as the reference
    if not decoded_sequences:
        print("No sequences were generated.")
    else:
        reference_sequence = decoded_sequences[0]
        reference_tokens = tokenizer.encode(reference_sequence)

        # Print the reference sequence without coloring
        print(f"Sequence 1 (Best Sequence): {reference_sequence}\n")

        # Compare and print other sequences with coloring
        for i in range(1, len(decoded_sequences)):
            current_sequence = decoded_sequences[i]
            current_tokens = tokenizer.encode(current_sequence)

            colored_output = []
            # Compare token by token with the reference sequence
            for j in range(max(len(current_tokens), len(reference_tokens))):
                current_token = current_tokens[j] if j < len(current_tokens) else None
                reference_token = reference_tokens[j] if j < len(reference_tokens) else None

                if current_token is not None and reference_token is not None and current_token == reference_token:
                    # Tokens are the same, print normally
                    colored_output.append(tokenizer.decode(current_token))
                elif current_token is not None:
                    # Tokens are different or only in this sequence, print in color (e.g., red)
                    colored_output.append(f"\033[91m{tokenizer.decode(current_token)}\033[0m") # Red color
                elif reference_token is not None:
                    # Token is only in the reference sequence (shouldn't happen if max_length is consistent)
                     colored_output.append(f"\033[94m{tokenizer.decode(reference_token)}\033[0m") # Blue color


            print(f"Sequence {i + 1}: {''.join(colored_output)}\n")

'''
% uv run gpt2_multi_layers_decoding/beam_search_decoding_demo.py
Prompt: The computer is

Performing Beam Search with num_beams=5 and num_return_sequences=5...

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Generated Sequences (Beam Search):
Sequence 1 (Best Sequence): The computer is connected to a USB port on the back of the computer, and the USB cable is attached to the motherboard.

The motherboard is powered by an Intel Core i7-6700K processor, which is capable of running Windows 7

Sequence 2: The computer is connected to a USB port on the back of the computer, and the USB cable is attached to the motherboard.

The motherboard is powered by an Intel Core i7-4790K CPU, which is capable of running Windows 7

Sequence 3: The computer is connected to a USB port on the back of the computer, and the USB cable is attached to the motherboard.

The motherboard is powered by an Intel Core i7-6700K processor, which is capable of running Windows 8

Sequence 4: The computer is connected to a USB port on the back of the computer, and the USB cable is attached to the motherboard.

The motherboard is powered by an Intel Core i7-6700K processor, which is capable of up to 4

Sequence 5: The computer is connected to a USB port on the back of the computer, and the USB cable is attached to the motherboard.

The motherboard is powered by an Intel Core i7-4790K CPU, which is capable of running Windows 8

'''
