# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 31 May 2025

import json
import os

class CharacterTokenizer:
    def __init__(self, chars=None):
        self.chars = sorted(list(set(chars))) if chars else []
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_token = '<pad>' # Using a simple pad token
        self.unk_token = '<unk>' # Using a simple unknown token
        self.pad_token_id = self.add_token(self.pad_token)
        self.unk_token_id = self.add_token(self.unk_token)


    def add_token(self, token):
        if token not in self.char_to_idx:
            self.chars.append(token)
            self.char_to_idx[token] = len(self.chars) - 1
            self.idx_to_char[len(self.chars) - 1] = token
            self.vocab_size = len(self.chars)
        return self.char_to_idx[token]

    def train(self, text_data):
        """Builds the vocabulary from the training data."""
        all_chars = sorted(list(set(text_data)))
        self.chars = all_chars
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_token_id = self.add_token(self.pad_token)
        self.unk_token_id = self.add_token(self.unk_token)


    def encode(self, text):
        """Encodes a string into a list of token IDs."""
        return [self.char_to_idx.get(ch, self.unk_token_id) for ch in text]

    def decode(self, token_ids):
        """Decodes a list of token IDs into a string."""
        return "".join([self.idx_to_char.get(idx, self.unk_token) for idx in token_ids])

    def save_pretrained(self, save_directory):
        """Saves the tokenizer vocabulary to a file."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        vocab_path = os.path.join(save_directory, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.char_to_idx, f, ensure_ascii=False, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory):
        """Loads the tokenizer vocabulary from a file."""
        vocab_path = os.path.join(save_directory, 'vocab.json')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            char_to_idx = json.load(f)
        chars = sorted(char_to_idx.keys())
        tokenizer = cls(chars=chars)
        return tokenizer

# Example Usage:
# tokenizer = CharacterTokenizer()
# tokenizer.train("hello world")
# encoded = tokenizer.encode("hi there")
# decoded = tokenizer.decode(encoded)
# print(f"Encoded: {encoded}")
# print(f"Decoded: {decoded}")
# tokenizer.save_pretrained("my_tokenizer")
# loaded_tokenizer = CharacterTokenizer.from_pretrained("my_tokenizer")
# print(f"Loaded vocab size: {loaded_tokenizer.vocab_size}")
