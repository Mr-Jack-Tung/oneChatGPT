# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Date: 31 May 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPT2Config:
    """
    Configuration class for GPT-2 models, manually defining standard configurations.
    """
    def __init__(self, model_type="small", vocab_size=None):
        self.model_type = model_type

        if model_type == "small":
            self.n_positions = 1024
            self.n_embd = 768 # 1020 | Changed from 768 to be divisible by n_head (12)
            self.n_layer = 12 # Standard GPT-2 small has 12 layers, but we only use 1
            self.n_head = 12
            self.embd_pdrop = 0.1
            self.attn_pdrop = 0.1
            self.resid_pdrop = 0.1
            self.layer_norm_epsilon = 1e-5
            self.initializer_range = 0.02
        elif model_type == "medium":
            self.n_positions = 1024
            self.n_embd = 1024
            self.n_layer = 24
            self.n_head = 16
            self.embd_pdrop = 0.1
            self.attn_pdrop = 0.1
            self.resid_pdrop = 0.1
            self.layer_norm_epsilon = 1e-5
            self.initializer_range = 0.02
        elif model_type == "large":
            self.n_positions = 1024
            self.n_embd = 1280
            self.n_layer = 36
            self.n_head = 20
            self.embd_pdrop = 0.1
            self.attn_pdrop = 0.1
            self.resid_pdrop = 0.1
            self.layer_norm_epsilon = 1e-5
            self.initializer_range = 0.02
        # Add other sizes (xl, etc.) if needed

        # Vocab size is determined by the tokenizer, override if provided
        self.vocab_size = vocab_size if vocab_size is not None else 50257 # Default to GPT-2 vocab size if not provided

        self.scale_attn_weights = True # GPT2 specific
        self.use_cache = False # Not using cache for simplicity

    def __repr__(self):
        return f"GPT2Config(model_type='{self.model_type}', vocab_size={self.vocab_size}, n_positions={self.n_positions}, n_embd={self.n_embd}, n_layer={self.n_layer}, n_head={self.n_head})"

def NewGELUActivation(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# Placeholder for MLP, TransformerBlock, and the main model class
# These will be implemented in subsequent steps.

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.split_size = self.n_embd
        self.register_buffer("bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1, 1024, 1024)) # Max position embedding size

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None):
        # q, k, v are (batch_size, num_heads, seq_len, head_dim)
        # attention_mask is (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = attn_weights / (float(v.size(-1)) ** 0.5)

        # Apply attention mask
        query_length, key_length = q.size(-2), k.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights, torch.tensor(-1e4, dtype=attn_weights.dtype, device=attn_weights.device))

        if attention_mask is not None:
             # Apply the attention mask
             attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        return attn_output, attn_weights

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states is (batch_size, seq_len, n_embd)

        # Linear projections for Q, K, V
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # Reshape for multi-head attention
        query = query.view(query.size(0), query.size(1), self.n_head, self.split_size // self.n_head).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.n_head, self.split_size // self.n_head).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.n_head, self.split_size // self.n_head).transpose(1, 2)

        # Calculate attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        # Concatenate heads and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1, self.n_embd)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.n_embd
        inner_dim = embed_dim * 4 # GPT2 uses 4 * embed_dim for the inner dimension

        self.c_fc = nn.Linear(embed_dim, inner_dim)
        self.c_proj = nn.Linear(inner_dim, embed_dim)
        self.act = NewGELUActivation
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = hidden_size * 4

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, hidden_states, attention_mask=None):
        # LayerNorm before attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs, attn_weights = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + attn_outputs # Add residual connection

        # LayerNorm before MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states # Add residual connection

        return hidden_states, attn_weights # Return hidden states and attention weights

class SingleBlockGPT2ModelNoDepend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding layers
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Single TransformerBlock
        self.h = TransformerBlock(config)

        # Final layer
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Get embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # Pass through the single TransformerBlock
        hidden_states, attn_weights = self.h(
            hidden_states,
            attention_mask=attention_mask,
        )

        # Pass through final layer norm and linear head
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        output = (lm_logits,)
        return ((loss,) + output) if loss is not None else output

    def generate(self, input_ids, max_length, pad_token_id, eos_token_id=None, temperature=1.0, top_k=None, top_p=None, device='cpu'):
        """Basic text generation method (greedy decoding for simplicity)."""
        self.eval()
        input_ids = input_ids.to(device)
        generated_ids = input_ids.clone()

        for _ in range(max_length - input_ids.shape[-1]):
            with torch.no_grad():
                outputs = self(generated_ids)
                logits = outputs[0][:, -1, :] / temperature # Get logits for the last token

                # Apply top-k and top-p filtering if specified (simplified)
                if top_k is not None:
                    # Get top k logits
                    v, _ = torch.topk(logits, top_k)
                    # Set logits below the k-th value to -inf
                    logits[logits < v[:, [-1]]] = -float('inf')

                if top_p is not None:
                    # Sort logits in descending order
                    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                    # Calculate cumulative probabilities
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    # Find the index where the cumulative probability exceeds top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    # Scatter indices to original shape
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    # Set logits to -inf for tokens to remove
                    logits[indices_to_remove] = -float('inf')

                # Sample the next token (using multinomial for temperature/top-p/top-k)
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)


            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)

            # Stop if the end-of-sequence token is generated (if provided)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

        return generated_ids

'''
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 1020)
  (wpe): Embedding(1024, 1020)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((1020,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=1020, out_features=3060, bias=True)
      (c_proj): Linear(in_features=1020, out_features=1020, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((1020,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=1020, out_features=4080, bias=True)
      (c_proj): Linear(in_features=4080, out_features=1020, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((1020,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=1020, out_features=34, bias=False)
)
Number of trainable params: 13,613,940
'''