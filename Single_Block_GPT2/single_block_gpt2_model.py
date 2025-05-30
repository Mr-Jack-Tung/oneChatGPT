import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Config

class SingleBlockGPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding layers
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop) # Add dropout layer

        # Single GPT2Block
        self.h = GPT2Block(config)

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
        hidden_states = self.drop(hidden_states) # Apply dropout

        # Pass through the single GPT2Block
        # GPT2Block returns a tuple. When use_cache=False and output_attentions=False, it returns only the output tensor.
        transformer_output = self.h(
            hidden_states,
            layer_past=None, # Not handling past key/values for simplicity in this basic example
            attention_mask=attention_mask,
            head_mask=None, # Not handling head mask
            use_cache=False, # Not using cache
            output_attentions=False # Not outputting attentions
        )
        hidden_states = transformer_output[0] # Get the output tensor

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

    def generate(self, input_ids, max_length, pad_token_id, eos_token_id, temperature=1.0, top_k=None, top_p=None, device='cpu'):
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

            # Stop if the end-of-sequence token is generated
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

        return generated_ids

'''
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

Number of trainable params: 85,070,592
'''