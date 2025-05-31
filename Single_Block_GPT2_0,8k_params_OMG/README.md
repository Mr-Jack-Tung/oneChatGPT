# Single Block GPT-2 OMG (Minimized Parameters) - The smallest GPT model in the world ?!

This project explores minimizing the parameter count of a custom GPT-2-like language model while maintaining accurate training results. A key achievement in this project is successfully reducing the model size to **780 parameters** while still being able to train the model to accurately reproduce the training data.

Đánh dấu một bước ngoặt trong quá trình nghiên cứu về việc tối ưu hóa kích thước mô hình GPT-2 để giảm thiểu số lượng tham số mà vẫn đảm bảo hiệu suất tốt trong việc học từ ngữ, đây là Lần đầu tiên chinh phục được transformers model size nhỏ nhất là 780 params và đạt độ chính xác cao trên tập dữ liệu huấn luyện ^^

```
SingleBlockGPT2ModelNoDepend(
  (wte): Embedding(34, 4)
  (wpe): Embedding(64, 4)
  (drop): Dropout(p=0.1, inplace=False)
  (h): TransformerBlock(
    (ln_1): LayerNorm((4,), eps=1e-05, elementwise_affine=True)
    (attn): SelfAttention(
      (c_attn): Linear(in_features=4, out_features=12, bias=True)
      (c_proj): Linear(in_features=4, out_features=4, bias=True)
      (attn_dropout): Dropout(p=0.1, inplace=False)
      (resid_dropout): Dropout(p=0.1, inplace=False)
    )
    (ln_2): LayerNorm((4,), eps=1e-05, elementwise_affine=True)
    (mlp): MLP(
      (c_fc): Linear(in_features=4, out_features=16, bias=True)
      (c_proj): Linear(in_features=16, out_features=4, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (ln_f): LayerNorm((4,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=4, out_features=34, bias=False)
)
Number of trainable params: 780
```

## Project Structure

-   `single_block_gpt2_model.py`: Defines the model architecture, including the `GPT2Config`, `SelfAttention`, `MLP`, `TransformerBlock`, and the main `SingleBlockGPT2ModelNoDepend` class.
-   `character_tokenizer.py`: Implements a simple character-level tokenizer.
-   `train_single_block_gpt2.py`: Script for training the model.
-   `inference_single_block_gpt2.py`: Script for generating text using a trained model.

## Getting Started

1.  Ensure you have Python and PyTorch installed. You can install PyTorch by following the instructions on the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2.  Navigate to the `Single_Block_GPT2_0,8k_params_OMG` directory in your terminal.

## Usage

### Training the Model

The `train_single_block_gpt2_35k_params.py` script is used for training the minimized model. The specific configuration and training methodology used to achieve the 34,704 parameter count and accurate results would be detailed here.

To train the model, run the appropriate training script (e.g., `train_single_block_gpt2_35k_params.py`) with the configuration that resulted in the minimized parameter count and accurate training.

```bash
uv run Single_Block_GPT2_0,8k_params_OMG/train_single_block_gpt2.py # Example command
```

### Generating Text (Inference)

The `inference_single_block_gpt2_35k_params.py` script uses a trained model to generate text.

To run inference using a trained model, run the following command:

```bash
uv run Single_Block_GPT2_0,8k_params_OMG/inference_single_block_gpt2.py # Example command
```

## Key Findings

-   **Model Minimization:** Successfully reduced the parameter count of a custom GPT-2-like model to **780 parameters** while still achieving accurate training results on the target data. This demonstrates the potential for creating very small yet effective language models for specific tasks.
-   **Training for Accuracy:** Achieving accurate training results with a minimized model requires careful consideration of the model architecture, training data, and training process.
