# Single Block GPT-2 (Minimized Parameters)

This project explores minimizing the parameter count of a custom GPT-2-like language model while maintaining accurate training results. A key achievement in this project is successfully reducing the model size to **34,704 parameters** while still being able to train the model to accurately reproduce the training data.

Đánh dấu một bước ngoặt trong quá trình nghiên cứu về việc tối ưu hóa kích thước mô hình GPT-2 để giảm thiểu số lượng tham số mà vẫn đảm bảo hiệu suất tốt trong việc học từ ngữ, đây là Lần đầu tiên chinh phục được transformers model size nhỏ nhất là 34,704 params và đạt độ chính xác cao trên tập dữ liệu huấn luyện ^^

## Project Structure

-   `single_block_gpt2_35k_params_model.py`: Defines the model architecture, including the `GPT2Config`, `SelfAttention`, `MLP`, `TransformerBlock`, and the main `SingleBlockGPT2ModelNoDepend` class.
-   `character_tokenizer.py`: Implements a simple character-level tokenizer.
-   `train_single_block_gpt2_35k_params.py`: Script for training the model.
-   `inference_single_block_gpt2_35k_params.py`: Script for generating text using a trained model.

## Getting Started

1.  Ensure you have Python and PyTorch installed. You can install PyTorch by following the instructions on the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2.  Navigate to the `Single_Block_GPT2_35k_params` directory in your terminal.

## Usage

### Training the Model

The `train_single_block_gpt2_35k_params.py` script is used for training the minimized model. The specific configuration and training methodology used to achieve the 34,704 parameter count and accurate results would be detailed here.

To train the model, run the appropriate training script (e.g., `train_single_block_gpt2_35k_params.py`) with the configuration that resulted in the minimized parameter count and accurate training.

```bash
uv run Single_Block_GPT2_35k_params/train_single_block_gpt2_35k_params.py # Example command
```

### Generating Text (Inference)

The `inference_single_block_gpt2_35k_params.py` script uses a trained model to generate text.

To run inference using a trained model, run the following command:

```bash
uv run Single_Block_GPT2_35k_params/inference_single_block_gpt2_35k_params.py # Example command
```

## Key Findings

-   **Model Minimization:** Successfully reduced the parameter count of a custom GPT-2-like model to **34,704 parameters** while still achieving accurate training results on the target data. This demonstrates the potential for creating very small yet effective language models for specific tasks.
-   **Training for Accuracy:** Achieving accurate training results with a minimized model requires careful consideration of the model architecture, training data, and training process.
