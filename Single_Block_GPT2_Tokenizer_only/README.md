# Single Block GPT-2 Model (Tokenizer Only Dependency)

This project implements a simplified version of the GPT-2 model using a single custom transformer block. **Crucially, this version is designed to depend ONLY on the `transformers` library for its tokenizer, with the rest of the model architecture and associated scripts being custom implementations.**

It demonstrates how to train this custom model with the standard GPT-2 tokenizer and use it for text inference.

Một thử nghiệm tạo ra mô hình Single_Block_GPT-2 mới với kiến trúc tùy chỉnh, chỉ sử dụng tokenizer từ thư viện `transformers`. Mục đích là để demo quá trình tạo ra một model tùy chỉnh và tích hợp nó với một tokenizer tiêu chuẩn.

## Files

- `single_block_gpt2_model.py`: Contains the custom implementation of the `SingleBlockGPT2Model` class, including the `GPT2Config`, `NewGELUActivation`, `SelfAttention`, `MLP`, and `TransformerBlock` classes.
- `train_single_block_gpt2.py`: Script for training the custom `SingleBlockGPT2Model` using the `transformers` tokenizer.
- `inference_single_block_gpt2.py`: Script for performing inference using the trained custom `SingleBlockGPT2Model` and the `transformers` tokenizer.

## Setup

1. Ensure you have Python and `uv` installed.
2. Navigate to the project directory in your terminal.
3. Create a virtual environment and install dependencies:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv add torch transformers
   ```
   *(Note: You might need a specific PyTorch installation command depending on your system. Refer to the official PyTorch website.)*

## Training

The `train_single_block_gpt2.py` script trains the custom `SingleBlockGPT2Model`. It uses the `transformers` GPT-2 tokenizer to process the training data.

- If the `TrainedSingleBlockGPT2_Tokenizer_only` directory does not exist, the script will initialize a new custom model with random weights and start training.
- If the `TrainedSingleBlockGPT2_Tokenizer_only` directory exists, the script will load the previously trained model from that directory and continue training.

To start or continue training, run:

```bash
uv run train_single_block_gpt2.py
```

The trained model's state dictionary and the `transformers` tokenizer will be saved in the `TrainedSingleBlockGPT2_Tokenizer_only` directory.

## Inference

The `inference_single_block_gpt2.py` script demonstrates how to load the trained custom `SingleBlockGPT2Model` and perform text generation using the saved `transformers` tokenizer.

To run the inference demo, ensure you have trained the model at least once (so the `TrainedSingleBlockGPT2_Tokenizer_only` directory exists) and run:

```bash
uv run inference_single_block_gpt2.py
```

The script will load the trained model and tokenizer and generate text based on a sample input.
