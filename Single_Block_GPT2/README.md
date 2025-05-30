# Single Block GPT-2 Model

This project implements a simplified version of the GPT-2 model using only a single transformer block. It demonstrates how to train this tiny model and use it for text inference.

Một thử nghiệm tạo ra mô hình Single_Block_GPT-2 mới dựa trên mô hình GPT-2 đơn giản với một khối transformers duy nhất và huấn luyện model đó với dataset rất nhỏ, chỉ một câu tiếng Việt duy nhất. Bạn hãy thử xem model Single_Block_GPT-2 hoạt động như thế nào nhé! ^^

~> Mục đích làm single_block_gpt2_model là để demo quá trình tạo ra một model mới, huấn luyện và sử dụng nó thế nào ^^

## Files

- `single_block_gpt2_model.py`: Defines the `SingleBlockGPT2Model` class.
- `train_single_block_gpt2.py`: Script for training the `SingleBlockGPT2Model`.
- `inference_single_block_gpt2.py`: Script for performing inference using the trained `SingleBlockGPT2Model`.

## Setup

1. Ensure you have Python and `uv` installed.
2. Navigate to the project directory in your terminal.
3. Create a virtual environment and install dependencies:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv add numpy torch transformers
   ```
   *(Note: You might need a specific PyTorch installation command depending on your system. Refer to the official PyTorch website.)*

## Training

The `train_single_block_gpt2.py` script supports continuous training.

- If the `TrainedSingleBlockGPT2` directory does not exist, the script will load initial weights from the original `gpt2` model and start training.
- If the `TrainedSingleBlockGPT2` directory exists, the script will load the previously trained model from that directory and continue training.

To start or continue training, run:

```bash
uv run Single_Block_GPT2/train_single_block_gpt2.py
```

The trained model will be saved in the `TrainedSingleBlockGPT2` directory.

## Inference

The `inference_single_block_gpt2.py` script demonstrates how to load the trained `SingleBlockGPT2Model` and perform text generation.

To run the inference demo, ensure you have trained the model at least once (so the `TrainedSingleBlockGPT2` directory exists) and run:

```bash
uv run Single_Block_GPT2/inference_single_block_gpt2.py
```

The script will load the trained model and generate text based on a sample input.
