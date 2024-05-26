# -*- coding: utf-8 -*-
# Author: Mr.Jack _ www.BICweb.vn
# Date: 26 May 2024

# https://huggingface.co/docs/trl/en/sft_trainer

import os, torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

os.environ["TOKENIZERS_PARALLELISM"] = "False"
RANDOM_SEED = 42 # 3407

# device = torch.device('mps')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = 'OneChatbotGPT2Vi'
# MODEL_NAME = './test_trainer'
# MODEL_NAME = './test_trainer/checkpoint-10'

print("MODEL_NAME:",MODEL_NAME)


# Step 1: Pretrained loading
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Step 2: Define the optimizer and learning rate scheduler
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

# Step 3: Fine-tune the model
# model.to(device)
# model.train()

model.config.use_cache = False

# pad_token_id=2
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Add new pad_token: [PAD]")

# text = 'Question: Xin chào\n Answer: Công ty BICweb kính chào quý khách!.'
text = "Question: Xin chào Answer: Dạ, em kính chào quý anh ạ!."

print("text:",text)

# data = [{"text": text}]
data = [{"input_ids": tokenizer.encode(text=text, add_special_tokens=True, return_tensors='pt')}]

from datasets import Dataset
dataset = Dataset.from_list(data)
# print(dataset)
# print(dataset[0])

dataset.set_format("torch")

EPOCHS = 100
LEARNING_RATE = 3e-4
OUTPUT_DIR = "test_trainer"

# print(model)

from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    r=16, # 16, 32, 64, 128, 256
    lora_alpha=32, # 32, 64, 128, 256
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    fan_in_fan_out=True,
    target_modules=[
        "attn.c_attn",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    ]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print("\n")

# RANK: r=16 ; epochs=50 ; checkpoint file: ~10MB ; target modules: Conv1D()
# trainable params: 589,824 || all params: 125,029,632 || trainable%: 0.4717473694555863

# (Ok) RANK: r=16 ; lora_alpha=32 ; epochs=100 ; checkpoint file: ~32MB ; adapter_model.safetensors: ~9.4MB; with target_modules: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", ]
# trainable params: 2,359,296 || all params: 126,799,104 || trainable%: 1.8606566809809635

# (Ok) RANK: r=32 ; lora_alpha=32 ; epochs=100 ; checkpoint file: ~60MB ; adapter_model.safetensors: ~18.9MB; with target_modules: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", ]
# trainable params: 4,718,592 || all params: 129,158,400 || trainable%: 3.653337297458005

# (Ok) RANK: r=64 ; epochs=100 ; checkpoint file: ~117MB ; with target_modules: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", ]
# trainable params: 9,437,184 || all params: 133,876,992 || trainable%: 7.049145532041831

# (Ok) RANK: r=128 ; epochs=100 ; checkpoint file: ~60MB ; target modules: Conv1D()
# trainable params: 4,718,592 || all params: 129,158,400 || trainable%: 3.653337297458005

# (Ok) RANK: r=64 ; epochs=150 ; checkpoint file: ~117MB ; adapter_model.safetensors: ~37.8MB; with target_modules: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", ]
# trainable params: 9,437,184 || all params: 133,876,992 || trainable%: 7.049145532041831

# (Ok) RANK: r=128 ; epochs=50 ; checkpoint file: ~230MB ; adapter_model.safetensors: ~75.5MB; with target_modules: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", ]
# trainable params: 18,874,368 || all params: 143,314,176 || trainable%: 13.169923957836522

# (Ok) RANK: r=256 ; epochs=50 ; checkpoint file: ~456MB ; with target_modules: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", ]
# trainable params: 37,748,736 || all params: 162,188,544 || trainable%: 23.27460070176103

args_config = TrainingArguments(
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=1,
    output_dir=OUTPUT_DIR,
    seed=RANDOM_SEED, #42,

    warmup_steps=1,
    weight_decay=0.01,

    overwrite_output_dir=True,
    save_steps=EPOCHS,
    save_strategy= 'steps', # 'steps'  'epoch' 
    save_total_limit=1,

    use_cpu=True, # default = False
    )

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="input_ids",
    max_seq_length=256,
    tokenizer=tokenizer,
    # optimizers=(optimizer, scheduler),
    args=args_config,
    peft_config=peft_config,
)
trainer.train()


# Generate responses to new questions
model.eval()

def generate_answer(question):
    # Encode the question using the tokenizer
    input_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors='pt').to(device)

    # Generate the answer using the model
    sample_output = model.generate(input_ids, pad_token_id=2, eos_token_id=50256, max_length=256, do_sample=True, top_k=50, top_p=1.0, temperature=0.6).to(device)

    # Decode the generated answer using the tokenizer
    answer = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    sentences = answer.split('.')

    return sentences[0]

# # Example usage
question = 'Question: Xin chào'
response = generate_answer(question)
print(f"\n{response}\n")
