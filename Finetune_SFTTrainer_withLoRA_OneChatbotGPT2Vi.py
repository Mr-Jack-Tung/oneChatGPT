# -*- coding: utf-8 -*-
# Author: Mr.Jack _ www.BICweb.vn
# Date: 26 May 2024

# https://huggingface.co/docs/trl/en/sft_trainer
# https://huggingface.co/docs/peft/en/developer_guides/model_merging

import os, torch
from trl import SFTTrainer

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "False"
RANDOM_SEED = 3407 # 3407 , 42

# device = torch.device('mps')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = 'OneChatbotGPT2Vi'
# MODEL_NAME = './test_trainer/checkpoint-10'

print("MODEL_NAME:",MODEL_NAME)

# Step 1: Pretrained loading
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model.config.use_cache = False

# pad_token_id=2
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Add new pad_token: [PAD]")

# text = 'Question: Xin chào\n Answer: Công ty BICweb kính chào quý khách!.'
# text = "Question: Xin chào Answer: Dạ, em chào anh, dạo này anh có khỏe không ạ!."
text = "Question: Xin chào Answer: Dạ, em kính chào quý anh ạ!."
print("text:",text)

data = [{"text": text}]
# data = [{"input_ids": tokenizer.encode(text=text, add_special_tokens=True, return_tensors='pt')}]

from datasets import Dataset
dataset = Dataset.from_list(data)

# dataset.set_format("torch")
# print(dataset)
# print(dataset[0])

EPOCHS = 80
LEARNING_RATE = 3e-4
OUTPUT_DIR = "test_trainer"

# print(model)

from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    init_lora_weights="gaussian",
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

# (Ok) RANK: r=16 ; lora_alpha=32 ; epochs=80 ; checkpoint file: ~32MB ; adapter_model.safetensors: ~9.4MB; with target_modules: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", ]
# trainable params: 2,359,296 || all params: 126,799,104 || trainable%: 1.8606566809809635

# (Ok) RANK: r=32 ; lora_alpha=32 ; epochs=80 ; checkpoint file: ~60MB ; adapter_model.safetensors: ~18.9MB; with target_modules: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", ]
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
    logging_steps=5, # 1, 5, 10
    output_dir=OUTPUT_DIR,
    seed=RANDOM_SEED, #42,

    optim='adamw_torch',
    lr_scheduler_type='constant_with_warmup',
    gradient_accumulation_steps=10,
    # per_device_train_batch_size=2,
    warmup_steps=0,
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
    dataset_text_field="text", # "input_ids",
    max_seq_length=256,
    # tokenizer=tokenizer,
    args=args_config,
    peft_config=peft_config,
)
trainer.train()


# Generate responses to new questions
model.eval()
# print(model)

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

"""
PRETRAINED_MODEL_NAME: gpt2
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

PRETRAINED_MODEL_NAME: OneChatbotGPT2Vi
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0-11): 12 x GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPT2Attention(
              (c_attn): lora.Linear(
                (base_layer): Conv1D()
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=768, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=2304, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (c_proj): lora.Linear(
                (base_layer): Conv1D()
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=768, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=768, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): lora.Linear(
                (base_layer): Conv1D()
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=768, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=3072, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (c_proj): lora.Linear(
                (base_layer): Conv1D()
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=768, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )
  )
)

"""
