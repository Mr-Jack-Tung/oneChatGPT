
# https://huggingface.co/docs/trl/en/sft_trainer

import os, torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, DataCollatorWithPadding

os.environ["TOKENIZERS_PARALLELISM"] = "False"

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

RANDOM_SEED = 42 # 3407
model.config.use_cache = False

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# text = 'Question: Xin chào\n Answer: Công ty BICweb kính chào quý khách!.'
text = "Question: Xin chào Answer: Dạ, em chào anh ạ!."

print("text:",text)

# data = [{"text": text}]
data = [{"input_ids": tokenizer.encode(text=text, add_special_tokens=True, return_tensors='pt')}]

from datasets import Dataset
dataset = Dataset.from_list(data)
# print(dataset)
# print(dataset[0])

dataset.set_format("torch")

# instruction_template = "Question:"
# response_template = " Answer:"
# collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False) # instruction_template=instruction_template, 

EPOCHS = 15
LEARNING_RATE = 3e-4
OUTPUT_DIR = "test_trainer"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="input_ids",
    max_seq_length=256,
    # optimizers=(optimizer, scheduler),
    tokenizer=tokenizer,
    # data_collator=collator,
    args = transformers.TrainingArguments(
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        output_dir=OUTPUT_DIR,
        seed=RANDOM_SEED, #42,

        # max_grad_norm=9.9,
        # resume_from_checkpoint='checkpoint-{0}'.format(EPOCHS), # default = None
        warmup_steps=1,
        weight_decay=0.01,

        overwrite_output_dir=True,
        # save_only_model=True, # default = False
        save_steps=EPOCHS, # 10, -1 is mean every step
        save_strategy= 'steps', # 'steps'  'epoch' 
        save_total_limit=1,
        
        no_cuda=True, # default = False
        use_cpu=True, # default = False
        # use_mps_device=True, # default = False
        ),
)
trainer.train()

# trainer.save_model(OUTPUT_DIR)
# trainer.model.save_pretrained(OUTPUT_DIR)
# trainer.tokenizer.save_pretrained(OUTPUT_DIR)

# Generate responses to new questions
model.eval()

def generate_answer(question):
    # Encode the question using the tokenizer
    input_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors='pt').to(device)

    # Generate the answer using the model
    sample_output = model.generate(input_ids, pad_token_id=2, eos_token_id=50256, max_length=256, do_sample=True, top_k=100, top_p=1.0, temperature=0.6).to(device)

    # Decode the generated answer using the tokenizer
    answer = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    sentences = answer.split('.')

    return sentences[0] # [answer] 

# # Example usage
question = 'Question: Xin chào'
response = generate_answer(question)
print(f"\n{response}\n")

# import gc
# torch.cuda.empty_cache()
# torch.mps.empty_cache()
# gc.collect()
