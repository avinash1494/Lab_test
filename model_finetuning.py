import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Restrict CUDA to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Model Configuration ===
BASE_MODEL = "meta-llama/Llama-2-13b-hf"
TOKEN_VALUE = "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"

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN_VALUE)

# === Configure BitsAndBytes for Quantization ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # bfloat16 for efficiency
)

# === Load Model with Quantization ===
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, token=TOKEN_VALUE,
    device_map="auto",  # Auto assigns to available GPUs
    quantization_config=bnb_config
)

# === Set Padding Token ===
tokenizer.pad_token_id = 0  # Set padding token to unk token
tokenizer.padding_side = "left"

# === Load Custom Dataset ===
file_path1 = "prompts_responses_train.xlsx"
file_path2 = "prompts_responses_test.xlsx"
df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)

df = pd.concat([df1, df2], ignore_index=True)
dataset = Dataset.from_pandas(df)

# === Tokenization Function ===
def tokenize_inputs(examples):
    input_texts = [f"{doc}\n{prompt}" for doc, prompt in zip(examples['sourceDocuments'], examples['prompt'])]
    target_texts = [answer for answer in examples['answer']]

    model_inputs = tokenizer(
        input_texts, padding="max_length", truncation=True, max_length=256, return_tensors=None
    )
    target_tokens = tokenizer(
        target_texts, padding="max_length", truncation=True, max_length=256, return_tensors=None
    )

    model_inputs['labels'] = target_tokens['input_ids']
    model_inputs['labels_attention_mask'] = target_tokens['attention_mask']
    
    return model_inputs

# === Tokenize Dataset ===
tokenized_datasets = dataset.map(
    tokenize_inputs, batched=True, batch_size=2, remove_columns=dataset.column_names, desc="Tokenizing datasets"
)

# === Split Dataset ===
train_test_split = tokenized_datasets.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_subset, eval_subset = train_test_split['train'], train_test_split['test']

print(f"Train dataset size: {len(train_subset)}")
print(f"Eval dataset size: {len(eval_subset)}")

# === Data Collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === LORA Configuration ===
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Enable Gradient Checkpointing **Before PEFT**
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# === Training Arguments (Optimized for Memory) ===
peft_training_args = TrainingArguments(
    output_dir="peft_FT_llama2_13b_on_prompt_res_dataset",
    num_train_epochs=1,
    per_device_train_batch_size=8,  # Increased batch size from 2 → 4
    per_device_eval_batch_size=4,   # Increased eval batch size from 1 → 2
    gradient_accumulation_steps=8,  # Reduced from 16 → 8 to speed up training
    learning_rate=1e-5,
    weight_decay=0.01,
    bf16=True,  # Use bf16 instead of fp16 if supported
    dataloader_num_workers=8,  # Increased workers for faster data loading
    logging_steps=10,
    evaluation_strategy='epoch',
    report_to="tensorboard",
    save_steps=500,  # Reduced frequent saves
)


# Disable Cache for Training
model.config.use_cache = False

# === Trainer ===
peft_trainer = Trainer(
    model=model,
    args=peft_training_args,
    train_dataset=train_subset,
    eval_dataset=eval_subset,
    data_collator=data_collator,
)

# === Train Model ===
with torch.cuda.amp.autocast():
    peft_trainer.train()

# === Save Model ===
output_dir = "peft_FT_llama2_13b_on_prompt_res_dataset"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
