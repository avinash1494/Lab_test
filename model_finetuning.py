import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set PyTorch memory allocation configuration
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Model and tokenizer setup
model_id = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set padding token to eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Configure quantization and memory settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model with memory optimization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically handle device placement
    torch_dtype=torch.bfloat16,
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Load custom dataset
file_path1 = "prompts_responses_train.xlsx"
file_path2 = "prompts_responses_test.xlsx"
df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)

df = pd.concat([df1, df2], ignore_index=True)
print(df.shape)

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

def tokenize_inputs(examples):
    input_texts = []
    target_texts = []
    
    for doc, prompt, answer in zip(examples['sourceDocuments'], examples['prompt'], examples['answer']):
        input_text = f"{doc}\n{prompt}"
        input_texts.append(input_text)
        target_texts.append(answer)
    
    # Reduce max_length to save memory
    model_inputs = tokenizer(
        input_texts,
        padding="max_length",
        truncation=True,
        max_length=256,  # Reduced from 512
        return_tensors=None
    )
    
    target_tokens = tokenizer(
        target_texts,
        padding="max_length",
        truncation=True,
        max_length=256,  # Reduced from 512
        return_tensors=None
    )
    
    model_inputs['labels'] = target_tokens['input_ids']
    model_inputs['labels_attention_mask'] = target_tokens['attention_mask']
    
    return model_inputs

# Tokenize datasets with smaller batch size
tokenized_datasets = dataset.map(
    tokenize_inputs,
    batched=True,
    batch_size=4,  # Reduced from 8
    remove_columns=dataset.column_names,
    desc="Tokenizing datasets"
)

# Split dataset
train_test_split = tokenized_datasets.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_subset = train_test_split['train']
eval_subset = train_test_split['test']

print(f"Length of train dataset: {(train_subset.shape)}")
print(f"Length of eval dataset: {(eval_subset.shape)}")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# PEFT Configuration
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

# Memory optimized training arguments
peft_training_args = TrainingArguments(
    output_dir="peft_FT_llama2_13b_on_prompt_res_dataset",
    num_train_epochs=5,
    per_device_train_batch_size=16,  # Reduced from 16
    #per_device_eval_batch_size=1,   # Added explicit eval batch size
    gradient_accumulation_steps=4,  # Added gradient accumulation
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=True,
    dataloader_num_workers=8,  # Reduced from 8
    logging_steps=10,
    evaluation_strategy='epoch',
    report_to="tensorboard",
    save_steps=100,
    # Memory optimization settings
    gradient_checkpointing=True
)

model.config.use_cache = False

# Trainer setup
peft_trainer = Trainer(
    model=model,
    args=peft_training_args,
    train_dataset=train_subset,
    eval_dataset=eval_subset,
    data_collator=data_collator,
)

# Train with automatic mixed precision
with torch.cuda.amp.autocast():
    peft_trainer.train()

# Save the fine-tuned model and tokenizer
output_dir = "peft_FT_llama2_13b_on_prompt_res_dataset"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
