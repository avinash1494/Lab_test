import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Set PyTorch memory allocation configuration
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Model and tokenizer setup
BASE_MODEL = "meta-llama/Llama-2-13b-hf"
TOKEN_VALUE="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"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
#tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Configure BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)
# Initialize the Llama model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,use_auth_token=token_value,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config
)
# Set pad_token_id and padding_side
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"
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

LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
# Prepare the model for K-Bit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
# Configure the Lora model
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
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
