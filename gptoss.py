from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Option 1: Use only first 3 GPUs with auto mapping
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Restrict to first 3 GPUs
device_map = "auto"

# Option 2: Manual device mapping for 3 GPUs (alternative approach)
# device_map = {
#     "model.embed_tokens": "cuda:0",
#     "model.layers.0": "cuda:0", "model.layers.1": "cuda:0", "model.layers.2": "cuda:0",
#     "model.layers.3": "cuda:0", "model.layers.4": "cuda:0", "model.layers.5": "cuda:0",
#     "model.layers.6": "cuda:0", "model.layers.7": "cuda:0", "model.layers.8": "cuda:1",
#     "model.layers.9": "cuda:1", "model.layers.10": "cuda:1", "model.layers.11": "cuda:1",
#     "model.layers.12": "cuda:1", "model.layers.13": "cuda:1", "model.layers.14": "cuda:1",
#     "model.layers.15": "cuda:1", "model.layers.16": "cuda:2", "model.layers.17": "cuda:2",
#     "model.layers.18": "cuda:2", "model.layers.19": "cuda:2", "model.layers.20": "cuda:2",
#     "model.layers.21": "cuda:2", "model.layers.22": "cuda:2", "model.layers.23": "cuda:2",
#     "model.norm": "cuda:2",
#     "lm_head": "cuda:2"
# }

# Load tokenizer and model
model_name = "openai/gpt-oss-20b"

# Check if tokenizer pad_token is set
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # More explicit than "auto"
    device_map=device_map,
    trust_remote_code=True,  # May be needed for some models
    low_cpu_mem_usage=True   # Helps with memory efficiency
)

print("Model loaded successfully!")

# Check model device
print(f"Model device: {next(model.parameters()).device}")

messages = [
    {"role": "user", "content": "Explain Spiral model"}
]

print("Tokenizing...")
# Step 1: Generate the prompt using the chat template
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(f"Prompt: {prompt}")

# Step 2: Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Move inputs to the same device as the model
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

print("Generating output...")
# Step 3: Generate output with better parameters
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# Step 4: Decode only the new tokens
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("Output:", response)
