from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load tokenizer and model
model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # since your GPU (NVIDIA L4) supports BF16
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded!")

messages = [
    {"role": "user", "content": "In my hostel, light is not working"}
]

print("Tokenizing...")

# Step 1: Generate the prompt using the chat template
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# Step 2: Tokenize the prompt to get a dict (input_ids, attention_mask)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating output...")

# Step 3: Generate output
outputs = model.generate(**inputs, max_new_tokens=40)

# Step 4: Decode
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print("Output:", response)
