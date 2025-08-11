from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "openai/gpt-oss-20b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded!")

# Just write your plain prompt here
prompt = "Explain Spiral model in software development."

# Tokenize directly
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating output...")

# Generate
outputs = model.generate(**inputs, max_new_tokens=200)

# Decode
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print("Output:", response)
