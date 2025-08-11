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

# Strict instruction: only answer the question, no extra commentary
question = "Explain Spiral model"
prompt = f"Answer the following question directly, without showing reasoning steps or extra text.\nQuestion: {question}\nAnswer:"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating output...")

# Generate with EOS token to stop at end of answer
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    eos_token_id=tokenizer.eos_token_id
)

# Decode
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print("Output:", response.strip())
