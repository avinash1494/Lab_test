from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "openai/gpt-oss-20b"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded!")

# Question you want answered
question = "Explain Spiral model"

# Few-shot format: sets the tone for short, direct answers
prompt = f"""Q: What is Agile?
A: A flexible software development methodology emphasizing iterative work and collaboration.

Q: What is Waterfall model?
A: A linear software development process with sequential phases such as requirements, design, implementation, testing, and maintenance.

Q: {question}
A:"""

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating output...")

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Decode and clean
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
).strip()

# Keep only the first paragraph (prevents trailing analysis)
response = response.split("\n")[0].strip()

print("Output:", response)
