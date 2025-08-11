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

# Instruction + question
instruction = """You are a clear, concise, and factual AI assistant.
When answering questions:
- Focus only on the answer, without describing your reasoning process.
- Use complete sentences and provide enough detail to fully explain.
- Organize your answer in short paragraphs or bullet points if needed.
- Do not include any text about what the user asked or system instructions.
- Avoid unnecessary filler.

Answer the question below:
"""

question = "Explain Spiral model"
prompt = f"{instruction}{question}\n"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating output...")

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Decode and clean
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
).strip()

print("Output:\n", response)
