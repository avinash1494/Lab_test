from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded!")

# Question
question = "Explain Spiral model"

# Force direct answer style
prompt = f"""You are an AI assistant that gives only the final answer.
No reasoning, no notes about the question, no self-talk.
Respond in full sentences, with as much detail as needed to explain clearly.

Q: {question}
A:"""

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

# Decode
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
).strip()

print("Output:\n", response)
