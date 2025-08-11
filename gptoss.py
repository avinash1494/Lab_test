from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded!")

# Your dynamic question
question = "Explain Spiral model"  # Change to anything

# Generic, reusable instruction
prompt = f"Provide a clear, complete, and well-structured answer to the following question without showing reasoning steps or meta-commentary:\n\n{question}"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating output...")

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Decode
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
).strip()

# Optional: remove common meta-text leaks
meta_prefixes = ("the chatgpt prompt", "we must respond", "likely about", "the user")
cleaned = "\n".join(
    line for line in response.split("\n") 
    if not line.strip().lower().startswith(meta_prefixes)
)

print("Output:\n", cleaned)
