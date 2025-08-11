from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded!")

question = "Explain Spiral model"

# Clean, strict instruction
prompt = f"""Answer the question below in clear, complete sentences.
Do not include reasoning steps, meta-commentary, or notes about the question.
Provide a direct, well-structured explanation.

Question: {question}
"""

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating output...")

outputs = model.generate(
    **inputs,
    max_new_tokens=300,  # More room for detailed explanation
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Decode
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
).strip()

print("Output:\n", response)
