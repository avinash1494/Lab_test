from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "openai/gpt-oss-20b"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded!")

question = "Explain Spiral model"  # You can change this to anything

# Chat messages list
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Always respond with only the final answer to the user’s question. "
            "Do not include reasoning steps, meta-commentary, or restate the question. "
            "Write in clear, complete sentences, and use multiple paragraphs if needed. "
            "Ensure the answer is comprehensive and factual."
        )
    },
    {"role": "user", "content": question}
]

# Apply the chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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

print("Output:\n", response)
