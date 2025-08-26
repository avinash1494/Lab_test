from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
device_map = {0: "cuda:0", 1: "cuda:1", 2: "cuda:2"}
# Load tokenizer and model
model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map
)
print("Model loaded!")
messages = [
    {"role": "user", "content": "Explain Spiral model"}
]
print("Tokenizing...")
# Step 1: Generate the prompt using the chat template
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
# Step 2: Tokenize the prompt to get a dict (input_ids, attention_mask)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print("Generating output...")
# Step 3: Generate output
outputs = model.generate(**inputs, max_new_tokens=200)
# Step 4: Decode
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("Output:", response)
