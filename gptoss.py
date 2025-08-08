from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch

model_name = "openai/gpt-oss-20b"  # or another verified chat model

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected.")

tokenizer = AutoTokenizer.from_pretrained(model_name)


# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,  # or torch.bfloat16 (if your GPU supports it)
#     bnb_4bit_quant_type="nf4",             # or "fp4"
#     bnb_4bit_use_double_quant=True
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
#     load_in_8bit=True   # or load_in_8bit=True if preferred
)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )

print("Model loaded!")

messages = [
    {"role": "user", "content": "Who are you?"},
]

print("Tokenizing...")

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(model.device)

print("Generating output...")
outputs = model.generate(**inputs, max_new_tokens=40)

print("Output:")
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
