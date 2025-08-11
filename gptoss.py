from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "openai/gpt-oss-20b"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Question and optimized prompt
question = "Explain the Spiral model in software engineering"
system_prompt = """You are a technical expert providing precise information. 
Respond with these characteristics:
1. Direct factual answer only
2. Comprehensive but concise
3. Well-structured with key points
4. No introductory phrases or self-references"""

prompt = f"""SYSTEM: {system_prompt}

QUESTION: {question}

ANSWER: The Spiral model is"""

# Tokenization and generation
try:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("\nGenerating response...")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean output
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response.split("ANSWER:")[-1].strip()
    
    print("\n=== Response ===")
    print(answer)
    
except Exception as e:
    print(f"Error during generation: {e}")
