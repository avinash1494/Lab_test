import time
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# === Model Configuration ===
BASE_MODEL = "meta-llama/Llama-3.2-3B"

TOKEN_VALUE = "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"

CACHE_DIR = "/home/models/"

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN_VALUE,cache_dir=CACHE_DIR)

# === Configure BitsAndBytes for Quantization ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# === Load Base Model ===
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=TOKEN_VALUE,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=CACHE_DIR
)

# === Set Padding ===
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# === Generate Response and Measure Performance ===
def generate_response(question):
    """
    Generates a response using the LLM and measures performance metrics.
    """
    try:
        # Encode input
        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)

        # Measure First Token Generation Time
        start_time = time.time()
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=1,  # Only generate the first token
                do_sample=False
            )
        first_token_time = time.time() - start_time

        # Measure Full Response Generation Time
        start_response_time = time.time()
        with torch.no_grad():
            output_generator = model.generate(
                input_ids,
                max_length=1000,
                top_p=0.9,
                temperature=0.3,
                repetition_penalty=1.2,
                return_dict_in_generate=True,
                output_scores=True
            )
        total_time = time.time() - start_response_time

        # Decode Output
        generated_text = tokenizer.decode(
            output_generator.sequences[0], skip_special_tokens=True
        ).strip()

        # Compute Token Metrics
        num_tokens = len(tokenizer.encode(generated_text))
        tokens_per_second = round(num_tokens / total_time, 2) if total_time > 0 else "N/A"

        return {
            "response": generated_text,
            "first_token_time": round(first_token_time, 3),
            "inference_time": round(total_time, 2),
            "tokens_generated": num_tokens,
            "tokens_per_second": tokens_per_second,
        }
    except Exception as e:
        print("Error in generate_response:", traceback.format_exc())
        return None

# === List of Questions ===
QUESTIONS = [
    "Explain SnapMirror",
    "How does NetApp ONTAP work?",
    "How are Snapshots useful in real-world applications?",
    "Explain the Spiral Model",
    "What is FlexClone?"
]

# === Loop Through Questions and Display Results ===
for i, question in enumerate(QUESTIONS):
    print("\n" + "=" * 50)
    print(f"Question {i+1}: {question}")

    result = generate_response(question)
    if result:
        print(f"Response: {result['response']}\n")
        print(f"Inference Time: {result['inference_time']} sec")
        print(f"First Token Generation Time: {result['first_token_time']} sec")
        print(f"Tokens Generated: {result['tokens_generated']}")
        print(f"Tokens Per Second: {result['tokens_per_second']}")

print("\nInference completed for all topics.")

