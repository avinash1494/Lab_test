import time
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from peft import PeftModel
import os

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Model Configuration
BASE_MODEL = "meta-llama/Llama-2-13b-hf"
ADAPTER_PATH = "peft_FT_llama2_13b_on_prompt_res_dataset"
TOKEN_VALUE = "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"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN_VALUE)

# Configure BitsAndBytes for Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, token=TOKEN_VALUE,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA Adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()

# Set Padding
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load Vector Database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vector_db = FAISS.load_local("second_vector_db", embeddings, allow_dangerous_deserialization=True)

# Retrieve Context from FAISS
def retrieve_context(query, top_k=3):
    try:
        return vector_db.similarity_search(query, top_k) or []
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return []

# Generate Response
def generate_response(question):
    try:
        source_knowledge = retrieve_context(question)
        context = "\n".join([x.page_content for x in source_knowledge]) if source_knowledge else ""
        input_text = f"{context}\nQuestion: {question}" if context else question
        
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        
        # Measure Inference Time
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=1000,
                top_p=0.9,
                temperature=0.3,
                repetition_penalty=1.2
            )
        total_time = round(time.time() - start_time, 2)
        
        # Decode Response
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        
        # Compute Token Metrics
        num_tokens = len(tokenizer.encode(generated_text))
        tokens_per_second = round(num_tokens / total_time, 2) if total_time > 0 else "N/A"
        
        return {
            "response": generated_text,
            "inference_time": total_time,
            "tokens_generated": num_tokens,
            "tokens_per_second": tokens_per_second
        }
    except Exception as e:
        print("Error in generate_response:", traceback.format_exc())
        return None

# List of Questions
QUESTIONS = [
    "Explain SnapMirror",
    "How does NetApp ONTAP work?",
    "How are Snapshots useful in real-world applications?",
    "Explain the Spiral Model",
    "What is FlexClone?"
]

# Run Inference
for i, question in enumerate(QUESTIONS):
    print("\n" + "=" * 50)
    print(f"Question {i+1}: {question}")
    result = generate_response(question)
    if result:
        print(f"Response: {result['response']}\n")
        print(f"Inference Time: {result['inference_time']} sec")
        print(f"Tokens Generated: {result['tokens_generated']}")
        print(f"Tokens Per Second: {result['tokens_per_second']}")

print("\nInference completed for all topics.")
