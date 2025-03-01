import time
import torch
import traceback
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from peft import PeftModel
import os

# === Environment Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# === Model Configuration ===
BASE_MODEL = "meta-llama/Llama-2-13b-hf"
ADAPTER_PATH = "peft_FT_llama2_13b_on_prompt_res_dataset"
TOKEN_VALUE = "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"

# === Load Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN_VALUE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, token=TOKEN_VALUE,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()

# === Create Inference Pipeline ===
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Load Vector Database ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vector_db_path = "second_vector_db"
vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

# === Retrieve Context from FAISS ===
def retrieve_context(query, top_k=3):
    try:
        retrieved_docs = vector_db.similarity_search(query, top_k)
        return retrieved_docs if retrieved_docs else []
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return []

# === Generate Response Using Pipeline ===
def generate_response(question):
    try:
        source_knowledge = retrieve_context(question)
        context = "\n".join([x.page_content for x in source_knowledge]) if source_knowledge else ""
        input_text = f"{context}\nQuestion: {question}" if context else question
        
        start_time = time.time()
        response = llm_pipeline(input_text, max_length=1000, top_p=0.9, temperature=0.3, repetition_penalty=1.2)[0]["generated_text"]
        total_time = time.time() - start_time
        response=response.split("Answer:")[-1]
        print("response:",response)
        num_tokens = len(tokenizer.encode(response))
        tokens_per_second = round(num_tokens / total_time, 2) if total_time > 0 else "N/A"
        
        return {
            "response": response,
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

# === Run Inference ===
for i, question in enumerate(QUESTIONS):
    print("\n" + "=" * 50)
    print(f"Question {i+1}: {question}")
    init_start_time = time.time()
    result = generate_response(question)
    init_end_time = time.time()
    api_elapsed_time=round(init_end_time - init_start_time, 4)

    if result:
        print(f"Response: {result['response']}\n")
        print(f"Inference Time: {result['inference_time']} sec")
        print(f"Tokens Generated: {result['tokens_generated']}")
        print(f"Tokens Per Second: {result['tokens_per_second']}")
        print("ealseped time for API call:",api_elapsed_time)
print("\nInference completed for all topics.")
