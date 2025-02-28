import time
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# === Model Configuration ===
BASE_MODEL = "meta-llama/Llama-2-13b"
TOKEN_VALUE = "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"

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN_VALUE)

# === Configure BitsAndBytes for Quantization ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# === Load Base Model ===
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, token=TOKEN_VALUE,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Uncomment below lines if using LoRA Adapter
# model = PeftModel.from_pretrained(model, ADAPTER_PATH)
# model = model.merge_and_unload()

# === Set Padding ===
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
# === Load Vector Database ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vector_db_path = "second_vector_db"
vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

 === Retrieve Context from FAISS ===
def retrieve_context(query, top_k=3):
    """
    Retrieves the most relevant documents from the FAISS vector database.
    """
    try:
        retrieved_docs = vector_db.similarity_search(query, top_k)
        return retrieved_docs if retrieved_docs else []
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return []

# === Prompt Augmentation with Retrieved Context ===
def augment_prompt(source_knowledge, query):
    """
    Constructs a prompt using retrieved knowledge.
    """
    try:
        if source_knowledge:
            context = "\n".join([x.page_content for x in source_knowledge])
            prompt_template = f"""
            [INST] <>
            Use the following context to answer the question at the end. Do not use any other information. 
            If you can't find the relevant information in the context, just say you don't have enough information.
            <>
            {context}
            Question: {query} [/INST]
            """
        else:
            prompt_template = "[INST] <> We are unable to answer the question. <> [/INST]"

        return prompt_template
    except Exception as e:
        print(f"Error in augment_prompt: {str(e)}")
        return "[INST] <> We encountered an error. <> [/INST]"

# === Generate Response and Measure Performance ===
def generate_response(question):
    """
    Generates a response using the LLM and measures performance metrics.
    """
    try:
        # Retrieve relevant knowledge
        source_knowledge = retrieve_context(question)

        # Construct Augmented Prompt
        full_prompt = augment_prompt(source_knowledge, question)
        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)

        # Measure First Token Generation Time
        start_time = time.time()

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

        first_token_time = time.time() - start_time  # Time for first token
        total_time = time.time() - start_time  # Total inference time

        # Decode Output
        generated_text = tokenizer.decode(output_generator.sequences[0], skip_special_tokens=True).strip()

        # Compute Token Metrics
        num_tokens = output_generator.sequences.shape[1] - input_ids.shape[1]
        tokens_per_second = round(num_tokens / total_time, 2) if total_time > 0 else "N/A"

        return {
            "response": generated_text,
            "inference_time": round(total_time, 2),
            "first_token_time": round(first_token_time, 3),
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

    # Generate Response
    result = generate_response(question)

    if result:
        print(f"Response: {result['response']}\n")
        print(f"Inference Time: {result['inference_time']} sec")
        print(f"First Token Generation Time: {result['first_token_time']} sec")
        print(f"Tokens Generated: {result['tokens_generated']}")
        print(f"Tokens Per Second: {result['tokens_per_second']}")

print("\nInference completed for all topics.")

