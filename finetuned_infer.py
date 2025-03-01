import os
import time
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from peft import PeftModel

# === Environment Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# === Model Configuration ===
BASE_MODEL = "meta-llama/Llama-2-13b-hf"
ADAPTER_PATH = "peft_FT_llama2_13b_on_prompt_res_dataset"
TOKEN_VALUE = "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"

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN_VALUE)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# === Configure BitsAndBytes for Quantization ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# === Load Base Model with LoRA Adapter ===
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, token=TOKEN_VALUE,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()

# === Load Vector Database ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vector_db_path = "second_vector_db"
vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

def retrieve_context(query, top_k=3):
    """Retrieves the most relevant documents from the FAISS vector database."""
    try:
        return vector_db.similarity_search(query, top_k) or []
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return []

def augment_prompt(source_knowledge, query):
    """Constructs a prompt using retrieved knowledge."""
    try:
        if source_knowledge:
            context = "\n".join([x.page_content for x in source_knowledge])
            return f"""
            [INST] <>
            Use the following context to answer the question at the end. Do not use any other information.
            If you can't find the relevant information in the context, just say you don't have enough information.
            <>
            {context}
            Question: {query} [/INST]
            """
        return "[INST] <> We are unable to answer the question. <> [/INST]"
    except Exception as e:
        print(f"Error in augment_prompt: {str(e)}")
        return "[INST] <> We encountered an error. <> [/INST]"

# === Set Up HuggingFacePipeline ===
hf_pipeline = pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    temperature=0.01,
    max_new_tokens=512,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# === Define Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="You are an AI assistant. Answer the following question based on your knowledge:\n\nQuestion: {question}\nAnswer:"
)

# === Initialize LLMChain ===
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

def generate_response(question):
    """Generates a response using the LLM and measures performance metrics."""
    try:
        source_knowledge = retrieve_context(question)
        full_prompt = augment_prompt(source_knowledge, question)
        
        # Measure Inference Time
        start_time = time.time()
        response = llm_chain.run({"question": question})
        total_time = round(time.time() - start_time, 2)

        num_tokens = len(tokenizer.encode(response))
        tokens_per_second = round(num_tokens / total_time, 2) if total_time > 0 else "N/A"

        return {
            "response": response,
            "inference_time": total_time,
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
        print(f"Tokens Generated: {result['tokens_generated']}")
        print(f"Tokens Per Second: {result['tokens_per_second']}")

print("\nInference completed for all topics.")
