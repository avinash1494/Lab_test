
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import time
import traceback
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp

# === Model Configuration ===
MODEL_NAME_OR_PATH = "TheBloke/Llama-2-13B-chat-GGML"
MODEL_BASENAME = "llama-2-13b-chat.ggmlv3.q5_1.bin"

# Download the model from Hugging Face
model_path = hf_hub_download(repo_id=MODEL_NAME_OR_PATH, filename=MODEL_BASENAME)

# GPU and batch configuration (adjust as needed)
N_GPU_LAYERS = -1  # Use GPU acceleration (-1 for full GPU)
N_BATCH = 512  # Adjust batch size based on your system

# === Initialize LLM Model ===
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=1840,
    n_gpu_layers=N_GPU_LAYERS,
    n_batch=N_BATCH,
    n_ctx=4900,
    callback_manager=callback_manager,
    verbose=True,
)
print("result:",llm("what is GLobal warming"))

import time
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# === Model Configuration ===
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
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
full_prompt = "Explain Global warming"
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
