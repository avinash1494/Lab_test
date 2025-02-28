# import os
# import time
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings

# # Specify the directory containing the PDFs
# pdf_directory = '/home/avinash_dataneuron_ai/jupy/jup_notebook/Netapp_vector_store_testong/'  # Replace with your actual directory path

# # Get a list of all PDF files in the directory
# pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# print("all files:",pdf_files)


# # Initialize embedding model once (to avoid loading it multiple times)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})


# # Start timer for the entire process
# start_time = time.time()

# total_docs=[]
# # Loop over all PDF files and process each one
# for pdf_file in pdf_files:
#     print(f"Processing file: {pdf_file}")
#     pdf_start_time = time.time()
#     # Start timer for the current PDF file
    
    
#     # Step 1: Load the PDF
#     load_start_time = time.time()
#     loader = PyPDFLoader(os.path.join(pdf_directory, pdf_file))
#     pages = loader.load_and_split()
#     load_end_time = time.time()
#     print(f"PDF loading and splitting time: {load_end_time - load_start_time:.2f} seconds")
    
#     # Step 2: Split the text
#     split_start_time = time.time()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     splited_docs = text_splitter.split_documents(pages)
#     total_docs.extend(splited_docs)
#     split_end_time = time.time()
#     print(f"Text splitting time: {split_end_time - split_start_time:.2f} seconds")

# print("total docs length:",len(total_docs))

# #input()
# print("pushoing docs to the faiss store started !!!")
# # Step 3: Generate embeddings
# embedding_start_time = time.time()
# db_pdf = FAISS.from_documents(total_docs, embeddings)
# embedding_end_time = time.time()
# print(f"Embedding generation time: {embedding_end_time - embedding_start_time:.2f} seconds")

# # Step 4: Save the vector store
# save_start_time = time.time()

# db_pdf.save_local("second_vector_db")
# save_end_time = time.time()
# print(f"Vector store saving time: {save_end_time - save_start_time:.2f} seconds")

# # End timer for the current PDF file
# pdf_end_time = time.time()
# print(f"Total processing time for {pdf_file}: {pdf_end_time - pdf_start_time:.2f} seconds\n")

# # End timer for the entire process
# end_time = time.time()
# print(f"Total elapsed time for all PDFs: {end_time - start_time:.2f} seconds")



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
N_GPU_LAYERS = 31  # Use GPU acceleration (-1 for full GPU)
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

# === Load Vector Database ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vector_db_path = "second_vector_db"
vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

# === Function to Augment Prompt and Measure Inference Time ===
def augment_prompt(source_knowledge, query):
    """
    Constructs a prompt using source knowledge and queries the LLM.
    Measures inference time and token usage.
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
            prompt_template = "[INST] <> Please provide a formal reply: 'We are unable to answer the question.' [/INST]"

        # === Measure Inference Time ===
        start_time = time.time()
        llm_result = llm(prompt_template)
        end_time = time.time()

        # === Token Usage (Estimated) ===
        num_tokens = len(prompt_template.split())  # Approximate token count

        # === Collect Performance Metrics ===
        metrics = {
            "inference_time": round(end_time - start_time, 4),  # Time in seconds
            "token_count": num_tokens,
            "tokens_per_second": round(num_tokens / (end_time - start_time), 2) if (end_time - start_time) > 0 else "N/A",
            "response": llm_result
        }

        return metrics
    except Exception as e:
        error_msg = {
            "task": "augment_prompt",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print("Error Message:", error_msg)
        return None

# === List of Questions ===
questions = [
    "Explain SnapMirror",
    "How does NetApp ONTAP work?",
    "How are Snapshots useful in real-world applications?",
    "Explain the Spiral Model",
    "What is FlexClone?"
]

# === Process Each Query ===
NEAREST_NEIGHBOR = 5

for query in questions:
    print("\n" + "="*50)
    print(f"Query: {query}")
    print("vector DB:",vector_db)
    # Retrieve similar documents
    source_knowledge = vector_db.similarity_search(query, k=NEAREST_NEIGHBOR)

    # Generate response and track performance
    result = augment_prompt(source_knowledge, query)

    # Print results
    if result:
        print(f"Response: {result['response']}")
        print(f"Inference Time: {result['inference_time']} seconds")
        print(f"Token Count: {result['token_count']}")
        print(f"Tokens per Second: {result['tokens_per_second']}")
    else:
        print("Failed to generate a response.")

print("\nProcessing Complete.")
