import os
import pandas as pd
import time
from sqlalchemy import create_engine
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import torch

# ---------------------------
# CONFIG
# ---------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
login(token=HF_TOKEN)
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

csv_files = ["titanic.csv"]
questions = import os
import pandas as pd
import time
from sqlalchemy import create_engine
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import torch

# ---------------------------
# CONFIG
# ---------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
login(token=HF_TOKEN)
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

csv_files = ["titanic.csv"]
questions = import os
import pandas as pd
import time
from sqlalchemy import create_engine
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import torch

# ---------------------------
# CONFIG
# ---------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
login(token=HF_TOKEN)
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

csv_files = ["titanic.csv"]
questions = [
    "What percentage of passengers survived?",
    "How many passengers were in each class (1st, 2nd, 3rd)?",
    "What was the average age of passengers on board?",
    "How many passengers traveled with siblings or spouses",
    "Fare paid by passengers",
    "How many passengers embarked from each port (S, C, Q)?",
    "What was the survival rate based on passenger class?",
    "What was the average fare paid by passengers in each class?"
]

db_file = "demo_rag.db"
results_file = "answers.csv"

# ---------------------------
# CREATE SQLITE DATABASE FROM CSV FILES
# ---------------------------
def create_db_from_csv_files(csv_files, db_file):
    import sqlite3
    conn = sqlite3.connect(db_file)
    for idx, file in enumerate(csv_files):
        df = pd.read_csv(file)
        table_name = f"table_{idx}"
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Created table: {table_name} from {file}")
    conn.close()

# ---------------------------
# LOAD MISTRAL 7B MODEL (CUDA)
# ---------------------------
def load_mistral_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3"
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        trust_remote_code=True,
        device_map="auto",  # Automatically load to available CUDA device
        torch_dtype=torch.float16  # Use FP16 for faster inference on NVIDIA GPUs
    )

    text_gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,  # Use CUDA device 0
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=True
    )

    llm_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return llm_pipeline

# ---------------------------
# RUN QUESTIONS THROUGH SQL + LLM CHAIN
# ---------------------------
def run_structured_rag(db_file, questions, llm_pipeline):
    db = SQLDatabase(create_engine(f"sqlite:///{db_file}"))
    chain = create_sql_query_chain(llm_pipeline, db, k=5)

    results = []
    for question in questions:
        start_time = time.time()
        output = chain.invoke({"question": question})
        end_time = time.time()

        answer = output if isinstance(output, str) else str(output)
        tokens = len(answer.split())
        tps = tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Tokens: {tokens}, Tokens/sec: {tps:.2f}")
        print("-" * 50)

        results.append({
            "question": question,
            "answer": answer,
            "tokens": tokens,
            "tokens_per_sec": tps
        })

    return results

# ---------------------------
# SAVE RESULTS TO CSV
# ---------------------------
def save_results(results, results_file):
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    create_db_from_csv_files(csv_files, db_file)
    mistral_pipeline = load_mistral_model()
    results = run_structured_rag(db_file, questions, mistral_pipeline)
    save_results(results, results_file)

db_file = "demo_rag.db"
results_file = "answers.csv"

# ---------------------------
# CREATE SQLITE DATABASE FROM CSV FILES
# ---------------------------
def create_db_from_csv_files(csv_files, db_file):
    import sqlite3
    conn = sqlite3.connect(db_file)
    for idx, file in enumerate(csv_files):
        df = pd.read_csv(file)
        table_name = f"table_{idx}"
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Created table: {table_name} from {file}")
    conn.close()

# ---------------------------
# LOAD MISTRAL 7B MODEL (CUDA)
# ---------------------------
def load_mistral_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3"
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        trust_remote_code=True,
        device_map="auto",  # Automatically load to available CUDA device
        torch_dtype=torch.float16  # Use FP16 for faster inference on NVIDIA GPUs
    )

    text_gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,  # Use CUDA device 0
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=True
    )

    llm_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return llm_pipeline

# ---------------------------
# RUN QUESTIONS THROUGH SQL + LLM CHAIN
# ---------------------------
def run_structured_rag(db_file, questions, llm_pipeline):
    db = SQLDatabase(create_engine(f"sqlite:///{db_file}"))
    chain = create_sql_query_chain(llm_pipeline, db, k=5)

    results = []
    for question in questions:
        start_time = time.time()
        output = chain.invoke({"question": question})
        end_time = time.time()

        answer = output if isinstance(output, str) else str(output)
        tokens = len(answer.split())
        tps = tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Tokens: {tokens}, Tokens/sec: {tps:.2f}")
        print("-" * 50)

        results.append({
            "question": question,
            "answer": answer,
            "tokens": tokens,
            "tokens_per_sec": tps
        })

    return results

# ---------------------------
# SAVE RESULTS TO CSV
# ---------------------------
def save_results(results, results_file):
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    create_db_from_csv_files(csv_files, db_file)
    mistral_pipeline = load_mistral_model()
    results = run_structured_rag(db_file, questions, mistral_pipeline)
    save_results(results, results_file)
db_file = "demo_rag.db"
results_file = "answers.csv"

# ---------------------------
# CREATE SQLITE DATABASE FROM CSV FILES
# ---------------------------
def create_db_from_csv_files(csv_files, db_file):
    import sqlite3
    conn = sqlite3.connect(db_file)
    for idx, file in enumerate(csv_files):
        df = pd.read_csv(file)
        table_name = f"table_{idx}"
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Created table: {table_name} from {file}")
    conn.close()

# ---------------------------
# LOAD MISTRAL 7B MODEL (CUDA)
# ---------------------------
def load_mistral_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3"
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        trust_remote_code=True,
        device_map="auto",  # Automatically load to available CUDA device
        torch_dtype=torch.float16  # Use FP16 for faster inference on NVIDIA GPUs
    )

    text_gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,  # Use CUDA device 0
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=True
    )

    llm_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return llm_pipeline

# ---------------------------
# RUN QUESTIONS THROUGH SQL + LLM CHAIN
# ---------------------------
def run_structured_rag(db_file, questions, llm_pipeline):
    db = SQLDatabase(create_engine(f"sqlite:///{db_file}"))
    chain = create_sql_query_chain(llm_pipeline, db, k=5)

    results = []
    for question in questions:
        start_time = time.time()
        output = chain.invoke({"question": question})
        end_time = time.time()

        answer = output if isinstance(output, str) else str(output)
        tokens = len(answer.split())
        tps = tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Tokens: {tokens}, Tokens/sec: {tps:.2f}")
        print("-" * 50)

        results.append({
            "question": question,
            "answer": answer,
            "tokens": tokens,
            "tokens_per_sec": tps
        })

    return results

# ---------------------------
# SAVE RESULTS TO CSV
# ---------------------------
def save_results(results, results_file):
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    create_db_from_csv_files(csv_files, db_file)
    mistral_pipeline = load_mistral_model()
    results = run_structured_rag(db_file, questions, mistral_pipeline)
    save_results(results, results_file)
