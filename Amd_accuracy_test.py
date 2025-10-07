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
questions =['How many passengers were on board?', 'How many male passengers were there?', 'How many female passengers were there?', 'What was the average age of all passengers?', 'What was the youngest passenger’s age?', 'What was the oldest passenger’s age?', 'What was the average fare paid?', 'How many passengers embarked at Southampton?', 'How many passengers embarked at Cherbourg?', 'How many passengers embarked at Queenstown?', 'What was the highest fare paid?', 'What was the lowest fare paid?', 'How many passengers survived?', 'How many passengers did not survive?', 'What was the survival rate among all passengers?', 'What was the average age by passenger class?', 'Which class had the highest average fare?', 'What percentage of male passengers survived?', 'What percentage of female passengers survived?', 'What was the average fare by embarkation port?', 'How many passengers traveled with family (SibSp + Parch > 0)?', 'How many passengers traveled alone?', 'Which passenger class had the most survivors?', 'What was the average age of survivors?', 'What was the average age of non-survivors?', 'What was the survival rate by class?', 'How many passengers were under 18 years old?', 'What was the average fare for survivors vs. non-survivors?', 'How many passengers had missing cabin data?', 'How many unique ticket numbers were there?', 'Find the most common surname on board.', 'Find the top 5 passengers who paid the highest fares.', 'List average age of survivors by gender.', 'Find how many passengers had family aboard but did not survive.', 'How many passengers paid above the overall average fare?', 'Find the number of male survivors in 1st class.', 'Find survival rate among passengers under 18.', 'How many different cabin codes exist?', 'What was the most common embarkation port among survivors?', 'What was the average age of passengers on board?', 'What was the average fare paid by passengers?', 'How many passengers were on board the Titanic?', 'How many passengers survived the disaster?', 'What percentage of passengers survived?', 'How many male passengers were on board?', 'How many female passengers were on board?', 'What was the average age of passengers who survived?', 'What was the average age of passengers who did not survive?', 'What was the age of the youngest passenger?', 'What was the survival rate among passengers in class 1?', 'What was the survival rate among passengers in class 2?', 'What was the survival rate among passengers in class 3?', 'What was the survival rate among male passengers?', 'What was the survival rate among female passengers?', 'How many passengers embarked from port S?', 'How many passengers embarked from port C?', 'How many passengers embarked from port Q?', 'What was the highest fare paid by a passenger?', 'What was the lowest fare paid by a passenger?', 'What was the median fare paid by passengers?', 'What was the median age of passengers?', 'How many passengers had siblings or spouses aboard?', 'How many passengers had parents or children aboard?', 'What was the average number of siblings/spouses aboard per passenger?', 'What was the average number of parents/children aboard per passenger?', 'What was the min of fare among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the max of parch among all passengers?', 'What was the max of fare among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the max of sibsp among all passengers?', 'What was the max of age among all passengers?', 'What was the max of sibsp among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the max of parch among all passengers?', 'What was the max of parch among all passengers?', 'What was the min of age among all passengers?', 'What was the min of parch among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the min of age among all passengers?', 'What was the max of sibsp among all passengers?', 'What was the max of sibsp among all passengers?', 'What was the max of age among all passengers?', 'What was the max of fare among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the max of age among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the min of fare among all passengers?', 'What was the min of parch among all passengers?', 'What was the min of age among all passengers?', 'What was the max of fare among all passengers?', 'What was the max of fare among all passengers?', 'What was the min of fare among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the min of parch among all passengers?', 'What was the max of age among all passengers?', 'What was the max of parch among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the max of parch among all passengers?', 'What was the max of age among all passengers?', 'What was the max of parch among all passengers?', 'What was the min of fare among all passengers?', 'What was the min of age among all passengers?', 'What was the min of parch among all passengers?', 'What was the max of fare among all passengers?', 'What was the max of parch among all passengers?', 'What was the min of parch among all passengers?', 'What was the min of parch among all passengers?', 'What was the max of sibsp among all passengers?', 'What was the max of age among all passengers?', 'What was the min of sibsp among all passengers?', 'What was the min of parch among all passengers?', 'What was the max of fare among all passengers?']

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
