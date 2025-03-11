@celery.task()
def finetuned_model_infernce_api(workflow_id,modelType,data):
    print("fientuned model inference APi called",workflow_id,modelType,data)
    workflow_title=get_workflow_title(workflow_id)
    query="Explain about Spiral model?"
    if workflow_title=="Structured RAG":
        print("Structured RAG workflow !!")
    else:
        doc_res=find_document_by_workflow_id(workflow_id)
        vectordb,embeddings,embeddings,modelname,search_type,nearest_neighbour,indexing_type,llm_model_to_use=get_process_work_flow_data_inference(workflow_id,doc_res)
        files_list=get_files_list(workflow_id)
        print("vector_db:",vectordb)
        print("embeddings:",embeddings)
        print("modelname:",modelname)
        print("serachtype:",search_type)
        print("nearestneighbours:",nearest_neighbour)
        print("indexing_type:",indexing_type)
        print("llm_model_to_use:",llm_model_to_use)
        is_finetuned=is_finetuning_completed(workflow_id)
        print("is finetuned mode is avaialble ?:",is_finetuned)
        if modelType=="finetuned_model":
                unload_model()
                unload_mistral_model()
                if is_finetuned:
                    pass
                    if llm_model_to_use=="llama":
                        BASE_MODEL = "meta-llama/Llama-2-7b-hf"
                        #ADAPTER_PATH = "peft_FT_llama2_13b_on_prompt_res_dataset"
                        TOKEN_VALUE = "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"
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

                        adaptive_layer_folder=download_and_extract_zip(workflow_id, "/home/avinash_dataneuron_ai/jupy/rag-env/Unififed_flow/rag-product-code/Celery_app/")

                        print("adaptive layerfolder:",adaptive_layer_folder)
                        # === Load Base Model with LoRA Adapter ===
                        model = AutoModelForCausalLM.from_pretrained(
                            BASE_MODEL,
                            token=TOKEN_VALUE,
                            quantization_config=bnb_config,  # ✅ Attach BitsAndBytesConfig here
                            torch_dtype=torch.bfloat16,
                            device_map="auto"
                        )
                        model = PeftModel.from_pretrained(model, adaptive_layer_folder+"/peft_model/")
                        model = model.merge_and_unload()
                        hf_pipeline = pipeline(
                            model=model, tokenizer=tokenizer,
                            return_full_text=True,
                            task='text-generation',
                            temperature=0.01,
                            max_new_tokens=512,
                            repetition_penalty=1.1
                        )
                        
                        llm = HuggingFacePipeline(pipeline=hf_pipeline)
                        prompt_template = PromptTemplate(
                        input_variables=["question"],
                        template="You are an AI assistant. Answer the following question based on your knowledge:\n\nQuestion: {question}\nAnswer:"
                        )
                        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
                        
                    workflow_embedding=load_embeddings_for_rag_inference(workflow_id,embeddings,modelname)
                    if indexing_type=="Langchain Indexing":
                        #vector_db_path=get_vector_db_path_while_loading(workflow_id,vectordb,files_list["fileType"],indexing_type,version)
                        vector_db_path=get_vector_db_path_while_loading(workflow_id,vectordb,files_list["fileType"],indexing_type)
                        print("700 vector db in langcain index loading:",vector_db_path,vectordb)
                        vector_db=load_vectordb_from_path_for_rag_inference(workflow_id,vectordb,vector_db_path,workflow_embedding)
                    else:
                        #vector_db_path=get_vector_db_path_while_loading(workflow_id,vectordb,files_list["fileType"],indexing_type,version)
                        vector_db_path=get_vector_db_path_while_loading(workflow_id,vectordb,files_list["fileType"],indexing_type)
                        print("704 vector db in lllma index loading:",vector_db_path,vectordb)
                        vector_db=load_vector_db_for_llama_index(workflow_id,vectordb,vector_db_path,workflow_embedding)
                    final_result=[]
                    for key, value in data.items():
                        print(f"id name: {key}, query: {value}")
                        if search_type == "similarity search":
                            if indexing_type=="Langchain Indexing":
                                vec_results = vector_db.similarity_search(query, k=int(nearest_neighbour))
                            
                            else:
                                query_engine = vector_db.as_query_engine(similarity_top_k=int(nearest_neighbour))
                                vec_results= query_engine.query(query)
                                vec_results=vec_results.source_nodes
                                vec_results = [langchain_Doc(page_content=node.node.text,metadata=node.node.metadata) for node in vec_results]
                        else:
                            if indexing_type=="Langchain Indexing":
                                vec_results = vector_db.max_marginal_relevance_search(query, k=int(nearest_neighbour))
                            else:
                                query_engine = vector_db.as_query_engine(vector_store_query_mode="mmr",similarity_top_k=int(nearest_neighbour))
                                vec_results= query_engine.query(query)
                                vec_results=vec_results.source_nodes
                                vec_results = [langchain_Doc(page_content=node.node.text,metadata=node.node.metadata) for node in vec_results]
                        full_prompt = augment_prompt(vec_results, query,indexing_type,prompt_layer_info={"last_part1":"","last_part2":""})
                        start_time = time.time()
                        response = llm_chain.run({"question": query})
                        total_time = round(time.time() - start_time, 2)
                        num_tokens = len(tokenizer.encode(response))
                        tokens_per_second = round(num_tokens / total_time, 2) if total_time > 0 else "N/A"
                        cals={
                            "Id":key,
                            "prompt":value,
                            "response": response,
                            "inference_time": total_time,
                            "tokens_generated": num_tokens,
                            "tokens_per_second": tokens_per_second,
                        }
                        final_result.append(cals)
                    del llm,llm_chain,model,tokenizer,hf_pipeline
                    final_response_to_send={
                        "predicted_results":final_result,
                        "result":"prediction success",
                        "state":"SUCCESS",
                        "message":"prediction success"
                    }
                    print("final_result:",final_response_to_send)
                    return final_response_to_send
                else:
                    print("fientuend model is not avaialble !!!")
        else:
            print("Need to infer the Base mdodel")
        
    
        

import os
import zipfile
from google.cloud import storage

def download_and_extract_zip(workflow_id, destination_folder):
    """
    Download the correct file from GCS, extract it, and return the extracted folder path.
    """
    try:
        storage_client = storage.Client.from_service_account_json(GOOGLE_CREDINTIALS_JSON)
        bucket = storage_client.bucket("bkt-dataneuron-prod-as1-dsalpartifacts")

        # ✅ Correct GCS path based on actual file name
        zip_blob_name = f"llm-model/{workflow_id}_adaptive_layer"  # No .zip
        local_zip_path = os.path.join(destination_folder, f"{workflow_id}_adaptive_layer.zip")
        extracted_folder_path = os.path.join(destination_folder, workflow_id)

        # ✅ Ensure local destination folder exists
        os.makedirs(destination_folder, exist_ok=True)

        # ✅ Check if file exists in GCS before downloading
        blob = bucket.blob(zip_blob_name)
        if not blob.exists(storage_client):
            raise FileNotFoundError(f"❌ GCS file not found: {zip_blob_name}")

        # ✅ Download file
        blob.download_to_filename(local_zip_path)
        print(f"✅ Downloaded: {zip_blob_name} -> {local_zip_path}")

        # ✅ Extract ZIP file
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_path)
        print(f"✅ Extracted to: {extracted_folder_path}")

        # ✅ Cleanup (optional: delete ZIP after extraction)
        os.remove(local_zip_path)

        return extracted_folder_path
    except Exception as e:
        print(f"❌ Error in download_and_extract_zip: {e}")
        return None


def is_finetuning_completed(workflow_id):
    """
    Check if the fine-tuning process is completed for a given workflowId.
    
    :param workflow_id: The workflowId to check.
    :return: Boolean indicating whether fine-tuning is completed.
    """
    db = client['dn-rag-db']
    collection = db["workflows"]
    
    workflow_data = collection.find_one({"workflowId": workflow_id})
    if not workflow_data:
        return False  # Workflow ID not found
    
    fine_tune_configs = workflow_data.get("dsInfo", {}).get("fineTuneConfigs", {})
    return fine_tune_configs.get("isFineTuningCompleted", False)



@app.route("/inference_for_finetuned_model",methods=["POST"])
#@prediction_token_required
def inference_for_finetuned_model():
    original_token = request.headers.get('x-access-token')  # Safer way to access headers
    content = request.json  # Remove duplicate assignment
    workflow_id = content.get('workflowId')
    modelType = content.get("modelType")
    data = content.get('data')

    if not workflow_id or not modelType or not data:
        return {"error": "Missing required fields"}, 400

    print("Got the request!")
    print("WorkflowID:", workflow_id)
    print("ModelType:", modelType)
    print("Data:", data)

    # Corrected task name
    r = simple_app.send_task('tasks.finetuned_model_infernce_api', 
                             kwargs={"workflow_id": workflow_id, "modelType": modelType, "data": data})

    app.logger.info(r.backend)
    return {"task_id": r.id}, 200    
