def generate_response(question):
    """
    Generates a response using the LLM and measures performance metrics accurately.
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
            first_token_output = model.generate(
                input_ids,
                max_new_tokens=1,  # Only generate the first token
                do_sample=False
            )
        
        first_token_time = time.time() - start_time  # Time for first token

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

        total_time = time.time() - start_response_time  # Only response generation time

        # Decode Output
        generated_text = tokenizer.decode(output_generator.sequences[0], skip_special_tokens=True).strip()
        generated_text = generated_text.split("Answer")[-1]

        # Compute Token Metrics
        num_tokens = len(tokenizer.encode(generated_text))  # More accurate token count
        tokens_per_second = round(num_tokens / total_time, 2) if total_time > 0 else "N/A"

        return {
            "response": generated_text,
            "first_token_time": round(first_token_time, 3),
            "inference_time": round(total_time, 2),  # Now only response generation time
            "tokens_generated": num_tokens,
            "tokens_per_second": tokens_per_second,
        }
    except Exception as e:
        print("Error in generate_response:", traceback.format_exc())
        return None
