from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def get_text_generator(model_path, **kwargs):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, **kwargs)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=False,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs
        )

        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=kwargs.get('device_map', 'auto')
        )

        return tokenizer, model, text_generator

    except OSError as e:
        print(f"Error: Could not find model files at the specified path: {model_path}")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")