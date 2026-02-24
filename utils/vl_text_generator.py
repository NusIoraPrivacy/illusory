from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM
import torch
import re

def extract_assistant_reply(text):
    # 匹配最后一个assistant role后的内容
    match = re.search(r"assistant\\n(.+)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"assistant: ?(.+)", text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return text.strip()

def get_vl_text_generator(model_path, model_name=None, return_model=False, **kwargs):
    name = model_name or model_path
    if "Qwen2.5-VL" in name or "qwen2.5-vl" in name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor as TransformersAutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float32, device_map="auto"
        )
        processor = TransformersAutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("Processor type:", type(processor))
        print("Has chat_template:", hasattr(processor, "chat_template"))
        if hasattr(processor, "chat_template"):
            print("chat_template preview:", str(processor.chat_template)[:200])
        else:
            print("No chat_template attribute in processor.")        
        def generate(messages, **gen_kwargs):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            # print("[主流程] pixel_values.shape:", inputs.get('pixel_values', None).shape if 'pixel_values' in inputs else None)

            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs.get('max_new_tokens', 1024)
                )
            if hasattr(processor, "batch_decode"):
                return processor.batch_decode(outputs[:, input_len:])[0].strip()
            else:
                return processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        if return_model:
            return generate, model, processor
        else:
            return generate
    elif "gemma-3-27b-it" in name or "gemma-3-27b-it" in name.lower():
        # gemma-3-27b-it 是一个多模态模型，支持图像处理
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch._dynamo as dynamo
            
            # 禁用torch编译优化以避免backend='inductor'错误
            dynamo.config.suppress_errors = True
            torch._dynamo.config.disable = True
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,  # gemma-3-27b-it 推荐使用 bfloat16
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            
            def generate(messages, **gen_kwargs):
                try:
                    # 确保torch编译被禁用
                    torch._dynamo.config.disable = True
                    
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(model.device, dtype=torch.bfloat16)
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                    input_len = inputs["input_ids"].shape[1]
                    
                    # 使用最简单的生成参数，避免所有可能导致错误的参数
                    with torch.inference_mode():
                        # 临时禁用torch编译
                        original_compile = torch._dynamo.config.disable
                        torch._dynamo.config.disable = True
                        
                        try:
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=gen_kwargs.get('max_new_tokens', 512),
                                do_sample=False,
                                use_cache=True,
                            )
                        finally:
                            # 恢复原始设置
                            torch._dynamo.config.disable = original_compile
                            
                    return processor.batch_decode(outputs[:, input_len:])[0].strip()
                except Exception as e:
                    print(f"gemma-3-27b-it 生成失败: {e}")
                    return f"模型生成错误: {str(e)}"
                    
            if return_model:
                return generate, model, processor
            else:
                return generate
        except Exception as e:
            print(f"加载 gemma-3-27b-it 模型时出错: {e}")
            error_msg = str(e)
            def error_generate(messages, **gen_kwargs):
                return f"模型加载错误: {error_msg}"
            if return_model:
                return error_generate, None, None
            else:
                return error_generate
    elif "Llama-4-Scout-17B-16E-Instruct" in name or "llama-4-scout-17b-16e-instruct" in name.lower():
        from transformers import AutoProcessor, Llama4ForConditionalGeneration
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_path,
            # attn_implementation="flex_attention",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        if hasattr(model, "vision_model"):
            model.vision_model = model.vision_model.half()
        processor = AutoProcessor.from_pretrained(model_path)
        def generate(messages, **gen_kwargs):
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs.get('max_new_tokens', 1024),
                    temperature=gen_kwargs.get('temperature', 1.0),
                    do_sample=True,
                    top_p=gen_kwargs.get('top_p', 0.95)
                )
            return processor.batch_decode(outputs[:, input_len:])[0].strip()
        if return_model:
            return generate, model, processor
        else:
            return generate
    else:
        def error_generate(messages, **gen_kwargs):
            return "不支持的模型类型"
        if return_model:
            return error_generate, None, None
        else:
            return error_generate
