import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import random
import os
import time
import argparse
import openai
from openai import OpenAI
from utils.model_path_map import LOCAL_MODEL_PATHS
from utils.text_generator import get_text_generator
import re
import gc
from explainability.integrated_gradients import integrated_gradients
from explainability.attention_rollout import attention_rollout
from explainability.rise_saliency import rise_saliency
from explainability.sae_attribution import sae_attribution


RANDOM_STATE = 42
N_EXPERIMENTS = 50 

prompt1 = """You are going to perform two tasks. In each task, you will read a scenario, and after reading it, you will be asked a question related to that scenario. My request is "You must provide your answers in each task."
Let me ask you a question to confirm your understanding:"""

prompt2="""How many tasks are you going to perform?"""

# Direct模式 - 只需要数字
def get_prompt_with_no_think(prompt_text, model_name, mode):
    """为 Qwen 3 模型添加 /no_think 标签"""
    if mode == "direct" and model_name and any(qwen_model in model_name.lower() for qwen_model in ["qwen3"]):
        return prompt_text + " /no_think"
    return prompt_text


prompt_instruction = "You are an objective and analytical assistant. When asked to assign a causal strength rating, you should (a) outline plausible alternative explanations, (b) weigh evidence for and against, (c) avoid overconfidence, and (d) justify your rating with acknowledgment of uncertainty."

prompt3_direct = """Now, please read Task 1: Imagine that you are one of the top administrators in your organization. You are in charge of running a number of aspects of the organization, including tracking the hours of all employees and their email and internet usage. You will soon be up for promotion. The day before your scheduled meeting with your superiors, you notice that the number of emails between your boss and the coworker sitting next to you jumps precipitously. 
When you meet with your boss, you are told you're not getting the promotion. 
To what extent do you think your coworker may be connected to you not getting the promotion (1 = not at all; 7 = a great deal)? Provide **only** a number between 1 and 7."""
prompt4_direct = """Please read Task 2: Imagine that you buy stock in one of the three construction companies that service your area. One day, your spouse, who runs the local bed and breakfast, notes that the families of all three company owners have checked into the B&B recently. Later, the prices all three companies offer for their services have risen drastically. 
Because of the higher prices, all three companies post very high profits, and you make a lot of money off of the stock you own. 
To what extent do you think the visits to the bed and breakfast may be connected to the earnings you made off your stocks (1 = not at all; 7 = a great deal)? Provide **only** your choice ("1", "2", "3", "4", "5", "6", "7")."""
prompt5_direct = """Now, please read Task 1: Imagine that you buy stock in one of the three construction companies that service your area. One day, your spouse, who runs the local bed and breakfast, notes that the families of all three company owners have checked into the B&B recently. Later, the prices all three companies offer for their services have risen drastically. 
Because of the higher prices, all three companies post very high profits, and you make a lot of money off of the stock you own. 
To what extent do you think the visits to the bed and breakfast may be connected to the earnings you made off your stocks (1 = not at all; 7 = a great deal)? Provide **only** a number between 1 and 7."""
prompt6_direct = """Please read Task 2: Imagine that you are one of the top administrators in your organization. You are in charge of running a number of aspects of the organization, including tracking the hours of all employees and their email and internet usage. You will soon be up for promotion. The day before your scheduled meeting with your superiors, you notice that the number of emails between your boss and the coworker sitting next to you jumps precipitously. 
When you meet with your boss, you are told you're not getting the promotion. 
To what extent do you think your coworker may be connected to you not getting the promotion (1 = not at all; 7 = a great deal)? Provide **only** your choice ("1", "2", "3", "4", "5", "6", "7")."""

# Explain模式 - 需要解释原因
prompt3_explain =  """Now, please read Task 1: Imagine that you are one of the top administrators in your organization. You are in charge of running a number of aspects of the organization, including tracking the hours of all employees and their email and internet usage. You will soon be up for promotion. The day before your scheduled meeting with your superiors, you notice that the number of emails between your boss and the coworker sitting next to you jumps precipitously. 
When you meet with your boss, you are told you're not getting the promotion. 
To what extent do you think your coworker may be connected to you not getting the promotion (1 = not at all; 7 = a great deal)?"""
prompt4_explain =  """Please read Task 2: Imagine that you buy stock in one of the three construction companies that service your area. One day, your spouse, who runs the local bed and breakfast, notes that the families of all three company owners have checked into the B&B recently. Later, the prices all three companies offer for their services have risen drastically. 
Because of the higher prices, all three companies post very high profits, and you make a lot of money off of the stock you own. 
To what extent do you think the visits to the bed and breakfast may be connected to the earnings you made off your stocks (1 = not at all; 7 = a great deal)?"""
prompt5_explain = """Now, please read Task 1: Imagine that you buy stock in one of the three construction companies that service your area. One day, your spouse, who runs the local bed and breakfast, notes that the families of all three company owners have checked into the B&B recently. Later, the prices all three companies offer for their services have risen drastically. 
Because of the higher prices, all three companies post very high profits, and you make a lot of money off of the stock you own. 
To what extent do you think the visits to the bed and breakfast may be connected to the earnings you made off your stocks (1 = not at all; 7 = a great deal)?"""
prompt6_explain = """Please read Task 2: Imagine that you are one of the top administrators in your organization. You are in charge of running a number of aspects of the organization, including tracking the hours of all employees and their email and internet usage. You will soon be up for promotion. The day before your scheduled meeting with your superiors, you notice that the number of emails between your boss and the coworker sitting next to you jumps precipitously. 
When you meet with your boss, you are told you're not getting the promotion. 
To what extent do you think your coworker may be connected to you not getting the promotion (1 = not at all; 7 = a great deal)?"""


class ChatGPT:
    def __init__(self, model, temperature=1):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()
    
    def ask(self, messages):
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature
        )
        return completion.choices[0].message.content

def run_task(model="Qwen2.5-14B-Instruct", temperature=1, n_cycles=100, mode="direct", max_new_tokens=1024, attribution_method="integrated_gradients"):
    if RANDOM_STATE:
        random.seed(RANDOM_STATE)
    
    repeat_times = n_cycles // 2
    if mode == "direct":
        # 为 Qwen 3 模型添加 /no_think 标签
        cycle_words1 = [prompt1, prompt2, get_prompt_with_no_think(prompt3_direct, model, mode), get_prompt_with_no_think(prompt4_direct, model, mode)]
        cycle_words2 = [prompt1, prompt2, get_prompt_with_no_think(prompt5_direct, model, mode), get_prompt_with_no_think(prompt6_direct, model, mode)]
    else:  
        cycle_words1 = [prompt1, prompt2, prompt3_explain, prompt4_explain]
        cycle_words2 = [prompt1, prompt2, prompt5_explain, prompt6_explain]
    
    all_cycles = [cycle_words1] * (n_cycles // 2) + [cycle_words2] * (n_cycles // 2)
    random.shuffle(all_cycles)
    
    # Initialize model
    if model in ["gpt-4-turbo", "gpt-4o", "GPT-3.5", "GPT-4"]:
        chat = ChatGPT(model, temperature)
        model_name = model
    else:
        model_path = LOCAL_MODEL_PATHS.get(model, model)
        model_name = model
        print(f"Loading model from: {model_path}")
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        tokenizer, model_hf, text_generator = get_text_generator(model_path, torch_dtype=torch.float16, device_map='auto')
    
    count_cycle1 = 0
    count_cycle2 = 0
    
    output_dir = './results/task3'
    explain_dir = './results/explainability/task3'
    os.makedirs(explain_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = f"model_{model_name}_mode_{mode}_temperature_{temperature}_cycles_{n_cycles}.txt"
    file_path = os.path.join(output_dir, file_name)
    explain_path = os.path.join(explain_dir, file_name)
    
    # 创建可解释性输出文件的头部
    out_prefix = f"results/explainability/task3/model_{model_name}_mode_{mode}_explainability"
    out_html = out_prefix + ".html"
    out_csv = out_prefix + ".csv"
    
    # 创建HTML文件头部
    with open(out_html, "w", encoding="utf-8") as f_html:
        f_html.write("<!DOCTYPE html>\n<html>\n<head>\n<title>Task3 Explainability Results</title>\n</head>\n<body>\n")
    
    # 创建CSV文件头部
    with open(out_csv, "w", encoding="utf-8", newline='') as f_csv:
        f_csv.write("cycle,token,score\n")
    
    with open(file_path, "w", encoding="utf-8") as f, open(explain_path, "w", encoding="utf-8") as fexp:
        def log_and_print(message):
            print(message, end='', flush=True)
            f.write(message)
        def log_explain(message):
            fexp.write(message)
        
        for i, current_prompts in enumerate(all_cycles):
            cycle_success = False
            if current_prompts == cycle_words1:
                cycle_info = f"====================\nCycle {i + 1}/{len(all_cycles)}: Selected cycle: 1\n====================\n"
                count_cycle1 += 1
            else:
                cycle_info = f"====================\nCycle {i + 1}/{len(all_cycles)}: Selected cycle: 2\n====================\n"
                count_cycle2 += 1
            log_and_print(cycle_info)
            log_explain(cycle_info)
            
            messages = []
            
            for prompt_content in current_prompts:
                log_and_print(f"[user]\n{prompt_content}\n--------------------\n")
                messages.append({"role": "user", "content": prompt_content})
                try:
                    # Closed-source
                    if model in ["gpt-4-turbo", "gpt-4o", "GPT-3.5", "GPT-4"]:
                        answer = chat.ask(messages)
                        messages.append({"role": "assistant", "content": answer})
                        log_and_print(f"[{model_name}]\n{answer}\n--------------------\n")
                    else:
                    # Open-source
                        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                            print("Applying Chat Template")
                            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.95)
                        else:
                            print("No Chat Template")
                            prompt_text = ""
                            for msg in messages:
                                if msg["role"] == "system":
                                    prompt_text += f"System: {msg['content']}\n\n"
                                elif msg["role"] == "user":
                                    prompt_text += f"User: {msg['content']}\n"
                                elif msg["role"] == "assistant":
                                    prompt_text += f"Assistant: {msg['content']}\n"
                            prompt_text += "Assistant: "
                            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.95)
                        outputs = text_generator(prompt_text, **gen_kwargs)
                        full_generated_text = outputs[0]["generated_text"]
                        answer = full_generated_text.split("<|im_start|>assistant\n")[-1].strip()
                        if "Llama-3.1-8B-Instruct" in model_name and mode == "direct":
                            answer = re.sub(r'<\|.*?\|>', '', answer)
                            answer = re.sub(r'(user|assistant|system):', '', answer)
                            paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
                            answer = paragraphs[-1] if paragraphs else answer
                        messages.append({"role": "assistant", "content": answer})
                        log_and_print(f"[{model_name}]\n{answer}\n--------------------\n")
                    cycle_success = True
                except Exception as e:
                    log_and_print(f"An error occurred during text generation: {e}\n")
                    log_explain(f"[Explain Error] {e}\n--------------------\n")
                    cycle_success = False
                    break

            # ====== 可解释性归因分析 ======
            # 只对用户的prompt进行归因
            if cycle_success and model not in ["gpt-4-turbo", "gpt-4o", "GPT-3.5", "GPT-4"] and attribution_method != "none":
                full_dialogue_text = ""
                for msg in messages:
                    if msg["role"] == "user":
                        full_dialogue_text += f"{msg['content']}\n"
                
                with open(out_html, "a", encoding="utf-8") as f_html, open(out_csv, "a", encoding="utf-8", newline='') as f_csv:
                    model_type = "llama" if "llama" in model_name.lower() else "qwen"
                    if attribution_method == "integrated_gradients":
                        html_snippet, filtered_tokens, rank_scores = integrated_gradients(model_hf, tokenizer, full_dialogue_text, out_prefix, mode="text", device=device, model_type=model_type, model_name=model_name)
                    elif attribution_method == "attention_rollout":
                        html_snippet, filtered_tokens, rank_scores = attention_rollout(model_hf, tokenizer, full_dialogue_text, out_prefix, mode="text", device=device, model_type=model_type, model_name=model_name, model_path=model_path)
                    elif attribution_method == "rise_saliency":
                        print("[RISE] RISE方法跳过文本归因")
                    elif attribution_method == "sae":
                        html_snippet, filtered_tokens, rank_scores = sae_attribution(model_hf, tokenizer, full_dialogue_text, out_prefix, mode="text", device=device, model_name=model_name, model_path=model_path, task=3, cycle_num=i+1)
                    else:
                        raise ValueError(f"不支持的归因方法: {attribution_method}")
                    f_html.write(f"<h2>Cycle {i+1}</h2>\n" + html_snippet + "<hr/>\n")
                    f_csv.write(f"Cycle {i+1}\n")
                    f_csv.write("token,score\n")
                    for token, score in zip(filtered_tokens, rank_scores):
                        f_csv.write(f"{token},{score}\n")
                    f_csv.write("====================\n")

            
            log_and_print("\n\n")
            log_explain("\n\n")
        
        log_and_print("\n====================\n")
        log_and_print("Experiment Complete.\n")
        log_and_print(f"Total Cycles: {len(all_cycles)}\n")
        log_and_print(f"Results logged to: {file_path}\n")
        log_and_print("====================\n")
        log_explain("\n====================\n")
        log_explain("Experiment Complete.\n")
        log_explain(f"Total Cycles: {len(all_cycles)}\n")
        log_explain(f"Results logged to: {explain_path}\n")
        log_explain("====================\n")
        
        # 添加HTML文件尾部
        with open(out_html, "a", encoding="utf-8") as f_html:
            f_html.write("</body>\n</html>")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2.5-14B-Instruct",
                       choices=["GPT-3.5", "GPT-4", "GPT-4o", 
                               "gemma-2-9b-it",
                               "gemma-3-27b-it",
                               "Llama-3-8B-Instruct",
                               "Llama-3.1-8B",
                               "Llama-3.0-8B-Instruct",
                               "Llama-3.1-8B-Instruct",
                               "Llama-3.2-1B-Instruct",
                               "Llama-3.2-3B-Instruct",
                               "Llama-3.2-11B-Vision",
                               "Llama-3.2-11B-Vision-Instruct",
                               "Llama-3.3-70B-Instruct",
                               "Llama-4-Scout-17B-16E-Instruct",
                               "Qwen2.5-7B-Instruct",
                               "Qwen2.5-14B-Instruct",
                               "Qwen2.5-VL-7B-Instruct",
                               "Qwen3-1.7B",
                               "Qwen3-4B",
                               "Qwen3-4B-Instruct-2507",
                               "Qwen3-8B",
                               "Qwen3-14B",
                               "DeepSeek-R1-Distill-Llama-8B"])
    parser.add_argument("--mode", type=str, default="direct", choices=['direct', 'explain'])
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--n_cycles", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--attribution_method", type=str, default="integrated_gradients", choices=["integrated_gradients", "attention_rollout", "rise_saliency", "sae", "none"])
    args = parser.parse_args()
    run_task(args.model, args.temperature, args.n_cycles, args.mode, args.max_new_tokens, args.attribution_method)

if __name__ == "__main__":
    main()

