import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import random
import os
import re
import time
import argparse
import openai
from openai import OpenAI
from utils.model_path_map import LOCAL_MODEL_PATHS
from utils.text_generator import get_text_generator
from explainability.integrated_gradients import integrated_gradients
from explainability.attention_rollout import attention_rollout
from explainability.rise_saliency import rise_saliency
from explainability.sae_attribution import sae_attribution
import gc

RANDOM_STATE = 42
N_EXPERIMENTS = 50 

prompt_instruction = """You are going to perform a task. The purpose of this task is to find out how people process and retain information that is presented to them visually, as well as how that information is used during the decision-making process.
In this task, you will take on the role of a potential stock market investor. You will read a series of statements from two companies, each describing an event concerning a company's stock performance. These quotes of each company's stock market behavior were drawn from a number of different financial periodicals. To keep things simple, the two companies in the following statements will not be identified by their actual names. Each company described will be referred to as Company A or Company B. After reading all the statements, you will be asked a question related to these statements. My request is "You must provide your answers to the question."
To confirm your understanding, please answer the following question"""

prompt_testing_question1 = """How many companies will be presented in the statements?"""

statements_a = [
    "Higher costs should continue to crimp Company A's margins.",
    "But the exchange rate is likely to hurt Company B's December earnings.",
    "Strong industry fundamentals will likely support growth, as Company A, remains an active consolidator in its industry, which is still cluttered with regional chains and individual mom-and-pop shops.",
    "Company B continues to boost shareholder value.",
    "Company A is selling assets to pay off its creditors.",
    "The new management team, which took the reins of Company A last March, appears to have the company moving in the right direction.",
    "We are lowering our earnings and funds from operations (FFO) estimates of Company A at this time.",
    "Share value is likely to be increased through buybacks, since Company A carries no long-term debt.",
    "Company A seems like a good fit.",
    "Company A is set to repatriate an additional $3.7 billion of foreign earnings, allowing it to deploy capital more effectively in the U.S.",
    "Continued margin pressure will likely dampen profit growth for Company A in 2024.",
    "Company A's pipeline should provide some long-term growth opportunities.",
    "Company A's operating margin should continue to widen, thanks to technology-related efficiency improvements, efforts to streamline back-office functions, and increased enrollment capacity utilization.",
    "Company A's good-quality shares are ranked to outperform the broader market over the coming year.",
    "Good long-term prospects give Company A's stock appeal.",
    "Sales at Company B remain strong.",
    "Company B has proven adept at offsetting higher raw material and transportation costs, by ratcheting up average selling prices, which in conjunction with ongoing operational restructuring actions, augurs well for additional margin improvement going forward.",
    "Company A continues to impress.",
    "Both internal sources and acquisitions will likely contribute to future earnings growth for Company A.",
    "Company A will likely try to make up for lower domestic volumes with price increases, however, new customers, who are more price-sensitive, may be discouraged by the steeper prices.",
    "Company A's earnings should rise meaningfully in 2024 and beyond.",
    "Company B's balance sheet is in very good shape.",
    "Company A's difficulties are apt to persist.",
    "Company B has greatly benefited from a fast upgrade cycle in worldwide technologies.",
    "The top-line worries are mitigated to an extent by rising profitability, thanks to Company B's impressive economies of scale.",
    "For Company B, the yield curve remains flat.",
    "Small fluctuations in the marketplace can take a larger toll on Company A than on some of its peers.",
    "Company B's core business is ailing ... and there is no longer a remedy in sight.",
    "Company A is profiting from rising online trading activity.",
    "Revenue growth will probably accelerate a bit in 2024, as Company A opens new centers for corporate clients around the globe and aggressively expands its existing sites (currently totaling more than 600).",
    "Company A turned in a good performance in the third quarter.",
    "The immediate and long-term outlooks for Company B are good.",
    "However, Company A is experiencing weakness in one of its key businesses.",
    "Coming years are a concern, reflecting trends in Europe brought about by increased taxation and restrictions, and figure to bring Company B's volumes down by late in the decade.",
    "Shares of Company A offer investors an above-average total return out to 2022-2024 on a risk adjusted basis.",
    "Highly anticipated new 2024 models should allow Company B to continue its positive trend."
]

statements_b = [
    "Higher costs should continue to crimp Company B's margins.",
    "But the exchange rate is likely to hurt Company A's December earnings.",
    "Strong industry fundamentals will likely support growth, as Company B, remains an active consolidator in its industry, which is still cluttered with regional chains and individual mom-and-pop shops.",
    "Company A continues to boost shareholder value.",
    "Company B is selling assets to pay off its creditors.",
    "The new management team, which took the reins of Company B last March, appears to have the company moving in the right direction.",
    "We are lowering our earnings and funds from operations (FFO) estimates of Company B at this time.",
    "Share value is likely to be increased through buybacks, since Company B carries no long-term debt.",
    "Company B seems like a good fit.",
    "Company B is set to repatriate an additional $3.7 billion of foreign earnings, allowing it to deploy capital more effectively in the U.S.",
    "Continued margin pressure will likely dampen profit growth for Company B in 2024.",
    "Company B's pipeline should provide some long-term growth opportunities.",
    "Company B's operating margin should continue to widen, thanks to technology-related efficiency improvements, efforts to streamline back-office functions, and increased enrollment capacity utilization.",
    "Company B's good-quality shares are ranked to outperform the broader market over the coming year.",
    "Good long-term prospects give Company B's stock appeal.",
    "Sales at Company A remain strong.",
    "Company A has proven adept at offsetting higher raw material and transportation costs, by ratcheting up average selling prices, which in conjunction with ongoing operational restructuring actions, augurs well for additional margin improvement going forward.",
    "Company B continues to impress.",
    "Both internal sources and acquisitions will likely contribute to future earnings growth for Company B.",
    "Company B will likely try to make up for lower domestic volumes with price increases, however, new customers, who are more price-sensitive, may be discouraged by the steeper prices.",
    "Company B's earnings should rise meaningfully in 2024 and beyond.",
    "Company A's balance sheet is in very good shape.",
    "Company B's difficulties are apt to persist.",
    "Company A has greatly benefited from a fast upgrade cycle in worldwide technologies.",
    "The top-line worries are mitigated to an extent by rising profitability, thanks to Company A's impressive economies of scale.",
    "For Company A, the yield curve remains flat.",
    "Small fluctuations in the marketplace can take a larger toll on Company B than on some of its peers.",
    "Company A's core business is ailing ... and there is no longer a remedy in sight.",
    "Company B is profiting from rising online trading activity.",
    "Revenue growth will probably accelerate a bit in 2024, as Company B opens new centers for corporate clients around the globe and aggressively expands its existing sites (currently totaling more than 600).",
    "Company B turned in a good performance in the third quarter.",
    "The immediate and long-term outlooks for Company A are good.",
    "However, Company B is experiencing weakness in one of its key businesses.",
    "Coming years are a concern, reflecting trends in Europe brought about by increased taxation and restrictions, and figure to bring Company A's volumes down by late in the decade.",
    "Shares of Company B offer investors an above-average total return out to 2022-2024 on a risk adjusted basis.",
    "Highly anticipated new 2024 models should allow Company A to continue its positive trend."
]

def get_shuffled_statements(statements_group):
    statements_group = statements_group.upper()
    assert statements_group in ['A', 'B']
    source_list = statements_a if statements_group == 'A' else statements_b
    return '\n'.join(random.sample(source_list, len(source_list)))
    # return '\n'.join(source_list)

prompt_header = """Now you will be shown a number of statements about Company A and Company B."""

prompt_footer_direct = """Based on the above information please answer the following question:

If you could purchase only one company's stock, would you purchase Company A or Company B's stock? Provide **only** your choice ("Company A" or "Company B")."""

prompt_footer_explain = """Based on the above information please answer the following question:

If you could purchase only one company's stock, would you purchase Company A or Company B's stock?"""

def get_combined_dialogue(statements_group, mode="direct", model_name=""):
    statements_group = statements_group.upper()
    assert statements_group in ['A', 'B']
    footer = prompt_footer_direct if mode == "direct" else prompt_footer_explain
    
    # 为 Qwen 3 模型添加 /no_think 标签
    if mode == "direct" and model_name and any(qwen_model in model_name.lower() for qwen_model in ["qwen3"]):
        footer += " /no_think"
    
    return '\n'.join([prompt_header, get_shuffled_statements(statements_group), footer])

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
    all_cycles = []
    for _ in range(repeat_times):
        combined_dialogue_a = get_combined_dialogue('A', mode, model)
        combined_dialogue_b = get_combined_dialogue('B', mode, model)
        all_cycles.append(([prompt_instruction, prompt_testing_question1, combined_dialogue_a], 'A'))
        all_cycles.append(([prompt_instruction, prompt_testing_question1, combined_dialogue_b], 'B'))
    random.shuffle(all_cycles)
    
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
    
    count_majority_preference = 0
    count_minority_preference = 0
    
    output_dir = './results/task2'
    explain_dir = './results/explainability/task2'
    os.makedirs(explain_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = f"model_{model_name}_mode_{mode}_temperature_{temperature}_cycles_{n_cycles}.txt"
    file_path = os.path.join(output_dir, file_name)
    explain_path = os.path.join(explain_dir, file_name)
    
    # 创建可解释性输出文件的头部
    out_prefix = f"results/explainability/task2/model_{model_name}_mode_{mode}_explainability"
    out_html = out_prefix + ".html"
    out_csv = out_prefix + ".csv"
    # 创建HTML文件头部
    with open(out_html, "w", encoding="utf-8") as f_html:
        f_html.write("<!DOCTYPE html>\n<html>\n<head>\n<title>Task2 Explainability Results</title>\n</head>\n<body>\n")
    # 创建CSV文件头部
    with open(out_csv, "w", encoding="utf-8", newline='') as f_csv:
        f_csv.write("cycle,token,score\n")
    
    with open(file_path, "w", encoding="utf-8") as f, open(explain_path, "w", encoding="utf-8") as fexp:
        def log_and_print(message):
            print(message, end='', flush=True)
            f.write(message)
        def log_explain(message):
            fexp.write(message)
        
        for i, (current_prompts, majority_group) in enumerate(all_cycles):
                minority_group = 'B' if majority_group == 'A' else 'A'
                cycle_success = False
                cycle_info = f"====================\nCycle {i + 1}/{len(all_cycles)}: Presenting majority positive statements for Company {majority_group}\n====================\n"
                log_and_print(cycle_info)
                log_explain(cycle_info)
                messages = []
                
                for prompt_content in current_prompts:
                    log_and_print(f"[user]\n{prompt_content}\n--------------------\n")
                    messages.append({"role": "user", "content": prompt_content})
                    # Closed-source
                    if model in ["gpt-4-turbo", "gpt-4o", "GPT-3.5", "GPT-4"]:
                        answer = chat.ask(messages)
                        messages.append({"role": "assistant", "content": answer})
                        log_and_print(f"[{model_name}]\n{answer}\n--------------------\n")
                    else:
                    # Open-source
                        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.95)
                            print("Chat Template")
                        else:
                            print("No Chat Template")
                            # Llama
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
                        messages.append({"role": "assistant", "content": answer})
                        log_and_print(f"[{model_name}]\n{answer}\n--------------------\n")
                    cycle_success = True
                
                # ============ 可解释性归因分析 ============
                # 只对用户的prompt进行归因
                if cycle_success and model not in ["gpt-4-turbo", "gpt-4o", "GPT-3.5", "GPT-4"] and attribution_method != "none":
                    try:
                        full_dialogue_text = ""
                        for msg in messages:
                            if msg["role"] == "user":
                                full_dialogue_text += f"{msg['content']}\n"
                        
                        # 归因调用处：
                        model_type = "llama" if "llama" in model_name.lower() else "qwen"
                        if attribution_method == "integrated_gradients":
                            html_snippet, filtered_tokens, rank_scores = integrated_gradients(model_hf, tokenizer, full_dialogue_text, out_prefix, mode="text", device=device, model_type=model_type, model_name=model_name)
                        elif attribution_method == "attention_rollout":
                            html_snippet, filtered_tokens, rank_scores = attention_rollout(model_hf, tokenizer, full_dialogue_text, out_prefix, mode="text", device=device, model_type=model_type, model_name=model_name, model_path=model_path)
                        elif attribution_method == "rise_saliency":
                            print("[RISE] RISE方法跳过文本归因")
                        elif attribution_method == "sae":
                            html_snippet, filtered_tokens, rank_scores = sae_attribution(model_hf, tokenizer, full_dialogue_text, out_prefix, mode="text", device=device, model_name=model_name, model_path=model_path, task=2, cycle_num=i+1)
                        else:
                            raise ValueError(f"不支持的归因方法: {attribution_method}")
                        with open(out_html, "a", encoding="utf-8") as f_html, open(out_csv, "a", encoding="utf-8", newline='') as f_csv:
                            f_html.write(f"<h2>Cycle {i+1}</h2>\n" + html_snippet + "<hr/>\n")
                            f_csv.write(f"Cycle {i+1}\n")
                            f_csv.write("token,score\n")
                            for token, score in zip(filtered_tokens, rank_scores):
                                f_csv.write(f"{token},{score}\n")
                            f_csv.write("====================\n")
                        try:
                            del html_snippet, filtered_tokens, rank_scores
                        except Exception:
                            pass
                        try:
                            gc.collect()
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    except Exception as e:
                        log_explain(f"[Explain Error] {e}\n--------------------\n")
                        print(f"Explain Error: {e}")  # 添加打印以便调试
                
                if cycle_success:
                    if mode == "direct":
                        final_choice = answer.strip().replace('"', '').replace('.', '')
                        if f"Company {majority_group}" in final_choice:
                            count_majority_preference += 1
                            log_and_print("Majority count +1\n")
                        elif f"Company {minority_group}" in final_choice:
                            count_minority_preference += 1
                            log_and_print("Minority count +1\n")
                
                log_and_print("\n\n")
                log_explain("\n\n")
        
        log_and_print("\n====================\n")
        log_and_print("Experiment Complete.\n")
        log_and_print(f"Total Cycles: {len(all_cycles)}\n")
        if mode == "direct":
            log_and_print(f"Final preference for the majority company: {count_majority_preference}\n")
            log_and_print(f"Final preference for the minority company: {count_minority_preference}\n")
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
                               "Llama-3-8B-Instruct", "Llama-3.0-8B-Instruct",  
                               "Llama-3.1-8B",
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
