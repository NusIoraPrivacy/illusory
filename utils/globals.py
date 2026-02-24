import os
from utils.keys import OPENAI_API_KEY

OPENAI_API_BASE=os.getenv('OPENAI_API_BASE', 'https://api.vectorengine.ai/v1')
# OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')

roles1 = ["female", "male", "Asian", "young child", "elderly", "African American"]
roles2 = ["elementary", "college"]

gpt_list = ["gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "GPT-3.5", "GPT-4", "gpt-5.2", "gpt-5-mini", "gpt-5"]
all_model_list = ["gpt-3.5-turbo", "GPT-4", "gpt-4o", "gpt-5.2", "gpt-5-mini", "gpt-5",
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
                               "DeepSeek-R1-Distill-Llama-8B"]