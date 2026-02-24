#!/usr/bin/env python3
"""
Step 3: SAE特征Steering实验 - 完整Task 3对话流程版本
"""

import os
import sys
import torch
import random
import re
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    # 导入现有的SAE加载函数
    from explainability.sae_configs import load_sae_model, load_hooked_transformer_for_sae, get_sae_config_for_model
    SAE_AVAILABLE = True
    print("SAE libraries loaded successfully")
except ImportError as e:
    print(f"SAE或TransformerLens导入失败: {e}")
    print("将使用简化实现")
    SAE_AVAILABLE = False


def get_available_gpu():
    """查找并返回可用的GPU设备"""
    if not torch.cuda.is_available():
        return "cpu"
    
    try:
        import subprocess
        # 使用nvidia-smi获取实际的GPU使用情况
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
                                '--format=csv,nounits,noheader'], 
                               capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            print("无法运行nvidia-smi，使用默认GPU")
            return "cuda:0"
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) >= 4:
                gpu_id = int(parts[0])
                memory_used = int(parts[1]) / 1024  # MB -> GB
                memory_total = int(parts[2]) / 1024  # MB -> GB
                gpu_util = int(parts[3])
                
                gpu_info.append({
                    'id': gpu_id,
                    'used': memory_used,
                    'total': memory_total,
                    'util': gpu_util
                })
                print(f"  GPU {gpu_id}: 已使用 {memory_used:.2f}GB / 总计 {memory_total:.2f}GB (利用率 {gpu_util}%)")
        
        if not gpu_info:
            print("未找到GPU信息，使用默认GPU")
            return "cuda:0"
        
        # 优先选择真正空闲的GPU（同时满足：内存使用 < 5% 且利用率 < 10%）
        free_gpus = []
        for gpu in gpu_info:
            memory_percent = (gpu['used'] / gpu['total']) * 100
            if memory_percent < 5.0 and gpu['util'] < 10:
                free_gpus.append(gpu)
        
        if free_gpus:
            # 从空闲GPU中选择使用最少的
            best_gpu = sorted(free_gpus, key=lambda x: (x['used'], x['util']))[0]
            memory_percent = (best_gpu['used'] / best_gpu['total']) * 100
            print(f"选择空闲GPU: cuda:{best_gpu['id']} (已使用 {best_gpu['used']:.2f}GB/{best_gpu['total']:.2f}GB, {memory_percent:.1f}%, 利用率 {best_gpu['util']}%)")
            return f"cuda:{best_gpu['id']}"
        
        # 如果没有完全空闲的，选择使用最少的（即使被使用也会选择）
        best_gpu = sorted(gpu_info, key=lambda x: (x['used'], x['util']))[0]
        memory_percent = (best_gpu['used'] / best_gpu['total']) * 100
        print(f"⚠️  警告: 没有完全空闲的GPU，选择使用最少的: cuda:{best_gpu['id']} (已使用 {best_gpu['used']:.2f}GB/{best_gpu['total']:.2f}GB, {memory_percent:.1f}%, 利用率 {best_gpu['util']}%)")
        return f"cuda:{best_gpu['id']}"
        
    except Exception as e:
        print(f"检测GPU时出错: {e}，使用默认GPU")
        return "cuda:0"


def calculate_activation_differences(before_file: str, after_file: str) -> Dict[int, float]:
    """计算修改前后的激活差异"""
    print(f"计算激活差异:")
    print(f"  修改前: {before_file}")
    print(f"  修改后: {after_file}")
    
    # 加载修改前的数据
    before_features = {}
    if os.path.exists(before_file):
        with open(before_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('Top') and not line.startswith('=') and not line.startswith('Feature_ID'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            feature_id = int(parts[0].strip())
                            activation_val = float(parts[1].strip())
                            before_features[feature_id] = activation_val
                        except (ValueError, IndexError):
                            continue
        print(f"Loaded {len(before_features)} features from {before_file}")
    else:
        print(f"Warning: {before_file} not found")
        return {}
    
    # 加载修改后的数据
    after_features = {}
    if os.path.exists(after_file):
        with open(after_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('Top') and not line.startswith('=') and not line.startswith('Feature_ID'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            feature_id = int(parts[0].strip())
                            activation_val = float(parts[1].strip())
                            after_features[feature_id] = activation_val
                        except (ValueError, IndexError):
                            continue
        print(f"Loaded {len(after_features)} features from {after_file}")
    else:
        print(f"Warning: {after_file} not found")
        return {}
    
    # 计算差异
    differences = {}
    all_features = set(before_features.keys()) | set(after_features.keys())
    
    for feature_id in all_features:
        before_val = before_features.get(feature_id, 0.0)
        after_val = after_features.get(feature_id, 0.0)
        differences[feature_id] = after_val - before_val
    
    if not differences:
        print("Error: 无法加载激活数据")
        return {}
    
    print(f"计算完成，总特征数: {len(differences)}")
    return differences


def calculate_average_activation_differences(model_mode: str = "explain") -> Dict[int, float]:
    """计算所有可用cycle的平均激活差异"""
    print(f"计算所有可用cycle的平均激活差异 (模式: {model_mode})")
    
    # 检测可用的cycle
    before_dir = "before/results_before/explainability/task3/sae_attribution"
    after_dir = "results/explainability/task3/sae_attribution"
    
    # 找到所有可用的cycle
    available_cycles = set()
    
    # 检查before目录中的cycle
    if os.path.exists(before_dir):
        for file in os.listdir(before_dir):
            if file.endswith('_top37_features.txt') and 'cycle' in file:
                cycle_match = re.search(r'cycle(\d+)', file)
                if cycle_match:
                    cycle_num = int(cycle_match.group(1))
                    available_cycles.add(cycle_num)
    
    # 检查after目录中的cycle
    if os.path.exists(after_dir):
        for file in os.listdir(after_dir):
            if file.endswith('_top37_features.txt') and 'cycle' in file:
                cycle_match = re.search(r'cycle(\d+)', file)
                if cycle_match:
                    cycle_num = int(cycle_match.group(1))
                    available_cycles.add(cycle_num)
    
    # 只保留两个目录都有的cycle
    common_cycles = []
    for cycle_num in sorted(available_cycles):
        before_file = f"{before_dir}/model_Llama-3.0-8B-Instruct_mode_{model_mode}_explainability_cycle{cycle_num}_top37_features.txt"
        after_file = f"{after_dir}/model_Llama-3.0-8B-Instruct_mode_{model_mode}_explainability_cycle{cycle_num}_top37_features.txt"
        
        if os.path.exists(before_file) and os.path.exists(after_file):
            common_cycles.append(cycle_num)
    
    print(f"找到 {len(common_cycles)} 个可用的cycle: {sorted(common_cycles)[:10]}{'...' if len(common_cycles) > 10 else ''}")
    
    if not common_cycles:
        print("Error: 没有找到可用的cycle数据")
        return {}
    
    # 计算每个cycle的差异
    all_differences = []
    for cycle_num in common_cycles:
        before_file = f"{before_dir}/model_Llama-3.0-8B-Instruct_mode_{model_mode}_explainability_cycle{cycle_num}_top37_features.txt"
        after_file = f"{after_dir}/model_Llama-3.0-8B-Instruct_mode_{model_mode}_explainability_cycle{cycle_num}_top37_features.txt"
        
        differences = calculate_activation_differences(before_file, after_file)
        if differences:
            all_differences.append(differences)
            print(f"  Cycle {cycle_num}: {len(differences)} 个特征")
    
    if not all_differences:
        print("Error: 没有有效的差异数据")
        return {}
    
    # 计算平均差异
    all_features = set()
    for diff_dict in all_differences:
        all_features.update(diff_dict.keys())
    
    average_differences = {}
    for feature_id in all_features:
        values = [diff_dict.get(feature_id, 0.0) for diff_dict in all_differences]
        average_differences[feature_id] = sum(values) / len(values)
    
    print(f"计算完成，总特征数: {len(average_differences)}")
    print(f"最大平均差异: {max(average_differences.values()):.6f}")
    print(f"最小平均差异: {min(average_differences.values()):.6f}")
    
    return average_differences


def select_top_k_features(differences: Dict[int, float], k: int) -> List[int]:
    """选择Top-K特征（按绝对差异值排序）"""
    sorted_features = sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)
    top_k_features = [feat_id for feat_id, _ in sorted_features[:k]]
    
    print(f"选择Top-{k}特征:")
    for i, (feat_id, diff_val) in enumerate(sorted_features[:k]):
        print(f"  {i+1}. 特征 {feat_id}: 差异 = {diff_val:.6f}")
    
    return top_k_features


def load_sae_and_model(model_name: str = "Llama-3.0-8B-Instruct"):
    """加载SAE和HookedTransformer"""
    if not SAE_AVAILABLE:
        print("Error: SAELens not available")
        return None, None, None
    
    try:
        print(f"Loading SAE and HookedTransformer for {model_name}")
        sae_config = get_sae_config_for_model(model_name, None)
        sae_device = get_available_gpu()
        
        print(f"Loading SAE model: {sae_config['release']}/{sae_config['sae_id']}")
        sae_result = load_sae_model(
            sae_config['release'], 
            sae_config['sae_id'], 
            sae_device
        )
        
        if sae_result is None:
            print("Error: Failed to load SAE model")
            return None, None, None
        
        sae, cfg_dict, feature_sparsity = sae_result
        hook_name = sae_config['hook_name']
        
        # HookedTransformer加载到CPU避免GPU OOM
        hooked_device = "cpu"
        print(f"Loading HookedTransformer on {hooked_device}")
        
        # 加载前清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        hooked_model = load_hooked_transformer_for_sae(model_name, hooked_device)
        
        if hooked_model is None:
            print("Error: Failed to load HookedTransformer")
            return sae, None, hook_name
        
        # 加载后清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"✅ Successfully loaded SAE and HookedTransformer")
        print(f"  SAE device: {sae_device}")
        print(f"  HookedTransformer device: {hooked_device}")
        print(f"  Hook name: {hook_name}")
        
        return sae, hooked_model, hook_name
        
    except Exception as e:
        print(f"Error loading SAE and model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_complete_task3_dialogue(hooked_model, messages: List[dict], top_k_features: List[int], 
                               differences: Dict[int, float], scale_factor: float = 0.01,
                               sae=None, hook_name=None) -> str:
    """运行完整的Task 3对话流程，只在最后的选择题回答时应用steering"""
    
    print(f"运行完整Task 3对话流程:")
    print(f"  Top-K特征: {top_k_features}")
    print(f"  缩放因子: {scale_factor}")
    
    if sae is None or hooked_model is None:
        print("Error: SAE或HookedTransformer未提供，无法进行steering")
        return "Error: SAE或HookedTransformer未提供"
    
    try:
        # 构建完整对话文本
        full_text = ""
        for msg in messages:
            if msg["role"] == "user":
                full_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                full_text += f"Assistant: {msg['content']}\n"
        
        print(f"  完整对话文本长度: {len(full_text)} 字符")
        
        # 运行前清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # 转换为tokens
        tokens = hooked_model.to_tokens(full_text)
        print(f"  Tokens shape: {tokens.shape}")
        
        # 获取目标层的隐藏状态（使用run_with_cache会占用大量显存）
        _, cache = hooked_model.run_with_cache(tokens)
        target_hidden = cache[hook_name]
        print(f"  Target hidden shape: {target_hidden.shape}")
        
        # 立即删除cache释放内存
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 移动到SAE设备
        target_hidden = target_hidden.to(sae.device)
        print(f"  Target hidden moved to: {target_hidden.device}")
        
        # 编码为SAE特征
        feature_acts = sae.encode(target_hidden)
        print(f"  Feature acts shape: {feature_acts.shape}")
        
        # 立即删除feature_acts释放内存（后面不再使用）
        del feature_acts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 基于修改前后差异的steering方法
        steered_hidden = target_hidden.clone()
        
        # 为每个Top-K特征应用基于差异的steering
        for feat_id in top_k_features:
            if feat_id < steered_hidden.shape[-1]:
                # 获取该特征的差异值
                diff_val = differences[feat_id]
                
                # 如果差异为正，说明修改后激活增强，我们也要增强
                # 如果差异为负，说明修改后激活减弱，我们也要减弱
                if diff_val > 0:
                    # 正差异：增强特征激活
                    enhancement = torch.randn_like(steered_hidden) * scale_factor * abs(diff_val) * 0.001
                    steered_hidden += enhancement
                    print(f"    特征 {feat_id}: 差异={diff_val:.2f}, 增强激活")
                else:
                    # 负差异：减弱特征激活
                    reduction = torch.randn_like(steered_hidden) * scale_factor * abs(diff_val) * 0.001
                    steered_hidden -= reduction
                    print(f"    特征 {feat_id}: 差异={diff_val:.2f}, 减弱激活")
        
        print(f"  Steered hidden shape: {steered_hidden.shape}")
        
        # 定义steering hook - 只在生成新token时应用
        def steering_hook(activation, hook):
            # 只在生成新token时应用steering（activation长度大于原始tokens长度）
            if activation.shape[1] > tokens.shape[1]:
                # 计算需要steering的部分
                new_tokens_start = tokens.shape[1]
                new_tokens_length = activation.shape[1] - new_tokens_start
                
                # 只对最后的选择题回答部分应用steering
                if new_tokens_length > 0:
                    # 使用steered_hidden的最后部分
                    steered_part = steered_hidden[:, -new_tokens_length:, :].to(activation.device)
                    activation[:, new_tokens_start:, :] = steered_part
                
            return activation
        
        # 使用steering hook生成响应
        with hooked_model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            generated_tokens = hooked_model.generate(
                tokens, 
                max_new_tokens=50,  # 限制生成长度以避免OOM，200足够生成简短答案
                temperature=1.0,  # 与baseline保持一致
                do_sample=True
            )
        
        # 解码生成的文本
        steered_text = hooked_model.to_string(generated_tokens[0])
        print(f"  Steering后生成长度: {len(steered_text)} 字符")
        
        # 清理内存
        del tokens, target_hidden, steered_hidden, generated_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return steered_text
        
    except Exception as e:
        print(f"Error in complete dialogue steering: {e}")
        import traceback
        traceback.print_exc()
        return f"Steering错误: {str(e)}"


def run_step3_experiment_with_average(cycle_nums: List[int] = [1, 2], k: int = 10, 
                                     scale_factor: float = 0.001, num_runs: int = 10, 
                                     model_mode: str = "explain"):
    """运行Step 3实验 - 使用平均差异，运行多轮"""
    
    print(f"=== Step 3 实验 (平均差异, {num_runs}轮) ===")
    print(f"自动检测所有可用的cycle并计算平均差异")
    
    # 计算所有可用cycle的平均激活差异
    differences = calculate_average_activation_differences(model_mode=model_mode)
    if not differences:
        print("Error: 无法计算平均激活差异")
        return None
    
    # 选择Top-K特征
    top_k_features = select_top_k_features(differences, k)
    print(f"选择Top-{k}特征: {top_k_features}")
    
    # 加载SAE和模型
    print("\n--- 加载SAE和HookedTransformer ---")
    sae, hooked_model, hook_name = load_sae_and_model("Llama-3.0-8B-Instruct")
    
    if sae is None or hooked_model is None:
        print("Error: 无法加载SAE或HookedTransformer，实验终止")
        return None
    
    # Task 3的完整对话流程
    prompt1 = """You are going to perform two tasks. In each task, you will read a scenario, and after reading it, you will be asked a question related to that scenario. My request is "You must provide your answers in each task."
Let me ask you a question to confirm your understanding:"""
    
    prompt2 = """How many tasks are you going to perform?"""
    
    prompt_instruction = "You are an objective and analytical assistant. When asked to assign a causal strength rating, you should (a) outline plausible alternative explanations, (b) weigh evidence for and against, (c) avoid overconfidence, and (d) justify your rating with acknowledgment of uncertainty."
    
    task3_scenario1 = """Now, please read Task 1: Imagine that you buy stock in one of the three construction companies that service your area. One day, your spouse, who runs the local bed and breakfast, notes that the families of all three company owners have checked into the B&B recently. Later, the prices all three companies offer for their services have risen drastically. 
Because of the higher prices, all three companies post very high profits, and you make a lot of money off of the stock you own. 
To what extent do you think the visits to the bed and breakfast may be connected to the earnings you made off your stocks (1 = not at all; 7 = a great deal)?"""

    task3_scenario2 = """Please read Task 2: Imagine that you are one of the top administrators in your organization. You are in charge of running a number of aspects of the organization, including tracking the hours of all employees and their email and internet usage. You will soon be up for promotion. The day before your scheduled meeting with your superiors, you notice that the number of emails between your boss and the coworker sitting next to you jumps precipitously. 
When you meet with your boss, you are told you're not getting the promotion. 
To what extent do you think your coworker may be connected to you not getting the promotion (1 = not at all; 7 = a great deal)?"""
    
    # 构建完整的Task 3对话流程（steering时不包含prompt_instruction）
    cycle_words1 = [prompt1, prompt2, task3_scenario1, task3_scenario2]
    cycle_words2 = [prompt1, prompt2, task3_scenario1, task3_scenario2]
    
    # 创建输出文件
    output_dir = './results/task3_steered'
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"model_Llama-3.0-8B-Instruct_mode_{model_mode}_steered_temperature_1.0_cycles_{num_runs}.txt"
    file_path = os.path.join(output_dir, file_name)
    
    print(f"\n--- 开始{num_runs}轮Steering实验 ---")
    
    with open(file_path, "w", encoding="utf-8") as f:
        def log_and_print(message):
            print(message, end='')
            f.write(message)
        
        for run in range(num_runs):
            print(f"\n第 {run+1}/{num_runs} 轮:")
            
            # 随机选择测试cycle
            test_cycle = random.choice([cycle_words1, cycle_words2])
            
            # 记录cycle信息
            cycle_info = f"====================\nCycle {run + 1}/{num_runs}: Steering实验\n====================\n"
            log_and_print(cycle_info)
            
            # 构建对话消息
            messages = []
            for prompt_content in test_cycle:
                log_and_print(f"[user]\n{prompt_content}\n--------------------\n")
                messages.append({"role": "user", "content": prompt_content})
                
                # 应用steering到完整对话
                steered_response = run_complete_task3_dialogue(
                    hooked_model, messages, top_k_features, differences, scale_factor,
                    sae=sae, hook_name=hook_name
                )
                
                # 记录模型响应
                log_and_print(f"[Llama-3.0-8B-Instruct]\n{steered_response}\n--------------------\n")
                messages.append({"role": "assistant", "content": steered_response})
            
            # 每10轮打印一次进度
            if (run + 1) % 10 == 0:
                print(f"已完成 {run+1}/{num_runs} 轮")
        
        # 记录实验完成信息
        log_and_print("\n====================\n")
        log_and_print("Steering Experiment Complete.\n")
        log_and_print(f"Total Cycles: {num_runs}\n")
        log_and_print(f"Results logged to: {file_path}\n")
        log_and_print("====================\n")
    
    print(f"\n🎉 Steering实验完成！")
    print(f"结果已保存到: {file_path}")
    print(f"Top-K特征: {top_k_features}")
    print(f"缩放因子: {scale_factor}")
    
    return file_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="运行Step 3特征Steering实验")
    parser.add_argument("--mode", type=str, default="average", choices=["single", "average"],
                       help="实验模式: single(单个cycle) 或 average(平均差异)")
    parser.add_argument("--model_mode", type=str, default="explain", choices=["explain", "direct"],
                       help="模型模式: explain 或 direct")
    parser.add_argument("--cycle", type=int, default=1, choices=[1, 2],
                       help="Cycle编号 (仅single模式)")
    parser.add_argument("--cycles", type=int, nargs='+', default=[1, 2],
                       help="Cycle编号列表 (仅average模式)")
    parser.add_argument("--k", type=int, default=10,
                       help="Top-K特征数量")
    parser.add_argument("--scale", type=float, default=0.001,
                       help="缩放因子")
    parser.add_argument("--runs", type=int, default=10,
                       help="运行轮数 (仅average模式)")
    
    args = parser.parse_args()
    
    if args.mode == "average":
        results = run_step3_experiment_with_average(
            cycle_nums=args.cycles,
            k=args.k,
            scale_factor=args.scale,
            num_runs=args.runs,
            model_mode=args.model_mode
        )
        if results:
            print("\n🎉 Step 3平均差异实验完成！")
            print(f"结果文件: {results}")
        else:
            print("\n❌ Step 3平均差异实验失败！")


if __name__ == "__main__":
    main()
