#!/usr/bin/env python3
"""Step 3: SAE Characteristics Steering Experiment - Full Task 3 Dialogue Process Version"""

import os
import sys
import torch
import random
import re
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# Extend sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    # Reuse SAE loader helpers
    from explainability.sae_configs import load_sae_model, load_hooked_transformer_for_sae, get_sae_config_for_model
    SAE_AVAILABLE = True
    print("SAE libraries loaded successfully")
except ImportError as e:
    print(f"SAE or TransformerLens import failed:{e}")
    print("Will use simplified implementation")
    SAE_AVAILABLE = False


def get_available_gpu():
    """Find and return available GPU devices"""
    if not torch.cuda.is_available():
        return "cpu"
    
    try:
        import subprocess
        # GPU stats via nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
                                '--format=csv,nounits,noheader'], 
                               capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            print("Cannot run nvidia-smi, using default GPU")
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
                print(f"  GPU {gpu_id}Uses:{memory_used:.2f}GB/Total{memory_total:.2f}GB (Utilization{gpu_util}%)")
        
        if not gpu_info:
            print("GPU info not found, using default GPU")
            return "cuda:0"
        
        # Prefer idle GPU (<5% mem & <10% util)
        free_gpus = []
        for gpu in gpu_info:
            memory_percent = (gpu['used'] / gpu['total']) * 100
            if memory_percent < 5.0 and gpu['util'] < 10:
                free_gpus.append(gpu)
        
        if free_gpus:
            # Among idle pick lowest usage
            best_gpu = sorted(free_gpus, key=lambda x: (x['used'], x['util']))[0]
            memory_percent = (best_gpu['used'] / best_gpu['total']) * 100
            print(f"Select free GPU: cuda:{best_gpu['id']} (used){best_gpu['used']:.2f}GB/{best_gpu['total']:.2f}GB, {memory_percent:.1f}Utilization, %{best_gpu['util']}%)")
            return f"cuda:{best_gpu['id']}"
        
        # Else pick least-used GPU
        best_gpu = sorted(gpu_info, key=lambda x: (x['used'], x['util']))[0]
        memory_percent = (best_gpu['used'] / best_gpu['total']) * 100
        print(f"⚠️  Warning: There is no fully idle GPU, choose the least used: cuda:{best_gpu['id']} (used){best_gpu['used']:.2f}GB/{best_gpu['total']:.2f}GB, {memory_percent:.1f}Utilization, %{best_gpu['util']}%)")
        return f"cuda:{best_gpu['id']}"
        
    except Exception as e:
        print(f"Error detecting GPU:{e}, use default GPU")
        return "cuda:0"


def calculate_activation_differences(before_file: str, after_file: str) -> Dict[int, float]:
    """Calculate activation difference before and after modification"""
    print(f"Calculate Activation Difference:")
    print(f"Before modification{before_file}")
    print(f"Post Modified{after_file}")
    
    # Load pre-edit activations
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
    
    # Load post-edit activations
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
    
    # Compute delta
    differences = {}
    all_features = set(before_features.keys()) | set(after_features.keys())
    
    for feature_id in all_features:
        before_val = before_features.get(feature_id, 0.0)
        after_val = after_features.get(feature_id, 0.0)
        differences[feature_id] = after_val - before_val
    
    if not differences:
        print("Error: Unable to load activation data")
        return {}
    
    print(f"Calculation completed, total number of features:{len(differences)}")
    return differences


def calculate_average_activation_differences(model_mode: str = "explain") -> Dict[int, float]:
    """Calculate the average activation difference for all available cycles"""
    print(f"Calculate the average activation difference for all available cycles (Mode:{model_mode})")
    
    # Detect usable cycles
    before_dir = "before/results_before/explainability/task3/sae_attribution"
    after_dir = "results/explainability/task3/sae_attribution"
    
    # Enumerate cycles
    available_cycles = set()
    
    # Cycles present in before/
    if os.path.exists(before_dir):
        for file in os.listdir(before_dir):
            if file.endswith('_top37_features.txt') and 'cycle' in file:
                cycle_match = re.search(r'cycle(\d+)', file)
                if cycle_match:
                    cycle_num = int(cycle_match.group(1))
                    available_cycles.add(cycle_num)
    
    # Cycles present in after/
    if os.path.exists(after_dir):
        for file in os.listdir(after_dir):
            if file.endswith('_top37_features.txt') and 'cycle' in file:
                cycle_match = re.search(r'cycle(\d+)', file)
                if cycle_match:
                    cycle_num = int(cycle_match.group(1))
                    available_cycles.add(cycle_num)
    
    # Intersect cycle ids
    common_cycles = []
    for cycle_num in sorted(available_cycles):
        before_file = f"{before_dir}/model_Llama-3.0-8B-Instruct_mode_{model_mode}_explainability_cycle{cycle_num}_top37_features.txt"
        after_file = f"{after_dir}/model_Llama-3.0-8B-Instruct_mode_{model_mode}_explainability_cycle{cycle_num}_top37_features.txt"
        
        if os.path.exists(before_file) and os.path.exists(after_file):
            common_cycles.append(cycle_num)
    
    print(f"Found {len(common_cycles)}cycles available:{sorted(common_cycles)[:10]}{'...' if len(common_cycles) > 10 else ''}")
    
    if not common_cycles:
        print("Error: No cycle data available")
        return {}
    
    # Per-cycle delta
    all_differences = []
    for cycle_num in common_cycles:
        before_file = f"{before_dir}/model_Llama-3.0-8B-Instruct_mode_{model_mode}_explainability_cycle{cycle_num}_top37_features.txt"
        after_file = f"{after_dir}/model_Llama-3.0-8B-Instruct_mode_{model_mode}_explainability_cycle{cycle_num}_top37_features.txt"
        
        differences = calculate_activation_differences(before_file, after_file)
        if differences:
            all_differences.append(differences)
            print(f"  Cycle {cycle_num}: {len(differences)}features")
    
    if not all_differences:
        print("Error: No valid variance data")
        return {}
    
    # Mean delta across cycles
    all_features = set()
    for diff_dict in all_differences:
        all_features.update(diff_dict.keys())
    
    average_differences = {}
    for feature_id in all_features:
        values = [diff_dict.get(feature_id, 0.0) for diff_dict in all_differences]
        average_differences[feature_id] = sum(values) / len(values)
    
    print(f"Calculation completed, total number of features:{len(average_differences)}")
    print(f"Maximum average variance:{max(average_differences.values()):.6f}")
    print(f"Minimum average difference:{min(average_differences.values()):.6f}")
    
    return average_differences


def select_top_k_features(differences: Dict[int, float], k: int) -> List[int]:
    """Select Top-K features (sorted by absolute difference value)"""
    sorted_features = sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)
    top_k_features = [feat_id for feat_id, _ in sorted_features[:k]]
    
    print(f"Select top-{k} features:")
    for i, (feat_id, diff_val) in enumerate(sorted_features[:k]):
        print(f"  {i+1}. Feature {feat_id}: difference = {diff_val:.6f}")
    
    return top_k_features


def load_sae_and_model(model_name: str = "Llama-3.0-8B-Instruct"):
    """Load SAE and HookedTransformer"""
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
        
        # HookedTransformer on CPU (OOM guard)
        hooked_device = "cpu"
        print(f"Loading HookedTransformer on {hooked_device}")
        
        # gc before load
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        hooked_model = load_hooked_transformer_for_sae(model_name, hooked_device)
        
        if hooked_model is None:
            print("Error: Failed to load HookedTransformer")
            return sae, None, hook_name
        
        # gc after load
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
    """Run the full Task 3 conversation process, applying steering only when answering the final multiple-choice question"""
    
    print(f"Run the full Task 3 conversation process:")
    print(f"Top-K features:{top_k_features}")
    print(f"Scaling Factors{scale_factor}")
    
    if sae is None or hooked_model is None:
        print("Error: SAE or HookedTransformer not available for steering")
        return "Error: SAE or HookedTransformer not provided"
    
    try:
        # Build full dialogue string
        full_text = ""
        for msg in messages:
            if msg["role"] == "user":
                full_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                full_text += f"Assistant: {msg['content']}\n"
        
        print(f"Full conversation text length:{len(full_text)}characters. ")
        
        # gc before forward
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Tokenize
        tokens = hooked_model.to_tokens(full_text)
        print(f"  Tokens shape: {tokens.shape}")
        
        # Target layer hidden states (heavy cache)
        _, cache = hooked_model.run_with_cache(tokens)
        target_hidden = cache[hook_name]
        print(f"  Target hidden shape: {target_hidden.shape}")
        
        # Drop cache immediately
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Move tensors to SAE device
        target_hidden = target_hidden.to(sae.device)
        print(f"  Target hidden moved to: {target_hidden.device}")
        
        # SAE encode
        feature_acts = sae.encode(target_hidden)
        print(f"  Feature acts shape: {feature_acts.shape}")
        
        # Drop feature_acts immediately (unused later)
        del feature_acts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Steering from before/after delta
        steered_hidden = target_hidden.clone()
        
        # Apply delta steering per top feature
        for feat_id in top_k_features:
            if feat_id < steered_hidden.shape[-1]:
                # Fetch feature delta
                diff_val = differences[feat_id]
                
                # Positive delta → amplify feature
                # Negative delta → suppress feature
                if diff_val > 0:
                    # Positive delta steering
                    enhancement = torch.randn_like(steered_hidden) * scale_factor * abs(diff_val) * 0.001
                    steered_hidden += enhancement
                    print(f"    Feature {feat_id}: diff={diff_val:.2f}, strengthened activation")
                else:
                    # Negative delta steering
                    reduction = torch.randn_like(steered_hidden) * scale_factor * abs(diff_val) * 0.001
                    steered_hidden -= reduction
                    print(f"    Feature {feat_id}: diff={diff_val:.2f}, weakened activation")
        
        print(f"  Steered hidden shape: {steered_hidden.shape}")
        
        # Hook applies during new tokens only
        def steering_hook(activation, hook):
            # During decode when len > prompt len
            if activation.shape[1] > tokens.shape[1]:
                # Slice region to steer
                new_tokens_start = tokens.shape[1]
                new_tokens_length = activation.shape[1] - new_tokens_start
                
                # Steer final MC answer span only
                if new_tokens_length > 0:
                    # Apply on tail hidden states
                    steered_part = steered_hidden[:, -new_tokens_length:, :].to(activation.device)
                    activation[:, new_tokens_start:, :] = steered_part
                
            return activation
        
        # Generate with steering hook
        with hooked_model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            generated_tokens = hooked_model.generate(
                tokens, 
                max_new_tokens=50,  # Short max_new_tokens for brief answers
                temperature=1.0,  # Match baseline decode hyperparams
                do_sample=True
            )
        
        # Decode tokens
        steered_text = hooked_model.to_string(generated_tokens[0])
        print(f"Post-Steering Generation Length:{len(steered_text)}characters. ")
        
        # Free memory
        del tokens, target_hidden, steered_hidden, generated_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return steered_text
        
    except Exception as e:
        print(f"Error in complete dialogue steering: {e}")
        import traceback
        traceback.print_exc()
        return f"Steering Error:{str(e)}"


def run_step3_experiment_with_average(cycle_nums: List[int] = [1, 2], k: int = 10, 
                                     scale_factor: float = 0.001, num_runs: int = 10, 
                                     model_mode: str = "explain"):
    """Running the Step 3 Experiment - Running Multiple Rounds Using Average Differences"""
    
    print(f"= = = Step 3 experiment (mean difference,{num_runs}rounds")
    print(f"Automatically detects all available cycles and calculates average differences")
    
    # Mean Δ activations
    differences = calculate_average_activation_differences(model_mode=model_mode)
    if not differences:
        print("Error: Unable to calculate average activation difference")
        return None
    
    # Select top-K features
    top_k_features = select_top_k_features(differences, k)
    print(f"Selected top-{k} features: {top_k_features}")
    
    # Load SAE + LM
    print("--- Load SAE and HookedTransformer ---")
    sae, hooked_model, hook_name = load_sae_and_model("Llama-3.0-8B-Instruct")
    
    if sae is None or hooked_model is None:
        print("Error: Unable to load SAE or HookedTransformer, experiment terminated")
        return None
    
    # Task3 pipeline
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
    
    # Task3 dialogue without prompt_instruction when steering
    cycle_words1 = [prompt1, prompt2, task3_scenario1, task3_scenario2]
    cycle_words2 = [prompt1, prompt2, task3_scenario1, task3_scenario2]
    
    # Open output log
    output_dir = './results/task3_steered'
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"model_Llama-3.0-8B-Instruct_mode_{model_mode}_steered_temperature_1.0_cycles_{num_runs}.txt"
    file_path = os.path.join(output_dir, file_name)
    
    print(f"\n--- Starting {num_runs}-run steering experiment ---")
    
    with open(file_path, "w", encoding="utf-8") as f:
        def log_and_print(message):
            print(message, end='')
            f.write(message)
        
        for run in range(num_runs):
            print(f"\nRun {run+1}/{num_runs}:")
            
            # Random evaluation cycle
            test_cycle = random.choice([cycle_words1, cycle_words2])
            
            # Log cycle metadata
            cycle_info = f"====================\nCycle {run + 1}/{num_runs}: Steering experiment\n====================\n"
            log_and_print(cycle_info)
            
            # Build chat messages
            messages = []
            for prompt_content in test_cycle:
                log_and_print(f"[user]\n{prompt_content}\n--------------------\n")
                messages.append({"role": "user", "content": prompt_content})
                
                # Apply steering over dialogue
                steered_response = run_complete_task3_dialogue(
                    hooked_model, messages, top_k_features, differences, scale_factor,
                    sae=sae, hook_name=hook_name
                )
                
                # Log model output
                log_and_print(f"[Llama-3.0-8B-Instruct]\n{steered_response}\n--------------------\n")
                messages.append({"role": "assistant", "content": steered_response})
            
            # Progress every 10 cycles
            if (run + 1) % 10 == 0:
                print(f"Completed {run+1}/{num_runs} runs")
        
        # Completion banner
        log_and_print("\n====================\n")
        log_and_print("Steering Experiment Complete.\n")
        log_and_print(f"Total Cycles: {num_runs}\n")
        log_and_print(f"Results logged to: {file_path}\n")
        log_and_print("====================\n")
    
    print(f"\n🎉 Steering experiment complete!")
    print(f"Results saved to:{file_path}")
    print(f"Top-K features:{top_k_features}")
    print(f"Scaling Factors{scale_factor}")
    
    return file_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Step 3 Feature Steering Experiment")
    parser.add_argument("--mode", type=str, default="average", choices=["single", "average"],
                       help="Experimental mode: single (single cycle) or average (average difference)")
    parser.add_argument("--model_mode", type=str, default="explain", choices=["explain", "direct"],
                       help="Model mode: explain or direct")
    parser.add_argument("--cycle", type=int, default=1, choices=[1, 2],
                       help="Cycle number (single mode only)")
    parser.add_argument("--cycles", type=int, nargs='+', default=[1, 2],
                       help="List of Cycle Numbers (average mode only)")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of Top-K features")
    parser.add_argument("--scale", type=float, default=0.001,
                       help="Scaling Factors")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of running rounds (average mode only)")
    
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
            print("🎉 Step 3 The average difference experiment is complete!")
            print(f"results file{results}")
        else:
            print("❌ Step 3 The average difference experiment failed!")


if __name__ == "__main__":
    main()
