import argparse
import subprocess
from utils.globals import all_model_list

def run_task(script_path, model, temperature, n_cycles, mode, max_new_tokens, attribution_method, role, cot):
    subprocess.run([
        "python", script_path,
        "--model", model,
        "--temperature", str(temperature),
        "--n_cycles", str(n_cycles),
        "--mode", mode,
        "--max_new_tokens", str(max_new_tokens),
        "--attribution_method", attribution_method,
        "--role", role,
        "--cot", str(cot)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="1", choices=["1", "2", "3", "4"])
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct",
        choices=all_model_list)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--n_cycles", type=int, default=100)
    parser.add_argument("--mode", type=str, default="direct", choices=['direct', 'explain'])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--attribution_method", type=str, default="integrated_gradients", choices=["integrated_gradients", "attention_rollout", "rise_saliency", "sae", "none"], help="归因方法名")
    parser.add_argument("--role", type=str, default="none")
    parser.add_argument("--cot", type=int, default=0)
    args = parser.parse_args()

    task_scripts = {
        "1": "task1_minority/task1_e.py",
        "2": "task2_investment/task2_e.py",
        "3": "task3_conspiracy/task3_e.py",
        "4": "task4_snowy/task4_e.py"
    }

    run_task(task_scripts[args.task], args.model, args.temperature, args.n_cycles, args.mode, args.max_new_tokens, args.attribution_method, args.role, args.cot)


if __name__ == "__main__":
    main()
