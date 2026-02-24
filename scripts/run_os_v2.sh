#!/bin/bash
# 禁用 CUDA 0，只使用 GPU 1-7
export CUDA_VISIBLE_DEVICES=1,2,3,4,5
# nohup bash run.sh > nohup.out 2>&1 &

# models=("gpt-5.2" "gpt-5-mini")
models=("gemma-2-9b-it" "Llama-3-8B-Instruct" "DeepSeek-R1-Distill-Llama-8B" "Qwen2.5-14B-Instruct")
# models=("Llama-4-Scout-17B-16E-Instruct")
# roles=("female" "male" "Asian" "young child" "elderly")
roles=("elementary" "college")

for model in ${models[@]}; do
    for role in ${roles[@]}; do
        python main_e.py --task 1 --model $model --attribution_method "none" --mode "explain" --temperature 1.0 --n_cycles 100 --max_new_tokens 6000 --role $role

        python main_e.py --task 2 --model $model --attribution_method "none" --mode "explain" --temperature 1.0 --n_cycles 100 --max_new_tokens 6000 --role $role

        python main_e.py --task 3 --model $model --attribution_method "none" --mode "explain" --temperature 1.0 --n_cycles 100 --max_new_tokens 6000 --role $role
    done
done