#!/bin/bash
# 禁用 CUDA 0，只使用 GPU 1-7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# nohup bash run.sh > nohup.out 2>&1 &

temperatures=(0.5 1.5)
models=("gemma-2-9b-it" "Llama-3-8B-Instruct" "DeepSeek-R1-Distill-Llama-8B" "Qwen2.5-14B-Instruct")
for model in ${models[@]}; do
    for temp in ${temperatures[@]}; do
        python main_e.py --task 1 --model $model --attribution_method "none" --mode "explain" --temperature $temp --n_cycles 100 --max_new_tokens 6000 

        python main_e.py --task 2 --model $model --attribution_method "none" --mode "explain" --temperature $temp --n_cycles 100 --max_new_tokens 6000 

        python main_e.py --task 3 --model $model --attribution_method "none" --mode "explain" --temperature $temp --n_cycles 100 --max_new_tokens 6000 
    done
done

for model in ${models[@]}; do
    python main_e.py --task 1 --model $model --attribution_method "none" --mode "explain" --temperature 1.0 --n_cycles 100 --max_new_tokens 6000 --cot 1

    python main_e.py --task 2 --model $model --attribution_method "none" --mode "explain" --temperature 1.0 --n_cycles 100 --max_new_tokens 6000 --cot 1

    python main_e.py --task 3 --model $model --attribution_method "none" --mode "explain" --temperature 1.0 --n_cycles 100 --max_new_tokens 6000 --cot 1
done