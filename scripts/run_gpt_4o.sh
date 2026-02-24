#!/bin/bash
# 禁用 CUDA 0，只使用 GPU 1-7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# nohup bash run.sh > nohup.out 2>&1 &
# models=("gpt-3.5-turbo" "gpt-4o")
# temperatures=(0.5 1 1.5)
temperatures=(1)
roles=("female" "male" "Asian" "child" "elderly" "elementary" "college")

models=("gpt-4o")

python main_e.py --task 2 --model "gpt-4o" --attribution_method "none" --mode "explain" --temperature 1 --n_cycles 100 --max_new_tokens 6000

for model in ${models[@]}; do
    for role in ${roles[@]}; do
        python main_e.py --task 2 --model $model --attribution_method "none" --mode "explain" --temperature 1.0 --n_cycles 100 --max_new_tokens 6000 --role $role
    done
done

for model in ${models[@]}; do
    python main_e.py --task 2 --model $model --attribution_method "none" --mode "explain" --temperature 1.0 --n_cycles 100 --max_new_tokens 6000 --role "African American"
done

for model in ${models[@]}; do
    python main_e.py --task 2 --model $model --attribution_method "none" --mode "explain" --temperature 1.0 --n_cycles 100 --max_new_tokens 6000 --cot 1
done