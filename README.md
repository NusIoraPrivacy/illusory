Illusory Pattern Perception
==================================

This repository contains code and analysis for the illusory pattern perception in LLM.

An example to run the code:
```
python main_e.py --task 1 --model "gpt-4o" --attribution_method "none" --mode "explain" --temperature 1 --n_cycles 100 --max_new_tokens 6000
```

## Data & Results

1. **Raw outputs** are stored under `results/` (per task, model, temperature, role, CoT).
2. **Extraction**:
   - Run `extract_results.py` to parse raw outputs and produce:
     - `extracted_results/average_encoded_by_combination.xlsx`
     - `extracted_results/weighted_averages_by_combination.xlsx`
     ```

