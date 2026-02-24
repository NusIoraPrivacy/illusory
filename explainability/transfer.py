import os
import csv
import numpy as np

def score_to_color(score, model_type_str="llama"):
    # 红色系，深浅随分数变化
    color_range = 0.95 if "llama" in model_type_str else 0.985
    intensity = abs(score)
    r = 255
    g = int(255 * (color_range - intensity))
    b = int(255 * (color_range - intensity))
    return f"rgb({r},{g},{b})"

def restore_token(token):
    return token.replace('Ċ', '\n').replace('Ġ', ' ')

def parse_cycles(csv_path):
    cycles = []
    with open(csv_path, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    split_indices = [i for i, line in enumerate(lines) if line.strip() == "===================="]
    if not split_indices or split_indices[0] != 0:
        split_indices = [0] + split_indices
    split_indices.append(len(lines))

    for idx in range(len(split_indices) - 1):
        start = split_indices[idx] + 1
        end = split_indices[idx + 1]
        cycle_lines = lines[start:end]
        if len(cycle_lines) < 3:
            continue
        token_score_lines = cycle_lines[2:]  # 跳过cycle和token,score
        tokens = []
        scores = []
        for line in token_score_lines:
            line = line.strip()
            if not line:
                continue
            # 跳过标题行
            if line.lower().startswith("cycle") or line.lower().startswith("token"):
                continue
            # 特殊处理逗号token
            if line.startswith(',,'):
                token = ','
                try:
                    score = float(line[2:].strip())
                except Exception:
                    continue
            else:
                # 只分割第一个逗号
                parts = line.split(',', 1)
                if len(parts) < 2:
                    continue
                token = parts[0]
                try:
                    score = float(parts[1])
                except Exception:
                    continue
            if token.strip() == "" or token.startswith("="):
                continue
            tokens.append(token)
            scores.append(score)
        if tokens:
            cycles.append((tokens, scores))
    return cycles

def get_task_number_from_path(csv_path):
    # 自动从路径中提取 task 编号
    for part in csv_path.split(os.sep):
        if part.lower().startswith("task"):
            num = ''.join(filter(str.isdigit, part))
            if num:
                return num
    return "?"

def csv_to_ig_html(csv_path, html_out_path, model_type_str="llama"):
    task_num = get_task_number_from_path(csv_path)
    cycles = parse_cycles(csv_path)
    html = f"<!DOCTYPE html>\n<html>\n<head>\n<title>Task {task_num} Explainability Results</title>\n</head>\n<body>\n"
    for idx, (tokens, scores) in enumerate(cycles):
        html += f"<h2>Task {task_num} Explainability Results - Cycle {idx+1}</h2>\n"
        html += '<div style="font-family:monospace;font-size:18px;">'
        scores = np.array(scores)
        max_abs = float(np.max(np.abs(scores))) if scores.size > 0 else 1.0
        if max_abs < 1e-8:
            norm_scores = np.zeros_like(scores)
        else:
            norm_scores = scores / max_abs
        for token, score in zip(tokens, norm_scores):
            token_text = restore_token(token)
            color = score_to_color(score, model_type_str)
            parts = token_text.split('\n')
            for idx2, part in enumerate(parts):
                if part:
                    html += f'<span style="background:{color};padding:2px;margin:1px;border-radius:3px;">{part}</span>'
                if idx2 < len(parts) - 1:
                    html += '<br/>'
        html += '</div><hr/>\n'
    html += "</body>\n</html>\n"
    with open(html_out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"IG explainability html saved to: {html_out_path}")

def get_cycle_titles(csv_path):
    titles = []
    base, ext = os.path.splitext(csv_path)
    html_out_path = base + "_ig_color.html"
    csv_to_ig_html(csv_path, html_out_path, model_type_str="llama")

if __name__ == "__main__":
    root_dir = "/opt/data/private/illusory_pattern_perception/results/explainability"
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                csv_path = os.path.join(dirpath, filename)
                print(f"Processing: {csv_path}")
                get_cycle_titles(csv_path)