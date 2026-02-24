import os
import re
import time
import json
import pandas as pd
import httpx
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError
from utils.keys import OPENAI_API_KEY
OPENAI_API_BASE='https://api.vectorengine.ai/v1'

# ===== Basic Configuration =====
INPUT_XLSX  = "./extracted_results/chatgpt_last_replies.xlsx"
OUTPUT_XLSX = "./extracted_results/chatgpt_choice_reason_with_encoded.xlsx"


def get_instruction(task: str) -> str:
    """
    Get task-specific instruction based on task name.
    """
    task_lower = task.lower()
    
    if "task1" in task_lower:
        return """You will be given the final ChatGPT reply from Task 1 (Group A vs Group B decision-making task).
Analyze the reply and do three things:

1) Determine whether the reply chooses "Group A" or "Group B". Output ONLY "Group A" or "Group B".

2) Identify the exact sentence or phrase in the text that is the most critical reason for this choice, and return it exactly as it appears (verbatim).

3) Provide a high-level reason summary in ≤ 15 words (English). Keep it concise and abstract (e.g., "A has more consistent prosocial behavior & B has severe negatives"). Avoid punctuation except "&" if needed.

Return a single JSON object with keys:
{
  "choice": "Group A" | "Group B",
  "reason": "<verbatim key sentence/phrase>",
  "summary": "<≤15 words>"
}

Text:
"""
    
    elif "task2" in task_lower:
        return """You will be given the final ChatGPT reply from Task 2 (Company A vs Company B stock investment decision-making task).
Analyze the reply and do three things:

1) Determine whether the reply chooses "Company A" or "Company B". Output ONLY "Company A" or "Company B".

2) Identify the exact sentence or phrase in the text that is the most critical reason for this choice, and return it exactly as it appears (verbatim).

3) Provide a high-level reason summary in ≤ 15 words (English). Keep it concise and abstract (e.g., "A has stronger financials & growth potential while B faces margin pressure"). Avoid punctuation except "&" if needed.

Return a single JSON object with keys:
{
  "choice": "Company A" | "Company B",
  "reason": "<verbatim key sentence/phrase>",
  "summary": "<≤15 words>"
}

Text:
"""
    
    elif "task3" in task_lower and "promotion" in task_lower:
        return """You will be given the final ChatGPT reply from Task 3 (Promotion scenario - conspiracy belief rating task).
Analyze the reply and do three things:

1) Extract the numerical rating (1-7) that the model provided for the promotion scenario question. Output ONLY the number (e.g., "3", "5", "7"). The rating indicates the extent to which the coworker may be connected to not getting the promotion (1 = not at all; 7 = a great deal).

2) Identify the exact sentence or phrase in the text that is the most critical reason for this rating, and return it exactly as it appears (verbatim).

3) Provide a high-level reason summary in ≤ 15 words (English). Keep it concise and abstract (e.g., "moderate conspiracy belief about coworker involvement" or "strong suspicion of coworker's role"). Avoid punctuation except "&" if needed.

Return a single JSON object with keys:
{
  "choice": "1" | "2" | "3" | "4" | "5" | "6" | "7",
  "reason": "<verbatim key sentence/phrase>",
  "summary": "<≤15 words>"
}

Text:
"""
    
    elif "task3" in task_lower and "stock" in task_lower:
        return """You will be given the final ChatGPT reply from Task 3 (Stock scenario - conspiracy belief rating task).
Analyze the reply and do three things:

1) Extract the numerical rating (1-7) that the model provided for the stock scenario question. Output ONLY the number (e.g., "3", "5", "7"). The rating indicates the extent to which the visits to the bed and breakfast may be connected to the earnings from stocks (1 = not at all; 7 = a great deal).

2) Identify the exact sentence or phrase in the text that is the most critical reason for this rating, and return it exactly as it appears (verbatim).

3) Provide a high-level reason summary in ≤ 15 words (English). Keep it concise and abstract (e.g., "high conspiracy belief about B&B connection" or "minimal suspicion of collusion"). Avoid punctuation except "&" if needed.

Return a single JSON object with keys:
{
  "choice": "1" | "2" | "3" | "4" | "5" | "6" | "7",
  "reason": "<verbatim key sentence/phrase>",
  "summary": "<≤15 words>"
}

Text:
"""
    
    else:
        # Default instruction for unknown tasks
        return """You will be given the final ChatGPT reply from a decision-making task.
Analyze the reply and do three things:

1) Determine the choice or rating from the reply. Output ONLY the choice (e.g., "Group A", "Company B") or rating (e.g., "3", "5", "7").

2) Identify the exact sentence or phrase in the text that is the most critical reason for this choice/rating, and return it exactly as it appears (verbatim).

3) Provide a high-level reason summary in ≤ 15 words (English). Keep it concise and abstract. Avoid punctuation except "&" if needed.

Return a single JSON object with keys:
{
  "choice": "<choice or rating>",
  "reason": "<verbatim key sentence/phrase>",
  "summary": "<≤15 words>"
}

Text:
"""

# ===== Proxy Configuration =====
PROXY = "http://127.0.0.1:7890"
os.environ["http_proxy"]  = PROXY
os.environ["https_proxy"] = PROXY

httpx_client = httpx.Client(
    timeout=httpx.Timeout(60.0, read=60.0, connect=60.0),
)

client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY, http_client=httpx_client)

def ask_once(text: str, task: str, max_retries: int = 6) -> str:
    """
    Send a request to GPT-4o with task-specific instruction.
    
    Args:
        text: The ChatGPT reply text to analyze
        task: The task name (e.g., "task1", "task2", "task3promotion", "task3stock")
        max_retries: Maximum number of retry attempts
    
    Returns:
        The response text from GPT-4o
    """
    instruction = get_instruction(task)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[{"role": "user", "content": instruction + "\n" + text}]
            )
            return resp.choices[0].message.content.strip()
        except (APIConnectionError, RateLimitError, APIStatusError, httpx.TransportError) as e:
            wait = min(5 * (2 ** attempt), 60)
            print(f"[warn] API call failed ({type(e).__name__}): {e}. Retry in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("API call failed after retries.")

def parse_json_output(output_text: str, original_reply: str):
    choice, reason, summary = "", "", ""
    # Remove markdown code block markers
    output_text = re.sub(r"^```json\s*|```$", "", output_text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(output_text)
        choice  = (data.get("choice") or "").strip()
        reason  = (data.get("reason") or "").strip()
        summary = (data.get("summary") or "").strip()
    except Exception:
        # 2) If not JSON, do simple fallback: find choice first, set reason and summary to empty
        # First try to find Group A/B
        m = re.search(r"\bGroup [AB]\b", output_text, re.I)
        if m:
            choice = m.group(0).title()
            reason = ""
            summary = ""
        else:
            # Then try to find numerical rating (1-7)
            m_num = re.search(r"\b([1-7])\b", output_text)
            if m_num:
                choice = m_num.group(1)
                reason = ""
                summary = ""
            else:
                # Fallback from original reply
                m2 = re.search(r"\bGroup [AB]\b", original_reply, re.I)
                if m2:
                    choice = m2.group(0).title()
                else:
                    # Find numbers from original reply
                    m2_num = re.search(r"\b([1-7])\b", original_reply)
                    if m2_num:
                        choice = m2_num.group(1)
                reason = ""
                summary = ""
    # 3) Normalize choice - support Group A/B, Company A/B and numbers 1-7 (not normalized)
    if choice not in ("Group A", "Group B", "Company A", "Company B") and choice not in ("1", "2", "3", "4", "5", "6", "7"):
        m3 = re.search(r"\b(?:Group|Company) [AB]\b", original_reply, re.I)
        if m3:
            choice = m3.group(0).title()
        else:
            m3_num = re.search(r"\b([1-7])\b", original_reply)
            if m3_num:
                choice = m3_num.group(1)
    # 4) Limit summary to no more than 10 words (trim again for safety)
    if summary:
        words = summary.split()
        if len(words) > 10:
            summary = " ".join(words[:10])
    return choice, reason, summary

def encode_by_cycle(majority_group: str, choice: str, task: str = None):
    # Task1: Group A/B 选择
    if choice in ("Group A", "Group B"):
        if majority_group == 'A':
            return 1 if choice == 'Group A' else 0
        elif majority_group == 'B':
            return 1 if choice == 'Group B' else 0
    
    # Task2: Company A/B 选择
    elif choice in ("Company A", "Company B"):
        if majority_group == 'A':
            return 1 if choice == 'Company A' else 0
        elif majority_group == 'B':
            return 1 if choice == 'Company B' else 0
    
    # Task3: 数字评分 (1-7)
    elif choice in ("1", "2", "3", "4", "5", "6", "7"):
        return int(choice)
    
    return None

def main():
    # 创建输出目录
    os.makedirs("./extracted_results", exist_ok=True)
    
    # 检查是否存在第一步生成的Excel文件
    input_excel = "./extracted_results/chatgpt_last_replies.xlsx"
    
    if not os.path.exists(input_excel):
        print(f"错误: 找不到输入文件 {input_excel}")
        print("请先运行第一步: python extract_txt_to_excel.py")
        return
    
    print("=" * 60)
    print("开始第二步：分析Excel中的回复")
    print("=" * 60)
    
    try:
        df = pd.read_excel(input_excel, engine="openpyxl")
        print(f"成功读取Excel文件，包含 {len(df)} 行数据")
        
        # 检查必要的列（包括新版的 Role, COT 和 Temperature）
        required_columns = ["Cycle", "ChatGPT最后回复", "Role", "COT", "Temperature"]
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Excel file missing required columns: {required_columns}")
            return
        
        # Check column structure
        print(f"Detected columns: {list(df.columns)}")
        
        # Ensure other key columns exist
        if "Task" not in df.columns:
            print("❌ Error: Missing Task column, but this is required")
            return
        
        if "Model" not in df.columns:
            print("❌ Error: Missing Model column, but this is required")
            return
            
        if "Filename" not in df.columns:
            print("❌ Error: Missing Filename column, but this is required")
            return
            
        print("✅ All required columns exist")

        # Display available tasks
        available_tasks = df['Task'].unique()
        print(f"Available tasks: {list(available_tasks)}")
        
        # Default to processing all tasks (skip interactive selection)
        target_task = "all"
        target_task_list = None
        
        # Filter data - process all tasks
        df = df.copy()
        print(f"⏳ Processing all tasks, total rows: {len(df)}")
            
    except Exception as e:
        print(f"Error: Failed to read Excel file: {e}")
        return
    
    # Fixed column structure
    columns = [
        "Task",
        "Cycle",
        "Chosen Group", 
        "Key Reason (Original Sentence)", 
        "Summary (≤10 words)", 
        "Encoded",
        "Model",
        "Filename",
        "Mode",
        "Role",
        "COT",
        "Temperature",
    ]
    
    output_file = "./extracted_results/chatgpt_choice_reason_with_encoded.xlsx"
    
    # Check if partial result file already exists, load if available
    existing_df = None
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_excel(output_file, engine="openpyxl")
            print(f"Found existing result file with {len(existing_df)} rows of processed data")
            
            # If processing specific task only, need to remove all data for that task first
            if target_task != "all":
                print(f"🔄 Will replace all data for {target_task} (Tasks={target_task_list}), keeping data for other tasks")
                # Keep data for non-target tasks
                if target_task_list:
                    existing_df = existing_df[~existing_df['Task'].isin(target_task_list)].copy()
                print(f"Keeping data for other tasks: {len(existing_df)} rows")
            else:
                # Get processed row indices (by Task+Model+Mode+Cycle+Role+COT+Temperature combination)
                processed_combinations = set()
                for _, row in existing_df.iterrows():
                    mode_value = str(row.get('Mode', 'unknown'))
                    role_value = str(row.get('Role', '')) if 'Role' in existing_df.columns else ''
                    cot_value = str(row.get('COT', '')) if 'COT' in existing_df.columns else ''
                    temp_value = str(row.get('Temperature', '')) if 'Temperature' in existing_df.columns else ''
                    key = f"{row['Task']}_{row['Model']}_{mode_value}_{row['Cycle']}_{role_value}_{cot_value}_{temp_value}"
                    processed_combinations.add(key)
                print(f"Number of processed combinations: {len(processed_combinations)}")
                print(f"Processed tasks: {existing_df['Task'].unique().tolist()}")
        except Exception as e:
            print(f"Warning: Failed to read existing result file: {e}")
            existing_df = None
    
    # Start processing from the last row, so we can see task3 results first
    for index in reversed(df.index):
        row = df.loc[index]
        cycle = str(row["Cycle"])
        reply = str(row["ChatGPT最后回复"])
        
        # Get all information first (needed for both checking and processing)
        task = str(row["Task"])
        model = str(row["Model"])
        mode = str(row.get("Mode", "unknown"))
        
        # 新增：Role, COT 和 Temperature 信息（来自第一步 Excel）
        role_value = row.get("Role", "")
        cot_value = row.get("COT", "")
        temp_value = row.get("Temperature", "")
        # 处理缺失值为字符串
        role = "" if pd.isna(role_value) else str(role_value)
        cot = "" if pd.isna(cot_value) else str(cot_value)
        temperature = "" if pd.isna(temp_value) else str(temp_value)
        
        # Check if this row has already been processed
        if existing_df is not None:
            # Need to check Task, Model, Mode, Cycle, Role, COT and Temperature together to uniquely identify a row
            # Check if this row has already been processed
            # Handle cases where Role, COT or Temperature columns might not exist in older result files
            mask = (
                (existing_df['Task'] == task) & 
                (existing_df['Model'] == model) & 
                (existing_df['Mode'] == mode) & 
                (existing_df['Cycle'].astype(str) == cycle)
            )
            
            # Add Role, COT and Temperature conditions if columns exist
            if 'Role' in existing_df.columns:
                mask = mask & (existing_df['Role'].astype(str) == role)
            if 'COT' in existing_df.columns:
                mask = mask & (existing_df['COT'].astype(str) == cot)
            if 'Temperature' in existing_df.columns:
                mask = mask & (existing_df['Temperature'].astype(str) == temperature)
            
            already_processed = existing_df[mask]
            
            if not already_processed.empty:
                print(f"Skipping already processed {task} - {model} - {mode} - Cycle {cycle} - Role {role} - COT {cot} - Temperature {temperature}")
                continue
        
        filename = str(row["Filename"])
        
        print(f"\nProcessing {task} - Cycle {cycle} - {model}")
        
        try:
            # Step 1: Call GPT API
            raw_out = ask_once(reply, task)
            # print(raw_out)
            # Step 2: Parse the output
            choice, reason, summary = parse_json_output(raw_out, reply)
            if not reason or not summary:
                print(f"[warn] Missing reason/summary, cycle={cycle}, model={model}, raw_out={raw_out}, reply={reply}")
            
            # Step 3: Encode the choice
            encoded = encode_by_cycle(row.get('majority_group', ''), choice, task)
            
            # Step 4: Build row data, including all information
            row_data = [task, cycle, choice, reason, summary, encoded, model, filename, mode, role, cot, temperature]
            
            print(f"  - Choice: {choice}, Encoded: {encoded}")
            
            # Step 5: Save immediately after GPT call completes and data is processed
            # This ensures we save after each combination, not just at the end
            if existing_df is not None:
                # Add new row to existing DataFrame
                new_row_df = pd.DataFrame([row_data], columns=columns)
                existing_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            else:
                # Create new DataFrame
                existing_df = pd.DataFrame([row_data], columns=columns)
            
            # Save to Excel file immediately after each GPT call completes
            existing_df.to_excel(output_file, index=False, engine="openpyxl")
            print(f"  ✅ Saved to {output_file} (total {len(existing_df)} rows)")
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  ❌ Error processing reply: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final statistics
    if existing_df is not None and len(existing_df) > 0:
        print("\n" + "=" * 60)
        print("Step 2 completed!")
        print(f"Total processed {len(existing_df)} replies")
        print(f"Results saved to: {output_file}")
        print("=" * 60)
        
        # Display statistics
        print("\n统计信息:")
        
        # Task distribution
        task_counts = existing_df['Task'].value_counts()
        print(f"Task distribution:")
        for task, count in task_counts.items():
            print(f"  {task}: {count} replies")
        
        # Model distribution
        model_counts = existing_df['Model'].value_counts()
        print(f"\nModel distribution:")
        for model, count in model_counts.items():
            print(f"  {model}: {count} replies")
        
        # Choice distribution
        choice_counts = existing_df['Chosen Group'].value_counts()
        print(f"\nChoice distribution:")
        for choice, count in choice_counts.items():
            print(f"  {choice}: {count} items")
            
        # New: Statistics for each task+model+mode preference
        print("\nModel Final preference statistics grouped by task:")
        summary_lines = []
        for task in existing_df['Task'].unique():
            print(f"\n{task}:")
            task_df = existing_df[existing_df['Task'] == task]
            for (model, mode) in task_df[['Model','Mode']].drop_duplicates().itertuples(index=False):
                model_df = task_df[(task_df['Model'] == model) & (task_df['Mode'] == mode)]
                
                if task.startswith('task1') or task.startswith('task2'):
                    # task1/task2 (包括 steered 版本): Statistics for majority/minority preference
                    majority_count = 0
                    minority_count = 0
                    for encoded in model_df['Encoded']:
                        if encoded == 1:
                            majority_count += 1
                        elif encoded == 0:
                            minority_count += 1
                    line = f"Task: {task} | Model: {model} | Mode: {mode} | Final preference for the majority group: {majority_count} | Final preference for the minority group: {minority_count}"
                elif task in ['task3promotion', 'task3stock', 'task3_steered_promotion', 'task3_steered_stock']:
                    # task3promotion/task3stock 及 steered 版本: Statistics for average conspiracy belief rating
                    ratings = [encoded for encoded in model_df['Encoded'] if encoded is not None and 1 <= encoded <= 7]
                    if ratings:
                        avg_rating = sum(ratings) / len(ratings)
                        line = f"Task: {task} | Model: {model} | Mode: {mode} | Average conspiracy belief rating: {avg_rating:.2f} | Total responses: {len(ratings)}"
                    else:
                        line = f"Task: {task} | Model: {model} | Mode: {mode} | No valid ratings found"
                else:
                    # Other task types
                    line = f"Task: {task} | Model: {model} | Mode: {mode} | Unknown task type"
                
                print(f"  {line}")
                summary_lines.append(line)
                # Save to file
        with open("./extracted_results/model_majority_preference_summary.txt", "w", encoding="utf-8") as f:
            for line in summary_lines:
                f.write(line + "\n")
            
    else:
        print("No valid replies found!")

if __name__ == "__main__":
    main()
