import os
import re
import pandas as pd
from pathlib import Path

MIN_CYCLE_COUNT = 90  # Allow files with slightly fewer than 100 cycles

def extract_role_and_cot_from_filename(filename: str):
    """
    从文件名中提取role和cot信息
    返回: (role, cot) 如果找到，否则返回 (None, None)
    """
    # Regex: _role_(.+?)_cot_(\d+)
    role_match = re.search(r'_role_(.+?)_cot_(\d+)', filename)
    if role_match:
        role = role_match.group(1)
        cot = role_match.group(2)
        return role, cot
    return None, None

def extract_temperature_from_filename(filename: str):
    """
    从文件名中提取temperature信息
    返回: temperature字符串，如果找到，否则返回 None
    """
    # Regex: _temperature_(.+?)_cycles_
    temp_match = re.search(r'_temperature_(.+?)_cycles_', filename)
    if temp_match:
        return temp_match.group(1)
    return None

def extract_final_response_only(response_text: str, model_name: str):
    """
    从可能包含对话历史的回复中提取仅最后的回复
    移除所有对话历史，只保留模型的最终回答
    """
    if not response_text:
        return response_text
    
    # Gemma: strip nested <start_of_turn>…, keep last model turn
    if "gemma" in model_name.lower():
        # Text after last <start_of_turn>model
        last_model_pos = response_text.rfind('<start_of_turn>model')
        if last_model_pos != -1:
            # Slice from last <start_of_turn>model
            segment = response_text[last_model_pos + len('<start_of_turn>model'):]
            # First <end_of_turn>/<start_of_turn>user marks history boundary
            end_positions = []
            end_turn_pos = segment.find('<end_of_turn>')
            if end_turn_pos != -1:
                end_positions.append(end_turn_pos)
            user_turn_pos = segment.find('<start_of_turn>user')
            if user_turn_pos != -1:
                end_positions.append(user_turn_pos)
            
            if end_positions:
                segment = segment[:min(end_positions)]
            
            # Drop leftover <start_of_turn> history
            # Non-greedy regex; avoid over-stripping
            # Remove user turns first
            segment = re.sub(r'<start_of_turn>user.*?<end_of_turn>', '', segment, flags=re.DOTALL)
            # Remove prior model turns
            segment = re.sub(r'<start_of_turn>model.*?<end_of_turn>', '', segment, flags=re.DOTALL)
            # Strip stray <start_of_turn> tags
            segment = re.sub(r'<start_of_turn>.*?</start_of_turn>', '', segment, flags=re.DOTALL)
            segment = re.sub(r'<end_of_turn>', '', segment)
            segment = re.sub(r'<bos>', '', segment)
            return segment.strip()
        # Fallback: strip every <start_of_turn>
        cleaned = re.sub(r'<start_of_turn>.*?</start_of_turn>', '', response_text, flags=re.DOTALL)
        cleaned = re.sub(r'<end_of_turn>', '', cleaned)
        cleaned = re.sub(r'<bos>', '', cleaned)
        return cleaned.strip()
    
    # Llama: strip nested <|…|>, keep final assistant span
    elif "llama" in model_name.lower():
        # Slice after last assistant header block
        last_assistant_tag = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        last_assistant_pos = response_text.rfind(last_assistant_tag)
        if last_assistant_pos != -1:
            segment = response_text[last_assistant_pos + len(last_assistant_tag):]
            # Remove nested chat history
            # Next user header starts new turn
            user_tag_pos = segment.find('<|start_header_id|>user')
            if user_tag_pos != -1:
                segment = segment[:user_tag_pos]
            # Remove <|begin_of_text|>…<|eot_id|> rounds
            segment = re.sub(r'<\|begin_of_text\|>.*?<\|eot_id\|>', '', segment, flags=re.DOTALL)
            # Strip leftover special tokens
            segment = re.sub(r'<\|.*?\|>', '', segment)
            return segment.strip()
        # Fallback strip all <|…|>
        cleaned = re.sub(r'<\|.*?\|>', '', response_text)
        return cleaned.strip()
    
    # Bracket models: slice after last [name]
    # Also strip embedded chat history
    else:
        # Find last [model_name] marker
        model_bracket_pattern = r'\[([^\]]+)\]\s*\n'
        matches = list(re.finditer(model_bracket_pattern, response_text))
        
        if matches:
            # Start from last match
            last_match = matches[-1]
            segment = response_text[last_match.end():]
            
            # Remove history markers
            # [user] begins fresh user turn
            user_bracket_pos = segment.find('[user]')
            if user_bracket_pos != -1:
                segment = segment[:user_bracket_pos]
            
            # Drop other dialogue wrappers
            # Remove [user] segments
            segment = re.sub(r'\[user\].*?(?=\[|$)', '', segment, flags=re.DOTALL)
            # Remove nested prior [model] replies
            segment = re.sub(r'\[[^\]]+\]\s*\n.*?(?=\[|$)', '', segment, flags=re.DOTALL)
            segment = re.sub(r'<\|.*?\|>', '', segment)
            segment = re.sub(r'<start_of_turn>.*?</start_of_turn>', '', segment, flags=re.DOTALL)
            segment = re.sub(r'<end_of_turn>', '', segment)
            
            return segment.strip()
        
        # If no bracket tag, return cleaned blob
        cleaned = re.sub(r'\[user\].*?(?=\[|$)', '', response_text, flags=re.DOTALL)
        cleaned = re.sub(r'<\|.*?\|>', '', cleaned)
        cleaned = re.sub(r'<start_of_turn>.*?</start_of_turn>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<end_of_turn>', '', cleaned)
        return cleaned.strip()

def extract_last_reply_from_txt(file_path: str, task_name: str):
    """
    从txt文件中提取每个cycle的模型最后一次回复
    返回: [(cycle, model_name, last_reply), ...]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Model id from filename
        filename = os.path.basename(file_path)
        model_match = re.search(r'model_(.+?)_mode_', filename)
        model_name = model_match.group(1) if model_match else "Unknown"
        
        # Locate cycle markers
        cycle_pattern = r'Cycle (\d+)/\d+:'
        cycle_matches = list(re.finditer(cycle_pattern, content))
        
        if not cycle_matches:
            print(f"警告: 在文件 {filename} 中未找到cycle信息")
            return []
        
        results = []
        
        for i, cycle_match in enumerate(cycle_matches):
            cycle_num = cycle_match.group(1)
            
            # End offset of current cycle
            if i + 1 < len(cycle_matches):
                # Not last cycle → ends at next cycle header
                end_pos = cycle_matches[i + 1].start()
            else:
                # Last cycle → EOF
                end_pos = len(content)
            
            # Slice cycle body
            cycle_start = cycle_match.start()
            cycle_content = content[cycle_start:end_pos].strip()
            
            # Parse majority_group
            majority_group = None
            # Match Group A/B (task1)
            majority_match = re.search(r'Presenting majority positive statements for Group ([AB])', cycle_content)
            if majority_match:
                majority_group = majority_match.group(1)
            else:
                # Match Company A/B (task2)
                majority_match = re.search(r'Presenting majority positive statements for Company ([AB])', cycle_content)
                if majority_match:
                    majority_group = majority_match.group(1)
            
            # Locate model reply
            # Check DeepSeek before Llama substring hits
            if "deepseek" in model_name.lower():
                # DeepSeek: between 2nd/last 20-dash rules
                # Locate 20-hyphen rules
                dash_pattern = r'^-{20}$'
                dash_matches = list(re.finditer(dash_pattern, cycle_content, re.MULTILINE))
                
                if len(dash_matches) >= 2:
                    # Between penultimate and final separators
                    second_last_pos = dash_matches[-2].end()  # Penultimate separator end offset
                    last_pos = dash_matches[-1].start()  # Final separator start offset
                    last_reply = cycle_content[second_last_pos:last_pos]
                    # Deliberately not strip()

                    # Keep only final Assistant thinking block
                    # From last <｜Assistant｜><think> (fallback <|Assistant|><think>)
                    assistant_think_tags = ["<｜Assistant｜><think>", "<|Assistant|><think>"]
                    cut_index = -1
                    for tag in assistant_think_tags:
                        idx = last_reply.rfind(tag)
                        if idx != -1 and idx > cut_index:
                            cut_index = idx
                    if cut_index != -1:
                        last_reply = last_reply[cut_index:]
                    
                    if last_reply:
                        results.append((cycle_num, model_name, last_reply, majority_group))
                        print(f"  - Cycle {cycle_num}: 提取到DeepSeek完整回复 (长度: {len(last_reply)})")
                    else:
                        print(f"  - Cycle {cycle_num}: DeepSeek回复内容为空")
                else:
                    print(f"  - Cycle {cycle_num}: DeepSeek模型未找到足够的--------------------分隔符 (找到{len(dash_matches)}个)")
            # Llama detection branch
            elif "llama" in model_name.lower():
                # Llama: take everything after final assistant header
                last_tag = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
                last_tag_backup = "<|eot|><|header_start|>assistant<|header_end|>"
                last_pos = cycle_content.rfind(last_tag)
                last_pos_backup = cycle_content.rfind(last_tag_backup)
                if last_pos != -1:
                    raw_reply = cycle_content[last_pos + len(last_tag):]
                    # Truncate at next rule line
                    cut_points = [p for p in [raw_reply.find('\n--------------------'), raw_reply.find('\n====================')] if p != -1]
                    if cut_points:
                        raw_reply = raw_reply[:min(cut_points)]
                    # Final reply only; strip history
                    last_reply = extract_final_response_only(raw_reply, model_name)
                    # Strip special markers
                    last_reply = re.sub(r'\n+', '\n', last_reply).strip()
                    results.append((cycle_num, model_name, last_reply, majority_group))
                    print(f"  - Cycle {cycle_num}: 提取到Llama完整回复 (长度: {len(last_reply)})")
                elif last_pos_backup != -1:
                    raw_reply = cycle_content[last_pos_backup + len(last_tag_backup):]
                    # Truncate at next rule line
                    cut_points = [p for p in [raw_reply.find('\n--------------------'), raw_reply.find('\n====================')] if p != -1]
                    if cut_points:
                        raw_reply = raw_reply[:min(cut_points)]
                    # Final reply only; strip history
                    last_reply = extract_final_response_only(raw_reply, model_name)
                    # Strip special markers
                    last_reply = re.sub(r'\n+', '\n', last_reply).strip()
                    results.append((cycle_num, model_name, last_reply, majority_group))
                    print(f"  - Cycle {cycle_num}: 提取到Llama完整回复 (长度: {len(last_reply)})")
                else:
                    # Fallback generic [model] pattern
                    reply_pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\[|\n--------------------|\n====================|\n\n|$)'
                    reply_matches = list(re.finditer(reply_pattern, cycle_content, re.DOTALL))
                    
                    if reply_matches:
                        # Keep last reply block
                        last_reply_match = reply_matches[-1]
                        raw_reply = last_reply_match.group(2)
                        # Final reply only; strip history
                        last_reply = extract_final_response_only(raw_reply, model_name)
                        
                        # Clean reply text
                        last_reply = re.sub(r'<think>.*?</think>', '', last_reply, flags=re.DOTALL)  # Remove <think> wrappers
                        last_reply = re.sub(r'Assistant:\s*', '', last_reply)  # Drop Assistant: prefix
                        last_reply = re.sub(r'\n+', '\n', last_reply).strip()  # Collapse blank lines
                        
                        if last_reply is not None:  # No min-length guard
                            results.append((cycle_num, model_name, last_reply, majority_group))
                            print(f"  - Cycle {cycle_num}: 提取到回复 (长度: {len(last_reply)})")
                        else:
                            print(f"  - Cycle {cycle_num}: 回复内容为None")
                    else:
                        print(f"  - Cycle {cycle_num}: 未找到Llama模型回复，cycle_content结尾如下：{cycle_content[-100:]}")
            elif "gemma" in model_name.lower():
                # Gemma final turn until separator
                tag_model = '<start_of_turn>model'
                tag_assistant = '<start_of_turn>assistant'
                tag_generic = '<start_of_turn>'
                pos_model = cycle_content.rfind(tag_model)
                pos_assistant = cycle_content.rfind(tag_assistant)
                pos_generic = cycle_content.rfind(tag_generic)

                use_tag = None
                use_pos = -1
                if pos_model != -1:
                    use_tag, use_pos = tag_model, pos_model
                elif pos_assistant != -1:
                    use_tag, use_pos = tag_assistant, pos_assistant
                elif pos_generic != -1:
                    use_tag, use_pos = tag_generic, pos_generic

                if use_pos != -1:
                    segment = cycle_content[use_pos + len(use_tag):]
                    # Clip at separator or EOF
                    cut_points = [p for p in [segment.find('\n--------------------'), segment.find('\n====================')] if p != -1]
                    if cut_points:
                        end_idx = min(cut_points)
                        segment = segment[:end_idx]
                    # Final reply only; strip history
                    last_reply = extract_final_response_only(segment, model_name)
                    # Remove placeholders/extra newlines
                    last_reply = re.sub(r'</?think>', '', last_reply, flags=re.DOTALL)
                    last_reply = re.sub(r'^\s*(model|assistant)\s*\n?', '', last_reply, flags=re.IGNORECASE)
                    last_reply = re.sub(r'\n+', '\n', last_reply).strip()
                    results.append((cycle_num, model_name, last_reply, majority_group))
                    print(f"  - Cycle {cycle_num}: 提取到Gemma完整回复 (长度: {len(last_reply)})")
                else:
                    print(f"  - Cycle {cycle_num}: 未找到Gemma模型回复起始标记，cycle_content结尾如下：{cycle_content[-100:]}")
            else:
                # Other models: [model_name] pattern
                # Find every [model] reply
                reply_pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\[|\n--------------------|\n====================|\n\n|$)'
                reply_matches = list(re.finditer(reply_pattern, cycle_content, re.DOTALL))
                
                if reply_matches:
                        # Keep last reply block
                        last_reply_match = reply_matches[-1]
                        
                        # Qwen: keep full last reply
                        if "qwen" in model_name.lower():
                            # Last [Qwen3-1.7B] offset
                            last_qwen_start = last_reply_match.start()
                            
                            # Next delimiter offset
                            next_separator = cycle_content.find('\n--------------------', last_qwen_start)
                            if next_separator == -1:
                                next_separator = cycle_content.find('\n====================', last_qwen_start)
                            if next_separator == -1:
                                next_separator = len(cycle_content)
                            
                            # Slice Qwen reply until delimiter
                            full_reply_start = last_qwen_start
                            full_reply_content = cycle_content[full_reply_start:next_separator]
                            
                            # Strip leading Qwen tag
                            full_reply_content = re.sub(r'^\[[^\]]+\]\n', '', full_reply_content)
                            
                            # Final reply only; strip history
                            full_reply_content = extract_final_response_only(full_reply_content, model_name)
                            
                            # Clean reply text
                            full_reply_content = re.sub(r'<think>.*?</think>', '', full_reply_content, flags=re.DOTALL)  # Remove <think> wrappers
                            full_reply_content = re.sub(r'Assistant:\s*', '', full_reply_content)  # Drop Assistant: prefix
                            # Keep markdown — encodes Group labels
                            # full_reply_content = re.sub(...markdown...)  # strip markdown (disabled)
                            full_reply_content = re.sub(r'\n+', '\n', full_reply_content).strip()  # Collapse blank lines
                            
                            if full_reply_content is not None:  # No min-length guard
                                results.append((cycle_num, model_name, full_reply_content, majority_group))
                                print(f"  - Cycle {cycle_num}: 提取到Qwen完整回复 (长度: {len(full_reply_content)})")
                            else:
                                print(f"  - Cycle {cycle_num}: Qwen回复内容为None")
                        else:
                            # Others: full final reply block
                            # Offset of last [model]
                            last_model_start = last_reply_match.start()
                            
                            # Next delimiter offset
                            next_separator = cycle_content.find('\n--------------------', last_model_start)
                            if next_separator == -1:
                                next_separator = cycle_content.find('\n====================', last_model_start)
                            if next_separator == -1:
                                # No delimiter → cycle tail
                                next_separator = len(cycle_content)
                            
                            # Slice bracket reply until delimiter
                            full_reply_start = last_model_start
                            full_reply_content = cycle_content[full_reply_start:next_separator]
                            
                            # Drop leading [model] tag
                            full_reply_content = re.sub(r'^\[[^\]]+\]\n?', '', full_reply_content)
                            
                            # Final reply only; strip history
                            last_reply = extract_final_response_only(full_reply_content, model_name)
                            
                            # Clean markers from reply
                            last_reply = re.sub(r'<think>.*?</think>', '', last_reply, flags=re.DOTALL)  # Remove <think> wrappers
                            last_reply = re.sub(r'Assistant:\s*', '', last_reply)  # Drop Assistant: prefix
                            # Keep markdown — encodes Group labels
                            # last_reply = re.sub(...markdown...)  # strip markdown (disabled)
                            last_reply = re.sub(r'\n+', '\n', last_reply).strip()  # Collapse blank lines
                            
                            if last_reply is not None:  # No min-length guard
                                results.append((cycle_num, model_name, last_reply, majority_group))
                                print(f"  - Cycle {cycle_num}: 提取到回复 (长度: {len(last_reply)})")
                            else:
                                print(f"  - Cycle {cycle_num}: 回复内容为None")
                else:
                    print(f"  - Cycle {cycle_num}: 未找到模型回复")
        
        return results
        
    except Exception as e:
        print(f"错误: 处理文件 {file_path} 时出错: {e}")
        return []

def extract_task3_replies(cycle_content: str, model_name_from_filename: str):
    """
    从task3的cycle内容中提取两个问题的回答
    根据内容识别是升职问题还是股票问题
    返回: ((model_name_promotion, reply_promotion), (model_name_stock, reply_stock))
    """
    promotion_reply = None
    stock_reply = None
    # Both answers share filename model id
    promotion_model_name = model_name_from_filename
    stock_model_name = model_name_from_filename

    # Locate 20-hyphen rules
    dash_positions = []
    for match in re.finditer(r'-{20}', cycle_content):
        dash_positions.append(match.start())

    if len(dash_positions) >= 3:
        # Q1 segment ends before 3rd-from-last rule
        task1_end_pos = dash_positions[-3]
        task1_segment = cycle_content[:task1_end_pos].strip()

        # Q2 segment between 3rd/last rules
        task2_start_pos = dash_positions[-3]
        task2_end_pos = dash_positions[-1]
        task2_segment = cycle_content[task2_start_pos:task2_end_pos].strip()

        # Extract both answers
        # Q1 segment already truncated → read to end
        # Q2 segment already truncated → read to end
        _, task1_reply = _extract_reply_from_segment_backup_style(task1_segment, model_name_from_filename, is_first_task=True)
        _, task2_reply = _extract_reply_from_segment_backup_style(task2_segment, model_name_from_filename, is_first_task=False)

        # Infer prompt type from wording
        # Promotion cues
        # Stock cues
        task1_is_promotion = any(keyword in task1_segment.lower() for keyword in 
                                ['promotion', 'boss', 'coworker', 'email', 'administrator', 'superiors'])
        task2_is_promotion = any(keyword in task2_segment.lower() for keyword in 
                                ['promotion', 'boss', 'coworker', 'email', 'administrator', 'superiors'])
        
        if task1_is_promotion and not task2_is_promotion:
            # Ordering: task1 promotion, task2 stock
            promotion_reply = task1_reply
            stock_reply = task2_reply
        elif task2_is_promotion and not task1_is_promotion:
            # Ordering: task2 promotion, task1 stock
            promotion_reply = task2_reply
            stock_reply = task1_reply
        else:
            # Unknown → default ordering
            promotion_reply = task1_reply
            stock_reply = task2_reply

    return (promotion_model_name, promotion_reply), (stock_model_name, stock_reply)

def _extract_reply_from_segment_backup_style(segment_content: str, model_name_from_filename: str, is_first_task: bool = False):
    """
    从segment中提取模型回复
    先尝试按问题定位，再提取到--------------------之间的内容
    
    Args:
        segment_content: 内容片段
        model_name_from_filename: 模型名
        is_first_task: 是否为第一个问题。第一个问题提取到segment末尾（因为segment已经截取到倒数第三个--------------------），
                      第二个问题也提取到segment末尾（因为segment已经截取到倒数第一个--------------------）
    """
    extracted_model_name = model_name_from_filename  # Default to filename model id
    extracted_reply = None
    
    # Highest priority: last question before terminator
    # Lenient regex templates for both prompts
    # Promotion regex variants
    promotion_patterns = [
        # With Provide-only clause
        r'To what extent do you think your coworker.*?connected.*?you not getting the promotion.*?\(1\s*=\s*not at all.*?7\s*=\s*a great deal\).*?\?.*?Provide.*?only.*?choice.*?\?',
        # Without Provide but full rubric
        r'To what extent do you think your coworker.*?connected.*?you not getting the promotion.*?\(1\s*=\s*not at all.*?7\s*=\s*a great deal\).*?\?',
        # Minimal regex fallback
        r'To what extent do you think your coworker.*?connected.*?you not getting the promotion.*?\?',
        r'coworker.*?connected.*?promotion.*?\?',  # Broadest pattern
    ]
    # Stock regex variants
    stock_patterns = [
        # Provide clause variant
        r'To what extent do you think the visits to the bed and breakfast.*?connected.*?earnings.*?stocks.*?\(1\s*=\s*not at all.*?7\s*=\s*a great deal\).*?\?.*?Provide.*?only.*?choice.*?\?',
        # Without Provide but full rubric
        r'To what extent do you think the visits to the bed and breakfast.*?connected.*?earnings.*?stocks.*?\(1\s*=\s*not at all.*?7\s*=\s*a great deal\).*?\?',
        # Minimal template variant
        r'To what extent do you think the visits to the bed and breakfast.*?connected.*?earnings.*?stocks.*?\?',
        r'bed and breakfast.*?connected.*?stocks.*?\?',  # Broadest pattern
    ]
    
    # Scan questions from end
    question_end_pos = -1
    last_match = None
    last_match_pos = -1
    
    # Try promotion patterns rear-first
    for pattern in promotion_patterns:
        matches = list(re.finditer(pattern, segment_content, re.DOTALL | re.IGNORECASE))
        if matches:
            # Keep match closest to terminator
            match = matches[-1]
            if match.end() > last_match_pos:
                last_match = match
                last_match_pos = match.end()
    
    # Try stock patterns rear-first
    for pattern in stock_patterns:
        matches = list(re.finditer(pattern, segment_content, re.DOTALL | re.IGNORECASE))
        if matches:
            # Keep match closest to terminator
            match = matches[-1]
            if match.end() > last_match_pos:
                last_match = match
                last_match_pos = match.end()
    
    if last_match:
        question_end_pos = last_match.end()
    
    if question_end_pos != -1:
        # Last Assistant: after question
        # Scan Assistant: from question end
        assistant_positions = []
        search_pos = question_end_pos
        while True:
            pos = segment_content.find('Assistant:', search_pos)
            if pos == -1:
                break
            assistant_positions.append(pos)
            search_pos = pos + len('Assistant:')
        
        if assistant_positions:
            # Use final Assistant: anchor
            last_assistant_pos = assistant_positions[-1]
            assistant_start = last_assistant_pos + len('Assistant:')
            # Segments already clipped at dash rules,
            # so read through segment end
            raw_reply = segment_content[assistant_start:].strip()
            # Final reply only; strip history
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
        else:
            # No Assistant: → still use whole segment
            raw_reply = segment_content[question_end_pos:].strip()
            # Final reply only; strip history
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
    
    # Return if question-anchored reply found
    if extracted_reply:
        return extracted_model_name, extracted_reply
    
    # Else fall back to legacy parser
    # Llama detection branch
    if "llama" in model_name_from_filename.lower():
        # Llama: take everything after final assistant header
        last_tag = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        last_tag_backup = "<|eot|><|header_start|>assistant<|header_end|>"
        last_pos = segment_content.rfind(last_tag)
        last_pos_backup = segment_content.rfind(last_tag_backup)
        if last_pos != -1:
            reply_start = last_pos + len(last_tag)
            # Truncated segment → take tail
            raw_reply = segment_content[reply_start:].strip()
            # Final reply only; strip history
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
            # Strip special markers
            extracted_reply = re.sub(r'\n+', '\n', extracted_reply).strip()
        elif last_pos_backup != -1:
            reply_start = last_pos_backup + len(last_tag_backup)
            # Truncated segment → take tail
            raw_reply = segment_content[reply_start:].strip()
            # Final reply only; strip history
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
            # Strip special markers
            extracted_reply = re.sub(r'\n+', '\n', extracted_reply).strip()
        else:
            # Fallback generic [model] pattern
            reply_pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\[|\n--------------------|\n====================|\n\n|$)'
            reply_matches = list(re.finditer(reply_pattern, segment_content, re.DOTALL))
            
            if reply_matches:
                # Keep last reply block
                last_reply_match = reply_matches[-1]
                extracted_model_name = last_reply_match.group(1).strip()  # Dynamic model name from reply
                raw_reply = last_reply_match.group(2)
                # Final reply only; strip history
                extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
                
                # Clean reply text
                extracted_reply = re.sub(r'<think>.*?</think>', '', extracted_reply, flags=re.DOTALL)
                extracted_reply = re.sub(r'Assistant:\s*', '', extracted_reply)
                extracted_reply = re.sub(r'\n+', '\n', extracted_reply).strip()
    elif "gemma" in model_name_from_filename.lower():
        # Gemma final turn until separator
        tag_model = '<start_of_turn>model'
        tag_assistant = '<start_of_turn>assistant'
        tag_generic = '<start_of_turn>'
        pos_model = segment_content.rfind(tag_model)
        pos_assistant = segment_content.rfind(tag_assistant)
        pos_generic = segment_content.rfind(tag_generic)

        use_tag = None
        use_pos = -1
        if pos_model != -1:
            use_tag, use_pos = tag_model, pos_model
        elif pos_assistant != -1:
            use_tag, use_pos = tag_assistant, pos_assistant
        elif pos_generic != -1:
            use_tag, use_pos = tag_generic, pos_generic

        if use_pos != -1:
            segment_start = use_pos + len(use_tag)
            # Truncated segment → take tail
            raw_reply = segment_content[segment_start:].strip()
            # Final reply only; strip history
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
            # Remove placeholders/extra newlines
            extracted_reply = re.sub(r'</?think>', '', extracted_reply, flags=re.DOTALL)
            extracted_reply = re.sub(r'^\s*(model|assistant)\s*\n?', '', extracted_reply, flags=re.IGNORECASE)
            extracted_reply = re.sub(r'\n+', '\n', extracted_reply).strip()
        else:
            # No Gemma tags → bracket fallback
            reply_pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\[|\n--------------------|\n====================|\n\n|$)'
            reply_matches = list(re.finditer(reply_pattern, segment_content, re.DOTALL))
            
            if reply_matches:
                # Keep last reply block
                last_reply_match = reply_matches[-1]
                extracted_model_name = last_reply_match.group(1).strip()  # Dynamic model name from reply
                raw_reply = last_reply_match.group(2)
                # Final reply only; strip history
                extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
                
                # Clean reply text
    return extracted_model_name, extracted_reply

def _extract_reply_from_segment(segment_content: str, model_name_from_filename: str):
    """
    从给定的内容片段中提取模型名和回复
    参考备份文件的逻辑，完全按照备份文件的提取方式
    """
    extracted_model_name = model_name_from_filename  # Default to filename model id
    extracted_reply = None

    # Try [MODEL] pattern first
    reply_pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\[|\n--------------------|\n====================|\n\n|$)'
    reply_matches = list(re.finditer(reply_pattern, segment_content, re.DOTALL))

    if reply_matches:
        last_reply_match = reply_matches[-1]
        extracted_model_name = last_reply_match.group(1).strip()
        raw_reply = last_reply_match.group(2)
        
        # Qwen: keep full last reply
        if "qwen" in extracted_model_name.lower():
            last_qwen_start = last_reply_match.start()
            next_separator = segment_content.find('\n--------------------', last_qwen_start)
            if next_separator == -1:
                next_separator = segment_content.find('\n====================', last_qwen_start)
            if next_separator == -1:
                next_separator = len(segment_content)
            full_reply_content = segment_content[last_qwen_start:next_separator]
            full_reply_content = re.sub(r'^\[[^\]]+\]\n', '', full_reply_content)
            # Final reply only; strip history
            processed_reply = extract_final_response_only(full_reply_content, extracted_model_name)
            extracted_reply = processed_reply.strip() if processed_reply and processed_reply.strip() else full_reply_content.strip()
        else:
            # Final reply only; strip history
            processed_reply = extract_final_response_only(raw_reply, extracted_model_name)
            extracted_reply = processed_reply.strip() if processed_reply and processed_reply.strip() else raw_reply.strip()
    else:
        # Try Llama specials
        llama_tag = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        llama_tag_backup = "<|eot|><|header_start|>assistant<|header_end|>"
        llama_pos = segment_content.rfind(llama_tag)
        llama_pos_backup = segment_content.rfind(llama_tag_backup)

        if llama_pos != -1:
            raw_reply = segment_content[llama_pos + len(llama_tag):].strip()
            # Final reply only; strip history
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
        elif llama_pos_backup != -1:
            raw_reply = segment_content[llama_pos_backup + len(llama_tag_backup):].strip()
            # Final reply only; strip history
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
        else:
            # Try Gemma specials
            gemma_tag_model = '<start_of_turn>model'
            gemma_tag_assistant = '<start_of_turn>assistant'
            gemma_tag_generic = '<start_of_turn>'
            gemma_tags = [gemma_tag_model, gemma_tag_assistant, gemma_tag_generic]

            for tag in gemma_tags:
                gemma_pos = segment_content.rfind(tag)
                if gemma_pos != -1:
                    raw_reply = segment_content[gemma_pos + len(tag):].strip()
                    # Final reply only; strip history
                    extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
                    break

    # Clean reply text
    if extracted_reply:
        extracted_reply = re.sub(r'</?think>', '', extracted_reply, flags=re.DOTALL)
        extracted_reply = re.sub(r'^\s*(model|assistant)\s*\n?', '', extracted_reply, flags=re.IGNORECASE)
        extracted_reply = re.sub(r'Assistant:\s*', '', extracted_reply)
        extracted_reply = re.sub(r'\n+', '\n', extracted_reply).strip()
        
    return extracted_model_name, extracted_reply

def scan_and_extract_results():
    """
    扫描results目录下的task1-3、task1_steered、task2_steered、task3_steered文件夹，提取所有txt文件的最后一次回复
    """
    results_dir = "./results"
    all_results = []
    
    # Scan task* folders
    for task_dir in ["task1", "task2", "task3", "task1_steered", "task2_steered", "task3_steered"]:
        task_path = os.path.join(results_dir, task_dir)
        if not os.path.exists(task_path):
            print(f"目录 {task_path} 不存在，跳过")
            continue
            
        print(f"\n正在处理 {task_dir}...")
        print(f"扫描目录: {task_path}")
        
        # Collect txt files
        txt_files = [f for f in os.listdir(task_path) if f.endswith('.txt')]
        print(f"找到 {len(txt_files)} 个txt文件")
        
        for i, filename in enumerate(txt_files):
            file_path = os.path.join(task_path, filename)
            print(f"\n处理文件 {i+1}/{len(txt_files)}: {filename}")
            
            # Skip empty files
            if os.path.getsize(file_path) == 0:
                print(f"  跳过空文件: {filename}")
                continue
            
            # Mode from filename
            mode = "direct" if "mode_direct" in filename else "explain" if "mode_explain" in filename else "unknown"
            
            # Role/COT from filename
            role, cot = extract_role_and_cot_from_filename(filename)
            
            # Temperature from filename
            temperature = extract_temperature_from_filename(filename)
            
            # Task3: dual-answer handling
            if task_dir == "task3" or task_dir == "task3_steered":
                # Re-read file contents
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Model id from filename
                model_match = re.search(r'model_(.+?)_mode_', filename)
                model_name_from_filename = model_match.group(1) if model_match else "Unknown"
                
                # Ensure enough task3 cycles
                cycle_pattern = r'Cycle (\d+)/\d+:'
                cycle_matches = list(re.finditer(cycle_pattern, file_content))
                if len(cycle_matches) < MIN_CYCLE_COUNT:
                    print(f"  跳过cycle数不足{MIN_CYCLE_COUNT}的文件: {filename} (cycle数: {len(cycle_matches)})")
                    continue
                
                # Re-parse task3 for both answers
                for cycle_num in range(1, 101):  # Task3 expects 100 cycles
                    cycle_pattern = rf'Cycle {cycle_num}/\d+:'
                    cycle_match = re.search(cycle_pattern, file_content)
                    
                    if not cycle_match:
                        continue
                    
                    cycle_start = cycle_match.start()
                    # Next cycle markers (complete/partial)
                    next_cycle_pattern = rf'Cycle {cycle_num + 1}/\d+:'
                    next_cycle_match = re.search(next_cycle_pattern, file_content)
                    
                    # Partial cycles: === line + Cycle row
                    incomplete_cycle_pattern = r'\n====================\nCycle\s*$'
                    incomplete_matches = list(re.finditer(incomplete_cycle_pattern, file_content, re.MULTILINE))
                    
                    # Candidate end offsets after cycle
                    candidate_positions = []
                    if next_cycle_match:
                        candidate_positions.append(next_cycle_match.start())
                    
                    # Include partial-cycle offsets
                    for incomplete_match in incomplete_matches:
                        if incomplete_match.start() > cycle_start:
                            # Locate === banner start
                            # Search lines before === banner
                            dash_pos = file_content.rfind('====================', cycle_start, incomplete_match.start())
                            if dash_pos != -1:
                                candidate_positions.append(dash_pos)
                    
                    # Pick nearest cutoff
                    if candidate_positions:
                        cycle_end = min(candidate_positions)
                    else:
                        cycle_end = len(file_content)
                    
                    cycle_content = file_content[cycle_start:cycle_end].strip()
                    
                    # Pull both Q&A bodies
                    (promotion_model, promotion_reply), (stock_model, stock_reply) = extract_task3_replies(cycle_content, model_name_from_filename)
                    
                    # One Excel row per answer
                    if promotion_reply is not None:
                        all_results.append({
                            'Task': f"{task_dir}_promotion" if task_dir == "task3_steered" else f"{task_dir}promotion",
                            'Filename': filename,
                            'Cycle': cycle_num,
                            'Model': promotion_model,
                            'ChatGPT最后回复': promotion_reply,
                            'majority_group': None,  # Task3 omits majority_group
                            'Mode': mode,
                            'Role': role,
                            'COT': cot,
                            'Temperature': temperature
                        })
                    
                    if stock_reply is not None:
                        all_results.append({
                            'Task': f"{task_dir}_stock" if task_dir == "task3_steered" else f"{task_dir}stock", 
                            'Filename': filename,
                            'Cycle': cycle_num,
                            'Model': stock_model,
                            'ChatGPT最后回复': stock_reply,
                            'majority_group': None,  # Task3 omits majority_group
                            'Mode': mode,
                            'Role': role,
                            'COT': cot,
                            'Temperature': temperature
                        })
                
                # Skip single-answer path
                continue
            else:
                # Task1/2 single-answer pipeline
                file_results = extract_last_reply_from_txt(file_path, task_dir)
                
                # Require ≥100 cycles
                if len(file_results) < MIN_CYCLE_COUNT:
                    print(f"  跳过cycle数不足{MIN_CYCLE_COUNT}的文件: {filename} (cycle数: {len(file_results)})")
                    continue
                
                # Append row to results
                for cycle, model_name, last_reply, majority_group in file_results:
                    # steered_* suffix for steered runs
                    task_name = f"{task_dir}_steered" if task_dir in ["task1_steered", "task2_steered"] else task_dir
                    all_results.append({
                        'Task': task_name,
                        'Filename': filename,
                        'Cycle': cycle,
                        'Model': model_name,
                        'ChatGPT最后回复': last_reply,
                        'majority_group': majority_group,
                        'Mode': mode,
                        'Role': role,
                        'COT': cot,
                        'Temperature': temperature
                    })
    
    return all_results

def main():
    print("=" * 60)
    print("开始从txt文件提取模型最后一次回复")
    print("=" * 60)
    
    # Create output directory
    output_dir = "./extracted_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Scan & extract
    all_results = scan_and_extract_results()
    
    if not all_results:
        print("\n未找到任何有效结果！")
        return
    
    # Save workbook
    output_file = os.path.join(output_dir, "chatgpt_last_replies.xlsx")
    
    df = pd.DataFrame(all_results)

    # Strip illegal chars before Excel write
    # Drop ASCII control chars
    illegal_char_pattern = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).apply(
                lambda x: illegal_char_pattern.sub("", x) if isinstance(x, str) else x
            )

    df.to_excel(output_file, index=False, engine="openpyxl")
    
    print("\n" + "=" * 60)
    print("提取完成！")
    print(f"总共提取了 {len(all_results)} 个回复")
    print(f"结果已保存到: {output_file}")
    print("=" * 60)
    
    # Print stats
    print("\n统计信息:")
    task_counts = df['Task'].value_counts()
    for task, count in task_counts.items():
        print(f"  {task}: {count} 个回复")
    
    model_counts = df['Model'].value_counts()
    print(f"\n模型分布:")
    for model, count in model_counts.head(10).items():  # Show top 10 models
        print(f"  {model}: {count} 个回复")

if __name__ == "__main__":
    main()
