import os
import re
import pandas as pd
from pathlib import Path

MIN_CYCLE_COUNT = 90  # 允许略少于100个cycle的文件通过

def extract_role_and_cot_from_filename(filename: str):
    """
    从文件名中提取role和cot信息
    返回: (role, cot) 如果找到，否则返回 (None, None)
    """
    # 匹配 pattern: _role_(.+?)_cot_(\d+)
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
    # 匹配 pattern: _temperature_(.+?)_cycles_
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
    
    # 对于Gemma模型：移除所有嵌套的<start_of_turn>标签及其内容，只保留最后一个model的回复
    if "gemma" in model_name.lower():
        # 找到最后一个<start_of_turn>model之后的内容
        last_model_pos = response_text.rfind('<start_of_turn>model')
        if last_model_pos != -1:
            # 从最后一个<start_of_turn>model开始提取
            segment = response_text[last_model_pos + len('<start_of_turn>model'):]
            # 找到第一个<end_of_turn>或<start_of_turn>user的位置（这些标记对话历史的开始）
            end_positions = []
            end_turn_pos = segment.find('<end_of_turn>')
            if end_turn_pos != -1:
                end_positions.append(end_turn_pos)
            user_turn_pos = segment.find('<start_of_turn>user')
            if user_turn_pos != -1:
                end_positions.append(user_turn_pos)
            
            if end_positions:
                segment = segment[:min(end_positions)]
            
            # 移除所有剩余的嵌套<start_of_turn>标签（这些是对话历史）
            # 使用非贪婪匹配，但需要确保不会移除太多
            # 先移除<start_of_turn>user...<end_of_turn>模式（这些是用户消息）
            segment = re.sub(r'<start_of_turn>user.*?<end_of_turn>', '', segment, flags=re.DOTALL)
            # 移除<start_of_turn>model...<end_of_turn>模式（这些是之前的模型回复）
            segment = re.sub(r'<start_of_turn>model.*?<end_of_turn>', '', segment, flags=re.DOTALL)
            # 移除其他<start_of_turn>标签
            segment = re.sub(r'<start_of_turn>.*?</start_of_turn>', '', segment, flags=re.DOTALL)
            segment = re.sub(r'<end_of_turn>', '', segment)
            segment = re.sub(r'<bos>', '', segment)
            return segment.strip()
        # 如果没有找到<start_of_turn>model，尝试移除所有<start_of_turn>标签
        cleaned = re.sub(r'<start_of_turn>.*?</start_of_turn>', '', response_text, flags=re.DOTALL)
        cleaned = re.sub(r'<end_of_turn>', '', cleaned)
        cleaned = re.sub(r'<bos>', '', cleaned)
        return cleaned.strip()
    
    # 对于Llama模型：移除所有嵌套的<|...|>标签及其内容，只保留最后一个assistant的回复
    elif "llama" in model_name.lower():
        # 找到最后一个<|eot_id|><|start_header_id|>assistant<|end_header_id|>之后的内容
        last_assistant_tag = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        last_assistant_pos = response_text.rfind(last_assistant_tag)
        if last_assistant_pos != -1:
            segment = response_text[last_assistant_pos + len(last_assistant_tag):]
            # 移除所有嵌套的对话历史
            # 找到第一个<|start_header_id|>user的位置（这标记新的用户消息开始）
            user_tag_pos = segment.find('<|start_header_id|>user')
            if user_tag_pos != -1:
                segment = segment[:user_tag_pos]
            # 移除所有<|begin_of_text|>...<|eot_id|>模式（这些是之前的对话轮次）
            segment = re.sub(r'<\|begin_of_text\|>.*?<\|eot_id\|>', '', segment, flags=re.DOTALL)
            # 清理所有剩余的标签
            segment = re.sub(r'<\|.*?\|>', '', segment)
            return segment.strip()
        # 如果没有找到，尝试移除所有<|...|>标签
        cleaned = re.sub(r'<\|.*?\|>', '', response_text)
        return cleaned.strip()
    
    # 对于使用[model_name]格式的模型：提取最后一个[model_name]标记后的内容
    # 但需要移除其中可能包含的对话历史
    else:
        # 查找最后一个[model_name]格式
        model_bracket_pattern = r'\[([^\]]+)\]\s*\n'
        matches = list(re.finditer(model_bracket_pattern, response_text))
        
        if matches:
            # 从最后一个匹配开始
            last_match = matches[-1]
            segment = response_text[last_match.end():]
            
            # 移除可能包含的对话历史标记
            # 对于可能包含的[user]标记（这标记新的用户消息开始）
            user_bracket_pos = segment.find('[user]')
            if user_bracket_pos != -1:
                segment = segment[:user_bracket_pos]
            
            # 移除其他可能的对话标记和嵌套内容
            # 移除[user]...模式（用户消息）
            segment = re.sub(r'\[user\].*?(?=\[|$)', '', segment, flags=re.DOTALL)
            # 移除嵌套的[model_name]模式（之前的模型回复）
            segment = re.sub(r'\[[^\]]+\]\s*\n.*?(?=\[|$)', '', segment, flags=re.DOTALL)
            segment = re.sub(r'<\|.*?\|>', '', segment)
            segment = re.sub(r'<start_of_turn>.*?</start_of_turn>', '', segment, flags=re.DOTALL)
            segment = re.sub(r'<end_of_turn>', '', segment)
            
            return segment.strip()
        
        # 如果没有找到[model_name]格式，直接返回清理后的内容
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
        
        # 从文件名提取模型名称
        filename = os.path.basename(file_path)
        model_match = re.search(r'model_(.+?)_mode_', filename)
        model_name = model_match.group(1) if model_match else "Unknown"
        
        # 查找所有cycle信息
        cycle_pattern = r'Cycle (\d+)/\d+:'
        cycle_matches = list(re.finditer(cycle_pattern, content))
        
        if not cycle_matches:
            print(f"警告: 在文件 {filename} 中未找到cycle信息")
            return []
        
        results = []
        
        for i, cycle_match in enumerate(cycle_matches):
            cycle_num = cycle_match.group(1)
            
            # 确定当前cycle的结束位置
            if i + 1 < len(cycle_matches):
                # 不是最后一个cycle，结束位置是下一个cycle的开始
                end_pos = cycle_matches[i + 1].start()
            else:
                # 最后一个cycle，结束位置是文件末尾
                end_pos = len(content)
            
            # 提取当前cycle的内容
            cycle_start = cycle_match.start()
            cycle_content = content[cycle_start:end_pos].strip()
            
            # 提取majority_group
            majority_group = None
            # 尝试匹配 Group A/B (task1)
            majority_match = re.search(r'Presenting majority positive statements for Group ([AB])', cycle_content)
            if majority_match:
                majority_group = majority_match.group(1)
            else:
                # 尝试匹配 Company A/B (task2)
                majority_match = re.search(r'Presenting majority positive statements for Company ([AB])', cycle_content)
                if majority_match:
                    majority_group = majority_match.group(1)
            
            # 查找模型回复
            # 注意：先检查DeepSeek，因为它可能也包含"llama"字符串
            if "deepseek" in model_name.lower():
                # DeepSeek模型：使用倒数第二个和倒数第一个--------------------之间的内容
                # 找到所有--------------------的位置（只匹配正好20个短横线）
                dash_pattern = r'^-{20}$'
                dash_matches = list(re.finditer(dash_pattern, cycle_content, re.MULTILINE))
                
                if len(dash_matches) >= 2:
                    # 倒数第二个--------------------之后到倒数第一个--------------------之前的内容
                    second_last_pos = dash_matches[-2].end()  # 倒数第二个--------------------的结束位置
                    last_pos = dash_matches[-1].start()  # 倒数第一个--------------------的开始位置
                    last_reply = cycle_content[second_last_pos:last_pos]
                    # 注意：这里不strip()，保留所有字符，包括前后的空白

                    # 仅保留最后一次 Assistant 思考的内容：
                    # 从最后一个 "<｜Assistant｜><think>"（或降级为 "<|Assistant|><think>"）开始
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
            # 检查是否是Llama模型
            elif "llama" in model_name.lower():
                # Llama模型：提取最后一个<|eot_id|><|start_header_id|>assistant<|end_header_id|>后面所有内容
                last_tag = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
                last_tag_backup = "<|eot|><|header_start|>assistant<|header_end|>"
                last_pos = cycle_content.rfind(last_tag)
                last_pos_backup = cycle_content.rfind(last_tag_backup)
                if last_pos != -1:
                    raw_reply = cycle_content[last_pos + len(last_tag):]
                    # 截断到下一个分隔线
                    cut_points = [p for p in [raw_reply.find('\n--------------------'), raw_reply.find('\n====================')] if p != -1]
                    if cut_points:
                        raw_reply = raw_reply[:min(cut_points)]
                    # 提取仅最后的回复，移除对话历史
                    last_reply = extract_final_response_only(raw_reply, model_name)
                    # 清理特殊标记
                    last_reply = re.sub(r'\n+', '\n', last_reply).strip()
                    results.append((cycle_num, model_name, last_reply, majority_group))
                    print(f"  - Cycle {cycle_num}: 提取到Llama完整回复 (长度: {len(last_reply)})")
                elif last_pos_backup != -1:
                    raw_reply = cycle_content[last_pos_backup + len(last_tag_backup):]
                    # 截断到下一个分隔线
                    cut_points = [p for p in [raw_reply.find('\n--------------------'), raw_reply.find('\n====================')] if p != -1]
                    if cut_points:
                        raw_reply = raw_reply[:min(cut_points)]
                    # 提取仅最后的回复，移除对话历史
                    last_reply = extract_final_response_only(raw_reply, model_name)
                    # 清理特殊标记
                    last_reply = re.sub(r'\n+', '\n', last_reply).strip()
                    results.append((cycle_num, model_name, last_reply, majority_group))
                    print(f"  - Cycle {cycle_num}: 提取到Llama完整回复 (长度: {len(last_reply)})")
                else:
                    # 如果没有找到Llama特殊标记，尝试使用通用的[model_name]格式
                    reply_pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\[|\n--------------------|\n====================|\n\n|$)'
                    reply_matches = list(re.finditer(reply_pattern, cycle_content, re.DOTALL))
                    
                    if reply_matches:
                        # 取最后一个回复
                        last_reply_match = reply_matches[-1]
                        raw_reply = last_reply_match.group(2)
                        # 提取仅最后的回复，移除对话历史
                        last_reply = extract_final_response_only(raw_reply, model_name)
                        
                        # 清理回复内容
                        last_reply = re.sub(r'<think>.*?</think>', '', last_reply, flags=re.DOTALL)  # 移除think标记
                        last_reply = re.sub(r'Assistant:\s*', '', last_reply)  # 移除Assistant前缀
                        last_reply = re.sub(r'\n+', '\n', last_reply).strip()  # 清理多余换行
                        
                        if last_reply is not None:  # 不再判断长度
                            results.append((cycle_num, model_name, last_reply, majority_group))
                            print(f"  - Cycle {cycle_num}: 提取到回复 (长度: {len(last_reply)})")
                        else:
                            print(f"  - Cycle {cycle_num}: 回复内容为None")
                    else:
                        print(f"  - Cycle {cycle_num}: 未找到Llama模型回复，cycle_content结尾如下：{cycle_content[-100:]}")
            elif "gemma" in model_name.lower():
                # Gemma模型：取最后一个 <start_of_turn>（优先 <start_of_turn>model，其次 assistant）到下一个分隔线之前
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
                    # 截断到下一个分隔线或文件末尾
                    cut_points = [p for p in [segment.find('\n--------------------'), segment.find('\n====================')] if p != -1]
                    if cut_points:
                        end_idx = min(cut_points)
                        segment = segment[:end_idx]
                    # 提取仅最后的回复，移除对话历史
                    last_reply = extract_final_response_only(segment, model_name)
                    # 清理常见占位符与多余换行
                    last_reply = re.sub(r'</?think>', '', last_reply, flags=re.DOTALL)
                    last_reply = re.sub(r'^\s*(model|assistant)\s*\n?', '', last_reply, flags=re.IGNORECASE)
                    last_reply = re.sub(r'\n+', '\n', last_reply).strip()
                    results.append((cycle_num, model_name, last_reply, majority_group))
                    print(f"  - Cycle {cycle_num}: 提取到Gemma完整回复 (长度: {len(last_reply)})")
                else:
                    print(f"  - Cycle {cycle_num}: 未找到Gemma模型回复起始标记，cycle_content结尾如下：{cycle_content[-100:]}")
            else:
                # 其他模型：匹配模式1: [model_name] 格式
                # 查找所有 [model_name] 格式的回复
                reply_pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\[|\n--------------------|\n====================|\n\n|$)'
                reply_matches = list(re.finditer(reply_pattern, cycle_content, re.DOTALL))
                
                if reply_matches:
                        # 取最后一个回复
                        last_reply_match = reply_matches[-1]
                        
                        # 对于Qwen模型，需要特殊处理：查找最后一个回复的完整内容
                        if "qwen" in model_name.lower():
                            # 找到最后一个 [Qwen3-1.7B] 的位置
                            last_qwen_start = last_reply_match.start()
                            
                            # 查找下一个分隔符的位置
                            next_separator = cycle_content.find('\n--------------------', last_qwen_start)
                            if next_separator == -1:
                                next_separator = cycle_content.find('\n====================', last_qwen_start)
                            if next_separator == -1:
                                next_separator = len(cycle_content)
                            
                            # 提取从 [Qwen3-1.7B] 到下一个分隔符的完整内容
                            full_reply_start = last_qwen_start
                            full_reply_content = cycle_content[full_reply_start:next_separator]
                            
                            # 移除开头的 [Qwen3-1.7B] 标记
                            full_reply_content = re.sub(r'^\[[^\]]+\]\n', '', full_reply_content)
                            
                            # 提取仅最后的回复，移除对话历史
                            full_reply_content = extract_final_response_only(full_reply_content, model_name)
                            
                            # 清理回复内容
                            full_reply_content = re.sub(r'<think>.*?</think>', '', full_reply_content, flags=re.DOTALL)  # 移除think标记
                            full_reply_content = re.sub(r'Assistant:\s*', '', full_reply_content)  # 移除Assistant前缀
                            # 不要移除markdown标记，因为可能包含重要的Group信息
                            # full_reply_content = re.sub(r'\*\*.*?\*\*', '', full_reply_content)  # 移除markdown标记
                            full_reply_content = re.sub(r'\n+', '\n', full_reply_content).strip()  # 清理多余换行
                            
                            if full_reply_content is not None:  # 不再判断长度
                                results.append((cycle_num, model_name, full_reply_content, majority_group))
                                print(f"  - Cycle {cycle_num}: 提取到Qwen完整回复 (长度: {len(full_reply_content)})")
                            else:
                                print(f"  - Cycle {cycle_num}: Qwen回复内容为None")
                        else:
                            # 其他模型（包括gpt-5-mini等）：提取最后一个回复的完整内容
                            # 找到最后一个 [model_name] 的位置
                            last_model_start = last_reply_match.start()
                            
                            # 查找下一个分隔符的位置
                            next_separator = cycle_content.find('\n--------------------', last_model_start)
                            if next_separator == -1:
                                next_separator = cycle_content.find('\n====================', last_model_start)
                            if next_separator == -1:
                                # 如果没有找到分隔符，提取到cycle末尾
                                next_separator = len(cycle_content)
                            
                            # 提取从 [model_name] 到下一个分隔符的完整内容
                            full_reply_start = last_model_start
                            full_reply_content = cycle_content[full_reply_start:next_separator]
                            
                            # 移除开头的 [model_name] 标记
                            full_reply_content = re.sub(r'^\[[^\]]+\]\n?', '', full_reply_content)
                            
                            # 提取仅最后的回复，移除对话历史
                            last_reply = extract_final_response_only(full_reply_content, model_name)
                            
                            # 清理回复内容（移除特殊标记）
                            last_reply = re.sub(r'<think>.*?</think>', '', last_reply, flags=re.DOTALL)  # 移除think标记
                            last_reply = re.sub(r'Assistant:\s*', '', last_reply)  # 移除Assistant前缀
                            # 不要移除markdown标记，因为可能包含重要的Group信息
                            # last_reply = re.sub(r'\*\*.*?\*\*', '', last_reply)  # 移除markdown标记
                            last_reply = re.sub(r'\n+', '\n', last_reply).strip()  # 清理多余换行
                            
                            if last_reply is not None:  # 不再判断长度
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
    # 两个问题都使用同一个模型名（从文件名提取）
    promotion_model_name = model_name_from_filename
    stock_model_name = model_name_from_filename

    # 找到所有--------------------的位置（只匹配正好20个短横线）
    dash_positions = []
    for match in re.finditer(r'-{20}', cycle_content):
        dash_positions.append(match.start())

    if len(dash_positions) >= 3:
        # 第一个问题的片段：从cycle开头到倒数第三个--------------------之前
        task1_end_pos = dash_positions[-3]
        task1_segment = cycle_content[:task1_end_pos].strip()

        # 第二个问题的片段：从倒数第三个--------------------到倒数第一个--------------------之前
        task2_start_pos = dash_positions[-3]
        task2_end_pos = dash_positions[-1]
        task2_segment = cycle_content[task2_start_pos:task2_end_pos].strip()

        # 提取两个问题的回复
        # task1_segment 已经截取到倒数第三个--------------------，所以提取到segment末尾
        # task2_segment 已经截取到倒数第一个--------------------，所以提取到segment末尾
        _, task1_reply = _extract_reply_from_segment_backup_style(task1_segment, model_name_from_filename, is_first_task=True)
        _, task2_reply = _extract_reply_from_segment_backup_style(task2_segment, model_name_from_filename, is_first_task=False)

        # 根据内容识别问题类型
        # 升职问题的关键词：promotion, boss, coworker, email, administrator
        # 股票问题的关键词：stock, construction companies, bed and breakfast, B&B, prices
        task1_is_promotion = any(keyword in task1_segment.lower() for keyword in 
                                ['promotion', 'boss', 'coworker', 'email', 'administrator', 'superiors'])
        task2_is_promotion = any(keyword in task2_segment.lower() for keyword in 
                                ['promotion', 'boss', 'coworker', 'email', 'administrator', 'superiors'])
        
        if task1_is_promotion and not task2_is_promotion:
            # task1是升职问题，task2是股票问题
            promotion_reply = task1_reply
            stock_reply = task2_reply
        elif task2_is_promotion and not task1_is_promotion:
            # task2是升职问题，task1是股票问题
            promotion_reply = task2_reply
            stock_reply = task1_reply
        else:
            # 如果无法识别，使用默认顺序
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
    extracted_model_name = model_name_from_filename  # 默认使用文件名中的模型名
    extracted_reply = None
    
    # 优先级最高：从后往前查找终止符之前最靠近的问题
    # 定义两个问题的问题模板（放松匹配以避免错过）
    # promotion问题：支持有或没有"Provide **only**..."的版本，使用更宽松的匹配
    promotion_patterns = [
        # 带Provide版本，允许各种变体
        r'To what extent do you think your coworker.*?connected.*?you not getting the promotion.*?\(1\s*=\s*not at all.*?7\s*=\s*a great deal\).*?\?.*?Provide.*?only.*?choice.*?\?',
        # 不带Provide版本，但包含完整的评分说明
        r'To what extent do you think your coworker.*?connected.*?you not getting the promotion.*?\(1\s*=\s*not at all.*?7\s*=\s*a great deal\).*?\?',
        # 最简版本（如果上面的都匹配不到）
        r'To what extent do you think your coworker.*?connected.*?you not getting the promotion.*?\?',
        r'coworker.*?connected.*?promotion.*?\?',  # 最宽松版本
    ]
    # stock问题：支持有或没有"Provide **only**..."的版本
    stock_patterns = [
        # 带Provide版本
        r'To what extent do you think the visits to the bed and breakfast.*?connected.*?earnings.*?stocks.*?\(1\s*=\s*not at all.*?7\s*=\s*a great deal\).*?\?.*?Provide.*?only.*?choice.*?\?',
        # 不带Provide版本，但包含完整的评分说明
        r'To what extent do you think the visits to the bed and breakfast.*?connected.*?earnings.*?stocks.*?\(1\s*=\s*not at all.*?7\s*=\s*a great deal\).*?\?',
        # 最简版本
        r'To what extent do you think the visits to the bed and breakfast.*?connected.*?earnings.*?stocks.*?\?',
        r'bed and breakfast.*?connected.*?stocks.*?\?',  # 最宽松版本
    ]
    
    # 从后往前查找问题（最高优先级）
    question_end_pos = -1
    last_match = None
    last_match_pos = -1
    
    # 尝试所有promotion模式，从后往前找
    for pattern in promotion_patterns:
        matches = list(re.finditer(pattern, segment_content, re.DOTALL | re.IGNORECASE))
        if matches:
            # 取最后一个匹配（最靠近终止符）
            match = matches[-1]
            if match.end() > last_match_pos:
                last_match = match
                last_match_pos = match.end()
    
    # 尝试所有stock模式，从后往前找
    for pattern in stock_patterns:
        matches = list(re.finditer(pattern, segment_content, re.DOTALL | re.IGNORECASE))
        if matches:
            # 取最后一个匹配（最靠近终止符）
            match = matches[-1]
            if match.end() > last_match_pos:
                last_match = match
                last_match_pos = match.end()
    
    if last_match:
        question_end_pos = last_match.end()
    
    if question_end_pos != -1:
        # 查找问题之后最后一个Assistant:标记（因为可能有多个Assistant回复）
        # 从问题结束位置开始，查找所有的Assistant:标记
        assistant_positions = []
        search_pos = question_end_pos
        while True:
            pos = segment_content.find('Assistant:', search_pos)
            if pos == -1:
                break
            assistant_positions.append(pos)
            search_pos = pos + len('Assistant:')
        
        if assistant_positions:
            # 使用最后一个Assistant标记
            last_assistant_pos = assistant_positions[-1]
            assistant_start = last_assistant_pos + len('Assistant:')
            # 由于segment已经正确截取（task1到倒数第三个--------------------，task2到倒数第一个--------------------），
            # 所以直接提取到segment末尾即可
            raw_reply = segment_content[assistant_start:].strip()
            # 提取仅最后的回复，移除对话历史
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
        else:
            # 如果没找到Assistant标记，直接提取到segment末尾（因为segment已经正确截取了）
            raw_reply = segment_content[question_end_pos:].strip()
            # 提取仅最后的回复，移除对话历史
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
    
    # 如果通过问题定位找到回复，直接返回
    if extracted_reply:
        return extracted_model_name, extracted_reply
    
    # 如果还是没找到回复，继续使用原有逻辑
    # 检查是否是Llama模型
    if "llama" in model_name_from_filename.lower():
        # Llama模型：提取最后一个<|eot_id|><|start_header_id|>assistant<|end_header_id|>后面所有内容
        last_tag = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        last_tag_backup = "<|eot|><|header_start|>assistant<|header_end|>"
        last_pos = segment_content.rfind(last_tag)
        last_pos_backup = segment_content.rfind(last_tag_backup)
        if last_pos != -1:
            reply_start = last_pos + len(last_tag)
            # 由于segment已经正确截取，直接提取到segment末尾
            raw_reply = segment_content[reply_start:].strip()
            # 提取仅最后的回复，移除对话历史
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
            # 清理特殊标记
            extracted_reply = re.sub(r'\n+', '\n', extracted_reply).strip()
        elif last_pos_backup != -1:
            reply_start = last_pos_backup + len(last_tag_backup)
            # 由于segment已经正确截取，直接提取到segment末尾
            raw_reply = segment_content[reply_start:].strip()
            # 提取仅最后的回复，移除对话历史
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
            # 清理特殊标记
            extracted_reply = re.sub(r'\n+', '\n', extracted_reply).strip()
        else:
            # 如果没有找到Llama特殊标记，尝试使用通用的[model_name]格式
            reply_pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\[|\n--------------------|\n====================|\n\n|$)'
            reply_matches = list(re.finditer(reply_pattern, segment_content, re.DOTALL))
            
            if reply_matches:
                # 取最后一个回复
                last_reply_match = reply_matches[-1]
                extracted_model_name = last_reply_match.group(1).strip()  # 动态提取模型名
                raw_reply = last_reply_match.group(2)
                # 提取仅最后的回复，移除对话历史
                extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
                
                # 清理回复内容
                extracted_reply = re.sub(r'<think>.*?</think>', '', extracted_reply, flags=re.DOTALL)
                extracted_reply = re.sub(r'Assistant:\s*', '', extracted_reply)
                extracted_reply = re.sub(r'\n+', '\n', extracted_reply).strip()
    elif "gemma" in model_name_from_filename.lower():
        # Gemma模型：取最后一个 <start_of_turn>（优先 <start_of_turn>model，其次 assistant）到下一个分隔线之前
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
            # 由于segment已经正确截取，直接提取到segment末尾
            raw_reply = segment_content[segment_start:].strip()
            # 提取仅最后的回复，移除对话历史
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
            # 清理常见占位符与多余换行
            extracted_reply = re.sub(r'</?think>', '', extracted_reply, flags=re.DOTALL)
            extracted_reply = re.sub(r'^\s*(model|assistant)\s*\n?', '', extracted_reply, flags=re.IGNORECASE)
            extracted_reply = re.sub(r'\n+', '\n', extracted_reply).strip()
        else:
            # 如果没有找到Gemma特殊标记，尝试使用通用的[model_name]格式
            reply_pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\[|\n--------------------|\n====================|\n\n|$)'
            reply_matches = list(re.finditer(reply_pattern, segment_content, re.DOTALL))
            
            if reply_matches:
                # 取最后一个回复
                last_reply_match = reply_matches[-1]
                extracted_model_name = last_reply_match.group(1).strip()  # 动态提取模型名
                raw_reply = last_reply_match.group(2)
                # 提取仅最后的回复，移除对话历史
                extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
                
                # 清理回复内容
    return extracted_model_name, extracted_reply

def _extract_reply_from_segment(segment_content: str, model_name_from_filename: str):
    """
    从给定的内容片段中提取模型名和回复
    参考备份文件的逻辑，完全按照备份文件的提取方式
    """
    extracted_model_name = model_name_from_filename  # 默认使用文件名中的模型名
    extracted_reply = None

    # 首先尝试匹配[MODEL_NAME]格式 - 这是最常见的格式
    reply_pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\[|\n--------------------|\n====================|\n\n|$)'
    reply_matches = list(re.finditer(reply_pattern, segment_content, re.DOTALL))

    if reply_matches:
        last_reply_match = reply_matches[-1]
        extracted_model_name = last_reply_match.group(1).strip()
        raw_reply = last_reply_match.group(2)
        
        # 对于Qwen模型，需要特殊处理：查找最后一个回复的完整内容
        if "qwen" in extracted_model_name.lower():
            last_qwen_start = last_reply_match.start()
            next_separator = segment_content.find('\n--------------------', last_qwen_start)
            if next_separator == -1:
                next_separator = segment_content.find('\n====================', last_qwen_start)
            if next_separator == -1:
                next_separator = len(segment_content)
            full_reply_content = segment_content[last_qwen_start:next_separator]
            full_reply_content = re.sub(r'^\[[^\]]+\]\n', '', full_reply_content)
            # 提取仅最后的回复，移除对话历史
            processed_reply = extract_final_response_only(full_reply_content, extracted_model_name)
            extracted_reply = processed_reply.strip() if processed_reply and processed_reply.strip() else full_reply_content.strip()
        else:
            # 提取仅最后的回复，移除对话历史
            processed_reply = extract_final_response_only(raw_reply, extracted_model_name)
            extracted_reply = processed_reply.strip() if processed_reply and processed_reply.strip() else raw_reply.strip()
    else:
        # 尝试匹配Llama特殊标记
        llama_tag = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        llama_tag_backup = "<|eot|><|header_start|>assistant<|header_end|>"
        llama_pos = segment_content.rfind(llama_tag)
        llama_pos_backup = segment_content.rfind(llama_tag_backup)

        if llama_pos != -1:
            raw_reply = segment_content[llama_pos + len(llama_tag):].strip()
            # 提取仅最后的回复，移除对话历史
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
        elif llama_pos_backup != -1:
            raw_reply = segment_content[llama_pos_backup + len(llama_tag_backup):].strip()
            # 提取仅最后的回复，移除对话历史
            extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
        else:
            # 尝试匹配Gemma特殊标记
            gemma_tag_model = '<start_of_turn>model'
            gemma_tag_assistant = '<start_of_turn>assistant'
            gemma_tag_generic = '<start_of_turn>'
            gemma_tags = [gemma_tag_model, gemma_tag_assistant, gemma_tag_generic]

            for tag in gemma_tags:
                gemma_pos = segment_content.rfind(tag)
                if gemma_pos != -1:
                    raw_reply = segment_content[gemma_pos + len(tag):].strip()
                    # 提取仅最后的回复，移除对话历史
                    extracted_reply = extract_final_response_only(raw_reply, model_name_from_filename)
                    break

    # 清理回复内容
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
    
    # 扫描task1、task2、task3、task1_steered、task2_steered、task3_steered目录
    for task_dir in ["task1", "task2", "task3", "task1_steered", "task2_steered", "task3_steered"]:
        task_path = os.path.join(results_dir, task_dir)
        if not os.path.exists(task_path):
            print(f"目录 {task_path} 不存在，跳过")
            continue
            
        print(f"\n正在处理 {task_dir}...")
        print(f"扫描目录: {task_path}")
        
        # 获取所有txt文件
        txt_files = [f for f in os.listdir(task_path) if f.endswith('.txt')]
        print(f"找到 {len(txt_files)} 个txt文件")
        
        for i, filename in enumerate(txt_files):
            file_path = os.path.join(task_path, filename)
            print(f"\n处理文件 {i+1}/{len(txt_files)}: {filename}")
            
            # 检查文件是否为空
            if os.path.getsize(file_path) == 0:
                print(f"  跳过空文件: {filename}")
                continue
            
            # 从文件名提取Mode
            mode = "direct" if "mode_direct" in filename else "explain" if "mode_explain" in filename else "unknown"
            
            # 从文件名提取Role和COT
            role, cot = extract_role_and_cot_from_filename(filename)
            
            # 从文件名提取Temperature
            temperature = extract_temperature_from_filename(filename)
            
            # 对于task3和task3_steered，需要特殊处理双答案
            if task_dir == "task3" or task_dir == "task3_steered":
                # 重新读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # 从文件名提取模型名称
                model_match = re.search(r'model_(.+?)_mode_', filename)
                model_name_from_filename = model_match.group(1) if model_match else "Unknown"
                
                # 检查task3文件是否有足够的cycles
                cycle_pattern = r'Cycle (\d+)/\d+:'
                cycle_matches = list(re.finditer(cycle_pattern, file_content))
                if len(cycle_matches) < MIN_CYCLE_COUNT:
                    print(f"  跳过cycle数不足{MIN_CYCLE_COUNT}的文件: {filename} (cycle数: {len(cycle_matches)})")
                    continue
                
                # 重新处理task3文件，提取两个问题的回答
                for cycle_num in range(1, 101):  # task3有100个cycles
                    cycle_pattern = rf'Cycle {cycle_num}/\d+:'
                    cycle_match = re.search(cycle_pattern, file_content)
                    
                    if not cycle_match:
                        continue
                    
                    cycle_start = cycle_match.start()
                    # 查找下一个cycle，包括完整的和不完整的格式
                    next_cycle_pattern = rf'Cycle {cycle_num + 1}/\d+:'
                    next_cycle_match = re.search(next_cycle_pattern, file_content)
                    
                    # 同时查找不完整的cycle格式：====================后跟单独的"Cycle"行
                    incomplete_cycle_pattern = r'\n====================\nCycle\s*$'
                    incomplete_matches = list(re.finditer(incomplete_cycle_pattern, file_content, re.MULTILINE))
                    
                    # 找到在当前cycle之后的所有可能的终止位置
                    candidate_positions = []
                    if next_cycle_match:
                        candidate_positions.append(next_cycle_match.start())
                    
                    # 添加所有不完整cycle的位置（在当前cycle之后）
                    for incomplete_match in incomplete_matches:
                        if incomplete_match.start() > cycle_start:
                            # 找到====================的开始位置（不是Cycle行的位置）
                            # 需要在====================之前的行查找
                            dash_pos = file_content.rfind('====================', cycle_start, incomplete_match.start())
                            if dash_pos != -1:
                                candidate_positions.append(dash_pos)
                    
                    # 选择最接近当前cycle的终止位置
                    if candidate_positions:
                        cycle_end = min(candidate_positions)
                    else:
                        cycle_end = len(file_content)
                    
                    cycle_content = file_content[cycle_start:cycle_end].strip()
                    
                    # 提取两个问题的回答
                    (promotion_model, promotion_reply), (stock_model, stock_reply) = extract_task3_replies(cycle_content, model_name_from_filename)
                    
                    # 为每个回答创建一行记录
                    if promotion_reply is not None:
                        all_results.append({
                            'Task': f"{task_dir}_promotion" if task_dir == "task3_steered" else f"{task_dir}promotion",
                            'Filename': filename,
                            'Cycle': cycle_num,
                            'Model': promotion_model,
                            'ChatGPT最后回复': promotion_reply,
                            'majority_group': None,  # task3没有majority_group
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
                            'majority_group': None,  # task3没有majority_group
                            'Mode': mode,
                            'Role': role,
                            'COT': cot,
                            'Temperature': temperature
                        })
                
                # 跳过原有的单答案处理逻辑
                continue
            else:
                # 对于task1和task2，使用原有的单答案处理逻辑
                file_results = extract_last_reply_from_txt(file_path, task_dir)
                
                # 只保留cycle数大于等于100的文件
                if len(file_results) < MIN_CYCLE_COUNT:
                    print(f"  跳过cycle数不足{MIN_CYCLE_COUNT}的文件: {filename} (cycle数: {len(file_results)})")
                    continue
                
                # 添加结果到all_results
                for cycle, model_name, last_reply, majority_group in file_results:
                    # 为steered任务添加steered后缀以区别于常规任务
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
    
    # 创建输出目录
    output_dir = "./extracted_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 扫描并提取结果
    all_results = scan_and_extract_results()
    
    if not all_results:
        print("\n未找到任何有效结果！")
        return
    
    # 保存到Excel
    output_file = os.path.join(output_dir, "chatgpt_last_replies.xlsx")
    
    df = pd.DataFrame(all_results)

    # 在写入Excel之前清理非法字符，避免 openpyxl IllegalCharacterError
    # 移除所有非法的控制字符：\x00-\x08, \x0B-\x0C, \x0E-\x1F
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
    
    # 显示统计信息
    print("\n统计信息:")
    task_counts = df['Task'].value_counts()
    for task, count in task_counts.items():
        print(f"  {task}: {count} 个回复")
    
    model_counts = df['Model'].value_counts()
    print(f"\n模型分布:")
    for model, count in model_counts.head(10).items():  # 显示前10个模型
        print(f"  {model}: {count} 个回复")

if __name__ == "__main__":
    main()
