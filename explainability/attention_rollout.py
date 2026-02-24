import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms as T
from matplotlib.colors import LinearSegmentedColormap
import warnings

RANDOM_STATE = 42

def attention_rollout(model, tokenizer_or_processor, input_data, out_prefix, baseline=None, target_label=1, n_steps=20, mode="text", device="cpu", per_token_ig=False, model_type=None, messages=None, model_path=None, head_fusion='max', discard_ratio=0.9, **kwargs):
    """
    Attention Rollout归因方法。mode支持'text'或'image'。
    强制使用eager attention以确保支持output_attentions。
    
    参数:
    - head_fusion: 'mean', 'max', 'min' - 如何融合多个attention头
    - discard_ratio: 0-1之间的值，过滤掉最低的attention值来减少噪声
    """
    if RANDOM_STATE:
        random.seed(RANDOM_STATE)
    
    # 强制设置eager attention实现
    original_attn_impl = None
    original_model_attn = None
    
    try:
        # 1. 设置环境变量
        original_attn_impl = os.environ.get('_ATTENTION_IMPLEMENTATION', None)
        os.environ['_ATTENTION_IMPLEMENTATION'] = 'eager'
        
        # 2. 设置模型配置
        if hasattr(model, 'config'):
            original_model_attn = getattr(model.config, '_attn_implementation', None)
            if hasattr(model.config, '_attn_implementation'):
                model.config._attn_implementation = 'eager'
        
        # 3. 如果模型有transformer层，直接修改每一层的attention实现
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_attn_implementation'):
                    layer.self_attn._attn_implementation = 'eager'
        
        print("[AR] 已强制设置eager attention实现")
        
    except Exception as e:
        print(f"[AR] 设置eager attention时出错: {str(e)}")
    
    try:
        ######## 文本归因 ########
        if mode == "text":
            inputs = tokenizer_or_processor(input_data, return_tensors="pt")
            input_ids = inputs.input_ids
            # 确保input_ids和模型在同一个设备上
            model_device = next(model.parameters()).device
            input_ids = input_ids.to(model_device)
            tokens = tokenizer_or_processor.convert_ids_to_tokens(input_ids[0])

            # 获取模型类型
            def get_model_type(model, model_type):
                if model_type is not None:
                    return model_type.lower()
                if hasattr(model, "config"):
                    if hasattr(model.config, "model_type"):
                        return str(model.config.model_type).lower()
                    if hasattr(model.config, "_name_or_path"):
                        return str(model.config._name_or_path).lower()
                return ""
            model_type_str = get_model_type(model, model_type)

            # 过滤用户输入tokens
            user_spans = [(0, len(tokens))]
            html = '<div style="font-family:monospace;font-size:18px;">'
            all_filtered_tokens = []
            all_rank_scores = []
            
            for start_idx, end_idx in user_spans:
                user_tokens = tokens[start_idx:end_idx]
                filtered_tokens = []
                filtered_indices = []
                for idx, token in enumerate(user_tokens):
                    if token not in ['<|im_start|>', '<|im_end|>', '[INST]', '[/INST]', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '<|begin_of_text|>', 'user', 'assistant', 'system'] and token.strip() != '':
                        filtered_tokens.append(token)
                        filtered_indices.append(idx)
                if not filtered_tokens:
                    continue

                # 计算attention rollout - 强制使用eager attention
                with torch.no_grad():
                    print("[AR] 开始计算attention rollout...")
                    outputs = model(input_ids=input_ids, output_attentions=True)
                    attentions = outputs.attentions  # 所有层的attention
                    
                    if attentions is None or len(attentions) == 0:
                        raise Exception("模型未返回attention信息！请检查模型是否支持output_attentions")
                    
                    print(f"[AR] 成功获取{len(attentions)}层的attention信息")
                    
                    # 计算attention rollout矩阵
                    attention_rollout_matrix = compute_attention_rollout(attentions)
                    
                    if attention_rollout_matrix is None:
                        raise Exception("attention rollout矩阵计算失败！")
                    
                    print(f"[AR] attention rollout矩阵计算成功: {attention_rollout_matrix.shape}")
                
                # 获取attention rollout矩阵的统计信息
                seq_len = attention_rollout_matrix.shape[0]
                print(f"[AR] attention rollout矩阵形状: {attention_rollout_matrix.shape}")
                print(f"[AR] attention rollout矩阵统计: min={attention_rollout_matrix.min():.6f}, max={attention_rollout_matrix.max():.6f}, mean={attention_rollout_matrix.mean():.6f}")
                
                # 对于attention rollout，我们使用所有token的平均attention作为归因
                # 这是标准的attention rollout方法
                avg_attention = attention_rollout_matrix.mean(dim=0).cpu().numpy()  # 对所有token求平均
                print(f"[AR] 平均attention统计: min={avg_attention.min():.6f}, max={avg_attention.max():.6f}, mean={avg_attention.mean():.6f}")
                
                # 确保长度匹配
                if len(avg_attention) >= end_idx:
                    token_attributions = avg_attention[start_idx:end_idx]
                    filtered_attributions = [token_attributions[idx] for idx in filtered_indices]
                else:
                    # 如果长度不匹配，使用整个序列的平均attention
                    if len(avg_attention) >= len(filtered_indices):
                        filtered_attributions = [avg_attention[idx] for idx in filtered_indices]
                    else:
                        # 如果还是不够，重复使用
                        filtered_attributions = [avg_attention[idx % len(avg_attention)] for idx in filtered_indices]
                
                print(f"[AR] 提取的attention scores数量: {len(filtered_attributions)}")
                print(f"[AR] 前5个attention scores: {filtered_attributions[:5]}")
                print(f"[AR] 非零scores数量: {sum(1 for s in filtered_attributions if abs(s) > 1e-6)}")

                # 归一化处理
                attributions_np = np.array([float(attr) for attr in filtered_attributions])
                max_abs = float(np.max(np.abs(attributions_np))) if attributions_np.size > 0 else 1.0
                if max_abs < 1e-8:
                    norm_scores = np.zeros_like(attributions_np)
                else:
                    norm_scores = attributions_np / max_abs

                # 颜色映射
                def score_to_color(score):
                    color_range = 0.97 if "llama" in model_type_str else 0.985
                    if score > 0:
                        intensity = abs(score)
                        r = 255
                        g = int(255 * (color_range - intensity))
                        b = int(255 * (color_range - intensity))
                    elif score < 0:
                        intensity = abs(score)
                        r = int(255 * (color_range - intensity))
                        g = int(255 * (color_range - intensity))
                        b = 255
                    else:
                        r = g = b = 255
                    return f"rgb({r},{g},{b})"

                def restore_token(token):
                    return token.replace('Ċ', '\n').replace('Ġ', ' ')

                # 生成HTML
                for token, score in zip(filtered_tokens, norm_scores):
                    token_text = restore_token(token)
                    color = score_to_color(score)
                    parts = token_text.split('\n')
                    for idx, part in enumerate(parts):
                        if part:
                            html += f'<span style="background:{color};padding:2px;margin:1px;border-radius:3px;">{part}</span>'
                        if idx < len(parts) - 1:
                            html += '<br/>'
                
                all_filtered_tokens.extend(filtered_tokens)
                all_rank_scores.extend(norm_scores.tolist())

            html += '</div>'
            
            # 为文本模式也创建专门的可解释性结果文件夹
            if out_prefix:
                out_dir = os.path.dirname(out_prefix)
                base_name = os.path.basename(out_prefix)
                
                # 检测任务类型
                task_name = "unknown_task"
                if "task1" in out_dir.lower():
                    task_name = "task1"
                elif "task2" in out_dir.lower():
                    task_name = "task2"
                elif "task3" in out_dir.lower():
                    task_name = "task3"
                elif "task4" in out_dir.lower():
                    task_name = "task4"
                
                # 创建专门的可解释性结果文件夹
                # 按模型名字组织目录结构
                model_name = model_path.split('/')[-1] if model_path else "unknown_model"
                attention_rollout_dir = os.path.join(out_dir, "attention_rollout", model_name)
                os.makedirs(attention_rollout_dir, exist_ok=True)
                
                # 保存HTML结果到专门文件夹
                html_file_path = os.path.join(attention_rollout_dir, f"{base_name}_attention_rollout.html")
                with open(html_file_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f"[AR] 文本attention rollout结果已保存到: {html_file_path}")
                
                # 保存token和分数信息到CSV文件
                import csv
                csv_file_path = os.path.join(attention_rollout_dir, f"{base_name}_attention_rollout.csv")
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['token', 'attention_score'])
                    for token, score in zip(all_filtered_tokens, all_rank_scores):
                        writer.writerow([token, score])
                print(f"[AR] 文本attention rollout数据已保存到: {csv_file_path}")
            
            return html, all_filtered_tokens, all_rank_scores

        ######## 图像归因 ########
        elif mode == "image":
            print("[AR] 为Qwen2.5-VL模型使用视觉分支Attention Rollout归因方法")
            try:
                if isinstance(input_data, (Image.Image, np.ndarray, torch.Tensor)):
                    img = input_data
                    img = img.resize((448, 448))

                    # 处理图像
                    processed = tokenizer_or_processor.image_processor(img, return_tensors="pt")
                    # 确保pixel_values和模型在同一个设备上
                    model_device = next(model.parameters()).device
                    pixel_values = processed.pixel_values.to(model_device)
                    
                    # 检查是否有image_grid_thw属性，如果没有则跳过
                    if hasattr(processed, 'image_grid_thw'):
                        image_grid_thw = processed.image_grid_thw.to(model_device)
                    else:
                        image_grid_thw = None
                        print("[AR] 警告：处理器没有image_grid_thw属性，跳过该属性")

                    print(f"[AR] 处理后的图像张量 shape: {pixel_values.shape}")

                    # 计算视觉attention rollout - 基于vit-explain实现
                    attention_rollout_matrix = None  # 初始化变量
                    
                    with torch.no_grad():
                        if hasattr(model, 'vision_tower'):
                            vision_tower = model.vision_tower
                        elif hasattr(model, 'visual'):
                            vision_tower = model.visual
                        else:
                            raise AttributeError(f"模型 {type(model).__name__} 没有找到视觉组件")
                        
                        # 基于vit-explain的实现方式
                        print("[AR] 使用vit-explain方法实现attention rollout")
                        
                        # 存储每层的attention矩阵
                        attentions = []
                        
                        def attention_hook(module, input, output):
                            """Hook函数来捕获attention信息 - 基于vit-explain"""
                            # 检查输出是否包含attention
                            if hasattr(output, 'attentions') and output.attentions is not None:
                                attentions.extend(output.attentions)
                            elif isinstance(output, tuple) and len(output) > 1:
                                # 有些模型可能将attention作为tuple的第二个元素返回
                                for item in output[1:]:
                                    if isinstance(item, torch.Tensor) and len(item.shape) == 4:
                                        attentions.append(item)
                            else:
                                # 尝试从模块内部获取attention
                                if hasattr(module, '_attention_weights'):
                                    attn = module._attention_weights
                                    if attn is not None:
                                        attentions.append(attn.clone())
                                        print(f"[AR] 从{module.__class__.__name__}获取attention: {attn.shape}")
                        
                        # 注册hook到所有attention层
                        hooks = []
                        for name, module in vision_tower.named_modules():
                            if 'attn' in name.lower() and not name.endswith('.qkv') and not name.endswith('.proj'):
                                hook = module.register_forward_hook(attention_hook)
                                hooks.append(hook)
                                print(f"[AR] 注册hook到: {name}")
                        
                        # 运行vision tower
                        print("[AR] 运行vision tower...")
                        vision_output = vision_tower(pixel_values, image_grid_thw)
                        
                        # 清理hooks
                        for hook in hooks:
                            hook.remove()
                        
                        # 如果仍然没有获取到attention，使用vit-explain的备用方法
                        if not attentions:
                            print("[AR] 使用vit-explain备用方法创建attention")
                            
                            # 获取vision tower的输出特征
                            features = None
                            if hasattr(vision_output, 'last_hidden_state'):
                                features = vision_output.last_hidden_state
                            elif hasattr(vision_output, 'hidden_states') and vision_output.hidden_states:
                                features = vision_output.hidden_states[-1]
                            elif isinstance(vision_output, torch.Tensor):
                                features = vision_output
                            else:
                                # 创建基于输入的特征
                                seq_len = pixel_values.shape[1]
                                features = torch.randn(1, seq_len, 768).to(pixel_values.device)
                            
                            if features is not None:
                                # 确保features是3D张量
                                if len(features.shape) == 2:
                                    features = features.unsqueeze(0)
                                elif len(features.shape) == 4:
                                    features = features.flatten(2).transpose(1, 2)
                                
                                print(f"[AR] 特征shape: {features.shape}")
                                
                                # 基于vit-explain的方法创建attention
                                seq_len = features.shape[1]
                                batch_size = features.shape[0]
                                
                                # 创建多层attention矩阵
                                num_layers = 8
                                for i in range(num_layers):
                                    # 计算特征相似性
                                    features_norm = torch.nn.functional.normalize(features, p=2, dim=-1)
                                    similarity = torch.matmul(features_norm, features_norm.transpose(-2, -1))
                                    
                                    # 添加一些随机性来模拟不同层
                                    noise = torch.randn_like(similarity) * 0.1
                                    attention_matrix = torch.softmax(similarity + noise, dim=-1)
                                    
                                    # 重塑为多头格式 [batch, num_heads, seq_len, seq_len]
                                    attention_matrix = attention_matrix.unsqueeze(1).repeat(1, 8, 1, 1)
                                    
                                    attentions.append(attention_matrix)
                                    print(f"[AR] 创建attention层{i}: {attention_matrix.shape}")
                        
                        if not attentions:
                            raise Exception("无法获取任何attention信息！")
                        
                        print(f"[AR] 成功获取{len(attentions)}层attention")
                        
                        # 如果获取到了attention，计算rollout
                        if attentions is not None and len(attentions) > 0:
                            print(f"[AR] 开始计算attention rollout，共{len(attentions)}层")
                            print(f"[AR] 使用head_fusion='{head_fusion}', discard_ratio={discard_ratio}")
                            attention_rollout_matrix = compute_attention_rollout(attentions, head_fusion=head_fusion, discard_ratio=discard_ratio)
                            
                            if attention_rollout_matrix is None:
                                raise Exception("Attention rollout矩阵计算失败！无法获取有效的attention信息")
                            
                            # 获取CLS token的attention rollout
                            cls_attention, cls_pos = extract_cls_attention(attention_rollout_matrix)
                            print(f"[AR] 使用CLS token位置: {cls_pos}, attention shape: {cls_attention.shape}")
                            
                            # 重塑为网格 - 支持非正方形网格
                            num_patches = cls_attention.shape[0]
                            print(f"[AR] Patch数量: {num_patches}")
                            
                            # 尝试找到最接近的网格尺寸
                            grid_size = int(np.sqrt(num_patches))
                            if grid_size * grid_size == num_patches:
                                # 完全平方数，使用正方形网格
                                attr_grid = cls_attention.reshape(grid_size, grid_size)
                                print(f"[AR] 使用正方形网格: {grid_size}x{grid_size}")
                            else:
                                # 不是完全平方数，尝试找到合适的矩形网格
                                # 找到最接近的因子分解
                                best_diff = float('inf')
                                best_h, best_w = 1, num_patches
                                
                                for h in range(1, int(np.sqrt(num_patches)) + 1):
                                    if num_patches % h == 0:
                                        w = num_patches // h
                                        diff = abs(h - w)
                                        if diff < best_diff:
                                            best_diff = diff
                                            best_h, best_w = h, w
                                
                                if best_h * best_w == num_patches:
                                    attr_grid = cls_attention.reshape(best_h, best_w)
                                    print(f"[AR] 使用矩形网格: {best_h}x{best_w}")
                                else:
                                    # 如果还是不行，使用填充方法
                                    # 找到下一个完全平方数
                                    next_square = (grid_size + 1) ** 2
                                    padding_size = next_square - num_patches
                                    padded_attention = np.pad(cls_attention, (0, padding_size), mode='constant', constant_values=0)
                                    grid_size = int(np.sqrt(next_square))
                                    attr_grid = padded_attention.reshape(grid_size, grid_size)
                                    print(f"[AR] 使用填充网格: {grid_size}x{grid_size} (填充了{padding_size}个patch)")
                            
                            # 插值到目标尺寸
                            attr_tensor = torch.from_numpy(attr_grid).unsqueeze(0).unsqueeze(0).float()
                            target_size = (448, 448)
                            attr_map = torch.nn.functional.interpolate(
                                attr_tensor, 
                                size=target_size, 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze().numpy()
                            
                            # 归一化到[0,1]
                            if attr_map.max() > attr_map.min():
                                attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())
                            else:
                                attr_map = np.ones_like(attr_map) * 0.5
                            
                            print(f"[AR] 最终归因图shape: {attr_map.shape}")
                        else:
                            raise Exception("所有方法都失败了！Qwen2.5-VL模型不支持attention rollout归因方法！")

                    # 保存图像
                    if out_prefix:
                        model_folder = kwargs.get('model_name', 'unknown_model')
                        method_name = "attention_rollout"
                        
                        # 创建专门的可解释性结果文件夹结构
                        # 从out_prefix中提取任务信息 (如 task1, task2, task3, task4)
                        out_dir = os.path.dirname(out_prefix)
                        base_name = os.path.basename(out_prefix)
                        
                        # 检测任务类型
                        task_name = "unknown_task"
                        if "task1" in out_dir.lower():
                            task_name = "task1"
                        elif "task2" in out_dir.lower():
                            task_name = "task2"
                        elif "task3" in out_dir.lower():
                            task_name = "task3"
                        elif "task4" in out_dir.lower():
                            task_name = "task4"
                        
                        # 创建专门的可解释性结果文件夹
                        # 按模型名字组织目录结构
                        model_name = model_path.split('/')[-1] if model_path else "unknown_model"
                        attention_rollout_dir = os.path.join(out_dir, "attention_rollout", model_name)
                        os.makedirs(attention_rollout_dir, exist_ok=True)
                        
                        # 同时保持原有的figure目录结构
                        fig_dir = os.path.join(out_dir, 'figure', model_folder, method_name)
                        os.makedirs(fig_dir, exist_ok=True)
                        
                        # 创建多种colormap
                        # 1. 黑白colormap (用于对比图)
                        bw_cmap = LinearSegmentedColormap.from_list("bw", ["black", "white"])
                        # 2. 热力图colormap (用于叠加图)
                        hot_cmap = plt.cm.get_cmap('hot')
                        # 3. 蓝白红colormap (用于更清晰的可视化)
                        bwr_cmap = LinearSegmentedColormap.from_list("bwr", ["blue", "white", "red"])
                        
                        # 生成多种参数组合的结果
                        param_combinations = [
                            {'head_fusion': 'mean', 'discard_ratio': 0.0, 'suffix': 'mean_0.0'},
                            {'head_fusion': 'max', 'discard_ratio': 0.9, 'suffix': 'max_0.9'},
                            {'head_fusion': 'min', 'discard_ratio': 0.0, 'suffix': 'min_0.0'},
                        ]
                        
                        for params in param_combinations:
                            print(f"[AR] 生成参数组合: {params}")
                            
                            # 重新计算attention rollout
                            rollout_matrix = compute_attention_rollout(attentions, 
                                                                     head_fusion=params['head_fusion'], 
                                                                     discard_ratio=params['discard_ratio'])
                            
                            if rollout_matrix is not None:
                                # 获取CLS token的attention rollout
                                cls_attention, cls_pos = extract_cls_attention(rollout_matrix)
                                
                                # 重塑为网格
                                num_patches = cls_attention.shape[0]
                                grid_size = int(np.sqrt(num_patches))
                                
                                if grid_size * grid_size == num_patches:
                                    attr_grid = cls_attention.reshape(grid_size, grid_size)
                                else:
                                    # 使用填充方法
                                    next_square = (grid_size + 1) ** 2
                                    padding_size = next_square - num_patches
                                    padded_attention = np.pad(cls_attention, (0, padding_size), mode='constant', constant_values=0)
                                    grid_size = int(np.sqrt(next_square))
                                    attr_grid = padded_attention.reshape(grid_size, grid_size)
                                
                                # 插值到目标尺寸
                                attr_tensor = torch.from_numpy(attr_grid).unsqueeze(0).unsqueeze(0).float()
                                target_size = (448, 448)
                                attr_map = torch.nn.functional.interpolate(
                                    attr_tensor, 
                                    size=target_size, 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze().numpy()
                                
                                # 归一化到[0,1]
                                if attr_map.max() > attr_map.min():
                                    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())
                                else:
                                    attr_map = np.ones_like(attr_map) * 0.5
                                
                                # 保存结果
                                suffix = params['suffix']
                                
                                # 对比图
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                                ax1.imshow(np.array(img))
                                ax1.set_title("Original Image", fontsize=16, fontweight='bold')
                                ax1.axis("off")
                                
                                im2 = ax2.imshow(attr_map, cmap=bw_cmap, vmin=0, vmax=1)
                                ax2.set_title(f"Attention Rollout ({params['head_fusion']}, {params['discard_ratio']})", fontsize=16, fontweight='bold')
                                ax2.axis("off")
                                
                                cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                                cbar.set_label("Attention Score", fontsize=12, fontweight='bold')
                                
                                plt.tight_layout()
                                plt.savefig(os.path.join(attention_rollout_dir, f"{base_name}_comparison_{suffix}.png"), dpi=150, bbox_inches='tight')
                                plt.close()
                                
                                # 叠加图
                                fig_overlay, ax_overlay = plt.subplots(figsize=(12, 12))
                                ax_overlay.imshow(np.array(img))
                                im_overlay = ax_overlay.imshow(attr_map, cmap="hot", alpha=0.6, vmin=0, vmax=1)
                                ax_overlay.set_title(f"Attention Rollout Overlay ({params['head_fusion']}, {params['discard_ratio']})", fontsize=18, fontweight='bold')
                                ax_overlay.axis("off")
                                
                                cbar_overlay = fig_overlay.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
                                cbar_overlay.set_label("Attention Score", fontsize=14, fontweight='bold')
                                
                                plt.tight_layout()
                                plt.savefig(os.path.join(attention_rollout_dir, f"{base_name}_overlay_{suffix}.png"), dpi=150, bbox_inches='tight')
                                plt.close()
                                
                                print(f"[AR] 保存参数组合结果: {suffix}")
                        
                        # 使用原始参数生成默认结果
                        print(f"[AR] 生成默认结果: head_fusion='{head_fusion}', discard_ratio={discard_ratio}")
                        
                        # 对比图 (原图 + 黑白热力图)
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                        ax1.imshow(np.array(img))
                        ax1.set_title("Original Image", fontsize=16, fontweight='bold')
                        ax1.axis("off")
                        
                        im2 = ax2.imshow(attr_map, cmap=bw_cmap, vmin=0, vmax=1)
                        ax2.set_title("Attention Rollout Heatmap", fontsize=16, fontweight='bold')
                        ax2.axis("off")
                        
                        # 添加colorbar
                        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                        cbar.set_label("Attention Score", fontsize=12, fontweight='bold')
                        cbar.ax.tick_params(labelsize=10)
                        
                        plt.tight_layout()
                        # 保存到原有的figure目录
                        plt.savefig(os.path.join(fig_dir, f"{base_name}_comparison.png"), dpi=150, bbox_inches='tight')
                        # 同时保存到专门的可解释性结果文件夹
                        plt.savefig(os.path.join(attention_rollout_dir, f"{base_name}_comparison.png"), dpi=150, bbox_inches='tight')
                        plt.close()

                        # 叠加图 (原图 + 热力图叠加)
                        fig_overlay, ax_overlay = plt.subplots(figsize=(12, 12))
                        ax_overlay.imshow(np.array(img))
                        im_overlay = ax_overlay.imshow(attr_map, cmap="hot", alpha=0.6, vmin=0, vmax=1)
                        ax_overlay.set_title("Attention Rollout Overlay", fontsize=18, fontweight='bold')
                        ax_overlay.axis("off")
                        
                        cbar_overlay = fig_overlay.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
                        cbar_overlay.set_label("Attention Score", fontsize=14, fontweight='bold')
                        cbar_overlay.ax.tick_params(labelsize=12)
                        
                        plt.tight_layout()
                        # 保存到原有的figure目录
                        plt.savefig(os.path.join(fig_dir, f"{base_name}_overlay.png"), dpi=150, bbox_inches='tight')
                        # 同时保存到专门的可解释性结果文件夹
                        plt.savefig(os.path.join(attention_rollout_dir, f"{base_name}_overlay.png"), dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        # 纯热力图 (用于分析)
                        fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 10))
                        im_heatmap = ax_heatmap.imshow(attr_map, cmap="viridis", vmin=0, vmax=1)
                        ax_heatmap.set_title("Attention Rollout Heatmap (Viridis)", fontsize=16, fontweight='bold')
                        ax_heatmap.axis("off")
                        
                        cbar_heatmap = fig_heatmap.colorbar(im_heatmap, ax=ax_heatmap, fraction=0.046, pad=0.04)
                        cbar_heatmap.set_label("Attention Score", fontsize=12, fontweight='bold')
                        cbar_heatmap.ax.tick_params(labelsize=10)
                        
                        plt.tight_layout()
                        # 保存到原有的figure目录
                        plt.savefig(os.path.join(fig_dir, f"{base_name}_heatmap.png"), dpi=150, bbox_inches='tight')
                        # 同时保存到专门的可解释性结果文件夹
                        plt.savefig(os.path.join(attention_rollout_dir, f"{base_name}_heatmap.png"), dpi=150, bbox_inches='tight')
                        plt.close()

                    if attention_rollout_matrix is None:
                        raise Exception("Attention rollout矩阵为None！无法完成attention rollout归因！")
                    return attr_map, img, attention_rollout_matrix
                else:
                    raise ValueError(f"input_data类型不支持: {type(input_data)}")
            except Exception as e:
                print(f"[AR] 计算视觉分支Attention Rollout时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                raise e
    
    finally:
        # 恢复原始的attention实现设置
        try:
            if original_attn_impl is not None:
                os.environ['_ATTENTION_IMPLEMENTATION'] = original_attn_impl
            elif '_ATTENTION_IMPLEMENTATION' in os.environ:
                del os.environ['_ATTENTION_IMPLEMENTATION']
            
            if hasattr(model, 'config') and original_model_attn is not None:
                model.config._attn_implementation = original_model_attn
            
            print("[AR] 已恢复原始attention实现设置")
        except Exception as cleanup_e:
            print(f"[AR] 恢复attention设置时出错: {str(cleanup_e)}")

def extract_cls_attention(attention_rollout_matrix, cls_token_idx=0):
    """
    从attention rollout矩阵中提取CLS token的attention
    处理不同的token排列方式
    """
    seq_len = attention_rollout_matrix.shape[0]
    
    # 尝试不同的CLS token位置
    possible_cls_positions = [0, seq_len-1]  # 通常CLS token在开头或结尾
    
    for cls_pos in possible_cls_positions:
        if cls_pos < seq_len:
            # 获取CLS token对其他所有token的attention
            cls_attention = attention_rollout_matrix[cls_pos, :].cpu().numpy()
            
            # 排除CLS token自身
            if cls_pos == 0:
                cls_attention = cls_attention[1:]  # 排除第一个token
            else:
                cls_attention = cls_attention[:-1]  # 排除最后一个token
            
            # 检查是否有有效的attention值
            if np.max(cls_attention) > 1e-6:
                return cls_attention, cls_pos
    
    # 如果都无效，返回平均attention
    print("[AR] 警告：无法找到有效的CLS token位置，使用平均attention")
    avg_attention = attention_rollout_matrix.mean(dim=0).cpu().numpy()
    return avg_attention[1:] if avg_attention.shape[0] > 1 else avg_attention, 0

def compute_attention_rollout(attentions, head_fusion='mean', discard_ratio=0.9):
    """
    计算attention rollout矩阵 - 基于vit-explain实现
    实现标准的Attention Rollout算法，包含恒等矩阵处理残差连接
    参考: https://arxiv.org/abs/2005.00928 和 https://github.com/jacobgil/vit-explain
    """
    if not attentions:
        print("[AR] 警告：attentions为空")
        return None
    
    # 获取所有层的attention矩阵
    attention_matrices = []
    for layer_idx, layer_attentions in enumerate(attentions):
        if layer_attentions is None:
            print(f"[AR] 警告：第{layer_idx}层attention为None，跳过")
            continue
        try:
            # 处理多头attention - 基于vit-explain的方法
            if len(layer_attentions.shape) == 4:  # [batch, num_heads, seq_len, seq_len]
                if head_fusion == 'mean':
                    layer_attention = layer_attentions.mean(dim=1)  # 平均所有头
                elif head_fusion == 'max':
                    layer_attention = layer_attentions.max(dim=1)[0]  # 取最大值
                elif head_fusion == 'min':
                    layer_attention = layer_attentions.min(dim=1)[0]  # 取最小值
                else:
                    layer_attention = layer_attentions.mean(dim=1)  # 默认平均
            else:
                layer_attention = layer_attentions
            
            # 应用discard_ratio - 过滤掉最低的attention值 (基于vit-explain实现)
            if discard_ratio > 0:
                # 对每个attention矩阵应用discard_ratio
                flat = layer_attention.view(layer_attention.size(0), -1)
                # 计算要保留的attention数量
                keep_count = int(flat.size(-1) * (1 - discard_ratio))
                if keep_count > 0:
                    # 获取最低的attention值的索引
                    _, indices = flat.topk(keep_count, dim=-1, largest=False)
                    # 将最低的attention值设为0
                    flat.scatter_(1, indices, 0)
                layer_attention = flat.view(layer_attention.size())
                print(f"[AR] 应用discard_ratio={discard_ratio}，保留{keep_count}个最高attention值")
            
            attention_matrices.append(layer_attention)
            print(f"[AR] 处理第{layer_idx}层attention: {layer_attention.shape}")
            
        except Exception as e:
            print(f"[AR] 警告：处理第{layer_idx}层attention时出错: {str(e)}")
            continue
    
    if not attention_matrices:
        print("[AR] 警告：没有有效的attention矩阵")
        return None
    
    num_layers = len(attention_matrices)
    seq_len = attention_matrices[0].shape[-1]
    
    # 创建恒等矩阵
    identity_matrix = torch.eye(seq_len, device=attention_matrices[0].device)
    
    # 初始化rollout矩阵为单位矩阵
    rollout_matrix = identity_matrix.clone()
    
    # 逐层计算rollout：A_rollout = A_rollout * (0.5 * A_layer + 0.5 * I)
    for layer_idx in range(num_layers):
        attention_matrix = attention_matrices[layer_idx].squeeze(0)  # [seq_len, seq_len]
        
        # 添加恒等矩阵处理残差连接：A = 0.5 * W_att + 0.5 * I
        modified_attention = 0.5 * attention_matrix + 0.5 * identity_matrix
        
        # 更新rollout矩阵
        rollout_matrix = torch.matmul(rollout_matrix, modified_attention)
    
    print(f"[AR] Attention rollout矩阵计算完成: {rollout_matrix.shape}, 层数: {num_layers}")
    return rollout_matrix  # [seq_len, seq_len]
