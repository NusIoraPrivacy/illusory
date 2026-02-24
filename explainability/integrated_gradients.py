import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import os
from torchvision import transforms as T
from matplotlib.colors import LinearSegmentedColormap

RANDOM_STATE = 42

def integrated_gradients(model, tokenizer_or_processor, input_data, out_prefix, baseline=None, target_label=1, n_steps=20, mode="text", device="cpu", per_token_ig=False, model_type=None, messages=None, model_path=None, batch_size=1, **kwargs):
    """
    通用集成梯度入口。mode支持'text'或'image'。
    - model: 已加载的transformers模型
    - tokenizer: transformers分词器
    - input_data: 文本(str)或图片tensor
    - out_prefix: 输出文件前缀（不带后缀），如results/explainability/task1/xxx
    - baseline: baseline输入（文本为全pad，图片为全零）
    - target_label: 目标类别（文本分类时用）
    - n_steps: IG步数
    - mode: 'text'或'image'
    - device: 运行设备
    - per_token_ig: 是否对每个user输入token逐个归因，极低显存消耗
    - model_type: 可选，指定模型类型（qwen/llama），否则自动判断
    - batch_size: 当per_token_ig=True时，每次处理的token数量，默认为1
    """
    # # 强制设置GPU设备，自动选择内存使用最少的GPU
    # if device == "cuda" and torch.cuda.is_available():
    #     # 检查CUDA_VISIBLE_DEVICES
    #     cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    #     if cuda_visible:
    #         # 找到内存使用最少的GPU（使用系统级检测）
    #         try:
    #             import pynvml
    #             pynvml.nvmlInit()
    #             min_memory_free = 0
    #             best_gpu = 0
                
    #             for i in range(torch.cuda.device_count()):
    #                 handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    #                 info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #                 memory_free = info.free / 1024**3  # GB
    #                 memory_used = info.used / 1024**3  # GB
    #                 print(f"[IG] GPU {i}: 已用 {memory_used:.2f}GB, 空闲 {memory_free:.2f}GB")
                    
    #                 if memory_free > min_memory_free:
    #                     min_memory_free = memory_free
    #                     best_gpu = i
                
    #             print(f"[IG] 选择空闲内存最多的GPU: cuda:{best_gpu} (空闲: {min_memory_free:.2f}GB)")
                
    #         except ImportError:
    #             # 如果没有pynvml，回退到PyTorch检测
    #             print("[IG] 警告: 未安装pynvml，使用PyTorch内存检测（可能不准确）")
    #             min_memory_used = float('inf')
    #             best_gpu = 0
                
    #             for i in range(torch.cuda.device_count()):
    #                 memory_used = torch.cuda.memory_allocated(i)
    #                 print(f"[IG] GPU {i}: PyTorch已用内存 {memory_used / 1024**3:.2f}GB")
    #                 if memory_used < min_memory_used:
    #                     min_memory_used = memory_used
    #                     best_gpu = i
            
    #         device = torch.device(f"cuda:{best_gpu}")
    #     else:
    #         device = torch.device("cuda")
    
    if RANDOM_STATE:
        random.seed(RANDOM_STATE)
    ######## 文本归因 ########
    if mode == "text":
        inputs = tokenizer_or_processor(input_data, return_tensors="pt")
        input_ids = inputs.input_ids
        if baseline is None:
            baseline_ids = input_ids * 0
        else:
            baseline_ids = baseline
        
        # 确保input_ids和模型在同一个设备上
        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
        baseline_ids = baseline_ids.to(model_device)
        
        if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
            embedding_layer = model.bert.embeddings.word_embeddings
            forward_func = model
            _target = target_label
        elif hasattr(model, "get_input_embeddings") and callable(model.get_input_embeddings):
            embedding_layer = model.get_input_embeddings()
            def forward_func(input_ids):
                outputs = model(input_ids=input_ids)
                return outputs.logits[:, -1, :]
            _target = input_ids[0, -1].item()
        else:
            raise ValueError("The model does not support automatically obtaining the embedding layer (it is neither BERT nor CausalLM).")
        tokens = tokenizer_or_processor.convert_ids_to_tokens(input_ids[0])

        # --------- 判断模型类型 ---------
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
            if not per_token_ig:
                lig = LayerIntegratedGradients(forward_func, embedding_layer)
                inputs_sub = input_ids[:, start_idx:end_idx]
                baseline_sub = baseline_ids[:, start_idx:end_idx]
                attributions, delta = lig.attribute(
                    inputs=inputs_sub,
                    baselines=baseline_sub,
                    target=_target,
                    return_convergence_delta=True,
                    n_steps=min(n_steps, 30)
                )
                token_attributions = attributions.sum(dim=-1).squeeze(0)
                filtered_attributions = [token_attributions[idx] for idx in filtered_indices]
            else:
                # 使用LayerIntegratedGradients，支持批量处理多个token
                batchsize = min(20, len(filtered_indices))  # 不超过段落大小
                filtered_attributions = []
                lig = LayerIntegratedGradients(forward_func, embedding_layer)
                
                # 将filtered_indices按batch_size分组
                for i in range(0, len(filtered_indices), batch_size):
                    batch_indices = filtered_indices[i:i+batch_size]
                    
                    # 计算当前batch的token范围
                    start_idx = batch_indices[0]
                    end_idx = batch_indices[-1] + 1
                    
                    # 取当前batch的tokens进行归因
                    inputs_sub = input_ids[:, start_idx:end_idx]
                    baseline_sub = baseline_ids[:, start_idx:end_idx]
                    
                    attributions, delta = lig.attribute(
                        inputs=inputs_sub,
                        baselines=baseline_sub,
                        target=_target,
                        return_convergence_delta=True,
                        n_steps=min(n_steps, 30)
                    )

                    # 提取当前batch中每个token的归因分数
                    token_attributions = attributions.sum(dim=-1).squeeze(0)
                    # 将batch结果映射回原始位置
                    for j, rel_idx in enumerate(batch_indices):
                        local_idx = rel_idx - start_idx  # 在当前batch中的位置
                        if local_idx < len(token_attributions):
                            filtered_attributions.append(token_attributions[local_idx])
                        else:
                            # 如果超出范围，使用0
                            filtered_attributions.append(torch.tensor(0.0))

            # --------- 对归因结果进行高亮处理 ---------
            attributions_np = np.array([float(attr.detach().cpu().numpy()) for attr in filtered_attributions])
            max_abs = float(np.max(np.abs(attributions_np))) if attributions_np.size > 0 else 1.0
            if max_abs < 1e-8:
                norm_scores = np.zeros_like(attributions_np)
            else:
                norm_scores = attributions_np / max_abs  
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
        return html, all_filtered_tokens, all_rank_scores

    ######## 图像归因 ########
    elif mode == "image":
        n_steps = 50
        print("[IG] 为Qwen2.5-VL模型使用视觉分支集成梯度归因方法")
        try:
            if isinstance(input_data, (Image.Image, np.ndarray, torch.Tensor)):
                # img = ensure_rgb_pil(input_data)
                img = input_data
                img = img.resize((448, 448))

                
                # 使用image_processor得到patch flatten的pixel_values和配套的image_grid_thw
                processed = tokenizer_or_processor.image_processor(img, return_tensors="pt")
                # 确保pixel_values和模型在同一个设备上
                model_device = next(model.parameters()).device
                pixel_values = processed.pixel_values.to(model_device)         # [num_patches, patch_dim]，如[1024, 1176]
                image_grid_thw = processed.image_grid_thw.to(model_device)     # [1, 3]

                print(f"[IG] 处理后的图像张量 shape: {pixel_values.shape}")
                print(f"[IG] image_grid_thw shape: {image_grid_thw.shape}")

                baseline = torch.zeros_like(pixel_values).to(model_device)

                def forward_vision_model(img_input):
                    # Captum可能会改变输入形状，我们需要动态处理
                    # 原始: [1024, 1176]，Captum可能变为: [5120, 1176] 等

                    # 确保输入在正确的设备上
                    img_input = img_input.to(model_device)
                    
                    # 动态计算batch size
                    if img_input.shape[1] == pixel_values.shape[1]: 
                        batch_size = img_input.shape[0] // pixel_values.shape[0]  
                        num_patches = pixel_values.shape[0]
                        
                        img_input_batch = img_input.reshape(batch_size, num_patches, -1)
                        
                        outputs = []
                        for i in range(batch_size):
                            if hasattr(model, 'vision_tower'):
                                vision_tower = model.vision_tower
                            elif hasattr(model, 'visual'):
                                vision_tower = model.visual
                            else:
                                raise AttributeError(f"模型 {type(model).__name__} 没有找到视觉组件")
                            vision_output = vision_tower(img_input_batch[i], image_grid_thw)
                            
                            if hasattr(vision_output, 'last_hidden_state'):
                                output = vision_output.last_hidden_state.mean()
                            elif isinstance(vision_output, torch.Tensor):
                                output = vision_output.mean()
                            else:
                                output = img_input_batch[i].mean()
                            outputs.append(output)
                        
                        result = torch.stack(outputs).mean()
                        return result.unsqueeze(0).to(model_device)  #
                print("[IG] 计算视觉分支集成梯度归因...")
                ig = IntegratedGradients(forward_vision_model)
                attributions, delta = ig.attribute(
                    inputs=pixel_values,
                    baselines=baseline,
                    return_convergence_delta=True,
                    n_steps=5
                )
                print(f"[IG] 收敛delta: {delta.mean().item() if isinstance(delta, torch.Tensor) and delta.numel() > 1 else delta.item() if isinstance(delta, torch.Tensor) else delta}")
                
                with torch.no_grad():
                    # 处理归因结果，attributions 形状是 [num_patches, patch_dim]
                    # 我们需要将patch归因重新映射到图像空间
                    attr_map = attributions.squeeze().abs().sum(dim=-1).cpu().numpy()  # [num_patches]
                    
                    # 将patch归因重新映射到图像空间
                    # Qwen2.5-VL使用32x32的patch网格，总共1024个patch
                    patch_size = 14  # Qwen2.5-VL的patch大小
                    grid_size = int(np.sqrt(attr_map.shape[0]))  # 应该是32
                    
                    if grid_size * grid_size == attr_map.shape[0]:
                        # 重塑为网格
                        attr_grid = attr_map.reshape(grid_size, grid_size)
                        
                        # 计算像素级别的归因分数
                        # 将每个patch的归因分数分配给该patch内的所有像素
                        pixel_attr_map = np.zeros((448, 448))
                        for i in range(grid_size):
                            for j in range(grid_size):
                                # 计算patch在图像中的位置
                                start_h = i * patch_size
                                end_h = min((i + 1) * patch_size, 448)
                                start_w = j * patch_size
                                end_w = min((j + 1) * patch_size, 448)
                                # 将该patch的归因分数分配给所有像素
                                pixel_attr_map[start_h:end_h, start_w:end_w] = attr_grid[i, j]
                        
                        # 使用torch的interpolate上采样到原始图像大小（保持原有逻辑）
                        attr_tensor = torch.from_numpy(attr_grid).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
                        target_size = (448, 448)  # 原始图像大小
                        attr_map_interpolated = torch.nn.functional.interpolate(
                            attr_tensor, 
                            size=target_size, 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze().numpy()
                    else:
                        # 如果patch数量不是完全平方数，回退到简单方法
                        print(f"[IG] 警告：patch数量 {attr_map.shape[0]} 不是完全平方数，使用简单reshape")
                        pixel_attr_map = attr_map.reshape(-1, 1)  # 临时reshape避免matplotlib错误
                        attr_map_interpolated = pixel_attr_map
                    
                    # 归一化
                    if pixel_attr_map.max() > pixel_attr_map.min():
                        pixel_attr_map = (pixel_attr_map - pixel_attr_map.min()) / (pixel_attr_map.max() - pixel_attr_map.min())
                    if attr_map_interpolated.max() > attr_map_interpolated.min():
                        attr_map_interpolated = (attr_map_interpolated - attr_map_interpolated.min()) / (attr_map_interpolated.max() - attr_map_interpolated.min())
                
                if out_prefix:
                    model_folder = kwargs.get('model_name', 'unknown_model')
                    method_name = "integrated_gradients"
                    fig_dir = os.path.join(os.path.dirname(out_prefix), 'figure', model_folder, method_name)
                    os.makedirs(fig_dir, exist_ok=True)
                    base_name = os.path.basename(out_prefix)
                    
                    # 定义黑白colormap
                    bw_cmap = LinearSegmentedColormap.from_list("bw", ["black", "white"])
                    
                    # 1. 拼合图：左原图，右黑白热力图（使用插值版本）
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    # 左边：原始图像
                    ax1.imshow(np.array(img))
                    ax1.set_title("Original Image", fontsize=16)
                    ax1.axis("off")
                    # 右边：只显示归因热力图
                    im2 = ax2.imshow(attr_map_interpolated, cmap=bw_cmap)
                    ax2.set_title("Attribution Heatmap (Interpolated)", fontsize=16)
                    ax2.axis("off")
                    # 颜色条
                    norm = plt.Normalize(attr_map_interpolated.min(), attr_map_interpolated.max())
                    sm = plt.cm.ScalarMappable(cmap=bw_cmap, norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
                    cbar.set_label("Attribution Score", fontsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, f"{base_name}_comparison.png"), dpi=150, bbox_inches='tight')
                    plt.close()

                    # 2. 叠加图：原图上叠加彩色热力图（使用插值版本）
                    fig_overlay, ax_overlay = plt.subplots(figsize=(10, 10))
                    ax_overlay.imshow(np.array(img))
                    im_overlay = ax_overlay.imshow(attr_map_interpolated, cmap="hot", alpha=0.5)
                    ax_overlay.set_title("Visual Branch Overlay Attribution")
                    ax_overlay.axis("off")
                    
                    # 为叠加图添加颜色条
                    cbar_overlay = fig_overlay.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
                    cbar_overlay.set_label("Attribution Score", fontsize=12)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, f"{base_name}_overlay.png"), dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # 3. 像素级归因图（新增）
                    fig_pixel, ax_pixel = plt.subplots(figsize=(10, 10))
                    im_pixel = ax_pixel.imshow(pixel_attr_map, cmap="viridis")
                    ax_pixel.set_title("Pixel-level Attribution Map", fontsize=16)
                    ax_pixel.axis("off")
                    
                    cbar_pixel = fig_pixel.colorbar(im_pixel, ax=ax_pixel, fraction=0.046, pad=0.04)
                    cbar_pixel.set_label("Pixel Attribution Score", fontsize=12)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, f"{base_name}_pixel_attribution.png"), dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # 保存像素级归因分数到文件
                    pixel_scores_file = os.path.join(fig_dir, f"{base_name}_pixel_scores.txt")
                    with open(pixel_scores_file, 'w') as f:
                        f.write("Pixel-level attribution scores (448x448)\n")
                        f.write("Format: row,column,score\n")
                        for i in range(448):
                            for j in range(448):
                                f.write(f"{i},{j},{pixel_attr_map[i,j]:.6f}\n")
                    
                return pixel_attr_map, img, attributions
            else:
                raise ValueError(f"input_data类型不支持: {type(input_data)}")
        except Exception as e:
            print(f"[IG] 计算视觉分支集成梯度时出错: {str(e)}")
            print(f"[IG] 错误类型: {type(e).__name__}")
            print("[IG] 错误追踪:")
            import traceback
            traceback.print_exc()
            raise e

# def ensure_rgb_pil(img):
#     if isinstance(img, torch.Tensor):
#         img = img.detach().cpu().numpy()
#     if isinstance(img, np.ndarray):
#         img = Image.fromarray(img)
#     if not isinstance(img, Image.Image):
#         raise ValueError(f"输入图片类型无法转为PIL.Image: {type(img)}")
#     print(f"[IG] 输入图片类型: {type(img)}, mode: {img.mode}, size: {img.size}")
#     if img.mode != 'RGB':
#         print(f"[IG] 自动将图片从{img.mode}转为RGB")
#         img = img.convert('RGB')
#     return img
