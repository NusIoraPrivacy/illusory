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
    """Generic integrated gradient entry. Mode supports' text 'or' image '.
    - model: loaded transformers model
    - tokenizer: transformers
    - input_data: text (str) or picture tensor
    - out_prefix: output file prefix (without suffix), e.g. results/explainability/task1/xxx
    - baseline: baseline input (text is full pad, image is full zero)
    - target_label: target category (for text categorization)
    - n_steps: IG steps
    - mode: 'text' or 'image'
    - device: Run the device
    - per_token_ig: whether token is individually attributed to each user input, very low memory consumption
    - model_type: optional, specify the model type (qwen/llama), otherwise it will be automatically judged
- batch_size: when per_token_ig = True, the number of tokens processed each time, the default is 1"""
    # # Force GPU: pick GPU with most free memory
    # if device == "cuda" and torch.cuda.is_available():
    #     # Check CUDA_VISIBLE_DEVICES
    #     cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    #     if cuda_visible:
    #         # GPU with least used memory (system)
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
    #                 print(f"[IG] GPU {i}: used {memory_used:.2f}GB, free {memory_free:.2f}GB")
                    
    #                 if memory_free > min_memory_free:
    #                     min_memory_free = memory_free
    #                     best_gpu = i
                
    #             print(f"[IG] Selected GPU with most free memory: cuda:{best_gpu} (free: {min_memory_free:.2f}GB)")
                
    #         except ImportError:
    #             # Fallback to PyTorch without pynvml
    #             print("[IG] Warning: pynvml missing, PyTorch memory stats may be inaccurate")
    #             min_memory_used = float('inf')
    #             best_gpu = 0
                
    #             for i in range(torch.cuda.device_count()):
    #                 memory_used = torch.cuda.memory_allocated(i)
    #                 print(f"[IG] GPU {i}: PyTorch allocated {memory_used / 1024**3:.2f}GB")
    #                 if memory_used < min_memory_used:
    #                     min_memory_used = memory_used
    #                     best_gpu = i
            
    #         device = torch.device(f"cuda:{best_gpu}")
    #     else:
    #         device = torch.device("cuda")
    
    if RANDOM_STATE:
        random.seed(RANDOM_STATE)
    ######## Text attribution ########
    if mode == "text":
        inputs = tokenizer_or_processor(input_data, return_tensors="pt")
        input_ids = inputs.input_ids
        if baseline is None:
            baseline_ids = input_ids * 0
        else:
            baseline_ids = baseline
        
        # Ensure input_ids and model are on the same device
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

        # --------- Model type ---------
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
                # LayerIntegratedGradients; batched tokens
                batchsize = min(20, len(filtered_indices))  # Cap by paragraph size
                filtered_attributions = []
                lig = LayerIntegratedGradients(forward_func, embedding_layer)
                
                # Chunk filtered_indices by batch_size
                for i in range(0, len(filtered_indices), batch_size):
                    batch_indices = filtered_indices[i:i+batch_size]
                    
                    # Token span for this batch
                    start_idx = batch_indices[0]
                    end_idx = batch_indices[-1] + 1
                    
                    # Attribution on current batch tokens
                    inputs_sub = input_ids[:, start_idx:end_idx]
                    baseline_sub = baseline_ids[:, start_idx:end_idx]
                    
                    attributions, delta = lig.attribute(
                        inputs=inputs_sub,
                        baselines=baseline_sub,
                        target=_target,
                        return_convergence_delta=True,
                        n_steps=min(n_steps, 30)
                    )

                    # Per-token scores in batch
                    token_attributions = attributions.sum(dim=-1).squeeze(0)
                    # Map batch scores back to sequence
                    for j, rel_idx in enumerate(batch_indices):
                        local_idx = rel_idx - start_idx  # Index within batch
                        if local_idx < len(token_attributions):
                            filtered_attributions.append(token_attributions[local_idx])
                        else:
                            # Out of range → 0
                            filtered_attributions.append(torch.tensor(0.0))

            # --------- Highlight attribution ---------
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

    ######## Image attribution ########
    elif mode == "image":
        n_steps = 50
        print("[IG] Integrated gradient attribution method using visual branches for Qwen2.5-VL model")
        try:
            if isinstance(input_data, (Image.Image, np.ndarray, torch.Tensor)):
                # img = ensure_rgb_pil(input_data)
                img = input_data
                img = img.resize((448, 448))

                
                # image_processor → flat patches + image_grid_thw
                processed = tokenizer_or_processor.image_processor(img, return_tensors="pt")
                # Ensure pixel_values and model share device
                model_device = next(model.parameters()).device
                pixel_values = processed.pixel_values.to(model_device)         # [num_patches, patch_dim] e.g. [1024,1176]
                image_grid_thw = processed.image_grid_thw.to(model_device)     # [1, 3]

                print(f"[IG] Processed image tensor shape:{pixel_values.shape}")
                print(f"[IG] image_grid_thw shape: {image_grid_thw.shape}")

                baseline = torch.zeros_like(pixel_values).to(model_device)

                def forward_vision_model(img_input):
                    # Captum may reshape inputs — handle dynamically
                    # Original [1024,1176]; Captum may use [5120,1176]

                    # Inputs on correct device
                    img_input = img_input.to(model_device)
                    
                    # Dynamic batch size
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
                                raise AttributeError(f"Models{type(model).__name__}No visual components found")
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
                print("[IG] Compute Visual Branch Integration Gradient Attribution...")
                ig = IntegratedGradients(forward_vision_model)
                attributions, delta = ig.attribute(
                    inputs=pixel_values,
                    baselines=baseline,
                    return_convergence_delta=True,
                    n_steps=5
                )
                print(f"[IG] Convergence delta:{delta.mean().item() if isinstance(delta, torch.Tensor) and delta.numel() > 1 else delta.item() if isinstance(delta, torch.Tensor) else delta}")
                
                with torch.no_grad():
                    # attributions shape [num_patches, patch_dim]
                    # Map patch attribution to image space
                    attr_map = attributions.squeeze().abs().sum(dim=-1).cpu().numpy()  # [num_patches]
                    
                    # Patch→image attribution map
                    # Qwen2.5-VL: 32×32 patch grid, 1024 patches
                    patch_size = 14  # Qwen2.5-VL patch size
                    grid_size = int(np.sqrt(attr_map.shape[0]))  # Expect 32
                    
                    if grid_size * grid_size == attr_map.shape[0]:
                        # Reshape to grid
                        attr_grid = attr_map.reshape(grid_size, grid_size)
                        
                        # Pixel-level attribution scores
                        # Spread patch score to its pixels
                        pixel_attr_map = np.zeros((448, 448))
                        for i in range(grid_size):
                            for j in range(grid_size):
                                # Patch location in image
                                start_h = i * patch_size
                                end_h = min((i + 1) * patch_size, 448)
                                start_w = j * patch_size
                                end_w = min((j + 1) * patch_size, 448)
                                # Assign patch score to pixels
                                pixel_attr_map[start_h:end_h, start_w:end_w] = attr_grid[i, j]
                        
                        # torch.interpolate upsample to original size
                        attr_tensor = torch.from_numpy(attr_grid).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
                        target_size = (448, 448)  # Original image size
                        attr_map_interpolated = torch.nn.functional.interpolate(
                            attr_tensor, 
                            size=target_size, 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze().numpy()
                    else:
                        # Non-square patch count → simple fallback
                        print(f"[IG] Warning: Number of patches{attr_map.shape[0]}Not exactly square, use simple reshape")
                        pixel_attr_map = attr_map.reshape(-1, 1)  # Temporary reshape for matplotlib
                        attr_map_interpolated = pixel_attr_map
                    
                    # Normalize
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
                    
                    # Grayscale colormap
                    bw_cmap = LinearSegmentedColormap.from_list("bw", ["black", "white"])
                    
                    # 1. Concat: left original, right B&W heatmap (interp)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    # Left: original image
                    ax1.imshow(np.array(img))
                    ax1.set_title("Original Image", fontsize=16)
                    ax1.axis("off")
                    # Right: attribution heatmap only
                    im2 = ax2.imshow(attr_map_interpolated, cmap=bw_cmap)
                    ax2.set_title("Attribution Heatmap (Interpolated)", fontsize=16)
                    ax2.axis("off")
                    # Colorbar
                    norm = plt.Normalize(attr_map_interpolated.min(), attr_map_interpolated.max())
                    sm = plt.cm.ScalarMappable(cmap=bw_cmap, norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
                    cbar.set_label("Attribution Score", fontsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, f"{base_name}_comparison.png"), dpi=150, bbox_inches='tight')
                    plt.close()

                    # 2. Overlay colored heatmap (interp)
                    fig_overlay, ax_overlay = plt.subplots(figsize=(10, 10))
                    ax_overlay.imshow(np.array(img))
                    im_overlay = ax_overlay.imshow(attr_map_interpolated, cmap="hot", alpha=0.5)
                    ax_overlay.set_title("Visual Branch Overlay Attribution")
                    ax_overlay.axis("off")
                    
                    # Colorbar on overlay
                    cbar_overlay = fig_overlay.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
                    cbar_overlay.set_label("Attribution Score", fontsize=12)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, f"{base_name}_overlay.png"), dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # 3. Pixel-level attribution
                    fig_pixel, ax_pixel = plt.subplots(figsize=(10, 10))
                    im_pixel = ax_pixel.imshow(pixel_attr_map, cmap="viridis")
                    ax_pixel.set_title("Pixel-level Attribution Map", fontsize=16)
                    ax_pixel.axis("off")
                    
                    cbar_pixel = fig_pixel.colorbar(im_pixel, ax=ax_pixel, fraction=0.046, pad=0.04)
                    cbar_pixel.set_label("Pixel Attribution Score", fontsize=12)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, f"{base_name}_pixel_attribution.png"), dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Save pixel-level scores
                    pixel_scores_file = os.path.join(fig_dir, f"{base_name}_pixel_scores.txt")
                    with open(pixel_scores_file, 'w') as f:
                        f.write("Pixel-level attribution scores (448x448)\n")
                        f.write("Format: row,column,score\n")
                        for i in range(448):
                            for j in range(448):
                                f.write(f"{i},{j},{pixel_attr_map[i,j]:.6f}\n")
                    
                return pixel_attr_map, img, attributions
            else:
                raise ValueError(f"the input_data type does not support:{type(input_data)}")
        except Exception as e:
            print(f"[IG] Error calculating visual branch integration gradient:{str(e)}")
            print(f"[IG] Error type:{type(e).__name__}")
            print("[IG] Bug Tracking:")
            import traceback
            traceback.print_exc()
            raise e

# def ensure_rgb_pil(img):
#     if isinstance(img, torch.Tensor):
#         img = img.detach().cpu().numpy()
#     if isinstance(img, np.ndarray):
#         img = Image.fromarray(img)
#     if not isinstance(img, Image.Image):
#         raise ValueError(f"Input image cannot be converted to PIL.Image: {type(img)}")
#     print(f"[IG] Input image type: {type(img)}, mode: {img.mode}, size: {img.size}")
#     if img.mode != 'RGB':
#         print(f"[IG] Auto-converting image from {img.mode} to RGB")
#         img = img.convert('RGB')
#     return img
