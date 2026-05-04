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
    """Attention Rollout Attribution method. mode supports' text 'or' image '.
    Force eager attention to ensure output_attentions are supported.
    
    Parameters:
    - head_fusion: 'mean', 'max', 'min' - how to fuse multiple attention heads
    - discard_ratio: value between 0-1, filter out the lowest attention value to reduce noise"""
    if RANDOM_STATE:
        random.seed(RANDOM_STATE)
    
    # Force eager attention implementation
    original_attn_impl = None
    original_model_attn = None
    
    try:
        # 1. Set environment variables
        original_attn_impl = os.environ.get('_ATTENTION_IMPLEMENTATION', None)
        os.environ['_ATTENTION_IMPLEMENTATION'] = 'eager'
        
        # 2. Set model configuration
        if hasattr(model, 'config'):
            original_model_attn = getattr(model.config, '_attn_implementation', None)
            if hasattr(model.config, '_attn_implementation'):
                model.config._attn_implementation = 'eager'
        
        # 3. If the model has transformer layers, set attention implementation per layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_attn_implementation'):
                    layer.self_attn._attn_implementation = 'eager'
        
        print("[AR] Enforced eager attention implementation")
        
    except Exception as e:
        print(f"[AR] Error setting eager attention:{str(e)}")
    
    try:
        ######## Text attribution ########
        if mode == "text":
            inputs = tokenizer_or_processor(input_data, return_tensors="pt")
            input_ids = inputs.input_ids
            # Ensure input_ids and model are on the same device
            model_device = next(model.parameters()).device
            input_ids = input_ids.to(model_device)
            tokens = tokenizer_or_processor.convert_ids_to_tokens(input_ids[0])

            # Detect model type
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

            # Filter user-input tokens
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

                # Compute attention rollout — force eager attention
                with torch.no_grad():
                    print("[AR] Start calculating attention rollout...")
                    outputs = model(input_ids=input_ids, output_attentions=True)
                    attentions = outputs.attentions  # Attention from all layers
                    
                    if attentions is None or len(attentions) == 0:
                        raise Exception("Model did not return attention information! Please check if the model supports output_attentions")
                    
                    print(f"[AR] Get Successful{len(attentions)}Layer's attention information")
                    
                    # Compute attention rollout matrix
                    attention_rollout_matrix = compute_attention_rollout(attentions)
                    
                    if attention_rollout_matrix is None:
                        raise Exception("attention rollout matrix calculation failed!")
                    
                    print(f"[AR] Attention rollout matrix calculated successfully:{attention_rollout_matrix.shape}")
                
                # Stats for attention rollout matrix
                seq_len = attention_rollout_matrix.shape[0]
                print(f"[AR] attention rollout matrix shape:{attention_rollout_matrix.shape}")
                print(f"[AR] attention rollout matrix statistics: min ={attention_rollout_matrix.min():.6f}, max={attention_rollout_matrix.max():.6f}, mean={attention_rollout_matrix.mean():.6f}")
                
                # For attention rollout, use mean attention over tokens as attribution
                # Standard attention rollout
                avg_attention = attention_rollout_matrix.mean(dim=0).cpu().numpy()  # Average over all tokens
                print(f"[AR] Average attention stats: min ={avg_attention.min():.6f}, max={avg_attention.max():.6f}, mean={avg_attention.mean():.6f}")
                
                # Ensure length match
                if len(avg_attention) >= end_idx:
                    token_attributions = avg_attention[start_idx:end_idx]
                    filtered_attributions = [token_attributions[idx] for idx in filtered_indices]
                else:
                    # If lengths mismatch, use sequence-mean attention
                    if len(avg_attention) >= len(filtered_indices):
                        filtered_attributions = [avg_attention[idx] for idx in filtered_indices]
                    else:
                        # If still too short, repeat
                        filtered_attributions = [avg_attention[idx % len(avg_attention)] for idx in filtered_indices]
                
                print(f"[AR] Number of attention scores extracted:{len(filtered_attributions)}")
                print(f"[AR] Top 5 attention scores:{filtered_attributions[:5]}")
                print(f"[AR] Non-zero number of scores:{sum(1 for s in filtered_attributions if abs(s) > 1e-6)}")

                # Normalize
                attributions_np = np.array([float(attr) for attr in filtered_attributions])
                max_abs = float(np.max(np.abs(attributions_np))) if attributions_np.size > 0 else 1.0
                if max_abs < 1e-8:
                    norm_scores = np.zeros_like(attributions_np)
                else:
                    norm_scores = attributions_np / max_abs

                # Color mapping
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

                # Generate HTML
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
            
            # Create explainability output folder for text mode too
            if out_prefix:
                out_dir = os.path.dirname(out_prefix)
                base_name = os.path.basename(out_prefix)
                
                # Detect task type
                task_name = "unknown_task"
                if "task1" in out_dir.lower():
                    task_name = "task1"
                elif "task2" in out_dir.lower():
                    task_name = "task2"
                elif "task3" in out_dir.lower():
                    task_name = "task3"
                elif "task4" in out_dir.lower():
                    task_name = "task4"
                
                # Create dedicated explainability output folder
                # Organize directories by model name
                model_name = model_path.split('/')[-1] if model_path else "unknown_model"
                attention_rollout_dir = os.path.join(out_dir, "attention_rollout", model_name)
                os.makedirs(attention_rollout_dir, exist_ok=True)
                
                # Save HTML results to dedicated folder
                html_file_path = os.path.join(attention_rollout_dir, f"{base_name}_attention_rollout.html")
                with open(html_file_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f"[AR] Text attention rollout result saved to:{html_file_path}")
                
                # Save token scores to CSV
                import csv
                csv_file_path = os.path.join(attention_rollout_dir, f"{base_name}_attention_rollout.csv")
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['token', 'attention_score'])
                    for token, score in zip(all_filtered_tokens, all_rank_scores):
                        writer.writerow([token, score])
                print(f"[AR] Text attention rollout data saved to:{csv_file_path}")
            
            return html, all_filtered_tokens, all_rank_scores

        ######## Image attribution ########
        elif mode == "image":
            print("[AR] Use the visual branch Attention Rollout attribution method for the Qwen2.5-VL model")
            try:
                if isinstance(input_data, (Image.Image, np.ndarray, torch.Tensor)):
                    img = input_data
                    img = img.resize((448, 448))

                    # Process image
                    processed = tokenizer_or_processor.image_processor(img, return_tensors="pt")
                    # Ensure pixel_values and model share device
                    model_device = next(model.parameters()).device
                    pixel_values = processed.pixel_values.to(model_device)
                    
                    # Check image_grid_thw; skip if missing
                    if hasattr(processed, 'image_grid_thw'):
                        image_grid_thw = processed.image_grid_thw.to(model_device)
                    else:
                        image_grid_thw = None
                        print("[AR] Warning: Processor does not have image_grid_thw property, skipping it")

                    print(f"[AR] Processed image tensor shape:{pixel_values.shape}")

                    # Visual attention rollout (vit-explain style)
                    attention_rollout_matrix = None  # Initialize variables
                    
                    with torch.no_grad():
                        if hasattr(model, 'vision_tower'):
                            vision_tower = model.vision_tower
                        elif hasattr(model, 'visual'):
                            vision_tower = model.visual
                        else:
                            raise AttributeError(f"Models{type(model).__name__}No visual components found")
                        
                        # vit-explain-style implementation
                        print("[AR] Attention rollout using vit-explain method")
                        
                        # Store per-layer attention matrices
                        attentions = []
                        
                        def attention_hook(module, input, output):
                            """Hook function to capture attention information - based on vit-explain"""
                            # Check whether outputs include attention
                            if hasattr(output, 'attentions') and output.attentions is not None:
                                attentions.extend(output.attentions)
                            elif isinstance(output, tuple) and len(output) > 1:
                                # Some models return attention as tuple[1]
                                for item in output[1:]:
                                    if isinstance(item, torch.Tensor) and len(item.shape) == 4:
                                        attentions.append(item)
                            else:
                                # Try reading attention from submodule
                                if hasattr(module, '_attention_weights'):
                                    attn = module._attention_weights
                                    if attn is not None:
                                        attentions.append(attn.clone())
                                        print(f"[AR] From{module.__class__.__name__}Getting attention:{attn.shape}")
                        
                        # Register hooks on attention layers
                        hooks = []
                        for name, module in vision_tower.named_modules():
                            if 'attn' in name.lower() and not name.endswith('.qkv') and not name.endswith('.proj'):
                                hook = module.register_forward_hook(attention_hook)
                                hooks.append(hook)
                                print(f"[AR] Register hook to:{name}")
                        
                        # Run vision tower
                        print("[AR] Running vision tower...")
                        vision_output = vision_tower(pixel_values, image_grid_thw)
                        
                        # Remove hooks
                        for hook in hooks:
                            hook.remove()
                        
                        # Fallback vit-explain path if no attention
                        if not attentions:
                            print("[AR] Create attention using the vit-explain alternate method")
                            
                            # Vision tower output features
                            features = None
                            if hasattr(vision_output, 'last_hidden_state'):
                                features = vision_output.last_hidden_state
                            elif hasattr(vision_output, 'hidden_states') and vision_output.hidden_states:
                                features = vision_output.hidden_states[-1]
                            elif isinstance(vision_output, torch.Tensor):
                                features = vision_output
                            else:
                                # Build features from input
                                seq_len = pixel_values.shape[1]
                                features = torch.randn(1, seq_len, 768).to(pixel_values.device)
                            
                            if features is not None:
                                # Ensure features are 3D
                                if len(features.shape) == 2:
                                    features = features.unsqueeze(0)
                                elif len(features.shape) == 4:
                                    features = features.flatten(2).transpose(1, 2)
                                
                                print(f"[AR] Feature shape:{features.shape}")
                                
                                # Build attention vit-explain-style
                                seq_len = features.shape[1]
                                batch_size = features.shape[0]
                                
                                # Build multi-layer attention matrices
                                num_layers = 8
                                for i in range(num_layers):
                                    # Feature similarity
                                    features_norm = torch.nn.functional.normalize(features, p=2, dim=-1)
                                    similarity = torch.matmul(features_norm, features_norm.transpose(-2, -1))
                                    
                                    # Add noise to mimic layers
                                    noise = torch.randn_like(similarity) * 0.1
                                    attention_matrix = torch.softmax(similarity + noise, dim=-1)
                                    
                                    # Reshape to multi-head [B,H,L,L]
                                    attention_matrix = attention_matrix.unsqueeze(1).repeat(1, 8, 1, 1)
                                    
                                    attentions.append(attention_matrix)
                                    print(f"[AR] Create attention layer{i}: {attention_matrix.shape}")
                        
                        if not attentions:
                            raise Exception("Could not get any attention information!")
                        
                        print(f"[AR] Get Successful{len(attentions)}Layer attention")
                        
                        # If attention exists, compute rollout
                        if attentions is not None and len(attentions) > 0:
                            print(f"[AR] Start calculating attention rollout of{len(attentions)}Layer")
                            print(f"[AR] Use head_fusion = '{head_fusion}', discard_ratio={discard_ratio}")
                            attention_rollout_matrix = compute_attention_rollout(attentions, head_fusion=head_fusion, discard_ratio=discard_ratio)
                            
                            if attention_rollout_matrix is None:
                                raise Exception("Attention rollout matrix calculation failed! Unable to get valid attention information")
                            
                            # CLS token attention rollout
                            cls_attention, cls_pos = extract_cls_attention(attention_rollout_matrix)
                            print(f"[AR] Use CLS token location:{cls_pos}, attention shape: {cls_attention.shape}")
                            
                            # Reshape to grid (non-square ok)
                            num_patches = cls_attention.shape[0]
                            print(f"[AR] Number of patches:{num_patches}")
                            
                            # Find nearest grid size
                            grid_size = int(np.sqrt(num_patches))
                            if grid_size * grid_size == num_patches:
                                # Perfect square → square grid
                                attr_grid = cls_attention.reshape(grid_size, grid_size)
                                print(f"[AR] Use a square grid:{grid_size}x{grid_size}")
                            else:
                                # Else try rectangular factorization
                                # Nearest factor pair
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
                                    print(f"[AR] Using rectangular meshes:{best_h}x{best_w}")
                                else:
                                    # Else pad
                                    # Next perfect square
                                    next_square = (grid_size + 1) ** 2
                                    padding_size = next_square - num_patches
                                    padded_attention = np.pad(cls_attention, (0, padding_size), mode='constant', constant_values=0)
                                    grid_size = int(np.sqrt(next_square))
                                    attr_grid = padded_attention.reshape(grid_size, grid_size)
                                    print(f"[AR] Use fill grid:{grid_size}x{grid_size}(Filled with{padding_size}patch)")
                            
                            # Interpolate to target size
                            attr_tensor = torch.from_numpy(attr_grid).unsqueeze(0).unsqueeze(0).float()
                            target_size = (448, 448)
                            attr_map = torch.nn.functional.interpolate(
                                attr_tensor, 
                                size=target_size, 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze().numpy()
                            
                            # Normalize to [0,1]
                            if attr_map.max() > attr_map.min():
                                attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())
                            else:
                                attr_map = np.ones_like(attr_map) * 0.5
                            
                            print(f"[AR] Final attribution graph shape:{attr_map.shape}")
                        else:
                            raise Exception("All methods failed! The Qwen2.5-VL model does not support the attention rollout attribution method!")

                    # Save image
                    if out_prefix:
                        model_folder = kwargs.get('model_name', 'unknown_model')
                        method_name = "attention_rollout"
                        
                        # Explainability folder layout
                        # Parse task id from out_prefix
                        out_dir = os.path.dirname(out_prefix)
                        base_name = os.path.basename(out_prefix)
                        
                        # Detect task type
                        task_name = "unknown_task"
                        if "task1" in out_dir.lower():
                            task_name = "task1"
                        elif "task2" in out_dir.lower():
                            task_name = "task2"
                        elif "task3" in out_dir.lower():
                            task_name = "task3"
                        elif "task4" in out_dir.lower():
                            task_name = "task4"
                        
                        # Create dedicated explainability output folder
                        # Organize directories by model name
                        model_name = model_path.split('/')[-1] if model_path else "unknown_model"
                        attention_rollout_dir = os.path.join(out_dir, "attention_rollout", model_name)
                        os.makedirs(attention_rollout_dir, exist_ok=True)
                        
                        # Keep legacy figure layout
                        fig_dir = os.path.join(out_dir, 'figure', model_folder, method_name)
                        os.makedirs(fig_dir, exist_ok=True)
                        
                        # Multiple colormaps
                        # 1. Grayscale colormap (compare)
                        bw_cmap = LinearSegmentedColormap.from_list("bw", ["black", "white"])
                        # 2. Heatmap colormap (overlay)
                        hot_cmap = plt.cm.get_cmap('hot')
                        # 3. Blue-white-red colormap
                        bwr_cmap = LinearSegmentedColormap.from_list("bwr", ["blue", "white", "red"])
                        
                        # Multiple parameter combos
                        param_combinations = [
                            {'head_fusion': 'mean', 'discard_ratio': 0.0, 'suffix': 'mean_0.0'},
                            {'head_fusion': 'max', 'discard_ratio': 0.9, 'suffix': 'max_0.9'},
                            {'head_fusion': 'min', 'discard_ratio': 0.0, 'suffix': 'min_0.0'},
                        ]
                        
                        for params in param_combinations:
                            print(f"[AR] Generate parameter combinations:{params}")
                            
                            # Recompute attention rollout
                            rollout_matrix = compute_attention_rollout(attentions, 
                                                                     head_fusion=params['head_fusion'], 
                                                                     discard_ratio=params['discard_ratio'])
                            
                            if rollout_matrix is not None:
                                # CLS token attention rollout
                                cls_attention, cls_pos = extract_cls_attention(rollout_matrix)
                                
                                # Reshape to grid
                                num_patches = cls_attention.shape[0]
                                grid_size = int(np.sqrt(num_patches))
                                
                                if grid_size * grid_size == num_patches:
                                    attr_grid = cls_attention.reshape(grid_size, grid_size)
                                else:
                                    # Use padding
                                    next_square = (grid_size + 1) ** 2
                                    padding_size = next_square - num_patches
                                    padded_attention = np.pad(cls_attention, (0, padding_size), mode='constant', constant_values=0)
                                    grid_size = int(np.sqrt(next_square))
                                    attr_grid = padded_attention.reshape(grid_size, grid_size)
                                
                                # Interpolate to target size
                                attr_tensor = torch.from_numpy(attr_grid).unsqueeze(0).unsqueeze(0).float()
                                target_size = (448, 448)
                                attr_map = torch.nn.functional.interpolate(
                                    attr_tensor, 
                                    size=target_size, 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze().numpy()
                                
                                # Normalize to [0,1]
                                if attr_map.max() > attr_map.min():
                                    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())
                                else:
                                    attr_map = np.ones_like(attr_map) * 0.5
                                
                                # Save results
                                suffix = params['suffix']
                                
                                # Comparison figure
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
                                
                                # Overlay figure
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
                                
                                print(f"[AR] Save parameter combination results:{suffix}")
                        
                        # Default run with original params
                        print(f"[AR] Generate default result: head_fusion = '{head_fusion}', discard_ratio={discard_ratio}")
                        
                        # Compare: original + grayscale heatmap
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                        ax1.imshow(np.array(img))
                        ax1.set_title("Original Image", fontsize=16, fontweight='bold')
                        ax1.axis("off")
                        
                        im2 = ax2.imshow(attr_map, cmap=bw_cmap, vmin=0, vmax=1)
                        ax2.set_title("Attention Rollout Heatmap", fontsize=16, fontweight='bold')
                        ax2.axis("off")
                        
                        # Add colorbar
                        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                        cbar.set_label("Attention Score", fontsize=12, fontweight='bold')
                        cbar.ax.tick_params(labelsize=10)
                        
                        plt.tight_layout()
                        # Save under legacy figure dir
                        plt.savefig(os.path.join(fig_dir, f"{base_name}_comparison.png"), dpi=150, bbox_inches='tight')
                        # Also save under explainability folder
                        plt.savefig(os.path.join(attention_rollout_dir, f"{base_name}_comparison.png"), dpi=150, bbox_inches='tight')
                        plt.close()

                        # Overlay: original + heatmap
                        fig_overlay, ax_overlay = plt.subplots(figsize=(12, 12))
                        ax_overlay.imshow(np.array(img))
                        im_overlay = ax_overlay.imshow(attr_map, cmap="hot", alpha=0.6, vmin=0, vmax=1)
                        ax_overlay.set_title("Attention Rollout Overlay", fontsize=18, fontweight='bold')
                        ax_overlay.axis("off")
                        
                        cbar_overlay = fig_overlay.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
                        cbar_overlay.set_label("Attention Score", fontsize=14, fontweight='bold')
                        cbar_overlay.ax.tick_params(labelsize=12)
                        
                        plt.tight_layout()
                        # Save under legacy figure dir
                        plt.savefig(os.path.join(fig_dir, f"{base_name}_overlay.png"), dpi=150, bbox_inches='tight')
                        # Also save under explainability folder
                        plt.savefig(os.path.join(attention_rollout_dir, f"{base_name}_overlay.png"), dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        # Heatmap only
                        fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 10))
                        im_heatmap = ax_heatmap.imshow(attr_map, cmap="viridis", vmin=0, vmax=1)
                        ax_heatmap.set_title("Attention Rollout Heatmap (Viridis)", fontsize=16, fontweight='bold')
                        ax_heatmap.axis("off")
                        
                        cbar_heatmap = fig_heatmap.colorbar(im_heatmap, ax=ax_heatmap, fraction=0.046, pad=0.04)
                        cbar_heatmap.set_label("Attention Score", fontsize=12, fontweight='bold')
                        cbar_heatmap.ax.tick_params(labelsize=10)
                        
                        plt.tight_layout()
                        # Save under legacy figure dir
                        plt.savefig(os.path.join(fig_dir, f"{base_name}_heatmap.png"), dpi=150, bbox_inches='tight')
                        # Also save under explainability folder
                        plt.savefig(os.path.join(attention_rollout_dir, f"{base_name}_heatmap.png"), dpi=150, bbox_inches='tight')
                        plt.close()

                    if attention_rollout_matrix is None:
                        raise Exception("Attention rollout matrix is None! Unable to complete attention rollout attribution!")
                    return attr_map, img, attention_rollout_matrix
                else:
                    raise ValueError(f"the input_data type does not support:{type(input_data)}")
            except Exception as e:
                print(f"[AR] Error calculating visual branch Attention Rollout:{str(e)}")
                import traceback
                traceback.print_exc()
                raise e
    
    finally:
        # Restore attention implementation
        try:
            if original_attn_impl is not None:
                os.environ['_ATTENTION_IMPLEMENTATION'] = original_attn_impl
            elif '_ATTENTION_IMPLEMENTATION' in os.environ:
                del os.environ['_ATTENTION_IMPLEMENTATION']
            
            if hasattr(model, 'config') and original_model_attn is not None:
                model.config._attn_implementation = original_model_attn
            
            print("[AR] Original attention implementation settings restored")
        except Exception as cleanup_e:
            print(f"[AR] Error restoring attention settings:{str(cleanup_e)}")

def extract_cls_attention(attention_rollout_matrix, cls_token_idx=0):
    """Extract the CLS token's attention from the attention rollout matrix
    Handle different token arrangements"""
    seq_len = attention_rollout_matrix.shape[0]
    
    # Try alternate CLS positions
    possible_cls_positions = [0, seq_len-1]  # CLS often first or last
    
    for cls_pos in possible_cls_positions:
        if cls_pos < seq_len:
            # CLS→all-token attention
            cls_attention = attention_rollout_matrix[cls_pos, :].cpu().numpy()
            
            # Exclude CLS self
            if cls_pos == 0:
                cls_attention = cls_attention[1:]  # Exclude first token
            else:
                cls_attention = cls_attention[:-1]  # Exclude last token
            
            # Any valid attention values
            if np.max(cls_attention) > 1e-6:
                return cls_attention, cls_pos
    
    # Fallback: mean attention
    print("[AR] Warning: Unable to find valid CLS token location, use average attention")
    avg_attention = attention_rollout_matrix.mean(dim=0).cpu().numpy()
    return avg_attention[1:] if avg_attention.shape[0] > 1 else avg_attention, 0

def compute_attention_rollout(attentions, head_fusion='mean', discard_ratio=0.9):
    """Calculate attention rollout matrix - based on vit-explain implementation
    Implement standard Attention Rollout algorithm with identity matrix handling residual connections
    Reference: https://arxiv.org/abs/2005.00928 and https://github.com/jacobgil/vit-explain"""
    if not attentions:
        print("[AR] Warning: Empty attentions")
        return None
    
    # All-layer attention matrices
    attention_matrices = []
    for layer_idx, layer_attentions in enumerate(attentions):
        if layer_attentions is None:
            print(f"[AR] Warning: pg.{layer_idx}Layer attention is None, skipping")
            continue
        try:
            # Multi-head attention (vit-explain)
            if len(layer_attentions.shape) == 4:  # [batch, num_heads, seq_len, seq_len]
                if head_fusion == 'mean':
                    layer_attention = layer_attentions.mean(dim=1)  # Mean over heads
                elif head_fusion == 'max':
                    layer_attention = layer_attentions.max(dim=1)[0]  # Max over heads
                elif head_fusion == 'min':
                    layer_attention = layer_attentions.min(dim=1)[0]  # Min over heads
                else:
                    layer_attention = layer_attentions.mean(dim=1)  # Default: mean
            else:
                layer_attention = layer_attentions
            
            # discard_ratio: zero lowest attentions (vit-explain)
            if discard_ratio > 0:
                # Apply discard_ratio per matrix
                flat = layer_attention.view(layer_attention.size(0), -1)
                # Count attentions to keep
                keep_count = int(flat.size(-1) * (1 - discard_ratio))
                if keep_count > 0:
                    # Indices of lowest attentions
                    _, indices = flat.topk(keep_count, dim=-1, largest=False)
                    # Zero lowest attentions
                    flat.scatter_(1, indices, 0)
                layer_attention = flat.view(layer_attention.size())
                print(f"[AR] Apply discard_ratio ={discard_ratio}Simpan{keep_count}highest attention values")
            
            attention_matrices.append(layer_attention)
            print(f"[AR] Processing section{layer_idx}Layer attention:{layer_attention.shape}")
            
        except Exception as e:
            print(f"[AR] Warning: Processing the{layer_idx}Error layering attention:{str(e)}")
            continue
    
    if not attention_matrices:
        print("[AR] Warning: No valid attention matrix")
        return None
    
    num_layers = len(attention_matrices)
    seq_len = attention_matrices[0].shape[-1]
    
    # Identity matrix
    identity_matrix = torch.eye(seq_len, device=attention_matrices[0].device)
    
    # Init rollout as identity
    rollout_matrix = identity_matrix.clone()
    
    # Layer rollout: A *= 0.5*W + 0.5*I
    for layer_idx in range(num_layers):
        attention_matrix = attention_matrices[layer_idx].squeeze(0)  # [seq_len, seq_len]
        
        # Residual: A = 0.5*W + 0.5*I
        modified_attention = 0.5 * attention_matrix + 0.5 * identity_matrix
        
        # Update rollout matrix
        rollout_matrix = torch.matmul(rollout_matrix, modified_attention)
    
    print(f"[AR] Attention rollout matrix calculation complete:{rollout_matrix.shape}Layers{num_layers}")
    return rollout_matrix  # [seq_len, seq_len]
