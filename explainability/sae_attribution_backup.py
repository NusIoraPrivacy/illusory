import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import time

try:
    from sae_lens import SAE
    from sae_dashboard.sae_vis_data import SaeVisConfig
    from sae_dashboard.sae_vis_runner import SaeVisRunner
    from sae_dashboard.data_writing_fns import save_feature_centric_vis
    SAE_AVAILABLE = True
except ImportError:
    SAE_AVAILABLE = False

def process_dialogue_text(full_dialogue_text):
    """保留完整的对话历史，包含所有提示词"""
    # Simple structural validation logging
    if full_dialogue_text:
        # # Check basic prompt keywords
        # has_instruction = any(keyword in full_dialogue_text.lower() for keyword in 
        #                     ['instruction', 'task', 'scenario', 'please'])
        # has_question = any(keyword in full_dialogue_text.lower() for keyword in 
        #                   ['question', 'answer', 'choose', 'select'])
        
        print(f"[SAE] Prompt: {full_dialogue_text}")
    
    return full_dialogue_text.strip()

def print_gpu_info():
    """打印GPU使用情况"""
    if torch.cuda.is_available():
        print(f"[SAE] GPU数量: {torch.cuda.device_count()}")
    else:
        print("[SAE] 无GPU可用")

def sae_attribution(model, tokenizer_or_processor, input_data, out_prefix, target_label=1, mode="text", device="cpu", model_type=None, messages=None, model_path=None, task=None, cycle_num=None, generate_top_features=True, **kwargs):
    device = "cpu"
    html_snippet = ""
    filtered_tokens = []
    rank_scores = []
    
    if mode not in ["text", "image"]:
        raise ValueError("SAE attribution supports text and image modes only")
    
    if not SAE_AVAILABLE:
        raise ImportError("SAELens is not installed")
    
    print_gpu_info()
    
    sae_release = kwargs.get('sae_release', 'llama-3-8b-it-res-jh')
    sae_id = kwargs.get('sae_id', 'blocks.25.hook_resid_post')
    
    try:
        print(f"[SAE] SAE使用设备: {device}")
        sae_result = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device
        )

        if isinstance(sae_result, tuple):
            if len(sae_result) >= 3:
                sae, cfg_dict, feature_sparsity = sae_result
            elif len(sae_result) == 2:
                sae, cfg_dict = sae_result
                feature_sparsity = None
            else:
                sae = sae_result[0] if len(sae_result) > 0 else sae_result
                cfg_dict = {'d_in': 768}
                feature_sparsity = None
        else:
            sae = sae_result
            cfg_dict = {'d_in': 768}
            feature_sparsity = None
            
        print(f"[SAE] Successfully loaded SAE model: {sae}")
        
    except Exception as e:
        print(f"[SAE] Failed to load SAE model: {str(e)}")
        raise RuntimeError("Failed to load SAE model")

    if mode == "text" and isinstance(input_data, str):
        input_data = process_dialogue_text(input_data)
        
        # Force all compute on CPU

        
        # Encode text
        inputs = tokenizer_or_processor(input_data, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        tokens = inputs['input_ids']
        if tokens.dim() == 3:  # [1, 1, seq_len]
            tokens = tokens.squeeze(0)  # Remove first dim only
        elif tokens.dim() == 2:  # [1, seq_len]
            tokens = tokens.squeeze(0)  # Remove batch dimension
        
        # Force tokens CPU
        tokens = tokens.to(device)


        # Hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states
        target_layer = kwargs.get('target_layer', 25)
        print(f"[SAE] 分析第{target_layer}层隐藏状态，shape: {hidden_states[target_layer].shape}")

        target_hidden = hidden_states[target_layer]


        d_model = cfg_dict.get('d_in', 768)
        if target_hidden.shape[-1] != d_model:
            print(f"[SAE] 维度不匹配，期望 {d_model}，实际 {target_hidden.shape[-1]}")
            target_hidden = target_hidden.to(torch.float32)
            projection = torch.nn.Linear(target_hidden.shape[-1], d_model, device=device)
            target_hidden = projection(target_hidden)
            print(f"[SAE] 投影后shape: {target_hidden.shape}")

        # Run SAE on CPU
        target_hidden = target_hidden.to(device)

        
        feature_acts = sae.encode(target_hidden)
        reconstructed = sae.decode(feature_acts)
        


        print(f"[SAE] 特征激活shape: {feature_acts.shape}")
        print(f"[SAE] 重构shape: {reconstructed.shape}")



        mse = F.mse_loss(target_hidden, reconstructed).item()
        cosine_sim = F.cosine_similarity(target_hidden, reconstructed, dim=-1).mean().item()
        print(f"[SAE] 重构MSE: {mse:.6f}, 余弦相似度: {cosine_sim:.6f}")


        l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
        print(f"[SAE] 平均L0: {l0.mean().item():.2f}")
        
        local_sparsity = (feature_acts == 0).float().mean().item()
        print(f"[SAE] 局部稀疏性: {local_sparsity:.4f}")


        # Token attribution via feature activation strength (standard)
        token_attributions = torch.norm(feature_acts, dim=-1)
        token_attributions = token_attributions.squeeze().detach().cpu()


        # html_snippet placeholder (replaced by minimal HTML)
        html_snippet = ""
        
        # Save viz with unique name per cycle
        if out_prefix:
            # Path: results/explainability/task{id}/sae_attribution/
            fig_dir = os.path.join('results', 'explainability', f'task{task}', 'sae_attribution')
            os.makedirs(fig_dir, exist_ok=True)
            base_name = os.path.basename(out_prefix)
            
            # Unique timestamp + cycle id per run
            timestamp = int(time.time())
            cycle_suffix = f"_cycle{cycle_num}" if cycle_num is not None else f"_ts{timestamp}"
            
            # Persist SAE config metadata
            config_file = os.path.join(fig_dir, f"{base_name}{cycle_suffix}_sae_config.txt")
            with open(config_file, 'w') as f:
                f.write(f"SAE Configuration:\n")
                f.write(f"Release: {sae_release}\n")
                f.write(f"SAE ID: {sae_id}\n")
                f.write(f"Target Layer: {target_layer}\n")
                f.write(f"Device: {device}\n")
                f.write(f"Sparsity: {local_sparsity:.6f}\n")
                f.write(f"Average L0: {l0.mean().item():.6f}\n")
                f.write(f"SAE Config: {sae.cfg}\n")
                if cycle_num is not None:
                    f.write(f"Cycle Number: {cycle_num}\n")
                f.write(f"Timestamp: {timestamp}\n")
            
            print(f"[SAE] 已保存配置信息到: {config_file}")
            
            # Active features from existing feature_acts
            
            
            # Check feature_acts shape
            if len(feature_acts.shape) == 2:
                # [batch_size, d_sae]
                total_features = feature_acts.shape[1]
                total_features = feature_acts.shape[1]
            elif len(feature_acts.shape) == 3:
                # [batch_size, seq_len, d_sae]
                total_features = feature_acts.shape[2]
            else:
                total_features = feature_acts.shape[-1]
            
            # feature_acts on correct device
            if feature_acts.device.type != device:
                feature_acts = feature_acts.to(device)
            
            # Per-feature strength (L1)
            feature_activations = []
            for i in range(total_features):
                try:
                    if len(feature_acts.shape) == 2:
                        # [batch_size, d_sae]
                        activation_strength = torch.norm(feature_acts[:, i], p=1).item()
                    elif len(feature_acts.shape) == 3:
                        # [batch_size, seq_len, d_sae]
                        activation_strength = torch.norm(feature_acts[:, :, i], p=1).item()
                    else:
                        # Use last dimension
                        activation_strength = torch.norm(feature_acts[..., i], p=1).item()
                    feature_activations.append((i, activation_strength))
                except Exception as e:
                    print(f"[SAE] 计算特征 {i} 激活强度时出错: {e}")
                    print(f"[SAE] feature_acts shape: {feature_acts.shape}")
                    print(f"[SAE] 尝试访问的索引: {i}")
                    raise
            
            # Sort by activation strength
            feature_activations.sort(key=lambda x: x[1], reverse=True)
            
            # top-k from local sparsity
            top_k = max(1, int((1 - local_sparsity) * total_features))
            # top_k = 1
            top_k = min(top_k, total_features)  # Clamp top-k to feature count
            top_features = [idx for idx, _ in feature_activations[:top_k]]
            
            print(f"[SAE] 局部稀疏性: {local_sparsity:.4f}, 直接识别出前{top_k}个最活跃特征 ({(1-local_sparsity)*100:.2f}% 的激活特征)")
            
            # Tutorial-style visualization
            try:
                from transformer_lens import HookedTransformer
                # CPU to avoid multi-GPU conflicts
                print("[SAE] 正在加载HookedTransformer模型到CPU以避免多GPU冲突...")
                hooked_model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device="cpu")
                
                # Optional top-feature subset
                if generate_top_features:
                    

                    top_config = SaeVisConfig(
                        hook_point=sae_id,
                        features=top_features,
                        minibatch_size_features=50,
                        minibatch_size_tokens=50,
                        verbose=True,
                        device="cpu",  # CPU fallback for GPU conflicts
                    )
                    # tokens [B,T] on CPU
                    if tokens.dim() == 1:
                        tokens_for_sae = tokens.unsqueeze(0).cpu()  # Add batch dim → CPU
                    else:
                        tokens_for_sae = tokens.cpu()
                    
                    # Keep SAE on CPU
                    sae_cpu = sae.cpu()
                    top_vis_data = SaeVisRunner(top_config).run(
                        encoder=sae_cpu,
                        model=hooked_model,
                        tokens=tokens_for_sae,
                    )
                    top_vis_file = os.path.join(fig_dir, f"{base_name}{cycle_suffix}_top{top_k}_features.html")
                    save_feature_centric_vis(sae_vis_data=top_vis_data, filename=top_vis_file)
                    print(f"[SAE] 已生成原始SAE Dashboard: {top_vis_file}")
                    
                    # Emit minimal HTML
                    try:
                        # Token tensor shape for decode
                        if tokens.dim() == 0:
                            tokens_for_decode = tokens.unsqueeze(0)
                        elif tokens.dim() == 1:
                            tokens_for_decode = tokens
                        else:
                            tokens_for_decode = tokens[0]
                        
                        token_texts = [tokenizer_or_processor.decode([token_id]) for token_id in tokens_for_decode.cpu().numpy()]
                        
                        # Feature activation tensors
                        feature_acts_np = feature_acts[0].detach().cpu().numpy()  # [seq_len, num_features]
                        
                        # Subset analyzed features only
                        selected_feature_acts = feature_acts_np[:, top_features]  # [seq_len, len(top_features)]
                        
                        # Minimal HTML export
                        simple_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SAE Features Analysis - Lightweight View</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ position: relative; margin-bottom: 30px; }}
        .feature-selector {{ position: absolute; top: 0; right: 0; }}
        .feature-selector select {{ padding: 8px; border-radius: 5px; border: 1px solid #ccc; font-size: 14px; }}
        .token {{ display: inline-block; margin: 1px; padding: 2px 4px; border-radius: 3px; border: 1px solid #ddd; }}
        .feature-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 8px; }}
        .feature-label {{ font-weight: bold; margin-bottom: 10px; color: #333; }}
        .stats {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
        .color-legend {{ margin: 20px 0; padding: 20px; background: #e9ecef; border-radius: 8px; width: 100%; }}
        .legend-item {{ display: block; margin: 10px 0; width: 100%; }}
        .legend-color {{ display: block; width: 100%; height: 30px; border: 1px solid #999; border-radius: 5px; }}
        .token-container {{ font-family: monospace; font-size: 12px; line-height: 1.6; margin: 10px 0; }}
        .activation-list {{ font-family: monospace; font-size: 11px; margin: 10px 0; }}
        .feature-page {{ display: none; }}
        .feature-page.active {{ display: block; }}
    </style>
    <script>
        function showFeature(featureId) {{
            // 隐藏所有特征页面
            const pages = document.querySelectorAll('.feature-page');
            pages.forEach(page => page.classList.remove('active'));
            
            // 显示选中的特征页面
            const selectedPage = document.getElementById('feature-' + featureId);
            if (selectedPage) {{
                selectedPage.classList.add('active');
            }}
            
            // 更新页面标题
            document.title = 'SAE Feature ' + featureId + ' Analysis';
        }}
        
        // 页面加载时显示第一个特征
        window.onload = function() {{
            const firstFeature = document.querySelector('.feature-page');
            if (firstFeature) {{
                firstFeature.classList.add('active');
                const featureId = firstFeature.id.replace('feature-', '');
                document.querySelector('#feature-select').value = featureId;
            }}
        }};
    </script>
</head>
<body>
    <div class="header">
        <h1>SAE Features Analysis - Lightweight View</h1>
        <div class="feature-selector">
            <label for="feature-select">Select Feature: </label>
            <select id="feature-select" onchange="showFeature(this.value)">
                {''.join([f'<option value="{feature_id}">Feature {feature_id}</option>' for feature_id in top_features])}
            </select>
        </div>
    </div>
    
    <div class="color-legend">
        <h3>Color Legend</h3>
        <div class="legend-item">
            <span class="legend-color" style="background: linear-gradient(to right, #ffffff, #ff0000); width: 100%; height: 30px; display: block; margin: 10px 0; border-radius: 5px;"></span>
            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                <span>White (low activation)</span>
                <span>Red (high activation)</span>
            </div>
        </div>
    </div>
"""
                        
                        # Viz each feature
                        for feat_idx, feature_id in enumerate(top_features):
                            feature_acts = selected_feature_acts[:, feat_idx]
                            
                            simple_html += f"""
    <div class="feature-page" id="feature-{feature_id}">
        <div class="feature-section">
            <div class="feature-label">Feature {feature_id}</div>
            
            <div class="stats">
                <p><strong>Max Activation:</strong> {feature_acts.max():.4f}</p>
                <p><strong>Min Activation:</strong> {feature_acts.min():.4f}</p>
                <p><strong>Mean Activation:</strong> {feature_acts.mean():.4f}</p>
                <p><strong>Active Tokens (>0.1):</strong> {(feature_acts > 0.1).sum()}</p>
            </div>
            
            <h4>Token Activations</h4>
            <div class="token-container">
"""
                            
                            # Per-token cells (improved palette)
                            for i, (token, activation) in enumerate(zip(token_texts, feature_acts)):
                                # Color intensity 0–1
                                max_act = feature_acts.max()
                                if max_act > 0:
                                    color_intensity = min(1.0, activation / max_act)
                                else:
                                    color_intensity = 0
                                
                                global_max_abs = abs(selected_feature_acts).max()
                                if global_max_abs > 0:
                                    normalized_intensity = min(1.0, abs(activation) / global_max_abs)
                                else:
                                    normalized_intensity = 0
                                
                                # Red gradient white→red
                                color_range = 0.97
                                intensity = abs(normalized_intensity)
                                r = 255
                                g = int(255 * (color_range - intensity))
                                b = int(255 * (color_range - intensity))
                                bg_color = f"rgb({r},{g},{b})"
                                
                                restored_token = token.replace('Ċ', '\n').replace('Ġ', ' ')
                                
                                if '\n' in restored_token:
                                    parts = restored_token.split('\n')
                                    for idx, part in enumerate(parts):
                                        if part:
                                            simple_html += f'<span class="token" style="background-color: {bg_color};" title="Token {i}: {activation:.4f}">{part}</span>'
                                        if idx < len(parts) - 1:
                                            simple_html += '<br/>'
                                else:
                                    simple_html += f'<span class="token" style="background-color: {bg_color};" title="Token {i}: {activation:.4f}">{restored_token}</span>'
                            
                            simple_html += """
            </div>
            
            <h4>Top 10 Most Active Tokens</h4>
            <div class="activation-list">
"""
                            
                            # List hottest tokens
                            top_indices = feature_acts.argsort()[-10:][::-1]
                            for i, idx in enumerate(top_indices):
                                activation = feature_acts[idx]
                                token = token_texts[idx]
                                simple_html += f'<div>{i+1:2d}. Token {idx:3d}: "{token}" (activation: {activation:.4f})</div>'
                            
                            simple_html += """
            </div>
        </div>
    </div>
"""
                        
                        simple_html += """
    <div style="margin-top: 30px; padding: 10px; background: #e8f4f8; border-radius: 5px;">
        <p><strong>Note:</strong> This is a lightweight view showing all analyzed features. 
        Colors range from white (low activation) to red (high activation).</p>
    </div>
</body>
</html>
"""
                        
                        # Persist minimal HTML
                        simple_html_file = os.path.join(fig_dir, f"{base_name}{cycle_suffix}_top{top_k}_features_simple.html")
                        with open(simple_html_file, 'w', encoding='utf-8') as f:
                            f.write(simple_html)
                        print(f"[SAE] 已生成超简化HTML: {simple_html_file}")
                        
                    except Exception as e:
                        print(f"[SAE] 超简化HTML生成失败: {e}")
                    
                    del top_vis_data
                    
                    # Save top-feature list
                    top_features_file = os.path.join(fig_dir, f"{base_name}{cycle_suffix}_top{top_k}_features.txt")
                    with open(top_features_file, 'w') as f:
                        f.write(f"Top {top_k} Most Active Features\n")
                        f.write("=" * 50 + "\n")
                        f.write("Feature_ID\tActivation_Score\n")
                        for i, (feature_id, score) in enumerate(feature_activations[:top_k]):
                            f.write(f"{feature_id}\t{score:.6f}\n")
                        f.write("\n" + "=" * 50 + "\n")
                        f.write(f"Total features analyzed: {len(feature_activations)}\n")
                        f.write(f"Top features selected: {top_k}\n")
                        f.write(f"Average activation of top features: {np.mean([score for _, score in feature_activations[:top_k]]):.6f}\n")
                        f.write(f"Average activation of all features: {np.mean([score for _, score in feature_activations]):.6f}\n")
                        f.write(f"Local sparsity: {local_sparsity:.6f}\n")
                        f.write(f"Average L0: {l0.mean().item():.6f}\n")
                    
                    print(f"[SAE] 已保存最活跃特征列表: {top_features_file}")
                
                # Free memory
                del hooked_model
                
            except Exception as e:
                print(f"[SAE] SAE可视化生成失败: {e}")
                import traceback
                traceback.print_exc()
                
            # Ensure list return type
        if isinstance(rank_scores, (float, np.float64)):
            rank_scores = [rank_scores]
        if isinstance(filtered_tokens, (str, int, float)):
            filtered_tokens = [filtered_tokens]
        return html_snippet, filtered_tokens, rank_scores

    ######## Image attribution ########
    elif mode == "image" and isinstance(input_data, (Image.Image, np.ndarray, torch.Tensor)):
        print(f"[SAE] 开始图像SAE归因分析...")

        # Process input image
        if isinstance(input_data, Image.Image):
            img = input_data
        elif isinstance(input_data, np.ndarray):
            img = Image.fromarray(input_data)
        elif isinstance(input_data, torch.Tensor):
            img = Image.fromarray(input_data.detach().cpu().numpy())

        img = img.resize((448, 448))
        img_array = np.array(img)

        # Convert to tensor
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        x = transform(img).unsqueeze(0).to(device)

        # HookedTransformer per tutorial
        try:
            from transformer_lens import HookedTransformer
            # HookedTransformer multi-GPU capable
            if torch.cuda.device_count() > 1:
                print(f"[SAE] 检测到 {torch.cuda.device_count()} 个GPU，HookedTransformer使用多GPU模式")
                hooked_model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
            else:
                print(f"[SAE] HookedTransformer使用单GPU模式")
                hooked_model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device=device)
            print(f"[SAE] 成功加载HookedTransformer: meta-llama/Meta-Llama-3-8B-Instruct")
        except Exception as e:
            print(f"[SAE] 无法加载HookedTransformer: {e}")
            raise RuntimeError("图像SAE分析需要HookedTransformer，请安装transformer-lens")

        # Tokens + cfg per tutorial
        prompt = "Describe the image."
        tokens = hooked_model.to_tokens(prompt)

        # Hook metadata per tutorial
        hook_name = sae_id  # hook_point = sae_id string
        print(f"[SAE] 使用hook点: {hook_name}")

        # Hidden states + sparsity
        target_hidden = hooked_model.run_with_cache(tokens)[1][hook_name]
        feature_acts = sae.encode(target_hidden)
        reconstructed = sae.decode(feature_acts)
        
        # Sparsity metrics
        local_sparsity = (feature_acts == 0).float().mean().item()
        l0 = (feature_acts > 0).float().sum(-1).detach()
        print(f"[SAE] 图像分析 - 局部稀疏性: {local_sparsity:.4f}, 平均L0: {l0.mean().item():.2f}")
        
        # Active features from existing feature_acts
        print(f"[SAE] 基于稀疏性分析直接识别图像活跃特征...")
        print(f"[SAE] feature_acts shape: {feature_acts.shape}")
        print(f"[SAE] feature_acts dtype: {feature_acts.dtype}")
        print(f"[SAE] feature_acts device: {feature_acts.device}")
        
        # Per-feature strength (L1)
        feature_activations = []
        total_features = feature_acts.shape[-1]
        print(f"[SAE] 总特征数: {total_features}")
        
        # feature_acts on correct device
        if feature_acts.device.type != device:
            feature_acts = feature_acts.to(device)
        
        for i in range(total_features):
            try:
                # Per-token strength for feature
                activation_strength = torch.norm(feature_acts[:, i], p=1).item()
                feature_activations.append((i, activation_strength))
            except Exception as e:
                print(f"[SAE] 计算特征 {i} 激活强度时出错: {e}")
                print(f"[SAE] feature_acts[:, {i}] shape: {feature_acts[:, i].shape}")
                raise
        
        # Sort by activation strength
        feature_activations.sort(key=lambda x: x[1], reverse=True)
        
        # top-k from local sparsity
        top_k = max(1, int((1 - local_sparsity) * total_features))
        top_k = 10
        top_k = min(top_k, total_features)  # Clamp top-k to feature count
        top_features = [idx for idx, _ in feature_activations[:top_k]]
        
        print(f"[SAE] 局部稀疏性: {local_sparsity:.4f}, 直接识别出前{top_k}个最活跃特征 ({(1-local_sparsity)*100:.2f}% 的激活特征)")

        # Save visualizations
        if out_prefix:
            # Path: results/explainability/task{id}/sae_attribution/
            fig_dir = os.path.join('results', 'explainability', f'task{task}', 'sae_attribution')
            os.makedirs(fig_dir, exist_ok=True)
            
            # Unique filename per cycle
            timestamp = int(time.time())
            cycle_suffix = f"_cycle{cycle_num}" if cycle_num is not None else f"_ts{timestamp}"
            
            # Optional top-feature subset
            if generate_top_features:
                # Viz top features only
                try:
                    print(f"[SAE] 开始生成图像最活跃特征子集可视化...")
                    
                    # Config for top features
                    top_config = SaeVisConfig(
                        hook_point=sae_id,
                        features=top_features,  # Analyze top activations only
                        minibatch_size_features=50,
                        minibatch_size_tokens=50,
                        verbose=False,
                        device=device,
                    )

                    # tokens [B,T] on CPU
                    if tokens.dim() == 1:
                        tokens_for_sae = tokens.unsqueeze(0).cpu()  # Add batch dim → CPU
                    else:
                        tokens_for_sae = tokens.cpu()
                    
                    # Keep SAE on CPU
                    sae_cpu = sae.cpu()
                    # Render top-feature viz
                    top_vis_data = SaeVisRunner(top_config).run(encoder=sae_cpu, model=hooked_model, tokens=tokens_for_sae)
                    
                    # Save top-feature figure
                    top_vis_file = os.path.join(fig_dir, f"{os.path.basename(out_prefix)}{cycle_suffix}_top_features.html")
                    save_feature_centric_vis(sae_vis_data=top_vis_data, filename=top_vis_file)
                    print(f"[SAE] 已生成图像最活跃特征可视化: {top_vis_file}")
                    
                    # Save top-feature list
                    top_features_file = os.path.join(fig_dir, f"{os.path.basename(out_prefix)}{cycle_suffix}_top_features.txt")
                    with open(top_features_file, 'w') as f:
                        f.write(f"Top {top_k} Most Active Features (Image Analysis)\n")
                        f.write("=" * 50 + "\n")
                        f.write("Feature_ID\tActivation_Score\n")
                        for i, (feature_id, score) in enumerate(feature_activations[:top_k]):
                            f.write(f"{feature_id}\t{score:.6f}\n")
                        f.write("\n" + "=" * 50 + "\n")
                        f.write(f"Total features analyzed: {len(feature_activations)}\n")
                        f.write(f"Top features selected: {top_k}\n")
                        f.write(f"Average activation of top features: {np.mean([score for _, score in feature_activations[:top_k]]):.6f}\n")
                        f.write(f"Average activation of all features: {np.mean([score for _, score in feature_activations]):.6f}\n")
                        f.write(f"Local sparsity: {local_sparsity:.6f}\n")
                        f.write(f"Average L0: {l0.mean().item():.6f}\n")
                    
                    print(f"[SAE] 已保存图像最活跃特征列表: {top_features_file}")
                    
                    # Emit minimal HTML
                    try:
                        # Decode token strings
                        # Token tensor shape for decode
                        if tokens.dim() == 0:
                            tokens_for_decode = tokens.unsqueeze(0)
                        elif tokens.dim() == 1:
                            tokens_for_decode = tokens
                        else:
                            tokens_for_decode = tokens[0]
                        
                        token_texts = [tokenizer_or_processor.decode([token_id]) for token_id in tokens_for_decode.cpu().numpy()]
                        
                        # Feature activation tensors
                        feature_acts_np = feature_acts[0].detach().cpu().numpy()  # [seq_len, num_features]
                        
                        # Subset analyzed features only
                        selected_feature_acts = feature_acts_np[:, top_features]  # [seq_len, len(top_features)]
                        
                        # Minimal HTML export
                        simple_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SAE Feature {top_features[0]} - Lightweight View</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .token {{ display: inline-block; margin: 1px; padding: 2px 4px; border-radius: 3px; }}
        .feature-row {{ margin: 10px 0; }}
        .feature-label {{ font-weight: bold; margin-bottom: 5px; }}
        .activation-bar {{ height: 20px; background: linear-gradient(to right, #e0e0e0, #ff4444); border-radius: 3px; }}
        .max-activation {{ background: #ff0000; }}
        .min-activation {{ background: #e0e0e0; }}
        .stats {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>SAE Feature {top_features[0]} - Lightweight Analysis</h1>
    
    <div class="stats">
        <h3>Feature Statistics</h3>
        <p><strong>Feature ID:</strong> {top_features[0]}</p>
        <p><strong>Sequence Length:</strong> {len(token_texts)}</p>
        <p><strong>Max Activation:</strong> {selected_feature_acts[:, 0].max():.4f}</p>
        <p><strong>Min Activation:</strong> {selected_feature_acts[:, 0].min():.4f}</p>
        <p><strong>Mean Activation:</strong> {selected_feature_acts[:, 0].mean():.4f}</p>
        <p><strong>Active Tokens:</strong> {(selected_feature_acts[:, 0] > 0.1).sum()}</p>
    </div>
    
    <h3>Token Activations</h3>
    <div class="feature-row">
        <div class="feature-label">Feature {top_features[0]}:</div>
        <div style="font-family: monospace; font-size: 12px; line-height: 1.4;">
"""
                        
                        # Append per-token activations
                        for i, (token, activation) in enumerate(zip(token_texts, selected_feature_acts[:, 0])):
                            # Color intensity 0–1
                            max_act = selected_feature_acts[:, 0].max()
                            color_intensity = min(1.0, activation / max_act if max_act > 0 else 0)
                            
                            global_max_abs = abs(selected_feature_acts).max()
                            if global_max_abs > 0:
                                normalized_intensity = min(1.0, abs(activation) / global_max_abs)
                            else:
                                normalized_intensity = 0
                            
                            # Red gradient white→red
                            color_range = 0.97
                            intensity = abs(normalized_intensity)
                            r = 255
                            g = int(255 * (color_range - intensity))
                            b = int(255 * (color_range - intensity))
                            bg_color = f"rgb({r},{g},{b})"
                            
                            restored_token = token.replace('Ċ', '\n').replace('Ġ', ' ')
                            
                            if '\n' in restored_token:
                                parts = restored_token.split('\n')
                                for idx, part in enumerate(parts):
                                    if part:
                                        simple_html += f'<span class="token" style="background-color: {bg_color};" title="Token {i}: {activation:.4f}">{part}</span>'
                                    if idx < len(parts) - 1:
                                        simple_html += '<br/>'
                            else:
                                simple_html += f'<span class="token" style="background-color: {bg_color};" title="Token {i}: {activation:.4f}">{restored_token}</span>'
                        
                        simple_html += """
        </div>
    </div>
    
    <h3>Activation Distribution</h3>
    <div style="margin: 20px 0;">
        <p><strong>Top 10 Most Active Tokens:</strong></p>
        <div style="font-family: monospace; font-size: 12px;">
"""
                        
                        # List hottest tokens
                        top_indices = selected_feature_acts[:, 0].argsort()[-10:][::-1]
                        for i, idx in enumerate(top_indices):
                            activation = selected_feature_acts[idx, 0]
                            token = token_texts[idx]
                            simple_html += f'<div>{i+1:2d}. Token {idx:3d}: "{token}" (activation: {activation:.4f})</div>'
                        
                        simple_html += """
        </div>
    </div>
    
    <div style="margin-top: 30px; padding: 10px; background: #e8f4f8; border-radius: 5px;">
        <p><strong>Note:</strong> This is a lightweight view showing only the essential data for feature analysis. 
        For full interactive features, use the original SAE Dashboard.</p>
    </div>
</body>
</html>
"""
                        
                        # Persist minimal HTML
                        simple_html_file = os.path.join(fig_dir, f"{base_name}{cycle_suffix}_top{top_k}_features_simple.html")
                        with open(simple_html_file, 'w', encoding='utf-8') as f:
                            f.write(simple_html)
                        print(f"[SAE] 已生成超简化HTML: {simple_html_file}")
                        
                    except Exception as e:
                        print(f"[SAE] 超简化HTML生成失败: {e}")
                    
                    del top_vis_data
                    
                except Exception as e:
                    print(f"[SAE] 图像最活跃特征子集可视化生成失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"[SAE] 图像特征统计信息:")
            print(f"  - 总特征数: {total_features}")
            print(f"  - 激活特征数: {int((1 - local_sparsity) * total_features)}")
            print(f"  - 平均激活强度: {np.mean([score for _, score in feature_activations]):.4f}")

        # Reconstruction compare (reuse tensors)
        reconstruction_error = torch.norm(target_hidden - reconstructed, dim=-1).mean().item()
        
        print(f"[SAE] 重构误差: {reconstruction_error:.6f}")

        # Save comparison figure
        if out_prefix:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(img_array)
            axs[0].set_title("Original Image")
            axs[0].axis('off')
            
            try:
                # Try visualize reconstruction
                if reconstructed.shape[-1] == 3:  # RGB image
                    reconstructed_img = reconstructed.detach().cpu().numpy()[0]
                    axs[1].imshow(reconstructed_img)
                else:
                    axs[1].text(0.5, 0.5, "Reconstruction not image-shaped", ha='center', va='center')
            except Exception as e:
                axs[1].text(0.5, 0.5, f"Reconstruction error: {str(e)}", ha='center', va='center')
            
            axs[1].set_title(f"Reconstructed (Error: {reconstruction_error:.4f})")
            axs[1].axis('off')
            plt.tight_layout()
            
            # Save with unique filename
            timestamp = int(time.time())
            cycle_suffix = f"_cycle{cycle_num}" if cycle_num is not None else f"_ts{timestamp}"
            comparison_file = f"{out_prefix}{cycle_suffix}_reconstruction_comparison.png"
            plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[SAE] 已保存重构对比图: {comparison_file}")

        return reconstruction_error, img_array, reconstructed.detach().cpu().numpy()
        
    else:
        raise ValueError(f"input_data类型不支持: {type(input_data)}")
