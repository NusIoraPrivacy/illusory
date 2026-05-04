
"""SAE Configuration Management Module
Contains all supported models and corresponding SAE configurations"""

import os


def get_sae_config_for_model(model_name, model_path=None):
    """Automatically select the corresponding SAE configuration according to the model name"""
    model_name_lower = model_name.lower()
    
    # SAE config registry
    sae_configs = {
        # Llama-3-8B-Instruct config
        'llama-3.0-8b-instruct': {
            'release': 'llama-3-8b-it-res-jh',
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        'llama-3.1-8b-instruct': {
            'release': 'llama-3-8b-it-res-jh',
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        'llama-3.2-1b-instruct': {
            'release': 'llama-3-8b-it-res-jh',  # Approximate with Llama-3-8B SAE
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        'llama-3.2-3b-instruct': {
            'release': 'llama-3-8b-it-res-jh',  # Approximate with Llama-3-8B SAE
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        'llama-3.2-11b-vision-instruct': {
            'release': 'llama-3-8b-it-res-jh',  # Approximate with Llama-3-8B SAE
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        'llama-3.3-70b-instruct': {
            'release': 'llama-3-8b-it-res-jh',  # Approximate with Llama-3-8B SAE
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        'llama-4-scout-17b-16e-instruct': {
            'release': 'llama-3-8b-it-res-jh',  # Approximate with Llama-3-8B SAE
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        
        # Gemma-2-9B-IT — layer 9
        'gemma-2-9b-it-layer9-131k-l0-121': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_9/width_131k/average_l0_121',
            'target_layer': 9,
            'hook_name': 'blocks.9.hook_resid_post'
        },
        'gemma-2-9b-it-layer9-131k-l0-13': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_9/width_131k/average_l0_13',
            'target_layer': 9,
            'hook_name': 'blocks.9.hook_resid_post'
        },
        'gemma-2-9b-it-layer9-131k-l0-22': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_9/width_131k/average_l0_22',
            'target_layer': 9,
            'hook_name': 'blocks.9.hook_resid_post'
        },
        'gemma-2-9b-it-layer9-131k-l0-39': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_9/width_131k/average_l0_39',
            'target_layer': 9,
            'hook_name': 'blocks.9.hook_resid_post'
        },
        'gemma-2-9b-it-layer9-131k-l0-67': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_9/width_131k/average_l0_67',
            'target_layer': 9,
            'hook_name': 'blocks.9.hook_resid_post'
        },
        'gemma-2-9b-it-layer9-16k-l0-14': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_9/width_16k/average_l0_14',
            'target_layer': 9,
            'hook_name': 'blocks.9.hook_resid_post'
        },
        'gemma-2-9b-it-layer9-16k-l0-186': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_9/width_16k/average_l0_186',
            'target_layer': 9,
            'hook_name': 'blocks.9.hook_resid_post'
        },
        'gemma-2-9b-it-layer9-16k-l0-26': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_9/width_16k/average_l0_26',
            'target_layer': 9,
            'hook_name': 'blocks.9.hook_resid_post'
        },
        'gemma-2-9b-it-layer9-16k-l0-47': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_9/width_16k/average_l0_47',
            'target_layer': 9,
            'hook_name': 'blocks.9.hook_resid_post'
        },
        'gemma-2-9b-it-layer9-16k-l0-88': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_9/width_16k/average_l0_88',
            'target_layer': 9,
            'hook_name': 'blocks.9.hook_resid_post'
        },
        
        # Gemma-2-9B-IT — layer 20
        'gemma-2-9b-it-layer20-131k-l0-13': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_20/width_131k/average_l0_13',
            'target_layer': 20,
            'hook_name': 'blocks.20.hook_resid_post'
        },
        'gemma-2-9b-it-layer20-131k-l0-153': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_20/width_131k/average_l0_153',
            'target_layer': 20,
            'hook_name': 'blocks.20.hook_resid_post'
        },
        'gemma-2-9b-it-layer20-131k-l0-24': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_20/width_131k/average_l0_24',
            'target_layer': 20,
            'hook_name': 'blocks.20.hook_resid_post'
        },
        'gemma-2-9b-it-layer20-131k-l0-43': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_20/width_131k/average_l0_43',
            'target_layer': 20,
            'hook_name': 'blocks.20.hook_resid_post'
        },
        'gemma-2-9b-it-layer20-131k-l0-81': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_20/width_131k/average_l0_81',
            'target_layer': 20,
            'hook_name': 'blocks.20.hook_resid_post'
        },
        
        # Gemma-2-9B-IT layer20 width 16k
        'gemma-2-9b-it-layer20-16k-l0-14': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_20/width_16k/average_l0_14',
            'target_layer': 20,
            'hook_name': 'blocks.20.hook_resid_post'
        },
        'gemma-2-9b-it-layer20-16k-l0-189': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_20/width_16k/average_l0_189',
            'target_layer': 20,
            'hook_name': 'blocks.20.hook_resid_post'
        },
        'gemma-2-9b-it-layer20-16k-l0-25': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_20/width_16k/average_l0_25',
            'target_layer': 20,
            'hook_name': 'blocks.20.hook_resid_post'
        },
        'gemma-2-9b-it-layer20-16k-l0-47': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_20/width_16k/average_l0_47',
            'target_layer': 20,
            'hook_name': 'blocks.20.hook_resid_post'
        },
        'gemma-2-9b-it-layer20-16k-l0-91': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_20/width_16k/average_l0_91',
            'target_layer': 20,
            'hook_name': 'blocks.20.hook_resid_post'
        },

        # Gemma-2-9B-IT — layer 31
        'gemma-2-9b-it-layer31-131k-l0-109': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_131k/average_l0_109',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        'gemma-2-9b-it-layer31-131k-l0-13': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_131k/average_l0_13',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        'gemma-2-9b-it-layer31-131k-l0-22': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_131k/average_l0_22',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        'gemma-2-9b-it-layer31-131k-l0-37': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_131k/average_l0_37',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        'gemma-2-9b-it-layer31-131k-l0-63': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_131k/average_l0_63',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        
        # Gemma-2-9B-IT layer31 width 16k
        'gemma-2-9b-it-layer31-16k-l0-14': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_16k/average_l0_14',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        'gemma-2-9b-it-layer31-16k-l0-142': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_16k/average_l0_142',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        'gemma-2-9b-it-layer31-16k-l0-24': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_16k/average_l0_24',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        'gemma-2-9b-it-layer31-16k-l0-43': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_16k/average_l0_43',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        'gemma-2-9b-it-layer31-16k-l0-76': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_16k/average_l0_76',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        
        # Gemma-2-9B-IT default (layer9 width131k l0_13)
        'gemma-2-9b-it': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_16k/average_l0_14',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        
        # Qwen2.5-7B-Instruct config
        # Note: Qwen ≠ Llama; no public Qwen SAE
        # Fallback Llama-3-8B SAE (approximate)
        # Update here when Qwen SAE releases
        'qwen2.5-7b-instruct': {
            'release': 'llama-3-8b-it-res-jh',
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        
        # DeepSeek-R1-Distill-Llama-8B config
        # llama_scope_r1_distill release, layer 15
        # Model name variants supported
        'deepseek-r1-distill-llama-8b': {
            'release': 'llama-3-8b-it-res-jh',
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
    }
    
    # Exact match
    if model_name_lower in sae_configs:
        config = sae_configs[model_name_lower].copy()
        print(f"[SAE] Found exact matching SAE configuration:{model_name} -> {config['release']}")
        return config
    
    # Fuzzy match
    for key, config in sae_configs.items():
        if key in model_name_lower or model_name_lower in key:
            config = config.copy()
            print(f"[SAE] Fuzzy matching SAE configuration found:{model_name} -> {config['release']}")
            return config
    
    # Default Llama-3-8B config
    default_config = sae_configs['llama-3.1-8b-instruct'].copy()
    print(f"[SAE] No matching SAE configuration found, use default configuration:{model_name} -> {default_config['release']}")
    return default_config

def get_available_gemma_configs():
    """Get all available Gemma SAE configurations"""
    gemma_configs = []
    for key in get_sae_config_for_model.__code__.co_names:
        if 'gemma-2-9b-it' in key:
            gemma_configs.append(key)
    return gemma_configs

def load_gemma_sae_from_local(npz_file_path, device="cpu"):
    """Load Gemma SAE from local npz file
    
    Args:
        npz_file_path: params.npz file path
        device: Device (cpu or cuda)
    
    Returns:
        tuple: (sae, cfg_dict, feature_sparsity) or None"""
    try:
        import numpy as np
        import torch
        
        # Try multiple import paths
        try:
            from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
        except ImportError:
            try:
                from sae_lens.standard_sae import StandardSAE, StandardSAEConfig
            except ImportError:
                try:
                    # Import from sae_lens
                    from sae_lens import SAE as StandardSAE, SAEConfig as StandardSAEConfig
                except ImportError:
                    # Fallback base class import
                    from sae_lens import SAE, SAEConfig
                    StandardSAE = SAE
                    StandardSAEConfig = SAEConfig
        
        print(f"[SAE] Load Gemma SAE from local:{npz_file_path}")
        
        # Load npz weights
        data = np.load(npz_file_path, allow_pickle=True)
        print(f"[SAE] Successfully loaded npz file containing:{list(data.keys())}")
        
        # Extract SAE tensors
        W_enc = torch.tensor(data['W_enc'], dtype=torch.float32, device=device)
        W_dec = torch.tensor(data['W_dec'], dtype=torch.float32, device=device)
        b_enc = torch.tensor(data['b_enc'], dtype=torch.float32, device=device)
        b_dec = torch.tensor(data['b_dec'], dtype=torch.float32, device=device)
        # threshold = torch.tensor(...)  # StandardSAE has no threshold field
        
        print(f"[SAE] SAE Parameter Dimension: d_in ={W_enc.shape[0]}, d_sae={W_enc.shape[1]}, d_out={W_dec.shape[0]}")
        
        # Build SAEConfig env-specific
        try:
            # Try StandardSAEConfig (newer)
            cfg = StandardSAEConfig(
                d_in=W_enc.shape[0],  # 3584
                d_sae=W_enc.shape[1],  # 131072
                dtype='float32',
                device=device,
                apply_b_dec_to_input=True,
                normalize_activations='none',
                reshape_activations='none'
            )
        except TypeError:
            # Else legacy SAEConfig
            cfg = StandardSAEConfig(
                architecture='standard',
                d_in=W_enc.shape[0],  # 3584
                d_sae=W_enc.shape[1],  # 131072
                activation_fn_str='relu',
                apply_b_dec_to_input=True,
                finetuning_scaling_factor=False,
                context_size=2048,
                model_name='gemma-2-9b-it',
                hook_name='blocks.9.hook_resid_post',
                hook_layer=9,
                hook_head_index=None,
                prepend_bos=True,
                dataset_path='',
                dataset_trust_remote_code=False,
                normalize_activations='none',
                dtype='float32',
                device=device,
                sae_lens_training_version=None
            )
        
        # Instantiate SAE
        sae = StandardSAE(cfg)
        
        # Assign weights manually
        sae.W_enc.data = W_enc
        sae.W_dec.data = W_dec
        sae.b_enc.data = b_enc
        sae.b_dec.data = b_dec
        # sae.threshold.data = threshold  # StandardSAE lacks threshold
        
        # Config dict per environment
        cfg_dict = {
            'd_in': cfg.d_in,
            'd_sae': cfg.d_sae,
            'dtype': cfg.dtype,
            'device': cfg.device,
            'apply_b_dec_to_input': cfg.apply_b_dec_to_input,
            'normalize_activations': cfg.normalize_activations
        }
        
        # Optional reshape_activations
        if hasattr(cfg, 'reshape_activations'):
            cfg_dict['reshape_activations'] = cfg.reshape_activations
        else:
            cfg_dict['reshape_activations'] = 'none'  # Default value
        
        # Match SAE.from_pretrained() return shape
        sae_result = (sae, cfg_dict, None)  # (sae, cfg_dict, feature_sparsity)
        
        print(f"[SAE] ✅ Successfully created SAE object from local npz file")
        data.close()
        
        return sae_result
        
    except Exception as e:
        print(f"[SAE] Failed to load local npz file:{e}")
        import traceback
        traceback.print_exc()
        return None

def generate_simple_html_visualization(token_texts, selected_feature_acts, top_features, base_name, cycle_suffix, fig_dir, top_k):
    """Generate ultra-simplified HTML visualizations"""
    try:
        print(f"[SAE] Start generating ultra-simplified HTML, number of features:{len(top_features)}")
        
        # Minimal HTML export
        simple_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SAE Features Analysis - Lightweight View</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f8f9fa; }}
        .header {{ position: relative; margin-bottom: 40px; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header h1 {{ font-size: 24px; color: #2c3e50; margin: 0 0 15px 0; }}
        .feature-selector {{ position: absolute; top: 25px; right: 25px; display: flex; align-items: center; gap: 10px; }}
        .feature-selector label {{ font-size: 14px; font-weight: 600; color: #34495e; margin-right: 10px; }}
        .feature-selector select {{ padding: 10px 14px; border-radius: 8px; border: 2px solid #3498db; font-size: 14px; background: white; cursor: pointer; }}
        .feature-selector select:focus {{ outline: none; border-color: #2980b9; box-shadow: 0 0 8px rgba(52, 152, 219, 0.3); }}
        .nav-button {{ padding: 8px 12px; border-radius: 6px; border: 2px solid #3498db; background: white; color: #3498db; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; }}
        .nav-button:hover {{ background: #3498db; color: white; }}
        .nav-button:disabled {{ opacity: 0.5; cursor: not-allowed; border-color: #bdc3c7; color: #bdc3c7; }}
        .token {{ display: inline-block; margin: 1px; padding: 2px 6px; border-radius: 6px; border: 1px solid #ddd; font-size: 11px; font-weight: 500; }}
        .feature-section {{ margin: 25px 0; padding: 25px; border: 1px solid #e1e8ed; border-radius: 12px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .feature-label {{ font-weight: bold; margin-bottom: 20px; color: #2c3e50; font-size: 22px; }}
        .stats {{ margin: 15px 0; padding: 20px; background: #ecf0f1; border-radius: 10px; border-left: 4px solid #3498db; }}
        .stats p {{ font-size: 14px; margin: 8px 0; color: #2c3e50; }}
        .stats strong {{ color: #34495e; }}
        .color-legend {{ margin: 25px 0; padding: 25px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .color-legend h3 {{ font-size: 20px; color: #2c3e50; margin-bottom: 20px; }}
        .legend-item {{ display: block; margin: 15px 0; width: 100%; }}
        .legend-color {{ display: block; width: 100%; height: 35px; border: 2px solid #999; border-radius: 6px; }}
        .token-container {{ font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 9px; line-height: 1.8; margin: 15px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .activation-list {{ font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 13px; margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .activation-list div {{ margin: 5px 0; padding: 5px 10px; background: white; border-radius: 4px; }}
        .feature-page {{ display: none; }}
        .feature-page.active {{ display: block; }}
        h4 {{ font-size: 20px; color: #34495e; margin: 20px 0 15px 0; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
        .note-box {{ margin-top: 25px; padding: 20px; background: #f5f5f5; color: #333; border-radius: 10px; border: 1px solid #ddd; }}
        .note-box p {{ font-size: 14px; margin: 0; line-height: 1.6; }}
        .note-box strong {{ color: #2c3e50; }}
    </style>
    <script>
        function showFeature(featureId) {{
            const pages = document.querySelectorAll('.feature-page');
            pages.forEach(page => page.classList.remove('active'));
            
            const selectedPage = document.getElementById('feature-' + featureId);
            if (selectedPage) {{
                selectedPage.classList.add('active');
            }}
            
            document.title = 'SAE Feature ' + featureId + ' Analysis';
            updateNavButtons();
        }}
        
        function nextFeature() {{
            const select = document.querySelector('#feature-select');
            const currentIndex = select.selectedIndex;
            if (currentIndex < select.options.length - 1) {{
                select.selectedIndex = currentIndex + 1;
                showFeature(select.value);
            }}
        }}
        
        function prevFeature() {{
            const select = document.querySelector('#feature-select');
            const currentIndex = select.selectedIndex;
            if (currentIndex > 0) {{
                select.selectedIndex = currentIndex - 1;
                showFeature(select.value);
            }}
        }}
        
        function updateNavButtons() {{
            const select = document.querySelector('#feature-select');
            const prevBtn = document.querySelector('#prev-btn');
            const nextBtn = document.querySelector('#next-btn');
            
            prevBtn.disabled = select.selectedIndex === 0;
            nextBtn.disabled = select.selectedIndex === select.options.length - 1;
        }}
        
        window.onload = function() {{
            const firstFeature = document.querySelector('.feature-page');
            if (firstFeature) {{
                firstFeature.classList.add('active');
                const featureId = firstFeature.id.replace('feature-', '');
                document.querySelector('#feature-select').value = featureId;
                updateNavButtons();
            }}
        }};
    </script>
</head>
<body>
    <div class="header">
        <h1>SAE Features Analysis - Lightweight View</h1>
        <div class="feature-selector">
            <button id="prev-btn" class="nav-button" onclick="prevFeature()">◀ Prev</button>
            <label for="feature-select">Select Feature: </label>
            <select id="feature-select" onchange="showFeature(this.value)">
                {''.join([f'<option value="{feature_id}">Feature {feature_id}</option>' for feature_id in top_features])}
            </select>
            <button id="next-btn" class="nav-button" onclick="nextFeature()">Next ▶</button>
        </div>
    </div>
    
    <div class="color-legend">
        <h3>Activation Level Legend</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 15px; margin: 15px 0;">
            <div style="display: flex; align-items: center; margin-right: 25px;">
                <div style="width: 28px; height: 28px; background-color: #ffffff; border: 2px solid #999; border-radius: 4px; margin-right: 12px;"></div>
                <span style="font-size: 16px; font-weight: 500;">Very Low (0-20%)</span>
            </div>
            <div style="display: flex; align-items: center; margin-right: 25px;">
                <div style="width: 28px; height: 28px; background-color: #ffcccc; border: 2px solid #999; border-radius: 4px; margin-right: 12px;"></div>
                <span style="font-size: 16px; font-weight: 500;">Low (20-40%)</span>
            </div>
            <div style="display: flex; align-items: center; margin-right: 25px;">
                <div style="width: 28px; height: 28px; background-color: #ff9999; border: 2px solid #999; border-radius: 4px; margin-right: 12px;"></div>
                <span style="font-size: 16px; font-weight: 500;">Medium (40-60%)</span>
            </div>
            <div style="display: flex; align-items: center; margin-right: 25px;">
                <div style="width: 28px; height: 28px; background-color: #ff6666; border: 2px solid #999; border-radius: 4px; margin-right: 12px;"></div>
                <span style="font-size: 16px; font-weight: 500;">High (60-80%)</span>
            </div>
            <div style="display: flex; align-items: center; margin-right: 25px;">
                <div style="width: 28px; height: 28px; background-color: #ff3333; border: 2px solid #999; border-radius: 4px; margin-right: 12px;"></div>
                <span style="font-size: 16px; font-weight: 500;">Very High (80-100%)</span>
            </div>
        </div>
    </div>
"""
        
        print(f"[SAE] HTML header generation complete, start generating feature page...")
        
        # Viz each feature
        for feat_idx, feature_id in enumerate(top_features):
            print(f"[SAE] Generating features{feature_id}Progress:{feat_idx+1}/{len(top_features)})")
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
            
            <h4>Top 10 Most Active Tokens</h4>
            <div class="activation-list">
"""
            
            # Top tokens section first
            top_indices = feature_acts.argsort()[-10:][::-1]
            for i, idx in enumerate(top_indices):
                activation = feature_acts[idx]
                token = token_texts[idx]
                simple_html += f'<div>{i+1:2d}. Token {idx:3d}: "{token}" (activation: {activation:.4f})</div>'
            
            simple_html += """
            </div>
            
            <h4>Token Activations</h4>
            <div class="token-container">
"""
            
            # Append per-token activations
            for i, (token, activation) in enumerate(zip(token_texts, feature_acts)):
                # Compute color intensity
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
                
                intensity = abs(normalized_intensity)
                
                if intensity < 0.2:
                    bg_color = "#ffffff"
                elif intensity < 0.4:
                    bg_color = "#ffcccc"
                elif intensity < 0.6:
                    bg_color = "#ff9999"
                elif intensity < 0.8:
                    bg_color = "#ff6666"
                else:
                    bg_color = "#ff3333"
                
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
            
            <div class="note-box">
                <p><strong>Note:</strong> This is a lightweight view showing all analyzed features. 
                Colors range from white (low activation) to red (high activation).</p>
            </div>
        </div>
    </div>
"""
            print(f"[SAE] Characteristics{feature_id}Build Complete")
        
        # Close HTML tags
        simple_html += """
</body>
</html>
"""
        
        print(f"[SAE] HTML generation complete, start saving file...")
        
        # Persist minimal HTML
        simple_html_file = os.path.join(fig_dir, f"{base_name}{cycle_suffix}_top{top_k}_features_simple.html")
        with open(simple_html_file, 'w', encoding='utf-8') as f:
            f.write(simple_html)
        print(f"[SAE] Generated super-simplified HTML:{simple_html_file}")
        
        return simple_html_file
        
    except Exception as e:
        print(f"[SAE] Hyper-Simplified HTML Generation Failed:{e}")
        import traceback
        traceback.print_exc()
        return None

def load_sae_model(sae_release, sae_id, sae_device="cuda:0"):
    """Load SAE models with multiple loading methods
    
    Args:
        sae_release: SAE Release Version
        sae_id: SAE ID
        sae_device: SAE device
    
    Returns:
        tuple: (sae, cfg_dict, feature_sparsity) or None"""
    try:
        from sae_lens import SAE
        
        print(f"[SAE] Start loading SAE model:{sae_release}/{sae_id}")
        print(f"[SAE] Target Device:{sae_device}")
        
        sae = None
        cfg_dict = {'d_in': 768}
        feature_sparsity = None
        
        # Method 1: local SAE mirror
        sae_base_dir = "/workspace/illusory/huggingface/SAEs"
        
        # HF repo name vs local folder
        # HF llama-3-8b-it-res-jh → local llama-3-8b-it-res
        local_release = sae_release
        if sae_release == "llama-3-8b-it-res-jh":
            local_release = "llama-3-8b-it-res"
        elif sae_release == "llama_scope_r1_distill":
            # llama_scope_r1_distill → Llama-Scope-R1-Distill locally
            local_release = "llama-scope-r1-distill"  # Try common local folder names
        elif sae_release in ["llama_scope_lxr_8x", "llama_scope_lxr_32x", "llama_scope_lxa_8x", "llama_scope_lxa_32x", "llama_scope_lxm_8x", "llama_scope_lxm_32x"]:
            # Llama Scope dirs vary
            local_release = sae_release.replace("_", "-")  # Replace underscores with hyphens
        
        local_sae_path = os.path.join(sae_base_dir, local_release)
        sae_config_path = os.path.join(local_sae_path, sae_id)
        
        # Probe SAE file layouts
        sae_files_exist = False
        if os.path.exists(sae_config_path):
            # Llama layout: cfg.json + safetensors
            cfg_file = os.path.join(sae_config_path, "cfg.json")
            safetensors_file = os.path.join(sae_config_path, "sae_weights.safetensors")
            if os.path.exists(cfg_file) and os.path.exists(safetensors_file):
                sae_files_exist = True
                print(f"[SAE] Found SAE file in Llama format: cfg.json + sae_weights.safetensors")
            else:
                # Gemma layout: params.npz
                npz_file = os.path.join(sae_config_path, "params.npz")
                if os.path.exists(npz_file):
                    sae_files_exist = True
                    print(f"[SAE] Found SAE file in Gemma format: params.npz")
        
        if sae_files_exist:
            try:
                print(f"[SAE] Attempt to load SAE model from local path:{sae_config_path}")
                
                # Loader per format
                if os.path.exists(os.path.join(sae_config_path, "cfg.json")):
                    # Llama: load_from_disk
                    print(f"[SAE] Load method using Llama format")
                    sae_result = SAE.load_from_disk(sae_config_path, device=sae_device)
                else:
                    # Gemma: custom npz loader
                    print(f"[SAE] Load method using Gemma format")
                    npz_file = os.path.join(sae_config_path, "params.npz")
                    if os.path.exists(npz_file):
                        sae_result = load_gemma_sae_from_local(npz_file, sae_device)
                    else:
                        print(f"[SAE] Local npz file does not exist:{npz_file}")
                        sae_result = None
                
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
                    
                print(f"[SAE] Successfully loaded SAE model from local path:{sae}")
                
            except Exception as e:
                print(f"[SAE] Failed to load local path:{str(e)}")
                sae = None
        
        # Method 2: HF cache
        if sae is None:
            standard_cache_path = f"/root/.cache/huggingface/hub/models--{sae_release.replace('/', '--')}"
            if os.path.exists(standard_cache_path):
                try:
                    print(f"[SAE] Attempt to load SAE model from standard cache:{standard_cache_path}")
                    
                    sae_result = SAE.from_pretrained(
                        release=sae_release,
                        sae_id=sae_id,
                        device=sae_device
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
                        
                    print(f"[SAE] Successfully loaded SAE model from standard cache:{sae}")
                    
                except Exception as e:
                    print(f"[SAE] Standard Cache Load Failed:{str(e)}")
                    sae = None
        
        # Method 3: HF download
        if sae is None:
            try:
                print(f"[SAE] Try downloading the SAE model from HuggingFace:{sae_release}/{sae_id}")
                
                sae_result = SAE.from_pretrained(
                    release=sae_release,
                    sae_id=sae_id,
                    device=sae_device
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
                
                print(f"[SAE] Successfully loaded SAE model from HuggingFace:{sae}")
                
            except Exception as e:
                print(f"[SAE] HuggingFace Download Failed:{str(e)}")
                print(f"[SAE] Unable to download SAE model due to network issues, skipping SAE analysis")
                print(f"[SAE] returned empty result, program continues to run")
                return None
        
        if sae is None:
            print(f"[SAE] Unable to load any SAE models, skipping SAE analysis")
            return None
        
        return (sae, cfg_dict, feature_sparsity)
        
    except ImportError:
        print(f"[SAE] sae_lens not installed, unable to load SAE model")
        return None
    except Exception as e:
        print(f"[SAE] An error occurred during SAE loading:{e}")
        import traceback
        traceback.print_exc()
        return None

def load_hooked_transformer_for_sae(model_name, hooked_device="cuda:0"):
    """Load HookedTransformer model for SAE analysis
    
    Args:
        model_name: Model name
        hooked_device: HookedTransformer device
    
    Returns:
        HookedTransformer object or None"""
    try:
        from transformer_lens import HookedTransformer
        
        print(f"[SAE] Start loading HookedTransformer for SAE analysis...")
        print(f"[SAE] Model Name:{model_name}")
        print(f"[SAE] Target Device:{hooked_device}")
        
        hooked_model = None
        
        # HookedTransformer: local snapshot
        try:
            print(f"[SAE] Try using a locally downloaded HookedTransformer model...")
            
            # Local path per model family
            sae_base_dir = "/workspace/illusory/huggingface/SAEs"
            if 'gemma' in model_name.lower():
                # IT HookedTransformer ↔ IT SAE
                hooked_model_path = os.path.join(sae_base_dir, "gemma-scope-9b-it-res", "hooked-transformer-it")
                model_name_for_hf = "google/gemma-2-9b-it"  # Instruct checkpoint
                print(f"[SAE] Gemma model detected, using IT version of HookedTransformer:{hooked_model_path}")
                print(f"[SAE] Align with IT version SAE using IT version HookedTransformer")
            else:
                hooked_model_path = os.path.join(sae_base_dir, "llama-3-8b-it-res", "hooked-transformer")
                model_name_for_hf = "meta-llama/Meta-Llama-3-8B-Instruct"  # Matching HF model id
                print(f"[SAE] Using local Llama HookedTransformer:{hooked_model_path}")
            
            # Verify local path
            print(f"[SAE] Check local path:{hooked_model_path}")
            print(f"[SAE] Path Exists:{os.path.exists(hooked_model_path)}")
            
            if os.path.exists(hooked_model_path):
                print(f"[SAE] Local HookedTransformer path exists, attempting to load...")
                
                # HF causal LM → HookedTransformer
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                print(f"[SAE] Load HuggingFace model from local path:{hooked_model_path}")
                try:
                    # Load HF weights on CPU first
                    print(f"[SAE] Load HuggingFace model with CPU to avoid memory conflicts")
                    
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        hooked_model_path,
                        device_map=None,  # Use CPU
                        torch_dtype="auto",
                        low_cpu_mem_usage=True
                    )
                    print(f"[SAE] HuggingFace model loaded successfully")
                    
                    # HookedTransformer.from_pretrained(..., hf_model=)
                    print(f"[SAE] Start creating HookedTransformer...")
                    hooked_model = HookedTransformer.from_pretrained(
                        model_name_for_hf,
                        hf_model=hf_model,
                        device=hooked_device
                    )
                    print(f"[SAE] HookedTransformer ✅ successfully created from local path:{hooked_model_path}")
                except Exception as inner_e:
                    print(f"[SAE] Error during local model loading:{inner_e}")
                    if "CUDA out of memory" in str(inner_e):
                        print(f"[SAE] Out of CUDA memory detected, skipping HookedTransformer creation")
                        print(f"[SAE] Recommendation: Release other GPU processes or use fewer GPUs")
                        hooked_model = None
                    else:
                        import traceback
                        traceback.print_exc()
                        raise inner_e
            else:
                print(f"[SAE] Local HookedTransformer path does not exist:{hooked_model_path}")
                raise Exception("Local HookedTransformer path does not exist")
        except Exception as e:
            print(f"[SAE] Failed to create local HookedTransformer:{e}")
            hooked_model = None
        
        # Fallback HF HookedTransformer
        if hooked_model is None:
            try:
                print(f"[SAE] Try downloading HookedTransformer from HuggingFace...")
                
                # HookedTransformer id per family
                if 'gemma' in model_name.lower():
                    hooked_model_name = "google/gemma-2-9b"  # Base weights align with SAE training
                    print(f"[SAE] Gemma model detected, using HookedTransformer:{hooked_model_name}")
                else:
                    hooked_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
                    print(f"[SAE] Use the default HookedTransformer:{hooked_model_name}")
                
                hooked_model = HookedTransformer.from_pretrained(
                    hooked_model_name, 
                    device=hooked_device
                )
                print(f"[SAE] ✅ Successfully created HookedTransformer from HuggingFace:{hooked_model_name}")
            except Exception as e:
                print(f"[SAE] HuggingFace creation failed:{e}")
                hooked_model = None
        
        
        # Verbose errors if all loaders fail
        if hooked_model is None:
            print(f"[SAE] ⚠️ All HookedTransformer creation methods failed")
            print(f"[SAE] Model Name:{model_name}")
            print(f"[SAE] Target Device:{hooked_device}")
            print(f"[SAE] Keep trying to use the original model for SAE analysis...")
        else:
            print(f"[SAE] ✅ HookedTransformer created successfully")
        
        return hooked_model
        
    except ImportError:
        print(f"[SAE] transformer_lens not installed, unable to load HookedTransformer")
        return None
    except Exception as e:
        print(f"[SAE] Error during loading of HookedTransformer:{e}")
        import traceback
        traceback.print_exc()
        return None

def list_all_sae_configs():
    """List all available SAE configurations"""
    return {
        'llama_models': ['llama-3.0-8b-instruct', 'llama-3.1-8b-instruct', 'llama-3.2-1b-instruct', 
                        'llama-3.2-3b-instruct', 'llama-3.2-11b-vision-instruct', 'llama-3.3-70b-instruct', 
                        'llama-4-scout-17b-16e-instruct'],
        'gemma_models': ['gemma-2-9b-it', 'gemma-2-9b-it-layer9-*', 'gemma-2-9b-it-layer20-*', 'gemma-2-9b-it-layer31-*'],
    }
