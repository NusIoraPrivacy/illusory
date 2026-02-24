
"""
SAE配置管理模块
包含所有支持的模型和对应的SAE配置
"""

import os


def get_sae_config_for_model(model_name, model_path=None):
    """
    根据模型名称自动选择对应的SAE配置
    """
    model_name_lower = model_name.lower()
    
    # SAE配置映射
    sae_configs = {
        # Llama-3-8B-Instruct 配置
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
            'release': 'llama-3-8b-it-res-jh',  # 使用Llama-3-8B的SAE作为近似
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        'llama-3.2-3b-instruct': {
            'release': 'llama-3-8b-it-res-jh',  # 使用Llama-3-8B的SAE作为近似
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        'llama-3.2-11b-vision-instruct': {
            'release': 'llama-3-8b-it-res-jh',  # 使用Llama-3-8B的SAE作为近似
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        'llama-3.3-70b-instruct': {
            'release': 'llama-3-8b-it-res-jh',  # 使用Llama-3-8B的SAE作为近似
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        'llama-4-scout-17b-16e-instruct': {
            'release': 'llama-3-8b-it-res-jh',  # 使用Llama-3-8B的SAE作为近似
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        
        # Gemma-2-9B-IT 配置 - Layer 9
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
        
        # Gemma-2-9B-IT 配置 - Layer 20
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
        
        # Gemma-2-9B-IT Layer 20 width_16k 配置
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

        # Gemma-2-9B-IT 配置 - Layer 31
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
        
        # Gemma-2-9B-IT Layer 31 width_16k 配置
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
        
        # Gemma-2-9B-IT 默认配置（使用layer 9, width 131k, l0_13）
        'gemma-2-9b-it': {
            'release': 'gemma-scope-9b-it-res',
            'sae_id': 'layer_31/width_16k/average_l0_14',
            'target_layer': 31,
            'hook_name': 'blocks.31.hook_resid_post'
        },
        
        # Qwen2.5-7B-Instruct 配置
        # 注意：Qwen 架构与 Llama 不同，目前没有公开的 Qwen 专用 SAE
        # 因此使用 Llama-3-8B 的 SAE 作为近似（可能不完全兼容，仅作参考）
        # 如果将来有 Qwen 专用的 SAE（如 qwen-scope-7b-instruct-res），可以在这里更新
        'qwen2.5-7b-instruct': {
            'release': 'llama-3-8b-it-res-jh',
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
        
        # DeepSeek-R1-Distill-Llama-8B 配置
        # 使用 llama_scope_r1_distill release，Layer 15
        # 注意：模型名称可能有多种变体（deepseek-r1-distill-llama-8b, deepseek-r1-llama-8b）
        'deepseek-r1-distill-llama-8b': {
            'release': 'llama-3-8b-it-res-jh',
            'sae_id': 'blocks.25.hook_resid_post',
            'target_layer': 25,
            'hook_name': 'blocks.25.hook_resid_post'
        },
    }
    
    # 尝试精确匹配
    if model_name_lower in sae_configs:
        config = sae_configs[model_name_lower].copy()
        print(f"[SAE] 找到精确匹配的SAE配置: {model_name} -> {config['release']}")
        return config
    
    # 尝试模糊匹配
    for key, config in sae_configs.items():
        if key in model_name_lower or model_name_lower in key:
            config = config.copy()
            print(f"[SAE] 找到模糊匹配的SAE配置: {model_name} -> {config['release']}")
            return config
    
    # 默认使用Llama-3-8B的配置
    default_config = sae_configs['llama-3.1-8b-instruct'].copy()
    print(f"[SAE] 未找到匹配的SAE配置，使用默认配置: {model_name} -> {default_config['release']}")
    return default_config

def get_available_gemma_configs():
    """
    获取所有可用的Gemma SAE配置
    """
    gemma_configs = []
    for key in get_sae_config_for_model.__code__.co_names:
        if 'gemma-2-9b-it' in key:
            gemma_configs.append(key)
    return gemma_configs

def load_gemma_sae_from_local(npz_file_path, device="cpu"):
    """
    从本地npz文件加载Gemma SAE
    
    Args:
        npz_file_path: params.npz文件路径
        device: 设备（cpu或cuda）
    
    Returns:
        tuple: (sae, cfg_dict, feature_sparsity) 或 None
    """
    try:
        import numpy as np
        import torch
        
        # 尝试多种导入方式
        try:
            from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
        except ImportError:
            try:
                from sae_lens.standard_sae import StandardSAE, StandardSAEConfig
            except ImportError:
                try:
                    # 尝试从sae_lens直接导入
                    from sae_lens import SAE as StandardSAE, SAEConfig as StandardSAEConfig
                except ImportError:
                    # 最后尝试导入基础类
                    from sae_lens import SAE, SAEConfig
                    StandardSAE = SAE
                    StandardSAEConfig = SAEConfig
        
        print(f"[SAE] 从本地加载Gemma SAE: {npz_file_path}")
        
        # 加载npz文件
        data = np.load(npz_file_path, allow_pickle=True)
        print(f"[SAE] 成功加载npz文件，包含: {list(data.keys())}")
        
        # 提取SAE参数
        W_enc = torch.tensor(data['W_enc'], dtype=torch.float32, device=device)
        W_dec = torch.tensor(data['W_dec'], dtype=torch.float32, device=device)
        b_enc = torch.tensor(data['b_enc'], dtype=torch.float32, device=device)
        b_dec = torch.tensor(data['b_dec'], dtype=torch.float32, device=device)
        # threshold = torch.tensor(data['threshold'], dtype=torch.float32, device=device)  # StandardSAE没有threshold属性
        
        print(f"[SAE] SAE参数维度: d_in={W_enc.shape[0]}, d_sae={W_enc.shape[1]}, d_out={W_dec.shape[0]}")
        
        # 创建SAE配置 - 根据环境使用不同的参数
        try:
            # 尝试使用StandardSAEConfig（较新版本）
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
            # 如果StandardSAEConfig失败，使用基础SAEConfig（需要更多参数）
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
        
        # 创建SAE对象
        sae = StandardSAE(cfg)
        
        # 手动设置权重
        sae.W_enc.data = W_enc
        sae.W_dec.data = W_dec
        sae.b_enc.data = b_enc
        sae.b_dec.data = b_dec
        # sae.threshold.data = threshold  # StandardSAE没有threshold属性
        
        # 创建配置字典 - 根据环境使用不同的属性
        cfg_dict = {
            'd_in': cfg.d_in,
            'd_sae': cfg.d_sae,
            'dtype': cfg.dtype,
            'device': cfg.device,
            'apply_b_dec_to_input': cfg.apply_b_dec_to_input,
            'normalize_activations': cfg.normalize_activations
        }
        
        # 尝试添加reshape_activations属性（如果存在）
        if hasattr(cfg, 'reshape_activations'):
            cfg_dict['reshape_activations'] = cfg.reshape_activations
        else:
            cfg_dict['reshape_activations'] = 'none'  # 默认值
        
        # 返回与SAE.from_pretrained()相同的格式
        sae_result = (sae, cfg_dict, None)  # (sae, cfg_dict, feature_sparsity)
        
        print(f"[SAE] ✅ 成功从本地npz文件创建SAE对象")
        data.close()
        
        return sae_result
        
    except Exception as e:
        print(f"[SAE] 本地npz文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_simple_html_visualization(token_texts, selected_feature_acts, top_features, base_name, cycle_suffix, fig_dir, top_k):
    """
    生成超简化HTML可视化
    """
    try:
        print(f"[SAE] 开始生成超简化HTML，特征数量: {len(top_features)}")
        
        # 生成超简化HTML
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
        
        print(f"[SAE] HTML头部生成完成，开始生成特征页面...")
        
        # 为每个特征生成可视化
        for feat_idx, feature_id in enumerate(top_features):
            print(f"[SAE] 正在生成特征 {feature_id} (进度: {feat_idx+1}/{len(top_features)})")
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
            
            # 先添加最活跃的token列表
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
            
            # 添加每个token的激活信息
            for i, (token, activation) in enumerate(zip(token_texts, feature_acts)):
                # 计算颜色强度
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
            print(f"[SAE] 特征 {feature_id} 生成完成")
        
        # 添加页面结束标签
        simple_html += """
</body>
</html>
"""
        
        print(f"[SAE] HTML生成完成，开始保存文件...")
        
        # 保存超简化HTML
        simple_html_file = os.path.join(fig_dir, f"{base_name}{cycle_suffix}_top{top_k}_features_simple.html")
        with open(simple_html_file, 'w', encoding='utf-8') as f:
            f.write(simple_html)
        print(f"[SAE] 已生成超简化HTML: {simple_html_file}")
        
        return simple_html_file
        
    except Exception as e:
        print(f"[SAE] 超简化HTML生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_sae_model(sae_release, sae_id, sae_device="cuda:0"):
    """
    加载SAE模型，支持多种加载方式
    
    Args:
        sae_release: SAE发布版本
        sae_id: SAE ID
        sae_device: SAE设备
    
    Returns:
        tuple: (sae, cfg_dict, feature_sparsity) 或 None
    """
    try:
        from sae_lens import SAE
        
        print(f"[SAE] 开始加载SAE模型: {sae_release}/{sae_id}")
        print(f"[SAE] 目标设备: {sae_device}")
        
        sae = None
        cfg_dict = {'d_in': 768}
        feature_sparsity = None
        
        # 方法1：尝试从SAE专用目录加载
        sae_base_dir = "/workspace/illusory/huggingface/SAEs"
        
        # 处理HuggingFace仓库名称与本地目录名称的差异
        # HuggingFace: llama-3-8b-it-res-jh -> 本地: llama-3-8b-it-res
        local_release = sae_release
        if sae_release == "llama-3-8b-it-res-jh":
            local_release = "llama-3-8b-it-res"
        elif sae_release == "llama_scope_r1_distill":
            # llama_scope_r1_distill 对应的本地目录可能是 Llama-Scope-R1-Distill 或类似名称
            local_release = "llama-scope-r1-distill"  # 尝试常见的本地目录名称
        elif sae_release in ["llama_scope_lxr_8x", "llama_scope_lxr_32x", "llama_scope_lxa_8x", "llama_scope_lxa_32x", "llama_scope_lxm_8x", "llama_scope_lxm_32x"]:
            # Llama Scope 系列可能使用不同的本地目录名称
            local_release = sae_release.replace("_", "-")  # 尝试将下划线替换为连字符
        
        local_sae_path = os.path.join(sae_base_dir, local_release)
        sae_config_path = os.path.join(local_sae_path, sae_id)
        
        # 检查不同格式的SAE文件
        sae_files_exist = False
        if os.path.exists(sae_config_path):
            # 检查Llama格式 (cfg.json + safetensors)
            cfg_file = os.path.join(sae_config_path, "cfg.json")
            safetensors_file = os.path.join(sae_config_path, "sae_weights.safetensors")
            if os.path.exists(cfg_file) and os.path.exists(safetensors_file):
                sae_files_exist = True
                print(f"[SAE] 找到Llama格式SAE文件: cfg.json + sae_weights.safetensors")
            else:
                # 检查Gemma格式 (params.npz)
                npz_file = os.path.join(sae_config_path, "params.npz")
                if os.path.exists(npz_file):
                    sae_files_exist = True
                    print(f"[SAE] 找到Gemma格式SAE文件: params.npz")
        
        if sae_files_exist:
            try:
                print(f"[SAE] 尝试从本地路径加载SAE模型: {sae_config_path}")
                
                # 根据文件格式选择加载方法
                if os.path.exists(os.path.join(sae_config_path, "cfg.json")):
                    # Llama格式：使用load_from_disk方法
                    print(f"[SAE] 使用Llama格式加载方法")
                    sae_result = SAE.load_from_disk(sae_config_path, device=sae_device)
                else:
                    # Gemma格式：使用专门的加载函数
                    print(f"[SAE] 使用Gemma格式加载方法")
                    npz_file = os.path.join(sae_config_path, "params.npz")
                    if os.path.exists(npz_file):
                        sae_result = load_gemma_sae_from_local(npz_file, sae_device)
                    else:
                        print(f"[SAE] 本地npz文件不存在: {npz_file}")
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
                    
                print(f"[SAE] 成功从本地路径加载SAE模型: {sae}")
                
            except Exception as e:
                print(f"[SAE] 本地路径加载失败: {str(e)}")
                sae = None
        
        # 方法2：如果SAE专用目录没有，尝试从标准HuggingFace缓存加载
        if sae is None:
            standard_cache_path = f"/root/.cache/huggingface/hub/models--{sae_release.replace('/', '--')}"
            if os.path.exists(standard_cache_path):
                try:
                    print(f"[SAE] 尝试从标准缓存加载SAE模型: {standard_cache_path}")
                    
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
                        
                    print(f"[SAE] 成功从标准缓存加载SAE模型: {sae}")
                    
                except Exception as e:
                    print(f"[SAE] 标准缓存加载失败: {str(e)}")
                    sae = None
        
        # 方法3：如果本地都没有，尝试从HuggingFace下载
        if sae is None:
            try:
                print(f"[SAE] 尝试从HuggingFace下载SAE模型: {sae_release}/{sae_id}")
                
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
                
                print(f"[SAE] 成功从HuggingFace加载SAE模型: {sae}")
                
            except Exception as e:
                print(f"[SAE] HuggingFace下载失败: {str(e)}")
                print(f"[SAE] 网络问题导致无法下载SAE模型，跳过SAE分析")
                print(f"[SAE] 返回空结果，程序继续运行")
                return None
        
        if sae is None:
            print(f"[SAE] 无法加载任何SAE模型，跳过SAE分析")
            return None
        
        return (sae, cfg_dict, feature_sparsity)
        
    except ImportError:
        print(f"[SAE] sae_lens未安装，无法加载SAE模型")
        return None
    except Exception as e:
        print(f"[SAE] SAE加载过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_hooked_transformer_for_sae(model_name, hooked_device="cuda:0"):
    """
    为SAE分析加载HookedTransformer模型
    
    Args:
        model_name: 模型名称
        hooked_device: HookedTransformer设备
    
    Returns:
        HookedTransformer对象或None
    """
    try:
        from transformer_lens import HookedTransformer
        
        print(f"[SAE] 开始加载HookedTransformer用于SAE分析...")
        print(f"[SAE] 模型名称: {model_name}")
        print(f"[SAE] 目标设备: {hooked_device}")
        
        hooked_model = None
        
        # 方法1：尝试使用本地下载的HookedTransformer模型
        try:
            print(f"[SAE] 尝试使用本地下载的HookedTransformer模型...")
            
            # 根据模型类型选择本地HookedTransformer路径
            sae_base_dir = "/workspace/illusory/huggingface/SAEs"
            if 'gemma' in model_name.lower():
                # 使用IT版本的HookedTransformer，与IT版本的SAE保持一致
                hooked_model_path = os.path.join(sae_base_dir, "gemma-scope-9b-it-res", "hooked-transformer-it")
                model_name_for_hf = "google/gemma-2-9b-it"  # IT版本
                print(f"[SAE] 检测到Gemma模型，使用IT版本的HookedTransformer: {hooked_model_path}")
                print(f"[SAE] 使用IT版本HookedTransformer，与IT版本SAE保持一致")
            else:
                hooked_model_path = os.path.join(sae_base_dir, "llama-3-8b-it-res", "hooked-transformer")
                model_name_for_hf = "meta-llama/Meta-Llama-3-8B-Instruct"  # 对应的HuggingFace模型名称
                print(f"[SAE] 使用本地Llama HookedTransformer: {hooked_model_path}")
            
            # 检查本地路径是否存在
            print(f"[SAE] 检查本地路径: {hooked_model_path}")
            print(f"[SAE] 路径存在: {os.path.exists(hooked_model_path)}")
            
            if os.path.exists(hooked_model_path):
                print(f"[SAE] 本地HookedTransformer路径存在，尝试加载...")
                
                # 先加载HuggingFace模型，然后转换为HookedTransformer
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                print(f"[SAE] 从本地路径加载HuggingFace模型: {hooked_model_path}")
                try:
                    # 简化策略：直接使用CPU加载HuggingFace模型，然后让HookedTransformer处理设备转换
                    print(f"[SAE] 使用CPU加载HuggingFace模型以避免内存冲突")
                    
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        hooked_model_path,
                        device_map=None,  # 使用CPU
                        torch_dtype="auto",
                        low_cpu_mem_usage=True
                    )
                    print(f"[SAE] HuggingFace模型加载成功")
                    
                    # 使用hf_model参数创建HookedTransformer
                    print(f"[SAE] 开始创建HookedTransformer...")
                    hooked_model = HookedTransformer.from_pretrained(
                        model_name_for_hf,
                        hf_model=hf_model,
                        device=hooked_device
                    )
                    print(f"[SAE] ✅ 成功从本地路径创建HookedTransformer: {hooked_model_path}")
                except Exception as inner_e:
                    print(f"[SAE] 本地模型加载过程中出错: {inner_e}")
                    if "CUDA out of memory" in str(inner_e):
                        print(f"[SAE] 检测到CUDA内存不足，跳过HookedTransformer创建")
                        print(f"[SAE] 建议：释放其他GPU进程或使用更少的GPU")
                        hooked_model = None
                    else:
                        import traceback
                        traceback.print_exc()
                        raise inner_e
            else:
                print(f"[SAE] 本地HookedTransformer路径不存在: {hooked_model_path}")
                raise Exception("本地HookedTransformer路径不存在")
        except Exception as e:
            print(f"[SAE] 本地HookedTransformer创建失败: {e}")
            hooked_model = None
        
        # 方法2：如果本地HookedTransformer失败，尝试从HuggingFace下载
        if hooked_model is None:
            try:
                print(f"[SAE] 尝试从HuggingFace下载HookedTransformer...")
                
                # 根据模型类型选择HookedTransformer模型名称
                if 'gemma' in model_name.lower():
                    hooked_model_name = "google/gemma-2-9b"  # 使用Base版本，与SAE训练模型一致
                    print(f"[SAE] 检测到Gemma模型，使用HookedTransformer: {hooked_model_name}")
                else:
                    hooked_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
                    print(f"[SAE] 使用默认HookedTransformer: {hooked_model_name}")
                
                hooked_model = HookedTransformer.from_pretrained(
                    hooked_model_name, 
                    device=hooked_device
                )
                print(f"[SAE] ✅ 成功从HuggingFace创建HookedTransformer: {hooked_model_name}")
            except Exception as e:
                print(f"[SAE] HuggingFace创建失败: {e}")
                hooked_model = None
        
        
        # 如果所有方法都失败，打印详细信息
        if hooked_model is None:
            print(f"[SAE] ⚠️ 所有HookedTransformer创建方法都失败")
            print(f"[SAE] 模型名称: {model_name}")
            print(f"[SAE] 目标设备: {hooked_device}")
            print(f"[SAE] 继续尝试使用原始模型进行SAE分析...")
        else:
            print(f"[SAE] ✅ HookedTransformer创建成功")
        
        return hooked_model
        
    except ImportError:
        print(f"[SAE] transformer_lens未安装，无法加载HookedTransformer")
        return None
    except Exception as e:
        print(f"[SAE] HookedTransformer加载过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def list_all_sae_configs():
    """
    列出所有可用的SAE配置
    """
    return {
        'llama_models': ['llama-3.0-8b-instruct', 'llama-3.1-8b-instruct', 'llama-3.2-1b-instruct', 
                        'llama-3.2-3b-instruct', 'llama-3.2-11b-vision-instruct', 'llama-3.3-70b-instruct', 
                        'llama-4-scout-17b-16e-instruct'],
        'gemma_models': ['gemma-2-9b-it', 'gemma-2-9b-it-layer9-*', 'gemma-2-9b-it-layer20-*', 'gemma-2-9b-it-layer31-*'],
    }
