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
    """Preserve full conversation history with all prompts"""
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
    """Print GPU Usage"""
    if torch.cuda.is_available():
        print(f"[SAE] Number of GPUs:{torch.cuda.device_count()}")
    else:
        print("[SAE] No GPU available")

# Import SAE config module
try:
    from .sae_configs import get_sae_config_for_model, load_gemma_sae_from_local, generate_simple_html_visualization, load_hooked_transformer_for_sae, load_sae_model
except ImportError:
    from sae_configs import get_sae_config_for_model, load_gemma_sae_from_local, generate_simple_html_visualization, load_hooked_transformer_for_sae, load_sae_model

    

def sae_attribution(model, tokenizer_or_processor, input_data, out_prefix, target_label=1, mode="text", device="cpu", model_type=None, messages=None, model_path=None, task=None, cycle_num=None, generate_top_features=True, **kwargs):
    # Ensure torch is available
    import torch
    
    # All-GPU: main compute on passed device
    main_device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
    # SAE device via sae_device; default another GPU than main model
    sae_device_param = kwargs.get('sae_device', None)
    if sae_device_param is not None:
        sae_device = sae_device_param if torch.cuda.is_available() and sae_device_param != "cpu" else "cpu"
    else:
        # Default: another GPU; try next GPU if main on GPU
        if torch.cuda.is_available() and main_device.startswith("cuda"):
            try:
                main_device_idx = int(main_device.split(":")[1]) if ":" in main_device else 0
                # Try next GPU; fall back to CPU if single GPU
                if torch.cuda.device_count() > 1:
                    next_device_idx = (main_device_idx + 1) % torch.cuda.device_count()
                    sae_device = f"cuda:{next_device_idx}"
                else:
                    sae_device = "cpu"  # Single GPU: run SAE on CPU
            except:
                sae_device = "cpu"
        else:
            sae_device = "cpu"
    
    html_snippet = ""
    filtered_tokens = []
    rank_scores = []
    
    if mode not in ["text", "image"]:
        raise ValueError("SAE attribution supports text and image modes only")
    
    if not SAE_AVAILABLE:
        raise ImportError("SAELens is not installed")
    
    print_gpu_info()
    
    # Auto-detect model and SAE config
    model_name = kwargs.get('model_name', 'unknown')
    if model_name == 'unknown' and model_path:
        # Infer model name from checkpoint path
        model_name = os.path.basename(model_path)
    
    print(f"[SAE] Model Name Detected:{model_name}")
    print(f"[SAE] Model path:{model_path}")
    
    sae_config = get_sae_config_for_model(model_name, model_path)
    
    # User may override default config
    sae_release = kwargs.get('sae_release', sae_config['release'])
    sae_id = kwargs.get('sae_id', sae_config['sae_id'])
    target_layer = kwargs.get('target_layer', sae_config['target_layer'])
    hook_name = kwargs.get('hook_name', sae_config['hook_name'])
    
    print(f"[SAE] Use SAE configuration: release ={sae_release}, sae_id={sae_id}, target_layer={target_layer}")
    
    # Load SAE via sae_configs helpers
    # Verbose GPU info
    if torch.cuda.is_available():
        if main_device.startswith("cuda"):
            try:
                device_idx = int(main_device.split(":")[1]) if ":" in main_device else 0
                gpu_name = torch.cuda.get_device_name(device_idx)
                gpu_memory = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
                if sae_device == main_device:
                    print(f"[SAE] Main model and SAE use GPU{device_idx}: {gpu_name} ({gpu_memory:.1f} GB)")
                else:
                    print(f"[SAE] Main model uses GPU{device_idx}: {gpu_name} ({gpu_memory:.1f} GB)")
            except:
                pass
        if sae_device.startswith("cuda") and sae_device != main_device:
            try:
                device_idx = int(sae_device.split(":")[1]) if ":" in sae_device else 0
                gpu_name = torch.cuda.get_device_name(device_idx)
                gpu_memory = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
                print(f"[SAE] The SAE model uses GPUs{device_idx}: {gpu_name} ({gpu_memory:.1f} GB)")
            except:
                pass
    sae_result = load_sae_model(sae_release, sae_id, sae_device)
    
    if sae_result is None:
        print(f"[SAE] Unable to load any SAE models, skipping SAE analysis")
        return "", [], []
    
    sae, cfg_dict, feature_sparsity = sae_result

    if mode == "text" and isinstance(input_data, str):
        input_data = process_dialogue_text(input_data)
        
        # Main on GPU if available; move acts to SAE device for SAE
        # Encode text
        inputs = tokenizer_or_processor(input_data, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(main_device) for k, v in inputs.items()}

        tokens = inputs['input_ids']
        if tokens.dim() == 3:  # [1, 1, seq_len]
            tokens = tokens.squeeze(0)  # Remove first dim only
        elif tokens.dim() == 2:  # [1, seq_len]
            tokens = tokens.squeeze(0)  # Remove batch dimension
        
        # Keep tokens on main device
        tokens = tokens.to(main_device)


        # Hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states
        print(f"[SAE] Analysis of{target_layer}Layer hidden state, shape:{hidden_states[target_layer].shape}")

        target_hidden = hidden_states[target_layer]

        d_model = cfg_dict.get('d_in', 768)
        if target_hidden.shape[-1] != d_model:
            print(f"[SAE] Dimension mismatch, expectation{d_model}actual{target_hidden.shape[-1]}")
            target_hidden = target_hidden.to(torch.float32)
            projection = torch.nn.Linear(target_hidden.shape[-1], d_model, device=main_device)
            target_hidden = projection(target_hidden)

        # Hidden states → SAE device for encoding
        if target_hidden.device != torch.device(sae_device):
            print(f"[SAE] Pull hidden states from{target_hidden.device}Transfer to{sae_device}Perform SAE calculation")
        target_hidden_sae = target_hidden.to(sae_device)

        feature_acts = sae.encode(target_hidden_sae)
        reconstructed = sae.decode(feature_acts)
        


        print(f"[SAE] Feature activation shape:{feature_acts.shape}, refactor shape:{reconstructed.shape}")



        mse = F.mse_loss(target_hidden_sae, reconstructed).item()
        cosine_sim = F.cosine_similarity(target_hidden_sae, reconstructed, dim=-1).mean().item()
        print(f"[SAE] Refactor MSE:{mse:.6f}, cosine similarity:{cosine_sim:.6f}")


        l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
        print(f"[SAE] Average L0:{l0.mean().item():.2f}")
        
        local_sparsity = (feature_acts == 0).float().mean().item()
        print(f"[SAE] Local sparsity:{local_sparsity:.4f}")


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
                f.write(f"Main Device: {main_device}\n")
                f.write(f"SAE Device: {sae_device}\n")
                f.write(f"Sparsity: {local_sparsity:.6f}\n")
                f.write(f"Average L0: {l0.mean().item():.6f}\n")
                f.write(f"SAE Config: {sae.cfg}\n")
                if cycle_num is not None:
                    f.write(f"Cycle Number: {cycle_num}\n")
                f.write(f"Timestamp: {timestamp}\n")
            
            print(f"[SAE] Saved configuration information to:{config_file}")
            
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
            
            # feature_acts on CPU (SAE output)
            if feature_acts.device.type != sae_device:
                feature_acts = feature_acts.to(sae_device)
            
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
                    print(f"[SAE] Calculation characteristics{i}Error activating intensity:{e}")
                    print(f"[SAE] feature_acts shape: {feature_acts.shape}")
                    print(f"[SAE] Index attempted to access:{i}")
                    raise
            
            # Sort by activation strength
            feature_activations.sort(key=lambda x: x[1], reverse=True)
            
            # top-k from local sparsity
            top_k = max(1, int((1 - local_sparsity) * total_features))
            # top_k = 1
            top_k = min(top_k, total_features)  # Clamp top-k to feature count
            top_features = [idx for idx, _ in feature_activations[:top_k]]
            
            print(f"[SAE] Local sparsity:{local_sparsity:.4f}, directly identified before{top_k}most active traits ({(1-local_sparsity)*100:.2f}% of activation characteristics)")
            
            # Tutorial-style visualization
            try:
                import gc
                # hooked_device for HookedTransformer; default distinct from main
                hooked_device_param = kwargs.get('hooked_device', None)
                if hooked_device_param is not None:
                    hooked_device = hooked_device_param if torch.cuda.is_available() and hooked_device_param != "cpu" else "cpu"
                else:
                    # Pick GPU with most free RAM (not main/SAE)
                    if torch.cuda.is_available() and main_device.startswith("cuda"):
                        try:
                            main_device_idx = int(main_device.split(":")[1]) if ":" in main_device else 0
                            sae_device_idx = int(sae_device.split(":")[1]) if sae_device.startswith("cuda") and ":" in sae_device else -1
                            
                            # GPU with most free memory excluding main & SAE
                            best_device_idx = -1
                            best_free_memory = 0
                            
                            for i in range(torch.cuda.device_count()):
                                if i != main_device_idx and i != sae_device_idx:
                                    try:
                                        # Inspect GPU memory usage
                                        memory_total = torch.cuda.get_device_properties(i).total_memory
                                        memory_allocated = torch.cuda.memory_allocated(i)
                                        memory_free = memory_total - memory_allocated
                                        
                                        # Need ≥10GB free for HookedTransformer
                                        if memory_free > best_free_memory and memory_free > 10 * 1024**3:
                                            best_free_memory = memory_free
                                            best_device_idx = i
                                    except:
                                        continue
                            
                            if best_device_idx >= 0:
                                hooked_device = f"cuda:{best_device_idx}"
                            elif torch.cuda.device_count() > 1:
                                # If none qualified, try next GPU
                                next_device_idx = (main_device_idx + 1) % torch.cuda.device_count()
                                if next_device_idx != sae_device_idx:
                                    hooked_device = f"cuda:{next_device_idx}"
                                else:
                                    hooked_device = "cpu"  # If next GPU hosts SAE, use CPU
                            else:
                                hooked_device = "cpu"  # Single GPU: HookedTransformer on CPU
                        except Exception as e:
                            print(f"[SAE] GPU selection error:{e}, HookedTransformer uses CPU")
                            hooked_device = "cpu"
                    else:
                        hooked_device = "cpu"
                
                # Detailed GPU logging
                if torch.cuda.is_available():
                    if hooked_device.startswith("cuda"):
                        try:
                            device_idx = int(hooked_device.split(":")[1]) if ":" in hooked_device else 0
                            gpu_name = torch.cuda.get_device_name(device_idx)
                            gpu_memory_used = torch.cuda.memory_allocated(device_idx) / 1024**3
                            gpu_memory_total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
                            print(f"[SAE] HookedTransformer uses GPU{device_idx}: {gpu_name}Memory:{gpu_memory_used:.2f}/{gpu_memory_total:.1f} GB)")
                        except:
                            print(f"[SAE] Using HookedTransformer{hooked_device}")
                    else:
                        print(f"[SAE] Using HookedTransformer{hooked_device}")
                else:
                    print("[SAE] No GPU available, HookedTransformer uses CPU")
                    if hooked_device != "cpu":
                        hooked_device = "cpu"
                
                # Load HookedTransformer via sae_configs
                hooked_model = load_hooked_transformer_for_sae(model_name, hooked_device)

                # Optional top-feature subset
                if generate_top_features and hooked_model is not None:
                    top_config = SaeVisConfig(
                        hook_point=hook_name,  # Use hook_point not sae_id
                        features=top_features,
                        minibatch_size_features=50,
                        minibatch_size_tokens=50,
                        verbose=True,
                        device=sae_device,  # SAE viz on first GPU
                    )
                    # tokens [B,T] on SAE device
                    if tokens.dim() == 1:
                        tokens_for_sae = tokens.unsqueeze(0).to(sae_device)  # Add batch dim → SAE device
                    else:
                        tokens_for_sae = tokens.to(sae_device)
                    
                    # SAE on expected device
                    sae_gpu = sae.to(sae_device)
                    
                    top_vis_data = SaeVisRunner(top_config).run(
                        encoder=sae_gpu,
                        model=hooked_model,
                        tokens=tokens_for_sae,
                    )
                    
                    top_vis_file = os.path.join(fig_dir, f"{base_name}{cycle_suffix}_top{top_k}_features.html")
                    save_feature_centric_vis(sae_vis_data=top_vis_data, filename=top_vis_file)
                    print(f"[SAE] Raw SAE Dashboard generated:{top_vis_file}")
                    
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
                        
                        # HTML via sae_configs helper
                        simple_html_file = generate_simple_html_visualization(
                            token_texts, selected_feature_acts, top_features, 
                            base_name, cycle_suffix, fig_dir, top_k
                        )
                        
                        if simple_html_file:
                            print(f"[SAE] Generated super-simplified HTML:{simple_html_file}")
                        
                    except Exception as e:
                        print(f"[SAE] Hyper-Simplified HTML Generation Failed:{e}")
                        import traceback
                        traceback.print_exc()
                        print(f"[SAE] Skip HTML generation and continue saving other results")
                    
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
                    
                    print(f"[SAE] Most active feature list saved:{top_features_file}")
                else:
                    print(f"[SAE] HookedTransformer not loaded, skipping SAE visualization generation")
                    print(f"[SAE] Save feature analysis results only")
                
                # Free memory
                if hooked_model is not None:
                    del hooked_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print("[SAE] HookedTransformer memory cleared")
                
            except Exception as e:
                print(f"[SAE] SAE visual generation failed:{e}")
                import traceback
                traceback.print_exc()
                
                # Ensure cleanup
                if 'hooked_model' in locals():
                    del hooked_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                
            # Ensure list return type
        if isinstance(rank_scores, (float, np.float64)):
            rank_scores = [rank_scores]
        if isinstance(filtered_tokens, (str, int, float)):
            filtered_tokens = [filtered_tokens]
        return html_snippet, filtered_tokens, rank_scores

    ######## Image attribution ########
    elif mode == "image" and isinstance(input_data, (Image.Image, np.ndarray, torch.Tensor)):
        print(f"[SAE] Starting image SAE attribution analysis...")

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
        x = transform(img).unsqueeze(0).to(main_device)

        # HookedTransformer per tutorial
        try:
            import gc
            
            # hooked_device for HookedTransformer; default distinct from main
            hooked_device_param = kwargs.get('hooked_device', None)
            if hooked_device_param is not None:
                hooked_device = hooked_device_param if torch.cuda.is_available() and hooked_device_param != "cpu" else "cpu"
            else:
                # Pick GPU with most free RAM (not main/SAE)
                if torch.cuda.is_available() and main_device.startswith("cuda"):
                    try:
                        main_device_idx = int(main_device.split(":")[1]) if ":" in main_device else 0
                        sae_device_idx = int(sae_device.split(":")[1]) if sae_device.startswith("cuda") and ":" in sae_device else -1
                        
                        # GPU with most free memory excluding main & SAE
                        best_device_idx = -1
                        best_free_memory = 0
                        
                        for i in range(torch.cuda.device_count()):
                            if i != main_device_idx and i != sae_device_idx:
                                try:
                                    # Inspect GPU memory usage
                                    memory_total = torch.cuda.get_device_properties(i).total_memory
                                    memory_allocated = torch.cuda.memory_allocated(i)
                                    memory_free = memory_total - memory_allocated
                                    
                                    # Need ≥10GB free for HookedTransformer
                                    if memory_free > best_free_memory and memory_free > 10 * 1024**3:
                                        best_free_memory = memory_free
                                        best_device_idx = i
                                except:
                                    continue
                        
                        if best_device_idx >= 0:
                            hooked_device = f"cuda:{best_device_idx}"
                        elif torch.cuda.device_count() > 1:
                            # If none qualified, try next GPU
                            next_device_idx = (main_device_idx + 1) % torch.cuda.device_count()
                            if next_device_idx != sae_device_idx:
                                hooked_device = f"cuda:{next_device_idx}"
                            else:
                                hooked_device = "cpu"  # If next GPU hosts SAE, use CPU
                        else:
                            hooked_device = "cpu"  # Single GPU: HookedTransformer on CPU
                    except Exception as e:
                        print(f"[SAE] GPU selection error:{e}, HookedTransformer uses CPU")
                        hooked_device = "cpu"
                else:
                    hooked_device = "cpu"
            
            # Detailed GPU logging
            if torch.cuda.is_available():
                if hooked_device.startswith("cuda"):
                    try:
                        device_idx = int(hooked_device.split(":")[1]) if ":" in hooked_device else 0
                        gpu_name = torch.cuda.get_device_name(device_idx)
                        gpu_memory_used = torch.cuda.memory_allocated(device_idx) / 1024**3
                        gpu_memory_total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
                        print(f"[SAE] HookedTransformer uses GPU{device_idx}: {gpu_name}Memory:{gpu_memory_used:.2f}/{gpu_memory_total:.1f} GB)")
                    except:
                        print(f"[SAE] Using HookedTransformer{hooked_device}")
                else:
                    print(f"[SAE] Using HookedTransformer{hooked_device}")
            else:
                print("[SAE] No GPU available, HookedTransformer uses CPU")
                if hooked_device != "cpu":
                    hooked_device = "cpu"
            
            # Load HookedTransformer via sae_configs
            hooked_model = load_hooked_transformer_for_sae(model_name, hooked_device)
        except Exception as e:
            print(f"[SAE] Unable to load HookedTransformer:{e}")
            import traceback
            traceback.print_exc()
            
            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            raise RuntimeError("HookedTransformer is required for image SAE analysis, please install transformer-lens")

        # Tokens + cfg per tutorial
        prompt = "Describe the image."
        tokens = hooked_model.to_tokens(prompt)

        # Hook name from config
        print(f"[SAE] Use hook points:{hook_name}")

        # Hidden states + sparsity
        target_hidden = hooked_model.run_with_cache(tokens)[1][hook_name]
        
        # Hidden states → SAE device for encoding
        print(f"[SAE] Pull hidden states from{target_hidden.device}Transfer to{sae_device}Perform SAE calculation")
        target_hidden_sae = target_hidden.to(sae_device)
        
        feature_acts = sae.encode(target_hidden_sae)
        reconstructed = sae.decode(feature_acts)
        
        # Sparsity metrics
        local_sparsity = (feature_acts == 0).float().mean().item()
        l0 = (feature_acts > 0).float().sum(-1).detach()
        print(f"[SAE] Image Analysis - Local Sparsity:{local_sparsity:.4f}, Mean L0:{l0.mean().item():.2f}")
        
        # Per-feature strength (L1)
        feature_activations = []
        total_features = feature_acts.shape[-1]
        
        # feature_acts on CPU (SAE output)
        if feature_acts.device.type != sae_device:
            feature_acts = feature_acts.to(sae_device)
        
        for i in range(total_features):
            try:
                # Per-token strength for feature
                activation_strength = torch.norm(feature_acts[:, i], p=1).item()
                feature_activations.append((i, activation_strength))
            except Exception as e:
                print(f"[SAE] Calculation characteristics{i}Error activating intensity:{e}")
                print(f"[SAE] feature_acts[:, {i}] shape: {feature_acts[:, i].shape}")
                raise
        
        # Sort by activation strength
        feature_activations.sort(key=lambda x: x[1], reverse=True)
        
        # top-k from local sparsity
        top_k = max(1, int((1 - local_sparsity) * total_features))
        top_k = 10
        top_k = min(top_k, total_features)  # Clamp top-k to feature count
        top_features = [idx for idx, _ in feature_activations[:top_k]]
        
        print(f"[SAE] Local sparsity:{local_sparsity:.4f}, directly identified before{top_k}most active traits ({(1-local_sparsity)*100:.2f}% of activation characteristics)")

        # Save visualizations
        if out_prefix:
            # Path: results/explainability/task{id}/sae_attribution/
            fig_dir = os.path.join('results', 'explainability', f'task{task}', 'sae_attribution')
            os.makedirs(fig_dir, exist_ok=True)
            base_name = os.path.basename(out_prefix)
            
            # Unique filename per cycle
            timestamp = int(time.time())
            cycle_suffix = f"_cycle{cycle_num}" if cycle_num is not None else f"_ts{timestamp}"
            
            # Optional top-feature subset
            if generate_top_features:
                # Viz top features only
                try:
                    # Config for top features
                    top_config = SaeVisConfig(
                        hook_point=hook_name,  # Use hook_point not sae_id
                        features=top_features,  # Analyze top activations only
                        minibatch_size_features=50,
                        minibatch_size_tokens=50,
                        verbose=False,
                        device=sae_device,  # SAE visualization on CPU
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
                    print(f"[SAE] Most Active Feature Visualizations for Generated Images:{top_vis_file}")
                    
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
                    
                    print(f"[SAE] List of most active features of saved images:{top_features_file}")
                    
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
                            
                            # Five red intensity buckets
                            intensity = abs(normalized_intensity)
                            
                            if intensity < 0.2:
                                # Very Low: white
                                bg_color = "#ffffff"
                            elif intensity < 0.4:
                                # Low: light red
                                bg_color = "#ffcccc"
                            elif intensity < 0.6:
                                # Medium: mid red
                                bg_color = "#ff9999"
                            elif intensity < 0.8:
                                # High: dark red
                                bg_color = "#ff6666"
                            else:
                                # Very High: deep red
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
                        print(f"[SAE] Generated super-simplified HTML:{simple_html_file}")
                        
                    except Exception as e:
                        print(f"[SAE] Hyper-Simplified HTML Generation Failed:{e}")
                    
                    del top_vis_data
                    
                except Exception as e:
                    print(f"[SAE] Image most active feature subset visualization generation failed:{e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"[SAE] Image Feature Statistics:")
            print(f"- Total number of features:{total_features}")
            print(f"- Number of active features:{int((1 - local_sparsity) * total_features)}")
            print(f"- Average activation intensity:{np.mean([score for _, score in feature_activations]):.4f}")

        # Reconstruction compare (reuse tensors)
        reconstruction_error = torch.norm(target_hidden_sae - reconstructed, dim=-1).mean().item()
        
        print(f"[SAE] Refactoring error:{reconstruction_error:.6f}")

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
            print(f"[SAE] Refactoring Comparison Chart Saved:{comparison_file}")

        return reconstruction_error, img_array, reconstructed.detach().cpu().numpy()
        
    else:
        raise ValueError(f"the input_data type does not support:{type(input_data)}")
