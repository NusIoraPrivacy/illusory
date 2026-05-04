import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms as T
from matplotlib.colors import LinearSegmentedColormap

RANDOM_STATE = 42

def _upsample_reflect(x, size, interpolate_mode="bilinear"):
    """
    上采样4D张量，使用反射填充
    参考TorchRay实现
    """
    assert len(x.shape) == 4
    orig_size = x.shape[2:]
    
    if not isinstance(size, tuple) and not isinstance(size, list):
        assert isinstance(size, int)
        size = (size, size)
    assert len(size) == 2
    
    # Ensure upsampling mode
    for i, o_s in enumerate(orig_size):
        assert o_s <= size[i]
    
    # Cell size after interpolation resize
    cell_size = [int(np.ceil(s / orig_size[i])) for i, s in enumerate(size)]
    
    # Interpolated input size incl. buffer
    pad_size = [int(cell_size[i] * (orig_size[i] + 2)) for i in range(len(orig_size))]
    
    # Reflect padding
    x_padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    
    # Input after interpolate+pad
    x_up = F.interpolate(x_padded, pad_size, mode=interpolate_mode, align_corners=False)
    
    # Crop/slice to target size
    x_new = x_up[:, :, cell_size[0]:cell_size[0] + size[0], cell_size[1]:cell_size[1] + size[1]]
    
    return x_new

def generate_rise_masks(n_masks, input_size, num_cells=7, p=0.5, device="cpu"):
    """
    生成RISE随机掩码
    - n_masks: 掩码数量
    - input_size: 输入尺寸 (H, W)
    - num_cells: 低分辨率网格的单元格数量
    - p: 每个单元格被设置为0的概率
    - device: 设备
    """
    masks = torch.zeros(n_masks, 1, input_size[0], input_size[1], device=device)
    
    # Low-res grid cell size
    cell_size = tuple([int(np.ceil(s / num_cells)) for s in input_size])
    
    # Upsampled mask size (input + cell)
    up_size = tuple([input_size[i] + cell_size[i] for i in range(2)])
    
    for i in range(n_masks):
        # Random low-res binary mask
        grid = (torch.rand(1, 1, num_cells, num_cells, device=device) < p).float()
        
        # Upsample mask to input+buffer shape
        mask_up = _upsample_reflect(grid, up_size)
        
        # Random shift offset
        shift_x = torch.randint(0, cell_size[0], (1,), device=device).item()
        shift_y = torch.randint(0, cell_size[1], (1,), device=device).item()
        
        # Extract final mask
        masks[i] = mask_up[0, :, shift_x:shift_x + input_size[0], shift_y:shift_y + input_size[1]]
    
    return masks

def rise_saliency(model, tokenizer_or_processor, input_data, out_prefix, target_label=1, n_masks=1024, num_cells=7, p=0.5, mode="image", device="cpu", model_type=None, messages=None, model_path=None, **kwargs):
    """
    RISE归因方法，仅支持图像归因。
    基于原始论文实现：https://arxiv.org/abs/1806.07421
    
    - model: 已加载的transformers模型
    - tokenizer_or_processor: 图像处理器
    - input_data: 图片tensor或PIL图像
    - out_prefix: 输出文件前缀（不带后缀）
    - target_label: 目标类别（图像分类时用）
    - n_masks: RISE掩码数量
    - num_cells: 低分辨率网格的单元格数量
    - p: 每个单元格被设置为0的概率
    - mode: 仅支持'image'
    - device: 运行设备
    """
    if RANDOM_STATE:
        random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)
    
    if mode != "image":
        raise ValueError("RISE方法仅支持图像归因，请使用mode='image'")
    
    print("[RISE] 开始图像归因...")
    try:
        if isinstance(input_data, (Image.Image, np.ndarray, torch.Tensor)):
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
            # Ensure inputs and model share device
            model_device = next(model.parameters()).device
            x = transform(img).unsqueeze(0).to(model_device)  # [1, 3, H, W]
            
            print(f"[RISE] 输入图像张量 shape: {x.shape}")
            
            # Forward closure
            def forward_fn(masked_img):
                """
                前向函数，计算目标类别的预测概率
                """
                masked_img = masked_img.to(model_device)
                
                # image_processor path
                # Tensor → PIL for processor
                masked_img_denorm = masked_img.clone()
                masked_img_denorm = masked_img_denorm * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(model_device) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(model_device)
                masked_img_denorm = torch.clamp(masked_img_denorm, 0, 1)
                
                # To PIL image
                masked_img_pil = T.ToPILImage()(masked_img_denorm.squeeze(0))
                
                # Run HF processor
                processed = tokenizer_or_processor.image_processor(masked_img_pil, return_tensors="pt")
                pixel_values = processed.pixel_values.to(model_device)
                
                # Match pixel_values dtype to model
                # Align dtypes
                if hasattr(model, 'vision_model'):
                    model_dtype = next(model.vision_model.parameters()).dtype
                elif hasattr(model, 'vision_tower'):
                    model_dtype = next(model.vision_tower.parameters()).dtype
                elif hasattr(model, 'visual'):
                    model_dtype = next(model.visual.parameters()).dtype
                else:
                    model_dtype = next(model.parameters()).dtype
                
                # Cast pixel_values
                pixel_values = pixel_values.to(model_dtype)
                
                # Optional image_grid_thw (Qwen2.5-VL)
                if hasattr(processed, 'image_grid_thw'):
                    image_grid_thw = processed.image_grid_thw.to(model_device)
                    # Forward — Qwen2.5-VL
                    with torch.no_grad():
                        if hasattr(model, 'vision_tower'):
                            vision_tower = model.vision_tower
                        elif hasattr(model, 'visual'):
                            vision_tower = model.visual
                        else:
                            raise AttributeError(f"模型 {type(model).__name__} 没有找到视觉组件")
                        vision_output = vision_tower(pixel_values, image_grid_thw)
                else:
                    # Forward — Llama-4-Scout
                    with torch.no_grad():
                        if hasattr(model, 'vision_model'):
                            vision_tower = model.vision_model
                        elif hasattr(model, 'vision_tower'):
                            vision_tower = model.vision_tower
                        elif hasattr(model, 'visual'):
                            vision_tower = model.visual
                        else:
                            raise AttributeError(f"模型 {type(model).__name__} 没有找到视觉组件")
                        vision_output = vision_tower(pixel_values)
                
                # Unified vision feature fetch
                if hasattr(vision_output, 'last_hidden_state'):
                    visual_features = vision_output.last_hidden_state
                elif isinstance(vision_output, torch.Tensor):
                    visual_features = vision_output
                else:
                    # Unknown vision outputs → error
                    raise ValueError(f"Unsupported vision output type: {type(vision_output)}")
                
                # Target-class score
                # Proxy score = mean vision features
                prediction_score = visual_features.mean()
                
                return prediction_score
            
            input_size = (x.shape[2], x.shape[3])  # (H, W)
            
            # Sample RISE masks
            masks = generate_rise_masks(n_masks, input_size, num_cells, p, model_device)
            
            print(f"[RISE] 使用 {n_masks} 个掩码计算归因...")
            
            # Baseline prediction
            with torch.no_grad():
                baseline_score = forward_fn(x).item()
            
            # Masked preds → accumulate map
            saliency = torch.zeros(1, input_size[0], input_size[1], device=model_device)
            
            batch_size = 32  # Batch size
            num_batches = (n_masks + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_masks)
                batch_masks = masks[start_idx:end_idx]
                
                for i, mask in enumerate(batch_masks):
                    # Apply mask to image
                    masked_img = x * mask
                    
                    # Masked forward prediction
                    with torch.no_grad():
                        masked_score = forward_fn(masked_img).item()
                    
                    # Accumulate attribution map
                    score_diff = masked_score - baseline_score
                    saliency += mask * score_diff
            
            # Normalize
            saliency = saliency.squeeze().cpu().numpy()
            if saliency.max() > saliency.min():
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            
            # Save visualizations
            if out_prefix:
                model_folder = kwargs.get('model_name', 'unknown_model')
                method_name = "rise_saliency"
                fig_dir = os.path.join(os.path.dirname(out_prefix), 'figure', model_folder, method_name)
                os.makedirs(fig_dir, exist_ok=True)
                base_name = os.path.basename(out_prefix)
                
                # Grayscale colormap
                bw_cmap = LinearSegmentedColormap.from_list("bw", ["black", "white"])
                
                # 1. Concat: original + B&W heatmap
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                # Left: original image
                ax1.imshow(img_array)
                ax1.set_title("Original Image", fontsize=16)
                ax1.axis("off")
                # Right: RISE heatmap
                im2 = ax2.imshow(saliency, cmap=bw_cmap)
                ax2.set_title("RISE Attribution Heatmap", fontsize=16)
                ax2.axis("off")
                # Colorbar
                norm = plt.Normalize(saliency.min(), saliency.max())
                sm = plt.cm.ScalarMappable(cmap=bw_cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
                cbar.set_label("RISE Attribution Score", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_rise_comparison.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 2. Overlay heatmap on original
                fig_overlay, ax_overlay = plt.subplots(figsize=(10, 10))
                ax_overlay.imshow(img_array)
                im_overlay = ax_overlay.imshow(saliency, cmap="hot", alpha=0.5)
                ax_overlay.set_title("RISE Overlay Attribution")
                ax_overlay.axis("off")
                
                # Colorbar on overlay
                cbar_overlay = fig_overlay.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
                cbar_overlay.set_label("RISE Attribution Score", fontsize=12)
                
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_rise_overlay.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 3. RISE-only map
                fig_rise, ax_rise = plt.subplots(figsize=(10, 10))
                im_rise = ax_rise.imshow(saliency, cmap="viridis")
                ax_rise.set_title("RISE Attribution Map", fontsize=16)
                ax_rise.axis("off")
                
                cbar_rise = fig_rise.colorbar(im_rise, ax=ax_rise, fraction=0.046, pad=0.04)
                cbar_rise.set_label("RISE Attribution Score", fontsize=12)
                
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_rise_attribution.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save RISE scores
                rise_scores_file = os.path.join(fig_dir, f"{base_name}_rise_scores.txt")
                with open(rise_scores_file, 'w') as f:
                    f.write(f"RISE attribution scores ({input_size[0]}x{input_size[1]})\n")
                    f.write("Format: row,column,score\n")
                    for i in range(input_size[0]):
                        for j in range(input_size[1]):
                            f.write(f"{i},{j},{saliency[i,j]:.6f}\n")
            
            return saliency, img, masks
        else:
            raise ValueError(f"input_data类型不支持: {type(input_data)}")
    except Exception as e:
        print(f"[RISE] 计算RISE归因时出错: {str(e)}")
        print(f"[RISE] 错误类型: {type(e).__name__}")
        print("[RISE] 错误追踪:")
        import traceback
        traceback.print_exc()
        raise e 