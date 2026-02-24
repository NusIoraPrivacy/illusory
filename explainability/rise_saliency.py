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
    
    # 确保是上采样
    for i, o_s in enumerate(orig_size):
        assert o_s <= size[i]
    
    # 计算插值后每个输入单元的大小
    cell_size = [int(np.ceil(s / orig_size[i])) for i, s in enumerate(size)]
    
    # 计算带缓冲区的插值输入大小
    pad_size = [int(cell_size[i] * (orig_size[i] + 2)) for i in range(len(orig_size))]
    
    # 使用反射填充
    x_padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    
    # 插值填充后的输入
    x_up = F.interpolate(x_padded, pad_size, mode=interpolate_mode, align_corners=False)
    
    # 切片到目标大小
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
    
    # 计算低分辨率网格单元格大小
    cell_size = tuple([int(np.ceil(s / num_cells)) for s in input_size])
    
    # 计算上采样掩码的大小（输入大小 + 单元格大小）
    up_size = tuple([input_size[i] + cell_size[i] for i in range(2)])
    
    for i in range(n_masks):
        # 生成低分辨率随机二进制掩码
        grid = (torch.rand(1, 1, num_cells, num_cells, device=device) < p).float()
        
        # 上采样低分辨率掩码到输入形状 + 缓冲区
        mask_up = _upsample_reflect(grid, up_size)
        
        # 随机偏移
        shift_x = torch.randint(0, cell_size[0], (1,), device=device).item()
        shift_y = torch.randint(0, cell_size[1], (1,), device=device).item()
        
        # 提取最终掩码
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
            # 处理输入图像
            if isinstance(input_data, Image.Image):
                img = input_data
            elif isinstance(input_data, np.ndarray):
                img = Image.fromarray(input_data)
            elif isinstance(input_data, torch.Tensor):
                img = Image.fromarray(input_data.detach().cpu().numpy())
            
            img = img.resize((448, 448))
            img_array = np.array(img)
            
            # 转换为tensor
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # 确保输入和模型在同一个设备上
            model_device = next(model.parameters()).device
            x = transform(img).unsqueeze(0).to(model_device)  # [1, 3, H, W]
            
            print(f"[RISE] 输入图像张量 shape: {x.shape}")
            
            # 定义前向函数
            def forward_fn(masked_img):
                """
                前向函数，计算目标类别的预测概率
                """
                masked_img = masked_img.to(model_device)
                
                # 使用image_processor处理图像
                # 将tensor转换回PIL图像进行处理
                masked_img_denorm = masked_img.clone()
                masked_img_denorm = masked_img_denorm * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(model_device) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(model_device)
                masked_img_denorm = torch.clamp(masked_img_denorm, 0, 1)
                
                # 转换为PIL图像
                masked_img_pil = T.ToPILImage()(masked_img_denorm.squeeze(0))
                
                # 使用processor处理
                processed = tokenizer_or_processor.image_processor(masked_img_pil, return_tensors="pt")
                pixel_values = processed.pixel_values.to(model_device)
                
                # 确保pixel_values的数据类型与模型一致
                # 检查模型的数据类型并匹配pixel_values
                if hasattr(model, 'vision_model'):
                    model_dtype = next(model.vision_model.parameters()).dtype
                elif hasattr(model, 'vision_tower'):
                    model_dtype = next(model.vision_tower.parameters()).dtype
                elif hasattr(model, 'visual'):
                    model_dtype = next(model.visual.parameters()).dtype
                else:
                    model_dtype = next(model.parameters()).dtype
                
                # 将pixel_values转换为与模型相同的数据类型
                pixel_values = pixel_values.to(model_dtype)
                
                # 检查是否有image_grid_thw属性（Qwen2.5-VL特有）
                if hasattr(processed, 'image_grid_thw'):
                    image_grid_thw = processed.image_grid_thw.to(model_device)
                    # 前向传播 - Qwen2.5-VL模式
                    with torch.no_grad():
                        if hasattr(model, 'vision_tower'):
                            vision_tower = model.vision_tower
                        elif hasattr(model, 'visual'):
                            vision_tower = model.visual
                        else:
                            raise AttributeError(f"模型 {type(model).__name__} 没有找到视觉组件")
                        vision_output = vision_tower(pixel_values, image_grid_thw)
                else:
                    # 前向传播 - Llama-4-Scout模式
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
                
                # 获取视觉特征（统一处理）
                if hasattr(vision_output, 'last_hidden_state'):
                    visual_features = vision_output.last_hidden_state
                elif isinstance(vision_output, torch.Tensor):
                    visual_features = vision_output
                else:
                    # 未知的视觉输出格式，抛出错误
                    raise ValueError(f"Unsupported vision output type: {type(vision_output)}")
                
                # 计算目标类别的预测分数
                # 这里使用视觉特征的平均值作为预测分数
                prediction_score = visual_features.mean()
                
                return prediction_score
            
            input_size = (x.shape[2], x.shape[3])  # (H, W)
            
            # 生成RISE掩码
            masks = generate_rise_masks(n_masks, input_size, num_cells, p, model_device)
            
            print(f"[RISE] 使用 {n_masks} 个掩码计算归因...")
            
            # 获取基准预测
            with torch.no_grad():
                baseline_score = forward_fn(x).item()
            
            # 计算掩码预测并累积归因图
            saliency = torch.zeros(1, input_size[0], input_size[1], device=model_device)
            
            batch_size = 32  # 批处理大小
            num_batches = (n_masks + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_masks)
                batch_masks = masks[start_idx:end_idx]
                
                for i, mask in enumerate(batch_masks):
                    # 应用掩码到图像
                    masked_img = x * mask
                    
                    # 计算掩码预测
                    with torch.no_grad():
                        masked_score = forward_fn(masked_img).item()
                    
                    # 累积归因图
                    score_diff = masked_score - baseline_score
                    saliency += mask * score_diff
            
            # 归一化
            saliency = saliency.squeeze().cpu().numpy()
            if saliency.max() > saliency.min():
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            
            # 保存可视化结果
            if out_prefix:
                model_folder = kwargs.get('model_name', 'unknown_model')
                method_name = "rise_saliency"
                fig_dir = os.path.join(os.path.dirname(out_prefix), 'figure', model_folder, method_name)
                os.makedirs(fig_dir, exist_ok=True)
                base_name = os.path.basename(out_prefix)
                
                # 定义黑白colormap
                bw_cmap = LinearSegmentedColormap.from_list("bw", ["black", "white"])
                
                # 1. 拼合图：左原图，右黑白热力图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                # 左边：原始图像
                ax1.imshow(img_array)
                ax1.set_title("Original Image", fontsize=16)
                ax1.axis("off")
                # 右边：RISE归因热力图
                im2 = ax2.imshow(saliency, cmap=bw_cmap)
                ax2.set_title("RISE Attribution Heatmap", fontsize=16)
                ax2.axis("off")
                # 颜色条
                norm = plt.Normalize(saliency.min(), saliency.max())
                sm = plt.cm.ScalarMappable(cmap=bw_cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
                cbar.set_label("RISE Attribution Score", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_rise_comparison.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 2. 叠加图：原图上叠加彩色热力图
                fig_overlay, ax_overlay = plt.subplots(figsize=(10, 10))
                ax_overlay.imshow(img_array)
                im_overlay = ax_overlay.imshow(saliency, cmap="hot", alpha=0.5)
                ax_overlay.set_title("RISE Overlay Attribution")
                ax_overlay.axis("off")
                
                # 为叠加图添加颜色条
                cbar_overlay = fig_overlay.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
                cbar_overlay.set_label("RISE Attribution Score", fontsize=12)
                
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_rise_overlay.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 3. 纯RISE归因图
                fig_rise, ax_rise = plt.subplots(figsize=(10, 10))
                im_rise = ax_rise.imshow(saliency, cmap="viridis")
                ax_rise.set_title("RISE Attribution Map", fontsize=16)
                ax_rise.axis("off")
                
                cbar_rise = fig_rise.colorbar(im_rise, ax=ax_rise, fraction=0.046, pad=0.04)
                cbar_rise.set_label("RISE Attribution Score", fontsize=12)
                
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_rise_attribution.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 保存RISE归因分数到文件
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