import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms as T
from matplotlib.colors import LinearSegmentedColormap

def gradient_saliency(model, tokenizer_or_processor, input_data, out_prefix, target_label=1, mode="image", device="cpu", model_type=None, messages=None, model_path=None, **kwargs):
    """
    基于梯度的归因方法，支持图像归因。
    实现包括：
    1. 普通梯度归因 (Gradient)
    2. 梯度×输入 (Gradient × Input)
    3. 引导反向传播 (Guided Backpropagation)
    
    - model: 已加载的transformers模型
    - tokenizer_or_processor: 图像处理器
    - input_data: 图片tensor或PIL图像
    - out_prefix: 输出文件前缀（不带后缀）
    - target_label: 目标类别（图像分类时用）
    - mode: 仅支持'image'
    - device: 运行设备
    """
    if mode != "image":
        raise ValueError("梯度归因方法仅支持图像归因，请使用mode='image'")
    
    print("[Gradient] 开始梯度归因...")
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
            
            # 转换为tensor并设置requires_grad
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # 确保输入和模型在同一个设备上
            model_device = next(model.parameters()).device
            x = transform(img).unsqueeze(0).to(model_device).requires_grad_(True)
            
            print(f"[Gradient] 输入图像张量 shape: {x.shape}")
            
            # 准备文本输入
            prompt = "What is in the image?"
            inputs = tokenizer_or_processor.tokenizer(prompt, return_tensors="pt").to(model_device)
            
            # 前向传播获取logits
            outputs = model(
                vision_inputs=x,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            logits = outputs.logits[:, -1, :]  # 最后一个token的logits
            
            # 选择目标token
            target_token_id = logits.argmax(dim=-1).item()
            score = logits[0, target_token_id]
            
            print(f"[Gradient] 目标token: {tokenizer_or_processor.tokenizer.convert_ids_to_tokens(target_token_id)}")
            
            # 计算梯度
            model.zero_grad()
            score.backward()
            
            # 获取输入梯度
            input_grad = x.grad.data.clone()
            
            # 1. 普通梯度归因
            gradient_saliency = input_grad.abs().max(dim=1)[0].squeeze().cpu()
            gradient_saliency = (gradient_saliency - gradient_saliency.min()) / (gradient_saliency.max() - gradient_saliency.min())
            
            # 2. 梯度×输入归因
            x_denorm = x.clone().detach()
            x_denorm = x_denorm * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(model_device) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(model_device)
            x_denorm = torch.clamp(x_denorm, 0, 1)
            
            gradient_input_saliency = (input_grad * x_denorm).abs().max(dim=1)[0].squeeze().cpu()
            gradient_input_saliency = (gradient_input_saliency - gradient_input_saliency.min()) / (gradient_input_saliency.max() - gradient_input_saliency.min())
            
            # 3. 引导反向传播（简化版本）
            guided_grad = input_grad.clone()
            guided_grad[guided_grad < 0] = 0  # ReLU on gradients
            guided_saliency = guided_grad.abs().max(dim=1)[0].squeeze().cpu()
            guided_saliency = (guided_saliency - guided_saliency.min()) / (guided_saliency.max() - guided_saliency.min())
            
            # 保存可视化结果
            if out_prefix:
                model_folder = kwargs.get('model_name', 'unknown_model')
                method_name = "gradient_saliency"
                fig_dir = os.path.join(os.path.dirname(out_prefix), 'figure', model_folder, method_name)
                os.makedirs(fig_dir, exist_ok=True)
                base_name = os.path.basename(out_prefix)
                
                # 定义黑白colormap
                bw_cmap = LinearSegmentedColormap.from_list("bw", ["black", "white"])
                
                # 1. 普通梯度归因
                fig_grad, ax_grad = plt.subplots(1, 2, figsize=(20, 10))
                # 左边：原始图像
                ax_grad[0].imshow(img_array)
                ax_grad[0].set_title("Original Image", fontsize=16)
                ax_grad[0].axis("off")
                # 右边：梯度归因热力图
                im_grad = ax_grad[1].imshow(gradient_saliency, cmap=bw_cmap)
                ax_grad[1].set_title("Gradient Attribution", fontsize=16)
                ax_grad[1].axis("off")
                # 颜色条
                norm_grad = plt.Normalize(gradient_saliency.min(), gradient_saliency.max())
                sm_grad = plt.cm.ScalarMappable(cmap=bw_cmap, norm=norm_grad)
                sm_grad.set_array([])
                cbar_grad = fig_grad.colorbar(sm_grad, ax=ax_grad[1], fraction=0.046, pad=0.04)
                cbar_grad.set_label("Gradient Score", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_gradient_comparison.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 2. 梯度×输入归因
                fig_grad_input, ax_grad_input = plt.subplots(1, 2, figsize=(20, 10))
                # 左边：原始图像
                ax_grad_input[0].imshow(img_array)
                ax_grad_input[0].set_title("Original Image", fontsize=16)
                ax_grad_input[0].axis("off")
                # 右边：梯度×输入归因热力图
                im_grad_input = ax_grad_input[1].imshow(gradient_input_saliency, cmap=bw_cmap)
                ax_grad_input[1].set_title("Gradient × Input Attribution", fontsize=16)
                ax_grad_input[1].axis("off")
                # 颜色条
                norm_grad_input = plt.Normalize(gradient_input_saliency.min(), gradient_input_saliency.max())
                sm_grad_input = plt.cm.ScalarMappable(cmap=bw_cmap, norm=norm_grad_input)
                sm_grad_input.set_array([])
                cbar_grad_input = fig_grad_input.colorbar(sm_grad_input, ax=ax_grad_input[1], fraction=0.046, pad=0.04)
                cbar_grad_input.set_label("Gradient × Input Score", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_gradient_input_comparison.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 3. 引导反向传播归因
                fig_guided, ax_guided = plt.subplots(1, 2, figsize=(20, 10))
                # 左边：原始图像
                ax_guided[0].imshow(img_array)
                ax_guided[0].set_title("Original Image", fontsize=16)
                ax_guided[0].axis("off")
                # 右边：引导反向传播归因热力图
                im_guided = ax_guided[1].imshow(guided_saliency, cmap=bw_cmap)
                ax_guided[1].set_title("Guided Backpropagation Attribution", fontsize=16)
                ax_guided[1].axis("off")
                # 颜色条
                norm_guided = plt.Normalize(guided_saliency.min(), guided_saliency.max())
                sm_guided = plt.cm.ScalarMappable(cmap=bw_cmap, norm=norm_guided)
                sm_guided.set_array([])
                cbar_guided = fig_guided.colorbar(sm_guided, ax=ax_guided[1], fraction=0.046, pad=0.04)
                cbar_guided.set_label("Guided Backprop Score", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_guided_comparison.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 4. 综合对比图
                fig_combined, axes = plt.subplots(2, 2, figsize=(20, 20))
                
                # 原图
                axes[0, 0].imshow(img_array)
                axes[0, 0].set_title("Original Image", fontsize=16)
                axes[0, 0].axis("off")
                
                # 梯度归因
                im1 = axes[0, 1].imshow(gradient_saliency, cmap="hot")
                axes[0, 1].set_title("Gradient Attribution", fontsize=16)
                axes[0, 1].axis("off")
                fig_combined.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                
                # 梯度×输入归因
                im2 = axes[1, 0].imshow(gradient_input_saliency, cmap="hot")
                axes[1, 0].set_title("Gradient × Input Attribution", fontsize=16)
                axes[1, 0].axis("off")
                fig_combined.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
                
                # 引导反向传播归因
                im3 = axes[1, 1].imshow(guided_saliency, cmap="hot")
                axes[1, 1].set_title("Guided Backpropagation Attribution", fontsize=16)
                axes[1, 1].axis("off")
                fig_combined.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_gradient_combined.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 保存归因分数到文件
                scores_file = os.path.join(fig_dir, f"{base_name}_gradient_scores.txt")
                with open(scores_file, 'w') as f:
                    f.write(f"Gradient attribution scores ({gradient_saliency.shape[0]}x{gradient_saliency.shape[1]})\n")
                    f.write("Format: row,column,gradient_score,gradient_input_score,guided_score\n")
                    for i in range(gradient_saliency.shape[0]):
                        for j in range(gradient_saliency.shape[1]):
                            f.write(f"{i},{j},{gradient_saliency[i,j]:.6f},{gradient_input_saliency[i,j]:.6f},{guided_saliency[i,j]:.6f}\n")
            
            return {
                'gradient': gradient_saliency.numpy(),
                'gradient_input': gradient_input_saliency.numpy(),
                'guided': guided_saliency.numpy()
            }, img, input_grad
        else:
            raise ValueError(f"input_data类型不支持: {type(input_data)}")
    except Exception as e:
        print(f"[Gradient] 计算梯度归因时出错: {str(e)}")
        print(f"[Gradient] 错误类型: {type(e).__name__}")
        print("[Gradient] 错误追踪:")
        import traceback
        traceback.print_exc()
        raise e 