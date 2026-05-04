import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms as T
from matplotlib.colors import LinearSegmentedColormap

def gradient_saliency(model, tokenizer_or_processor, input_data, out_prefix, target_label=1, mode="image", device="cpu", model_type=None, messages=None, model_path=None, **kwargs):
    """Gradient-based attribution methods that support image attribution.
    Implementation includes:
    1. General Gradient Attribution
    2. Gradient × Input
    3. Guided Backpropagation
    
    - model: loaded transformers model
    - tokenizer_or_processor: image processor
    - input_data: image tensor or PIL image
    - out_prefix: output file prefix (without suffix)
    - target_label: target category (for image classification)
    - mode: only 'image' supported
    - device: Run the device"""
    if mode != "image":
        raise ValueError("Gradient attribution method only supports image attribution, please use mode = 'image'")
    
    print("[Gradient] Starting gradient attribution...")
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
            
            # To tensor + requires_grad
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # Ensure inputs and model share device
            model_device = next(model.parameters()).device
            x = transform(img).unsqueeze(0).to(model_device).requires_grad_(True)
            
            print(f"[Gradient] Input image tensor shape:{x.shape}")
            
            # Prepare text input
            prompt = "What is in the image?"
            inputs = tokenizer_or_processor.tokenizer(prompt, return_tensors="pt").to(model_device)
            
            # Forward → logits
            outputs = model(
                vision_inputs=x,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            logits = outputs.logits[:, -1, :]  # Logits at last token
            
            # Pick target token
            target_token_id = logits.argmax(dim=-1).item()
            score = logits[0, target_token_id]
            
            print(f"[Gradient] Target token:{tokenizer_or_processor.tokenizer.convert_ids_to_tokens(target_token_id)}")
            
            # Backprop gradients
            model.zero_grad()
            score.backward()
            
            # Input gradients
            input_grad = x.grad.data.clone()
            
            # 1. Plain gradient attribution
            gradient_saliency = input_grad.abs().max(dim=1)[0].squeeze().cpu()
            gradient_saliency = (gradient_saliency - gradient_saliency.min()) / (gradient_saliency.max() - gradient_saliency.min())
            
            # 2. Gradient × input attribution
            x_denorm = x.clone().detach()
            x_denorm = x_denorm * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(model_device) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(model_device)
            x_denorm = torch.clamp(x_denorm, 0, 1)
            
            gradient_input_saliency = (input_grad * x_denorm).abs().max(dim=1)[0].squeeze().cpu()
            gradient_input_saliency = (gradient_input_saliency - gradient_input_saliency.min()) / (gradient_input_saliency.max() - gradient_input_saliency.min())
            
            # 3. Guided backprop (simplified)
            guided_grad = input_grad.clone()
            guided_grad[guided_grad < 0] = 0  # ReLU on gradients
            guided_saliency = guided_grad.abs().max(dim=1)[0].squeeze().cpu()
            guided_saliency = (guided_saliency - guided_saliency.min()) / (guided_saliency.max() - guided_saliency.min())
            
            # Save visualizations
            if out_prefix:
                model_folder = kwargs.get('model_name', 'unknown_model')
                method_name = "gradient_saliency"
                fig_dir = os.path.join(os.path.dirname(out_prefix), 'figure', model_folder, method_name)
                os.makedirs(fig_dir, exist_ok=True)
                base_name = os.path.basename(out_prefix)
                
                # Grayscale colormap
                bw_cmap = LinearSegmentedColormap.from_list("bw", ["black", "white"])
                
                # 1. Plain gradient attribution
                fig_grad, ax_grad = plt.subplots(1, 2, figsize=(20, 10))
                # Left: original image
                ax_grad[0].imshow(img_array)
                ax_grad[0].set_title("Original Image", fontsize=16)
                ax_grad[0].axis("off")
                # Right: gradient heatmap
                im_grad = ax_grad[1].imshow(gradient_saliency, cmap=bw_cmap)
                ax_grad[1].set_title("Gradient Attribution", fontsize=16)
                ax_grad[1].axis("off")
                # Colorbar
                norm_grad = plt.Normalize(gradient_saliency.min(), gradient_saliency.max())
                sm_grad = plt.cm.ScalarMappable(cmap=bw_cmap, norm=norm_grad)
                sm_grad.set_array([])
                cbar_grad = fig_grad.colorbar(sm_grad, ax=ax_grad[1], fraction=0.046, pad=0.04)
                cbar_grad.set_label("Gradient Score", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_gradient_comparison.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 2. Gradient × input attribution
                fig_grad_input, ax_grad_input = plt.subplots(1, 2, figsize=(20, 10))
                # Left: original image
                ax_grad_input[0].imshow(img_array)
                ax_grad_input[0].set_title("Original Image", fontsize=16)
                ax_grad_input[0].axis("off")
                # Right: grad×input heatmap
                im_grad_input = ax_grad_input[1].imshow(gradient_input_saliency, cmap=bw_cmap)
                ax_grad_input[1].set_title("Gradient × Input Attribution", fontsize=16)
                ax_grad_input[1].axis("off")
                # Colorbar
                norm_grad_input = plt.Normalize(gradient_input_saliency.min(), gradient_input_saliency.max())
                sm_grad_input = plt.cm.ScalarMappable(cmap=bw_cmap, norm=norm_grad_input)
                sm_grad_input.set_array([])
                cbar_grad_input = fig_grad_input.colorbar(sm_grad_input, ax=ax_grad_input[1], fraction=0.046, pad=0.04)
                cbar_grad_input.set_label("Gradient × Input Score", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_gradient_input_comparison.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 3. Guided backprop attribution
                fig_guided, ax_guided = plt.subplots(1, 2, figsize=(20, 10))
                # Left: original image
                ax_guided[0].imshow(img_array)
                ax_guided[0].set_title("Original Image", fontsize=16)
                ax_guided[0].axis("off")
                # Right: guided backprop heatmap
                im_guided = ax_guided[1].imshow(guided_saliency, cmap=bw_cmap)
                ax_guided[1].set_title("Guided Backpropagation Attribution", fontsize=16)
                ax_guided[1].axis("off")
                # Colorbar
                norm_guided = plt.Normalize(guided_saliency.min(), guided_saliency.max())
                sm_guided = plt.cm.ScalarMappable(cmap=bw_cmap, norm=norm_guided)
                sm_guided.set_array([])
                cbar_guided = fig_guided.colorbar(sm_guided, ax=ax_guided[1], fraction=0.046, pad=0.04)
                cbar_guided.set_label("Guided Backprop Score", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_guided_comparison.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # 4. Combined comparison
                fig_combined, axes = plt.subplots(2, 2, figsize=(20, 20))
                
                # Original
                axes[0, 0].imshow(img_array)
                axes[0, 0].set_title("Original Image", fontsize=16)
                axes[0, 0].axis("off")
                
                # Gradient attribution
                im1 = axes[0, 1].imshow(gradient_saliency, cmap="hot")
                axes[0, 1].set_title("Gradient Attribution", fontsize=16)
                axes[0, 1].axis("off")
                fig_combined.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                
                # Grad × input attribution
                im2 = axes[1, 0].imshow(gradient_input_saliency, cmap="hot")
                axes[1, 0].set_title("Gradient × Input Attribution", fontsize=16)
                axes[1, 0].axis("off")
                fig_combined.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
                
                # Guided backprop attribution
                im3 = axes[1, 1].imshow(guided_saliency, cmap="hot")
                axes[1, 1].set_title("Guided Backpropagation Attribution", fontsize=16)
                axes[1, 1].axis("off")
                fig_combined.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{base_name}_gradient_combined.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save attribution scores
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
            raise ValueError(f"the input_data type does not support:{type(input_data)}")
    except Exception as e:
        print(f"[Gradient] Error calculating gradient attribution:{str(e)}")
        print(f"[Gradient] Error type:{type(e).__name__}")
        print("[Gradient] Bug Tracking:")
        import traceback
        traceback.print_exc()
        raise e 