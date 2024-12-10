"""
Mochi Model Checkpoint Converter

This script converts Mochi model checkpoints from their original format to the Diffusers format.
It handles three main components:
1. The transformer model
2. The VAE (encoder and decoder)
3. The text encoder (T5)

The script provides utility functions to:
- Convert state dict keys between formats
- Handle weight transformations and reshaping
- Create and save a complete Diffusers pipeline

Usage:
    python convert_mochi_to_diffusers.py

Configuration is handled via the MochiWeightsSettings class (see configs/mochi_weights.py)
"""

from contextlib import nullcontext
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLMochi, FlowMatchEulerDiscreteScheduler, MochiPipeline, MochiTransformer3DModel
from diffusers.utils.import_utils import is_accelerate_available
from configs.mochi_weights import MochiWeightsSettings

CTX = init_empty_weights if is_accelerate_available else nullcontext

def swap_scale_shift(weight, dim):
    """
    Swaps the order of scale and shift parameters in normalization layers.
    
    Args:
        weight (torch.Tensor): Input tensor containing scale and shift parameters
        dim (int): Dimension along which to split the parameters
        
    Returns:
        torch.Tensor: Reordered tensor with scale parameters first, then shift parameters
    """
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight

def swap_proj_gate(weight):
    """
    Swaps projection and gate weights in attention mechanisms.
    
    Args:
        weight (torch.Tensor): Input tensor containing projection and gate weights
        
    Returns:
        torch.Tensor: Reordered tensor with gate weights first, then projection weights
    """
    proj, gate = weight.chunk(2, dim=0)
    new_weight = torch.cat([gate, proj], dim=0)
    return new_weight

def convert_mochi_transformer_checkpoint_to_diffusers(ckpt_path):
    """
    Converts a Mochi transformer checkpoint to the Diffusers format.
    
    This function handles the conversion of:
    - Embedding layers
    - Transformer blocks
    - Attention mechanisms
    - Normalization layers
    - Output projections
    
    Args:
        ckpt_path (str): Path to the original Mochi checkpoint file
        
    Returns:
        dict: State dictionary in Diffusers format
    """
    original_state_dict = load_file(ckpt_path, device="cpu")
    new_state_dict = {}
    new_state_dict["patch_embed.proj.weight"] = original_state_dict.pop("x_embedder.proj.weight")
    new_state_dict["patch_embed.proj.bias"] = original_state_dict.pop("x_embedder.proj.bias")
    new_state_dict["time_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop("t_embedder.mlp.0.weight")
    new_state_dict["time_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop("t_embedder.mlp.0.bias")
    new_state_dict["time_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop("t_embedder.mlp.2.weight")
    new_state_dict["time_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop("t_embedder.mlp.2.bias")
    new_state_dict["time_embed.pooler.to_kv.weight"] = original_state_dict.pop("t5_y_embedder.to_kv.weight")
    new_state_dict["time_embed.pooler.to_kv.bias"] = original_state_dict.pop("t5_y_embedder.to_kv.bias")
    new_state_dict["time_embed.pooler.to_q.weight"] = original_state_dict.pop("t5_y_embedder.to_q.weight")
    new_state_dict["time_embed.pooler.to_q.bias"] = original_state_dict.pop("t5_y_embedder.to_q.bias")
    new_state_dict["time_embed.pooler.to_out.weight"] = original_state_dict.pop("t5_y_embedder.to_out.weight")
    new_state_dict["time_embed.pooler.to_out.bias"] = original_state_dict.pop("t5_y_embedder.to_out.bias")
    new_state_dict["time_embed.caption_proj.weight"] = original_state_dict.pop("t5_yproj.weight")
    new_state_dict["time_embed.caption_proj.bias"] = original_state_dict.pop("t5_yproj.bias")

    num_layers = 48
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        old_prefix = f"blocks.{i}."

        new_state_dict[block_prefix + "norm1.linear.weight"] = original_state_dict.pop(old_prefix + "mod_x.weight")
        new_state_dict[block_prefix + "norm1.linear.bias"] = original_state_dict.pop(old_prefix + "mod_x.bias")
        if i < num_layers - 1:
            new_state_dict[block_prefix + "norm1_context.linear.weight"] = original_state_dict.pop(
                old_prefix + "mod_y.weight"
            )
            new_state_dict[block_prefix + "norm1_context.linear.bias"] = original_state_dict.pop(
                old_prefix + "mod_y.bias"
            )
        else:
            new_state_dict[block_prefix + "norm1_context.linear_1.weight"] = original_state_dict.pop(
                old_prefix + "mod_y.weight"
            )
            new_state_dict[block_prefix + "norm1_context.linear_1.bias"] = original_state_dict.pop(
                old_prefix + "mod_y.bias"
            )

        qkv_weight = original_state_dict.pop(old_prefix + "attn.qkv_x.weight")
        q, k, v = qkv_weight.chunk(3, dim=0)

        new_state_dict[block_prefix + "attn1.to_q.weight"] = q
        new_state_dict[block_prefix + "attn1.to_k.weight"] = k
        new_state_dict[block_prefix + "attn1.to_v.weight"] = v
        new_state_dict[block_prefix + "attn1.norm_q.weight"] = original_state_dict.pop(
            old_prefix + "attn.q_norm_x.weight"
        )
        new_state_dict[block_prefix + "attn1.norm_k.weight"] = original_state_dict.pop(
            old_prefix + "attn.k_norm_x.weight"
        )
        new_state_dict[block_prefix + "attn1.to_out.0.weight"] = original_state_dict.pop(
            old_prefix + "attn.proj_x.weight"
        )
        new_state_dict[block_prefix + "attn1.to_out.0.bias"] = original_state_dict.pop(old_prefix + "attn.proj_x.bias")

        qkv_weight = original_state_dict.pop(old_prefix + "attn.qkv_y.weight")
        q, k, v = qkv_weight.chunk(3, dim=0)

        new_state_dict[block_prefix + "attn1.add_q_proj.weight"] = q
        new_state_dict[block_prefix + "attn1.add_k_proj.weight"] = k
        new_state_dict[block_prefix + "attn1.add_v_proj.weight"] = v
        new_state_dict[block_prefix + "attn1.norm_added_q.weight"] = original_state_dict.pop(
            old_prefix + "attn.q_norm_y.weight"
        )
        new_state_dict[block_prefix + "attn1.norm_added_k.weight"] = original_state_dict.pop(
            old_prefix + "attn.k_norm_y.weight"
        )
        if i < num_layers - 1:
            new_state_dict[block_prefix + "attn1.to_add_out.weight"] = original_state_dict.pop(
                old_prefix + "attn.proj_y.weight"
            )
            new_state_dict[block_prefix + "attn1.to_add_out.bias"] = original_state_dict.pop(
                old_prefix + "attn.proj_y.bias"
            )

        new_state_dict[block_prefix + "ff.net.0.proj.weight"] = swap_proj_gate(
            original_state_dict.pop(old_prefix + "mlp_x.w1.weight")
        )
        new_state_dict[block_prefix + "ff.net.2.weight"] = original_state_dict.pop(old_prefix + "mlp_x.w2.weight")
        if i < num_layers - 1:
            new_state_dict[block_prefix + "ff_context.net.0.proj.weight"] = swap_proj_gate(
                original_state_dict.pop(old_prefix + "mlp_y.w1.weight")
            )
            new_state_dict[block_prefix + "ff_context.net.2.weight"] = original_state_dict.pop(
                old_prefix + "mlp_y.w2.weight"
            )

    new_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        original_state_dict.pop("final_layer.mod.weight"), dim=0
    )
    new_state_dict["norm_out.linear.bias"] = swap_scale_shift(original_state_dict.pop("final_layer.mod.bias"), dim=0)
    new_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    new_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
    new_state_dict["pos_frequencies"] = original_state_dict.pop("pos_frequencies")

    return new_state_dict

def convert_mochi_vae_state_dict_to_diffusers(encoder_ckpt_path, decoder_ckpt_path):
    """
    Converts Mochi VAE encoder and decoder checkpoints to the Diffusers format.
    
    This function handles:
    - Input/output projections
    - ResNet blocks
    - Up/down sampling blocks
    - Attention layers
    - Normalization layers
    
    Args:
        encoder_ckpt_path (str): Path to the VAE encoder checkpoint
        decoder_ckpt_path (str): Path to the VAE decoder checkpoint
        
    Returns:
        dict: Combined state dictionary for both encoder and decoder in Diffusers format
    """
    encoder_state_dict = load_file(encoder_ckpt_path, device="cpu")
    decoder_state_dict = load_file(decoder_ckpt_path, device="cpu")
    new_state_dict = {}

    # Decoder section
    prefix = "decoder."

    new_state_dict[f"{prefix}conv_in.weight"] = decoder_state_dict.pop("blocks.0.0.weight")
    new_state_dict[f"{prefix}conv_in.bias"] = decoder_state_dict.pop("blocks.0.0.bias")

    for i in range(3):
        new_state_dict[f"{prefix}block_in.resnets.{i}.norm1.norm_layer.weight"] = decoder_state_dict.pop(
            f"blocks.0.{i+1}.stack.0.weight"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.norm1.norm_layer.bias"] = decoder_state_dict.pop(
            f"blocks.0.{i+1}.stack.0.bias"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.conv1.conv.weight"] = decoder_state_dict.pop(
            f"blocks.0.{i+1}.stack.2.weight"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.conv1.conv.bias"] = decoder_state_dict.pop(
            f"blocks.0.{i+1}.stack.2.bias"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.norm2.norm_layer.weight"] = decoder_state_dict.pop(
            f"blocks.0.{i+1}.stack.3.weight"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.norm2.norm_layer.bias"] = decoder_state_dict.pop(
            f"blocks.0.{i+1}.stack.3.bias"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.conv2.conv.weight"] = decoder_state_dict.pop(
            f"blocks.0.{i+1}.stack.5.weight"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.conv2.conv.bias"] = decoder_state_dict.pop(
            f"blocks.0.{i+1}.stack.5.bias"
        )

    down_block_layers = [6, 4, 3]
    for block in range(3):
        for i in range(down_block_layers[block]):
            new_state_dict[f"{prefix}up_blocks.{block}.resnets.{i}.norm1.norm_layer.weight"] = decoder_state_dict.pop(
                f"blocks.{block+1}.blocks.{i}.stack.0.weight"
            )
            new_state_dict[f"{prefix}up_blocks.{block}.resnets.{i}.norm1.norm_layer.bias"] = decoder_state_dict.pop(
                f"blocks.{block+1}.blocks.{i}.stack.0.bias"
            )
            new_state_dict[f"{prefix}up_blocks.{block}.resnets.{i}.conv1.conv.weight"] = decoder_state_dict.pop(
                f"blocks.{block+1}.blocks.{i}.stack.2.weight"
            )
            new_state_dict[f"{prefix}up_blocks.{block}.resnets.{i}.conv1.conv.bias"] = decoder_state_dict.pop(
                f"blocks.{block+1}.blocks.{i}.stack.2.bias"
            )
            new_state_dict[f"{prefix}up_blocks.{block}.resnets.{i}.norm2.norm_layer.weight"] = decoder_state_dict.pop(
                f"blocks.{block+1}.blocks.{i}.stack.3.weight"
            )
            new_state_dict[f"{prefix}up_blocks.{block}.resnets.{i}.norm2.norm_layer.bias"] = decoder_state_dict.pop(
                f"blocks.{block+1}.blocks.{i}.stack.3.bias"
            )
            new_state_dict[f"{prefix}up_blocks.{block}.resnets.{i}.conv2.conv.weight"] = decoder_state_dict.pop(
                f"blocks.{block+1}.blocks.{i}.stack.5.weight"
            )
            new_state_dict[f"{prefix}up_blocks.{block}.resnets.{i}.conv2.conv.bias"] = decoder_state_dict.pop(
                f"blocks.{block+1}.blocks.{i}.stack.5.bias"
            )
        new_state_dict[f"{prefix}up_blocks.{block}.proj.weight"] = decoder_state_dict.pop(
            f"blocks.{block+1}.proj.weight"
        )
        new_state_dict[f"{prefix}up_blocks.{block}.proj.bias"] = decoder_state_dict.pop(f"blocks.{block+1}.proj.bias")

    for i in range(3):
        new_state_dict[f"{prefix}block_out.resnets.{i}.norm1.norm_layer.weight"] = decoder_state_dict.pop(
            f"blocks.4.{i}.stack.0.weight"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.norm1.norm_layer.bias"] = decoder_state_dict.pop(
            f"blocks.4.{i}.stack.0.bias"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.conv1.conv.weight"] = decoder_state_dict.pop(
            f"blocks.4.{i}.stack.2.weight"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.conv1.conv.bias"] = decoder_state_dict.pop(
            f"blocks.4.{i}.stack.2.bias"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.norm2.norm_layer.weight"] = decoder_state_dict.pop(
            f"blocks.4.{i}.stack.3.weight"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.norm2.norm_layer.bias"] = decoder_state_dict.pop(
            f"blocks.4.{i}.stack.3.bias"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.conv2.conv.weight"] = decoder_state_dict.pop(
            f"blocks.4.{i}.stack.5.weight"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.conv2.conv.bias"] = decoder_state_dict.pop(
            f"blocks.4.{i}.stack.5.bias"
        )

    new_state_dict[f"{prefix}proj_out.weight"] = decoder_state_dict.pop("output_proj.weight")
    new_state_dict[f"{prefix}proj_out.bias"] = decoder_state_dict.pop("output_proj.bias")

    # Encoder section
    prefix = "encoder."
    new_state_dict[f"{prefix}proj_in.weight"] = encoder_state_dict.pop("layers.0.weight")
    new_state_dict[f"{prefix}proj_in.bias"] = encoder_state_dict.pop("layers.0.bias")

    for i in range(3):
        new_state_dict[f"{prefix}block_in.resnets.{i}.norm1.norm_layer.weight"] = encoder_state_dict.pop(
            f"layers.{i+1}.stack.0.weight"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.norm1.norm_layer.bias"] = encoder_state_dict.pop(
            f"layers.{i+1}.stack.0.bias"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.conv1.conv.weight"] = encoder_state_dict.pop(
            f"layers.{i+1}.stack.2.weight"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.conv1.conv.bias"] = encoder_state_dict.pop(
            f"layers.{i+1}.stack.2.bias"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.norm2.norm_layer.weight"] = encoder_state_dict.pop(
            f"layers.{i+1}.stack.3.weight"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.norm2.norm_layer.bias"] = encoder_state_dict.pop(
            f"layers.{i+1}.stack.3.bias"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.conv2.conv.weight"] = encoder_state_dict.pop(
            f"layers.{i+1}.stack.5.weight"
        )
        new_state_dict[f"{prefix}block_in.resnets.{i}.conv2.conv.bias"] = encoder_state_dict.pop(
            f"layers.{i+1}.stack.5.bias"
        )

    down_block_layers = [3, 4, 6]
    for block in range(3):
        new_state_dict[f"{prefix}down_blocks.{block}.conv_in.conv.weight"] = encoder_state_dict.pop(
            f"layers.{block+4}.layers.0.weight"
        )
        new_state_dict[f"{prefix}down_blocks.{block}.conv_in.conv.bias"] = encoder_state_dict.pop(
            f"layers.{block+4}.layers.0.bias"
        )

        for i in range(down_block_layers[block]):
            new_state_dict[f"{prefix}down_blocks.{block}.resnets.{i}.norm1.norm_layer.weight"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.stack.0.weight"
            )
            new_state_dict[f"{prefix}down_blocks.{block}.resnets.{i}.norm1.norm_layer.bias"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.stack.0.bias"
            )
            new_state_dict[f"{prefix}down_blocks.{block}.resnets.{i}.conv1.conv.weight"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.stack.2.weight"
            )
            new_state_dict[f"{prefix}down_blocks.{block}.resnets.{i}.conv1.conv.bias"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.stack.2.bias"
            )
            new_state_dict[f"{prefix}down_blocks.{block}.resnets.{i}.norm2.norm_layer.weight"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.stack.3.weight"
            )
            new_state_dict[f"{prefix}down_blocks.{block}.resnets.{i}.norm2.norm_layer.bias"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.stack.3.bias"
            )
            new_state_dict[f"{prefix}down_blocks.{block}.resnets.{i}.conv2.conv.weight"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.stack.5.weight"
            )
            new_state_dict[f"{prefix}down_blocks.{block}.resnets.{i}.conv2.conv.bias"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.stack.5.bias"
            )

            new_state_dict[f"{prefix}down_blocks.{block}.norms.{i}.norm_layer.weight"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.attn_block.norm.weight"
            )
            new_state_dict[f"{prefix}down_blocks.{block}.norms.{i}.norm_layer.bias"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.attn_block.norm.bias"
            )

            qkv_weight = encoder_state_dict.pop(f"layers.{block+4}.layers.{i+1}.attn_block.attn.qkv.weight")
            q, k, v = qkv_weight.chunk(3, dim=0)

            new_state_dict[f"{prefix}down_blocks.{block}.attentions.{i}.to_q.weight"] = q
            new_state_dict[f"{prefix}down_blocks.{block}.attentions.{i}.to_k.weight"] = k
            new_state_dict[f"{prefix}down_blocks.{block}.attentions.{i}.to_v.weight"] = v
            new_state_dict[f"{prefix}down_blocks.{block}.attentions.{i}.to_out.0.weight"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.attn_block.attn.out.weight"
            )
            new_state_dict[f"{prefix}down_blocks.{block}.attentions.{i}.to_out.0.bias"] = encoder_state_dict.pop(
                f"layers.{block+4}.layers.{i+1}.attn_block.attn.out.bias"
            )

    for i in range(3):
        new_state_dict[f"{prefix}block_out.resnets.{i}.norm1.norm_layer.weight"] = encoder_state_dict.pop(
            f"layers.{i+7}.stack.0.weight"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.norm1.norm_layer.bias"] = encoder_state_dict.pop(
            f"layers.{i+7}.stack.0.bias"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.conv1.conv.weight"] = encoder_state_dict.pop(
            f"layers.{i+7}.stack.2.weight"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.conv1.conv.bias"] = encoder_state_dict.pop(
            f"layers.{i+7}.stack.2.bias"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.norm2.norm_layer.weight"] = encoder_state_dict.pop(
            f"layers.{i+7}.stack.3.weight"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.norm2.norm_layer.bias"] = encoder_state_dict.pop(
            f"layers.{i+7}.stack.3.bias"
        )

        new_state_dict[f"{prefix}block_out.resnets.{i}.conv2.conv.weight"] = encoder_state_dict.pop(
            f"layers.{i+7}.stack.5.weight"
        )
        new_state_dict[f"{prefix}block_out.resnets.{i}.conv2.conv.bias"] = encoder_state_dict.pop(
            f"layers.{i+7}.stack.5.bias"
        )

        new_state_dict[f"{prefix}block_out.norms.{i}.norm_layer.weight"] = encoder_state_dict.pop(
            f"layers.{i+7}.attn_block.norm.weight"
        )
        new_state_dict[f"{prefix}block_out.norms.{i}.norm_layer.bias"] = encoder_state_dict.pop(
            f"layers.{i+7}.attn_block.norm.bias"
        )

        qkv_weight = encoder_state_dict.pop(f"layers.{i+7}.attn_block.attn.qkv.weight")
        q, k, v = qkv_weight.chunk(3, dim=0)

        new_state_dict[f"{prefix}block_out.attentions.{i}.to_q.weight"] = q
        new_state_dict[f"{prefix}block_out.attentions.{i}.to_k.weight"] = k
        new_state_dict[f"{prefix}block_out.attentions.{i}.to_v.weight"] = v
        new_state_dict[f"{prefix}block_out.attentions.{i}.to_out.0.weight"] = encoder_state_dict.pop(
            f"layers.{i+7}.attn_block.attn.out.weight"
        )
        new_state_dict[f"{prefix}block_out.attentions.{i}.to_out.0.bias"] = encoder_state_dict.pop(
            f"layers.{i+7}.attn_block.attn.out.bias"
        )

    new_state_dict[f"{prefix}norm_out.norm_layer.weight"] = encoder_state_dict.pop("output_norm.weight")
    new_state_dict[f"{prefix}norm_out.norm_layer.bias"] = encoder_state_dict.pop("output_norm.bias")
    new_state_dict[f"{prefix}proj_out.weight"] = encoder_state_dict.pop("output_proj.weight")

    return new_state_dict

def main(settings: MochiWeightsSettings = MochiWeightsSettings()):
    """
    Main execution function that orchestrates the conversion process.
    
    This function:
    1. Configures the computation dtype (fp16, bf16, or fp32)
    2. Converts the transformer model if a checkpoint is provided
    3. Converts the VAE if encoder/decoder checkpoints are provided
    4. Loads the T5 text encoder and tokenizer
    5. Creates and saves a complete Diffusers pipeline
    
    Args:
        settings (MochiWeightsSettings): Configuration object containing paths and options
            - dtype: Computation precision ("fp16", "bf16", "fp32", or None)
            - transformer_checkpoint_path: Path to transformer checkpoint
            - vae_encoder_checkpoint_path: Path to VAE encoder checkpoint
            - vae_decoder_checkpoint_path: Path to VAE decoder checkpoint
            - text_encoder_cache_dir: Cache directory for T5 model
            - output_path: Where to save the converted pipeline
            - push_to_hub: Whether to push the converted model to HuggingFace Hub
    
    Raises:
        ValueError: If an unsupported dtype is specified
    """
    if settings.dtype is None:
        dtype = None
    elif settings.dtype == "fp16":
        dtype = torch.float16
    elif settings.dtype == "bf16":
        dtype = torch.bfloat16
    elif settings.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {settings.dtype}")

    transformer = None
    vae = None

    if settings.transformer_checkpoint_path is not None:
        print(f"Converting transformer from: {settings.transformer_checkpoint_path}")
        converted_transformer_state_dict = convert_mochi_transformer_checkpoint_to_diffusers(
            str(settings.transformer_checkpoint_path)
        )
        transformer = MochiTransformer3DModel()
        transformer.load_state_dict(converted_transformer_state_dict, strict=True)
        if dtype is not None:
            transformer = transformer.to(dtype=dtype)

    if settings.vae_encoder_checkpoint_path is not None and settings.vae_decoder_checkpoint_path is not None:
        print(f"Converting VAE from encoder: {settings.vae_encoder_checkpoint_path}")
        print(f"and decoder: {settings.vae_decoder_checkpoint_path}")
        vae = AutoencoderKLMochi(latent_channels=12, out_channels=3)
        converted_vae_state_dict = convert_mochi_vae_state_dict_to_diffusers(
            str(settings.vae_encoder_checkpoint_path), 
            str(settings.vae_decoder_checkpoint_path)
        )
        vae.load_state_dict(converted_vae_state_dict, strict=True)
        if dtype is not None:
            vae = vae.to(dtype=dtype)

    print("Loading text encoder and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(
        "google/t5-v1_1-xxl", 
        model_max_length=256
    )
    text_encoder = T5EncoderModel.from_pretrained(
        "google/t5-v1_1-xxl",
        cache_dir=settings.text_encoder_cache_dir
    )

    for param in text_encoder.parameters():
        param.data = param.data.contiguous()

    print("Creating pipeline...")
    pipe = MochiPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(invert_sigmas=True),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    
    print(f"Saving pipeline to: {settings.output_path}")
    pipe.save_pretrained(
        str(settings.output_path), 
        safe_serialization=True, 
        max_shard_size="5GB", 
        push_to_hub=settings.push_to_hub
    )
    print("Done!")

if __name__ == "__main__":
    settings = MochiWeightsSettings()
    main(settings)