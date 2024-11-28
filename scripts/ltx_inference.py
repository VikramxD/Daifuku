"""
Inference module for LTX video generation.

This module provides the main inference functionality for the LTX video generation model,
using the configuration defined in the settings module.
"""

import json
import random
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional
import imageio
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
from ltx.ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx.ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx.ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx.ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx.ltx_video.utils.conditioning_method import ConditioningMethod
from configs.ltx_settings import LTXVideoSettings

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LTXInference:
    """
    LTX Video Generation Inference Class
    
    Handles model loading, inference, and video generation using the LTX pipeline.
    """
    
    def __init__(self, config: LTXVideoSettings):
        """Initialize with settings"""
        self.config = config
        self.setup_random_seeds()
        self.pipeline = self._initialize_pipeline()
        
    def setup_random_seeds(self) -> None:
        """Set random seeds for reproducibility"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

    def _load_vae(self, vae_dir: Path) -> CausalVideoAutoencoder:
        """Load and configure the VAE model"""
        vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
        vae_config_path = vae_dir / "config.json"
        
        with open(vae_config_path, "r") as f:
            vae_config = json.load(f)
            
        vae = CausalVideoAutoencoder.from_config(vae_config)
        vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
        vae.load_state_dict(vae_state_dict)
        
        if torch.cuda.is_available():
            vae = vae.cuda()
        return vae.to(torch.bfloat16)

    def _load_unet(self, unet_dir: Path) -> Transformer3DModel:
        """Load and configure the UNet model"""
        unet_ckpt_path = unet_dir / "unet_diffusion_pytorch_model.safetensors"
        unet_config_path = unet_dir / "config.json"
        
        transformer_config = Transformer3DModel.load_config(unet_config_path)
        transformer = Transformer3DModel.from_config(transformer_config)
        unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
        transformer.load_state_dict(unet_state_dict, strict=True)
        
        if torch.cuda.is_available():
            transformer = transformer.cuda()
        return transformer

    def _initialize_pipeline(self) -> LTXVideoPipeline:
        """Initialize pipeline with all components"""
        unet_dir, vae_dir, scheduler_dir = self.config.get_model_paths()
        
        # Load models
        vae = self._load_vae(vae_dir)
        unet = self._load_unet(unet_dir)
        scheduler = self._load_scheduler(scheduler_dir)
        patchifier = SymmetricPatchifier(patch_size=1)
        
        # Load text encoder and tokenizer from PixArt
        text_encoder = T5EncoderModel.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", 
            subfolder="text_encoder"
        )
        if torch.cuda.is_available():
            text_encoder = text_encoder.to("cuda")
            
        tokenizer = T5Tokenizer.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", 
            subfolder="tokenizer"
        )

        if self.config.bfloat16 and unet.dtype != torch.bfloat16:
            unet = unet.to(torch.bfloat16)

        # Initialize pipeline
        pipeline = LTXVideoPipeline(
            transformer=unet,
            patchifier=patchifier,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            vae=vae,
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            
        return pipeline

    def _load_scheduler(self, scheduler_dir: Path):
        """Load and configure the scheduler"""
        from ltx.ltx_video.schedulers.rf import RectifiedFlowScheduler
        scheduler_config_path = scheduler_dir / "scheduler_config.json"
        scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
        return RectifiedFlowScheduler.from_config(scheduler_config)

    def load_input_image(self) -> Optional[torch.Tensor]:
        """Load and preprocess input image if provided"""
        if not self.config.input_image_path:
            return None
            
        image = Image.open(self.config.input_image_path).convert("RGB")
        target_height, target_width = self.config.height, self.config.width
        
        # Calculate aspect ratio and resize
        input_width, input_height = image.size
        aspect_ratio_target = target_width / target_height
        aspect_ratio_frame = input_width / input_height
        
        if aspect_ratio_frame > aspect_ratio_target:
            new_width = int(input_height * aspect_ratio_target)
            new_height = input_height
            x_start = (input_width - new_width) // 2
            y_start = 0
        else:
            new_width = input_width
            new_height = int(input_width / aspect_ratio_target)
            x_start = 0
            y_start = (input_height - new_height) // 2

        image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
        image = image.resize((target_width, target_height))
        
        frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
        frame_tensor = (frame_tensor / 127.5) - 1.0
        return frame_tensor.unsqueeze(0).unsqueeze(2)

    def generate(self) -> None:
        """Run generation pipeline"""
        # Load input image if provided
        media_items_prepad = self.load_input_image()
        
        # Calculate dimensions with padding
        height = self.config.height
        width = self.config.width
        num_frames = self.config.num_frames
        
        # Validate dimensions
        if height > self.config.MAX_HEIGHT or width > self.config.MAX_WIDTH or num_frames > self.config.MAX_NUM_FRAMES:
            logger.warning(
                f"Input resolution or number of frames {height}x{width}x{num_frames} is too big, "
                f"it is suggested to use the resolution below {self.config.MAX_HEIGHT}x{self.config.MAX_WIDTH}x{self.config.MAX_NUM_FRAMES}."
            )

        # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
        
        logger.info(f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}")
        
        # Calculate and apply padding
        padding = self._calculate_padding(height, width, height_padded, width_padded)
        if media_items_prepad is not None:
            media_items = F.pad(media_items_prepad, padding, mode="constant", value=-1)
        else:
            media_items = None

        # Set up generator
        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(self.config.seed)

        # Run pipeline
        images = self.pipeline(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            num_inference_steps=self.config.num_inference_steps,
            num_images_per_prompt=self.config.num_images_per_prompt,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
            output_type="pt",
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=self.config.frame_rate,
            media_items=media_items,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=(
                ConditioningMethod.FIRST_FRAME
                if media_items is not None
                else ConditioningMethod.UNCONDITIONAL
            ),
            mixed_precision=not self.config.bfloat16,
        ).images

        # Process and save outputs
        self._save_outputs(images, padding, media_items_prepad)

    def _calculate_padding(
        self,
        source_height: int,
        source_width: int,
        target_height: int,
        target_width: int
    ) -> tuple[int, int, int, int]:
        """Calculate padding values for input tensors"""
        pad_height = target_height - source_height
        pad_width = target_width - source_width
        
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        return (pad_left, pad_right, pad_top, pad_bottom)

    def _save_outputs(
        self,
        images: torch.Tensor,
        padding: tuple[int, int, int, int],
        media_items_prepad: Optional[torch.Tensor]
    ) -> None:
        """Save generated outputs as videos and/or images"""

        output_dir = (
            Path(self.config.output_path)
            if self.config.output_path
            else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Crop padding
        pad_left, pad_right, pad_top, pad_bottom = padding
        pad_bottom = -pad_bottom if pad_bottom != 0 else images.shape[3]
        pad_right = -pad_right if pad_right != 0 else images.shape[4]
        
        images = images[
            :, :,
            :self.config.num_frames,
            pad_top:pad_bottom,
            pad_left:pad_right
        ]

        # Save each generated sequence
        for i in range(images.shape[0]):
            video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
            video_np = (video_np * 255).astype(np.uint8)
            
            # Save as image if single frame
            if video_np.shape[0] == 1:
                self._save_single_frame(video_np[0], i, output_dir)
            else:
                self._save_video(video_np, i, output_dir, media_items_prepad)

        logger.info(f"Outputs saved to {output_dir}")

    def _save_single_frame(self, frame: np.ndarray, index: int, output_dir: Path) -> None:
        """Save a single frame as an image"""
        output_filename = self._get_unique_filename(
            f"image_output_{index}",
            ".png",
            output_dir
        )
        imageio.imwrite(output_filename, frame)

    def _save_video(
        self,
        video_np: np.ndarray,
        index: int,
        output_dir: Path,
        media_items_prepad: Optional[torch.Tensor]
    ) -> None:
        """Save video frames and optional condition image"""
        # Save video
        base_filename = f"img_to_vid_{index}" if self.config.input_image_path else f"text_to_vid_{index}"
        output_filename = self._get_unique_filename(base_filename, ".mp4", output_dir)
        
        with imageio.get_writer(output_filename, fps=self.config.frame_rate) as video:
            for frame in video_np:
                video.append_data(frame)

        # Save condition image if provided
        if media_items_prepad is not None:
            reference_image = (
                (media_items_prepad[0, :, 0].permute(1, 2, 0).cpu().data.numpy() + 1.0)
                / 2.0 * 255
            )
            condition_filename = self._get_unique_filename(
                base_filename,
                ".png",
                output_dir,
                "_condition"
            )
            imageio.imwrite(condition_filename, reference_image.astype(np.uint8))

    def _get_unique_filename(
        self,
        base: str,
        ext: str,
        output_dir: Path,
        suffix: str = ""
    ) -> Path:
        """Generate a unique filename for outputs"""
        prompt_str = self._convert_prompt_to_filename(self.config.prompt or "no_prompt")
        base_filename = f"{base}_{prompt_str}_{self.config.seed}_{self.config.height}x{self.config.width}x{self.config.num_frames}"
        
        for i in range(1000):
            filename = output_dir / f"{base_filename}_{i}{suffix}{ext}"
            if not filename.exists():
                return filename
                
        raise FileExistsError("Could not find a unique filename after 1000 attempts.")

    @staticmethod
    def _convert_prompt_to_filename(text: str, max_len: int = 30) -> str:
        """Convert prompt text to a valid filename"""
        clean_text = "".join(
            char.lower() for char in text if char.isalpha() or char.isspace()
        )
        words = clean_text.split()
        
        result = []
        current_length = 0
        
        for word in words:
            new_length = current_length + len(word)
            if new_length <= max_len:
                result.append(word)
                current_length += len(word)
            else:
                break
                
        return "-".join(result)

def main():
    """Main entry point for inference"""
    config = LTXVideoSettings()
    inference = LTXInference(config)
    inference.generate()

if __name__ == "__main__":
    main()