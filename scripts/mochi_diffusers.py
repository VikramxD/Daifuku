"""
Inference class for Mochi video generation model.
Tested on A6000 /A40 / A100 
Generates Video for 20 inference steps at 480 resolution in 10 minutes
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple
import torch
from loguru import logger
from diffusers import MochiPipeline, MochiTransformer3DModel, AutoencoderKLMochi
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from configs.mochi_settings import MochiSettings
import gc

class MochiInference:
    """
    Production-ready inference class for Mochi video generation model.
    
    Args:
        settings (MochiSettings): Configuration settings for the model.
        
    Attributes:
        settings (MochiSettings): Model and inference configuration.
        pipe (MochiPipeline): Loaded diffusers pipeline.
        transformer (MochiTransformer3DModel): Loaded transformer model.
    """
    
    def __init__(self, settings: MochiSettings):
        """Initialize Mochi inference with provided settings."""
        self.settings = settings
        self.pipe = None
        self.transformer = None
        
        logger.info(f"Initializing {settings.model_name} inference pipeline")
        self._setup_pipeline()
        
    def _setup_pipeline(self) -> None:
        """
        Set up the Mochi pipeline with specified configuration.
        
        Raises:
            RuntimeError: If pipeline setup fails.
        """
        try:
            # Validate paths and CUDA availability
            self.settings.validate_paths()
            
            logger.info("Loading transformer model from {}", self.settings.transformer_path)
            self.transformer = MochiTransformer3DModel.from_pretrained(
                self.settings.transformer_path,
                torch_dtype=self.settings.dtype
            )
            
            logger.info("Loading pipeline from {}", self.settings.pipeline_path)
            self.pipe = MochiPipeline.from_pretrained(
                self.settings.pipeline_path,
                torch_dtype=self.settings.dtype,
                transformer=self.transformer
            )
            
            # Move to device
            logger.info("Moving pipeline to {}", self.settings.device)
            self.pipe.to(self.settings.device)
            
            # Apply optimizations based on settings
            if self.settings.enable_vae_tiling:
                logger.info("Enabling VAE tiling")
                self.pipe.enable_vae_tiling()
                
            if self.settings.enable_model_cpu_offload:
                logger.info("Enabling model CPU offload")
                self.pipe.enable_model_cpu_offload()
                
            logger.success("Pipeline setup complete")
            
        except Exception as e:
            logger.exception("Failed to setup pipeline")
            raise RuntimeError(f"Pipeline setup failed: {str(e)}") from e

    def generate(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Union[str, List[torch.Tensor]]:
        """
        Generate video from text prompt.
        
        Args:
            prompt: Text description of the video to generate.
            output_path: Path to save the output video. If None, returns frames.
            num_inference_steps: Override default number of inference steps.
            guidance_scale: Override default guidance scale.
            height: Override default height.
            width: Override default width.
            num_frames: Override default number of frames.
            fps: Override default frames per second.
            negative_prompt: Text description of what to avoid in generation.
            seed: Random seed for reproducibility.
            
        Returns:
            Union[str, List[torch.Tensor]]: Either path to saved video or list of frames.
            
        Raises:
            RuntimeError: If video generation fails.
        """
        logger.info("Starting video generation for prompt: {}", prompt)
        try:
            # Set random seed if provided
            if seed is not None:
                logger.info("Setting random seed to {}", seed)
                torch.manual_seed(seed)
                
            # Encode prompt first and free text encoder memory
            with torch.no_grad():
                logger.debug("Encoding prompt")
                prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
                    self.pipe.encode_prompt(prompt=prompt)
                )
            
            logger.debug("Freeing text encoder memory")
            del self.pipe.text_encoder
            gc.collect()
            
            # Generate latents
            logger.debug("Generating latents")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                frames = self.pipe(
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                    guidance_scale=guidance_scale or self.settings.guidance_scale,
                    num_inference_steps=num_inference_steps or self.settings.num_inference_steps,
                    height=height or self.settings.height,
                    width=width or self.settings.width,
                    num_frames=num_frames or self.settings.num_frames,
                    output_type="latent",
                    return_dict=False,
                )[0]

            logger.debug("Freeing transformer memory")
            del self.pipe.transformer
            gc.collect()

            # Setup VAE and process latents
            logger.debug("Loading VAE model")
            vae = AutoencoderKLMochi.from_pretrained(
                self.settings.pipeline_path, 
                subfolder="vae"
            ).to(self.settings.device)
            vae._enable_framewise_decoding()
            
            # Scale latents appropriately
            logger.debug("Scaling latents")
            if hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std"):
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, 12, 1, 1, 1).to(frames.device, frames.dtype)
                latents_std = torch.tensor(vae.config.latents_std).view(1, 12, 1, 1, 1).to(frames.device, frames.dtype)
                frames = frames * latents_std / vae.config.scaling_factor + latents_mean
            else:
                frames = frames / vae.config.scaling_factor

            # Decode frames
            logger.debug("Decoding frames")
            with torch.no_grad():
                video = vae.decode(frames.to(vae.dtype), return_dict=False)[0]

            logger.debug("Post-processing video")
            video_processor = VideoProcessor(vae_scale_factor=8)
            video = video_processor.postprocess_video(video)[0]

            # Save or return video
            if output_path:
                logger.info("Saving video to {}", output_path)
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                fps = fps or self.settings.fps
                export_to_video(video, str(output_path), fps=fps)
                logger.success("Video saved to: {}", output_path)
                return str(output_path)
            
            logger.success("Video generation completed successfully")
            return video

        except Exception as e:
            logger.exception("Video generation failed")
            raise RuntimeError(f"Video generation failed: {str(e)}") from e

   
            
    def get_memory_usage(self) -> Tuple[float, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Tuple[float, float]: (allocated_memory_gb, max_memory_gb)
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            logger.debug("Memory usage: {:.2f}GB (peak: {:.2f}GB)", allocated, max_allocated)
            return (allocated, max_allocated)
        return (0.0, 0.0)
    
    def clear_memory(self) -> None:
        """Clear CUDA memory cache and reset memory statistics."""
        if torch.cuda.is_available():
            logger.info("Clearing CUDA memory cache")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    from configs.mochi_settings import MochiSettings

    settings = MochiSettings()

    # Initialize inference class
    mochi_inference = MochiInference(settings)

    # Define prompt and output path
    prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
    output_path = "/home/ubuntu/Minimochi/outputs/output.mp4"

    # Generate video
    try:
        video_path = mochi_inference.generate(
            prompt=prompt,
            negative_prompt='((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))',
            output_path=output_path,
            num_inference_steps=30,  
            guidance_scale=3.5,
            height=480,
            width=848,
            num_frames=150,
            fps=30,
        )
        print(f"Video saved to: {video_path}")
    except RuntimeError as e:
        print(f"Failed to generate video: {e}")
    allocated, max_allocated = mochi_inference.get_memory_usage()
    print(f"Memory usage: {allocated:.2f}GB (peak: {max_allocated:.2f}GB)")
    mochi_inference.clear_memory()
