import torch
from diffusers import AutoencoderKLAllegro, AllegroPipeline
from diffusers.utils import export_to_video
from loguru import logger
from configs.allegro_settings import AllegroSettings

class AllegroInference:
    """
    A class for managing the Allegro inference pipeline for generating videos based on textual prompts.

    This class encapsulates the initialization, configuration, and video generation processes
    for the Allegro model pipeline. It provides a streamlined way to handle prompts, model setup,
    and output file management in a production-grade environment.
    """

    def __init__(self, settings: AllegroSettings):
        """
        Initialize the AllegroInference class with the given settings.

        :param settings: An instance of AllegroSettings containing model, device, and generation parameters.
        """
        self.settings = settings
        self.pipe = None

        logger.info(f"Initializing {self.settings.model_name} inference pipeline")
        self._setup_pipeline()

    def _setup_pipeline(self):
        """
        Set up the Allegro model pipeline by loading the VAE and the pipeline with specified configurations.

        This method loads the models, moves them to the specified device, and enables tiling for
        efficient memory usage during inference.

        :raises Exception: If there is an error during the model loading or configuration process.
        """
        try:
            # Load VAE
            logger.info("Loading VAE model...")
            vae = AutoencoderKLAllegro.from_pretrained(
                self.settings.model_name, 
                subfolder="vae", 
                torch_dtype=torch.float32
            )

            # Load Allegro pipeline
            logger.info("Loading Allegro pipeline...")
            self.pipe = AllegroPipeline.from_pretrained(
                self.settings.model_name, 
                vae=vae, 
                torch_dtype=torch.bfloat16
            )
            
            # Move pipeline to the specified device
            self.pipe.to(self.settings.device)

            # Enable tiling for efficient memory usage
            self.pipe.vae.enable_tiling()

            logger.info("Pipeline successfully initialized")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise

    def generate_video(self, prompt: str, positive_prompt: str, negative_prompt: str, output_path: str):
        """
        Generate a video based on the provided prompts and save it to the specified path.

        :param prompt: The main textual description of the video scene.
        :param positive_prompt: Additional positive prompts to enhance quality and style.
        :param negative_prompt: Prompts to avoid undesirable features in the generated video.
        :param output_path: File path to save the generated video.
        :raises Exception: If there is an error during video generation or export.
        """
        try:
            logger.info("Preparing prompts...")
            prompt = positive_prompt.format(prompt.lower().strip())

            logger.info("Generating video...")
            generator = torch.Generator(device=self.settings.device).manual_seed(self.settings.seed)
            video_frames = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=self.settings.guidance_scale,
                max_sequence_length=self.settings.max_sequence_length,
                num_inference_steps=self.settings.num_inference_steps,
                generator=generator
            ).frames[0]

            logger.info(f"Exporting video to {output_path}...")
            export_to_video(video_frames, output_path, fps=self.settings.fps)

            logger.info("Video generation completed successfully")
        except Exception as e:
            logger.error(f"Error during video generation: {e}")
            raise


if __name__ == "__main__":
    settings = AllegroSettings()
    inference = AllegroInference(settings)

    prompt = "A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this location might be a popular spot for docking fishing boats."
    positive_prompt = """
    (masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
    {} 
    emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
    sharp focus, high budget, cinemascope, moody, epic, gorgeous
    """

    negative_prompt = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """

    output_path = "output.mp4"
    inference.generate_video(prompt, positive_prompt, negative_prompt, output_path)
