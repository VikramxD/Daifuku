from pydantic_settings import BaseSettings

class AllegroSettings(BaseSettings):
    """
    A Pydantic settings class for Allegro inference configuration.

    This class uses Pydantic to provide validation and easy environment-based configuration
    for Allegro inference pipeline settings.
    """

    model_name:str = "rhymes-ai/Allegro"
    device: str = "cuda"
    seed: int = 42
    guidance_scale: float = 7.5
    max_sequence_length: int = 512
    num_inference_steps: int = 100
    fps: int = 15

    class Config:
        """
        Pydantic configuration class for environment variable support.
        """
        env_prefix = "ALLEGRO_"  # Prefix for environment variables
        validate_assignment = True

    def __repr__(self):
        """
        Return a string representation of the settings for debugging purposes.

        :return: A string summarizing the settings.
        """
        return (f"AllegroSettings(model_name={self.model_name}, device={self.device}, seed={self.seed}, "
                f"guidance_scale={self.guidance_scale}, max_sequence_length={self.max_sequence_length}, "
                f"num_inference_steps={self.num_inference_steps}, fps={self.fps})")
