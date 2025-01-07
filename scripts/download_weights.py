import os
from pathlib import Path
from configs.mochi_weights import MochiWeightsSettings
def download_weights(settings: MochiWeightsSettings = MochiWeightsSettings()):
    """
    Download model weights using configuration from pydantic settings.
    
    Args:
        settings: Settings instance containing configuration
    """
    if not settings.output_dir.exists():
        print(f"Creating output directory: {settings.output_dir}")
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    def download_file(repo_id: str, output_dir: Path, filename: str, description: str):
        file_path = output_dir / filename
        if not file_path.exists():
            print(f"Downloading mochi {description} to: {file_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{filename}*"],
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
        else:
            print(f"{description} already exists in: {file_path}")
        assert file_path.exists()

    download_file(settings.repo_id, settings.output_dir, settings.model_file, "model")
    download_file(settings.repo_id, settings.output_dir, settings.decoder_file, "decoder")
    download_file(settings.repo_id, settings.output_dir, settings.encoder_file, "encoder")

if __name__ == "__main__":
    download_weights()