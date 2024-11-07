from scripts.s3_manager import S3ManagerService
import uuid
import io
from io import BytesIO




def mp4_to_s3_json(video_bytes: io.BytesIO, file_name: str) -> dict:
    """
    Upload an MP4 video to Amazon S3 and return a JSON object with the video ID and signed URL.

    Args:
        video_bytes (io.BytesIO): The video data as bytes.
        file_name (str): The name of the file.

    Returns:
        dict: A JSON object containing the video ID and the signed URL.
    """
    video_id = str(uuid.uuid4())
    s3_uploader = S3ManagerService()

    unique_file_name = s3_uploader.generate_unique_file_name(file_name)
    s3_uploader.upload_file(video_bytes, unique_file_name)
    signed_url = s3_uploader.generate_signed_url(unique_file_name, exp=43200)  # 12 hours
    return {"video_id": video_id, "url": signed_url}
