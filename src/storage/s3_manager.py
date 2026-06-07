import os
import joblib
import logging
import tempfile
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from src import config

logger = logging.getLogger(__name__)

class S3Manager:
    def __init__(self):
        self.bucket_name = config.S3_BUCKET_NAME
        self.local_dir = Path(config.LOCAL_STORAGE_DIR)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.s3_client = None
        self.use_s3 = False

        # Attempt to initialize boto3 client
        try:
            # If endpoint URL is specified, try to connect to MinIO/local S3 mockup
            kwargs = {
                "aws_access_key_id": config.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": config.AWS_SECRET_ACCESS_KEY,
                "region_name": config.AWS_DEFAULT_REGION,
            }
            if config.S3_ENDPOINT_URL:
                kwargs["endpoint_url"] = config.S3_ENDPOINT_URL
            
            self.s3_client = boto3.client("s3", **kwargs)
            
            # Verify bucket existence or try creating it
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                logger.info(f"Connected to S3. Bucket '{self.bucket_name}' exists.")
                self.use_s3 = True
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                if error_code == '404' or error_code == 'NoSuchBucket':
                    logger.info(f"Bucket '{self.bucket_name}' not found. Attempting to create it...")
                    # For region constraint, us-east-1 does not need LocationConstraint
                    if config.AWS_DEFAULT_REGION == "us-east-1":
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': config.AWS_DEFAULT_REGION}
                        )
                    self.use_s3 = True
                    logger.info(f"Bucket '{self.bucket_name}' created successfully.")
                else:
                    logger.warning(f"S3 access test failed: {e}. Falling back to local storage.")
        except Exception as e:
            logger.warning(f"Could not initialize S3 client: {e}. Falling back to local storage.")

    def upload_file(self, local_path: str, s3_key: str):
        """Uploads a local file to S3 or moves it to the local store fallback."""
        if self.use_s3:
            try:
                self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
                logger.info(f"Uploaded {local_path} to S3 bucket {self.bucket_name} as {s3_key}")
                return True
            except ClientError as e:
                logger.error(f"Failed to upload to S3: {e}")
                # Fallback to local
        
        # Local fallback
        dest_path = self.local_dir / s3_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Check if they point to the exact same physical path
        if Path(local_path).resolve() != dest_path.resolve():
            import shutil
            shutil.copy2(local_path, dest_path)
            logger.info(f"Saved locally to {dest_path}")
        else:
            logger.info(f"File already exists in local storage path: {dest_path}")
        return True

    def download_file(self, s3_key: str, local_path: str):
        """Downloads a file from S3 or retrieves it from the local store fallback."""
        if self.use_s3:
            try:
                self.s3_client.download_file(self.bucket_name, s3_key, local_path)
                logger.info(f"Downloaded S3 key {s3_key} from bucket {self.bucket_name} to {local_path}")
                return True
            except ClientError as e:
                logger.error(f"Failed to download from S3: {e}")
                # Fallback to local
        
        # Local fallback
        src_path = self.local_dir / s3_key
        if src_path.exists():
            # Check if they point to the exact same physical path
            if src_path.resolve() != Path(local_path).resolve():
                import shutil
                shutil.copy2(src_path, local_path)
                logger.info(f"Retrieved locally from {src_path} to {local_path}")
            else:
                logger.info(f"File already exists at local destination: {local_path}")
            return True
        else:
            logger.error(f"Local file {src_path} not found.")
            return False

    def save_pkl(self, obj, s3_key: str):
        """Saves a python object directly as a PKL file in S3 or local directory."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            joblib.dump(obj, temp_path)
            self.upload_file(temp_path, s3_key)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def load_pkl(self, s3_key: str):
        """Loads a python object directly from a PKL file in S3 or local directory."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            if self.download_file(s3_key, temp_path):
                return joblib.load(temp_path)
            else:
                return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def list_files(self, prefix: str = ""):
        """Lists files with given prefix."""
        if self.use_s3:
            try:
                response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
                return [obj['Key'] for obj in response.get('Contents', [])]
            except Exception as e:
                logger.error(f"Failed to list from S3: {e}")
        
        # Local fallback
        search_dir = self.local_dir / prefix
        if search_dir.is_dir():
            return [str(p.relative_to(self.local_dir)) for p in search_dir.rglob("*") if p.is_file()]
        elif (self.local_dir).exists():
            return [str(p.relative_to(self.local_dir)) for p in self.local_dir.rglob(f"{prefix}*") if p.is_file()]
        return []
