import boto3
import os
from src.constants import AWS_SECRET_ACCESS_KEY_ENV_KEY, AWS_ACCESS_KEY_ID_ENV_KEY, REGION_NAME


class S3Client:

    s3_client=None
    s3_resource = None
    def __init__(self, region_name=REGION_NAME):
        """ 
        This Class gets aws credentials from env_variable and creates an connection with s3 bucket 
        and raise exception when environment variable is not set
        """

        if S3Client.s3_resource == None or S3Client.s3_client == None:
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

            # If one is set but not the other, fail fast with a useful error.
            if bool(__access_key_id) ^ bool(__secret_access_key):
                raise Exception(
                    f"Both {AWS_ACCESS_KEY_ID_ENV_KEY} and {AWS_SECRET_ACCESS_KEY_ENV_KEY} must be set together "
                    "(or set neither to use AWS default credential chain)."
                )

            # Prefer explicit env creds when present; otherwise fall back to boto3 default chain
            # (shared credentials file, AWS_PROFILE, IAM role, etc.).
            if __access_key_id and __secret_access_key:
                S3Client.s3_resource = boto3.resource(
                    's3',
                    aws_access_key_id=__access_key_id,
                    aws_secret_access_key=__secret_access_key,
                    region_name=region_name
                )
                S3Client.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=__access_key_id,
                    aws_secret_access_key=__secret_access_key,
                    region_name=region_name
                )
            else:
                S3Client.s3_resource = boto3.resource('s3', region_name=region_name)
                S3Client.s3_client = boto3.client('s3', region_name=region_name)
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client