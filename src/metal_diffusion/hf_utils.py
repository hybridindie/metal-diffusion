import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

load_dotenv()

class HFManager:
    def __init__(self, token=None):
        self.token = token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.token)

    def login_check(self):
        """Verifies if the user is logged in."""
        try:
            user = self.api.whoami()
            print(f"Logged in as: {user['name']}")
            return True
        except Exception as e:
            print(f"Not logged in or invalid token: {e}")
            print("Please run 'huggingface-cli login' or provide a token.")
            return False

    def download_model(self, repo_id, local_dir=None):
        """Downloads a model from Hugging Face."""
        print(f"Downloading {repo_id}...")
        try:
            path = snapshot_download(repo_id=repo_id, local_dir=local_dir, token=self.token)
            print(f"Model downloaded to: {path}")
            return path
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    def upload_model(self, local_path, repo_id, private=True):
        """Uploads a converted model to Hugging Face."""
        print(f"Uploading {local_path} to {repo_id}...")
        try:
            try:
                self.api.repo_info(repo_id)
            except RepositoryNotFoundError:
                print(f"Creating repository {repo_id}...")
                create_repo(repo_id, private=private, token=self.token)
            
            self.api.upload_folder(
                folder_path=local_path,
                repo_id=repo_id,
                repo_type="model",
                token=self.token
            )
            print("Upload complete!")
        except Exception as e:
            print(f"Error uploading model: {e}")
            raise
