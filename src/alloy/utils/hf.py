import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from alloy.exceptions import HuggingFaceError, GatedModelError
from alloy.logging import get_logger
from alloy.utils.errors import get_download_suggestions, is_gated_model_error, get_gated_model_suggestions

load_dotenv()

logger = get_logger(__name__)


class HFManager:
    def __init__(self, token=None):
        self.token = token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.token)

    def login_check(self):
        """Verifies if the user is logged in."""
        try:
            user = self.api.whoami()
            logger.info("Logged in as: %s", user['name'])
            return True
        except Exception as e:
            logger.warning("Not logged in or invalid token: %s", e)
            logger.info("Please run 'huggingface-cli login' or provide a token.")
            return False

    def download_model(self, repo_id, local_dir=None):
        """Downloads a model from Hugging Face."""
        logger.info("Downloading %s...", repo_id)
        try:
            path = snapshot_download(repo_id=repo_id, local_dir=local_dir, token=self.token)
            logger.info("Model downloaded to: %s", path)
            return path
        except Exception as e:
            # Check if this is a gated model access issue
            if is_gated_model_error(e):
                raise GatedModelError(
                    f"Access denied to gated model: {e}",
                    repo_id=repo_id,
                    original_error=e,
                    suggestions=get_gated_model_suggestions(repo_id),
                ) from e
            raise HuggingFaceError(
                f"Failed to download model: {e}",
                repo_id=repo_id,
                original_error=e,
                suggestions=get_download_suggestions(repo_id),
            ) from e

    def upload_model(self, local_path, repo_id, private=True):
        """Uploads a converted model to Hugging Face."""
        logger.info("Uploading %s to %s...", local_path, repo_id)
        try:
            try:
                self.api.repo_info(repo_id)
            except RepositoryNotFoundError:
                logger.info("Creating repository %s...", repo_id)
                create_repo(repo_id, private=private, token=self.token)

            self.api.upload_folder(
                folder_path=local_path,
                repo_id=repo_id,
                repo_type="model",
                token=self.token
            )
            logger.info("Upload complete!")
        except Exception as e:
            logger.error("Error uploading model: %s", e)
            raise
