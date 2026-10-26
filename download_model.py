# download_model.py
from huggingface_hub import snapshot_download
snapshot_download("mlx-community/whisper-large-v3-turbo")
print("Modèle téléchargé.")