import kagglehub

# Download latest version
path = kagglehub.dataset_download("mozillaorg/common-voice")

print("Path to dataset files:", path)