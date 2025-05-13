import kagglehub

# Download latest version
# path = kagglehub.dataset_download("mozillaorg/common-voice")
path = kagglehub.dataset_download("jbuchner/synthetic-speech-commands-dataset")
print("Path to dataset files:", path)