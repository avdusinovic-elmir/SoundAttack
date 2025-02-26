import torch
import torchaudio
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import matplotlib.pyplot as plt

# Load a pre-trained ASR model with CTC loss
model_name = "facebook/wav2vec2-large-960h"  # Wav2Vec2 ASR model
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode


def extract_audio_fragment(audio_path, start_sample, end_sample, min_samples=1600, save_path=None):
    """
    Extracts a specific fragment of an audio file from start_sample to end_sample.
    Ensures the fragment is long enough for Wav2Vec2.
    Optionally saves the extracted fragment.
    """
    waveform, sample_rate = torchaudio.load(audio_path)  # Load audio
    extracted_fragment = waveform[:, start_sample:end_sample]  # Extract segment

    # Ensure the fragment is long enough
    if extracted_fragment.shape[1] < min_samples:
        raise ValueError(f"Audio fragment too short ({extracted_fragment.shape[1]} samples). Increase segment length!")

    # Save the extracted fragment if a save path is provided
    if save_path:
        # extracted_fragment = extracted_fragment / extracted_fragment.abs().max()
        # torchaudio.save(save_path, extracted_fragment, sample_rate)
        print(f"Extracted fragment saved at: {save_path}")

    return extracted_fragment, sample_rate


def compute_letter_probabilities(audio_tensor, sample_rate):
    """
    Converts an audio fragment to letter probabilities using a pre-trained CTC model.
    """
    # Resample if needed
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_tensor = transform(audio_tensor)

    # Process audio and get logits
    input_values = processor(audio_tensor.squeeze(), sampling_rate=16000, return_tensors="pt").input_values

    if input_values.shape[1] < 10:
        raise ValueError("Processed audio is too short for ASR model.")

    with torch.no_grad():
        logits = model(input_values).logits  # Model output logits

    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Average over time steps to get letter-level probabilities
    avg_probs = probabilities.mean(dim=0)

    # Define alphabet (26 letters + blank token + space)
    alphabet = ["_"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

    # Map probabilities to letters
    letter_probs = {alphabet[i]: avg_probs[0, i].item() for i in range(len(alphabet))}

    return letter_probs


def plot_waveform(audio_tensor, sample_rate):
    plt.figure(figsize=(10, 3))
    plt.plot(audio_tensor.squeeze().numpy())
    plt.title("Extracted Audio Waveform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()


# Example usage
audio_path = "Dataset/mozillaorg/common-voice/versions/2/cv-valid-test/cv-valid-test/sample-000002.mp3"  # Replace with your audio file
start = 11988
start_sample = start
end_sample = 24624  # w2v requires at least 1600 samples (3200 to make it audible)

save_path = "extracted_fragment.wav"  # Set a file name for saving the fragment

# Extract audio fragment and save it
audio_fragment, sr = extract_audio_fragment(audio_path, start_sample, end_sample, save_path=save_path)

# Plot before saving
plot_waveform(audio_fragment, sr)

# Compute letter probabilities
letter_probs = compute_letter_probabilities(audio_fragment, sr)

# Print results
print("Letter Probabilities for Extracted Fragment:")
for letter, prob in letter_probs.items():
    print(f"{letter}: {prob:.4f}")
