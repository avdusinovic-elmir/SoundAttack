import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load pre-trained model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Load and preprocess audio file
audio_path = "stop_1.wav"  # Replace with your audio file
waveform, sr = librosa.load(audio_path, sr=16000)
waveform = waveform[12010:]
# Convert to tensor and add batch dimension
input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values

# Get logits (raw character scores)
with torch.no_grad():
    logits = model(input_values).logits  # Shape: (1, time_steps, vocab_size)

# Convert logits to probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)

# Decode predicted text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

# Get vocab mapping (token ID -> character)
id_to_char = {v: k for k, v in processor.tokenizer.get_vocab().items()}

# Extract character-level probabilities
char_probs = probs[0].cpu().numpy()  # Convert to NumPy
predicted_ids = predicted_ids[0].cpu().numpy()

# Print character probabilities
print("Predicted Text:", transcription)
print("\nCharacter Probabilities:\n")
for i, char_id in enumerate(predicted_ids):
    char = id_to_char.get(char_id, "[UNK]")  # Get corresponding character
    prob = char_probs[i, char_id]  # Probability of the selected character
    print(f"Char: {char}, Probability: {prob:.4f}")
