from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import soundfile as sf

# Load Pretrained Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Load Audio File
audio_file = "stop_1.wav"
speech_array, sampling_rate = torchaudio.load(audio_file)
speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
speech_array = speech_array.squeeze().numpy()

# Process Input Features
input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values

# Perform Inference
with torch.no_grad():
    logits = model(input_values).logits

# Decode Transcription and Probabilities
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
print("Transcription:", transcription)

# Optional: Generate Alignment
# Get frame-level probabilities and map them to characters
probs = torch.softmax(logits, dim=-1).squeeze(0)
char_vocab = processor.tokenizer.convert_ids_to_tokens(range(len(processor.tokenizer)))
print("PROBS: ", probs.shape) # its size is: [49, 32]
'''The number 49 means the model outputs 49 frames for this 1-second audio, 
meaning the model has a frame rate of approximately 50 frames per second.
'''

frame_duration = (len(speech_array)/16000)/probs.shape[0]
samples_per_frame = frame_duration * len(speech_array)

# Each frame represents a sliding window of samples_per_frame raw audio samples

# Create Alignment
alignment = []
for frame_idx, prob_distribution in enumerate(probs):
    char_id = torch.argmax(prob_distribution).item()
    char = char_vocab[char_id]
    if char != "<pad>" and char != processor.tokenizer.pad_token:
        alignment.append((frame_idx * samples_per_frame, char))  # Frame index and character
print("Alignment:", alignment)


import matplotlib.pyplot as plt
import numpy as np

# Plot waveform
plt.figure(figsize=(15, 5))
plt.plot(speech_array, alpha=0.7, label="Waveform")
for frame_idx, char in alignment:
    time = frame_idx #* (1 / 16000)  # Assuming frame rate = 16kHz
    plt.axvline(x=time, color="r", linestyle="--", alpha=0.5)
    plt.text(time, max(speech_array) * 0.8, char, rotation=90, color="red", fontsize=8)
plt.title("CTC Alignment")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


### TRANSCRIPT ONLY FIRST LETTER:
# Step 1: Find the frame range for the first "S"
for frame_idx, char in alignment:
    if char == "S":
        s_frame_idx = frame_idx
        break

# Step 2: Map frame indices to raw audio samples
frame_rate = len(probs) / 1.0  # Frames per second (assuming audio is 1 second long)
samples_per_frame = int(16000 / frame_rate)  # Raw samples per frame

# Calculate start and end sample indices for "S"
start_sample = int(s_frame_idx)
end_sample = int((s_frame_idx + samples_per_frame))

print("START: ",start_sample)
print("STOP: ", end_sample)

# Step 3: Extract the audio segment for "S"
s_audio_segment = speech_array[:8489]

# Step 4: Save the extracted segment to a new audio file
sf.write("s_audio.wav", s_audio_segment, 16000)
print(f"Extracted 'S' audio segment saved to s_audio.wav")