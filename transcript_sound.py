from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import soundfile as sf

# Load Pretrained Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Load Audio File
audio_file = "2830-3980-0043.wav"
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