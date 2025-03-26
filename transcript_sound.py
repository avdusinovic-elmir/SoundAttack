from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import soundfile as sf

# Load Pretrained Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Load Audio File
audio_file = "gridSearch/generated_audio_epoch_234.wav"
speech_array, sampling_rate = torchaudio.load(audio_file)
# speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
# speech_array = speech_array.squeeze().numpy()

# Process Input Features
# input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values

# Perform Inference
with torch.no_grad():
    logits = model(speech_array).logits

# Decode Transcription and Probabilities
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
print("Transcription:", transcription)

import torchaudio.transforms as T
def perceptual_loss_combined(original_audio, adversarial_noise):
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,  # Increase FFT size to capture more frequencies
        win_length=400,  # Standard for 16kHz speech
        hop_length=160,  # 10ms step (160 samples for 16kHz)
        n_mels=80  # Reduce from 128 to 80 for stability
    )

    # Compute Mel spectrograms
    original_mel = mel_transform(original_audio)
    adversarial_mel = mel_transform(adversarial_noise)

    # Compute log-Mel loss
    log_original = torch.log(original_mel + 1e-6)
    log_adversarial = torch.log(adversarial_mel + 1e-6)
    mel_loss = torch.nn.functional.mse_loss(log_original, log_adversarial)

    # Compute waveform loss (to capture transient noises)
    waveform_loss = torch.nn.functional.mse_loss(original_audio, adversarial_noise)

    # Combine losses (adjust weight if needed)
    total_loss = 0.7 * mel_loss + 0.3 * waveform_loss

    return total_loss.item()

original_audio = "YES.wav"
speech_array2, sampling_rate2 = torchaudio.load(original_audio)
print(perceptual_loss_combined(speech_array2, speech_array))
