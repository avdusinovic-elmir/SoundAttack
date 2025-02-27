from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

from EA_wav3vec import EA



# ATTACK
audio_file = "stop_1.wav"
target_text = "STAR"  # Target transcription for the adversarial sample
population = 30
elits = 10
epochs = 100
mutatation_range = 0.015
epsilon = 0.08

speech_array, sampling_rate = torchaudio.load(audio_file)
speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
speech_array = speech_array.squeeze().numpy()
print("speech_array" , speech_array.shape)

# plt.plot(speech_array, label='Clean Audio')
# plt.legend()
# plt.show()

# Process Input Features
preprocessed_audio = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values


attackEA = EA(target=target_text, pop=population, elits=elits, mutatation_range=mutatation_range, epsilon=epsilon)

result, noise, fitness, ctc_loss = attackEA.attack_speech(org=preprocessed_audio, adv=target_text, epochs=epochs)


with torch.no_grad():
    logits = model(result).logits

# Decode Transcription and Probabilities
predicted_ids = torch.argmax(logits, dim=-1)
result_text = processor.batch_decode(predicted_ids)[0]

print("Transcription:", result_text)
print("Result:", result_text)
print("Fitness:", fitness)
print("CTC Loss:", ctc_loss)
#
result_array = speech_array+noise#result.squeeze(0).numpy()
# noise = noise.squeeze(0).numpy()

plt.plot(result_array, label='Adversarial Audio')
plt.plot(speech_array, label='Clean Audio')
plt.plot(noise, label='Adversarial Noise')
plt.legend()
plt.show()
print("Result Type:", type(result))
print("Result Shape:", result.shape)
result = result.detach().cpu().numpy()
result = np.asarray(result, dtype=np.float32)
result = result.squeeze()
print("Type of result:", type(result))

if isinstance(result, np.ndarray):
    print("Shape of result:", result.shape)
    print("Data type of result:", result.dtype)
sf.write("s_audio.wav", result, 16000)



