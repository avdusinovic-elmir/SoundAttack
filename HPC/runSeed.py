import os

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
print("TensorFlow version:", tf.version)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Using GPU:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

from EA_seed import EA

commands = ["no"]

population_array = [50]
elites_array = [0.3]
thresholds = [0.3, 0.2, 0.1, 0.05, 0.01]
mutation_list = [0.7]
epsilon_list = [0.5]

print("Seed run")

command = "yes"
# ATTACK
audio_file = "../Dataset/augmented_dataset/yes/197.wav"
print(os.listdir())
target_text = "no".upper()  # Target transcription for the adversarial sample
population = 50
elits = int(population*0.3)
epochs = 1
mutatation_range = 0.7
epsilon = 0.5
start = 0
end = 16000

speech_array, sampling_rate = torchaudio.load(audio_file)
speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
speech_array = speech_array.squeeze().numpy()
print("speech_array", speech_array.shape)
print("speech_array", speech_array)

speech_array_tensor = torch.tensor(speech_array, dtype=torch.float32).unsqueeze(0)
print("speech_array", speech_array_tensor.shape)
print("speech_array", speech_array_tensor)

# for threshold in thresholds:
sample_name = command + "_"+target_text
attackEA = EA(target=target_text, pop=population, elits=elits, mutatation_range=mutatation_range,
                epsilon=epsilon,
                start=start, end=end, sample_name=sample_name)

result, noise, fitness, ctc_loss, final_epoch, perceptual_loss = attackEA.attack_speech(org=speech_array_tensor, adv=target_text,
                                                            epochs=epochs, threshold=0)
result = result.squeeze().numpy()
result = processor(result, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
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
result_array = np.clip(speech_array + noise, -1, 1)  # result.squeeze(0).numpy()
# noise = noise.squeeze(0).numpy()

folder_path = "HPC/"

plt.figure()
plt.plot(result_array, label='Adversarial Audio')
plt.plot(speech_array, label='Clean Audio')
# plt.plot(noise, label='Adversarial Noise')
plt.title(f"{result_text}_{perceptual_loss}_{ctc_loss}")
plt.legend()
plt.savefig(f"{folder_path}{result_text}.png")
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
sf.write(f"{folder_path}{result_text}.wav", speech_array + noise, sampling_rate)
