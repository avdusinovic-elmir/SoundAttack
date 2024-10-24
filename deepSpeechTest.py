import deepspeech
import wave
import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
from EA import EA
import librosa
import soundfile as sf

# Load the deepspeech model
model_file_path = 'Targets/Deepspeech/deepspeech-0.9.3-models.pbmm'
model = deepspeech.Model(model_file_path)

# Load the scorer
scorer_file_path = 'Targets/Deepspeech/deepspeech-0.9.3-models.scorer'
model.enableExternalScorer(scorer_file_path)

# Load the audio file
audio_file_path = 'Dataset/mozillaorg/common-voice/versions/2/cv-valid-dev/cv-valid-dev/sample-000000.mp3'

if os.path.exists(audio_file_path):
    print("File exists")
else:
    print("File does not exist")

#--------------------MP3 files------------------------------------------
# Define a function to convert MP3 to WAV using audioread
# Export the audio as WAV
wav_file_path = 'sample_converted.wav'
# Load the MP3 file with librosa
audio_data, sample_rate = librosa.load(audio_file_path, sr=None)

# Save as a WAV file using soundfile
sf.write(wav_file_path, audio_data, sample_rate)

# Now open the WAV file with wave
w = wave.open(wav_file_path, 'r')
rate = w.getframerate()

frames = w.getnframes()
buffer = w.readframes(frames)
data16 = np.frombuffer(buffer, dtype=np.int16)
print(data16)
#--------------------MP3 files------------------------------------------

#--------------------WAV files------------------------------------------
# w = wave.open(audio_file_path, 'r')
# rate = w.getframerate()
#
# frames = w.getnframes()
# buffer = w.readframes(frames)
# data16 = np.frombuffer(buffer, dtype=np.int16)
# print(data16.dtype)
#--------------------WAV files------------------------------------------

# data16[0] = 50

x = data16.copy()
# # x[15000:30000] = 0
# print(type(x))
text = model.stt(x)
print(text)
#
# target = "Facts validate this."
# print(random.random())

# alg = EA(target=target)
# result = alg.attack_speech(org=x, adv=target, model=model, epochs=300)
# print(result)
#
# plt.plot(x)
# plt.show()
# Transcribe the audio file