import deepspeech
import wave
import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

from EA import EA
import librosa
import soundfile as sf
import torch
import tensorflow as tf
import torchaudio
from deepspeech import Model as DeepSpeechModel  # Ensure this matches your DeepSpeech version


# Load DeepSpeech model
ds_model_path = 'Targets/Deepspeech/deepspeech-0.9.3-models.pbmm'
ds = DeepSpeechModel(ds_model_path)

# # Load the deepspeech model
# model_file_path = 'Targets/Deepspeech/deepspeech-0.9.3-models.pbmm'
# model = deepspeech.Model(model_file_path)
#
# Load the scorer
scorer_file_path = 'Targets/Deepspeech/deepspeech-0.9.3-models.scorer'
# model.enableExternalScorer(scorer_file_path)
ds.enableExternalScorer(scorer_path=scorer_file_path)


####Single file test

# # Load the audio file
# audio_file_path = 'Dataset/mozillaorg/common-voice/versions/2/cv-valid-test/cv-valid-test/sample-000002.mp3'
#
# if os.path.exists(audio_file_path):
#     print("File exists")
# else:
#     print("File does not exist")
#
# #--------------------MP3 files------------------------------------------
# # Define a function to convert MP3 to WAV using audioread
# # Export the audio as WAV
# wav_file_path = 'sample_converted.wav'
# # Load the MP3 file with librosa
# audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
#
# # Save as a WAV file using soundfile
# sf.write(wav_file_path, audio_data, sample_rate)
#
# # Load and preprocess clean audio
# waveform, sample_rate = torchaudio.load(wav_file_path)
# target_text = "where are you from"  # Target transcription for the adversarial sample
#
# # Resample if necessary to match DeepSpeech sample rate 16000Hz
# if sample_rate != ds.sampleRate():
#     print("Sample rate changed")
#     transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=ds.sampleRate())
#     waveform = transform(waveform)
#
# # Normalize the waveform using TensorFlow functions
# waveform = waveform / tf.reduce_max(tf.abs(waveform))  # Normalize to [-1, 1]
#
# # Convert waveform to int16 for DeepSpeech
# waveform_int16 = (waveform.numpy()[0] * 32767).astype(np.int16)
#
# # Now pass waveform_int16 to the stt function
# original_transcription = ds.stt(waveform_int16)
# print("Orignal transcription: ", original_transcription)

# # Now open the WAV file with wave
# w = wave.open(wav_file_path, 'r')
# rate = w.getframerate()
#
# frames = w.getnframes()
# buffer = w.readframes(frames)
# data16 = np.frombuffer(buffer, dtype=np.int16)
# print(data16)
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

# x = data16.copy()
# # # x[15000:30000] = 0
# # print(type(x))
# text = model.stt(x)
# print(text)
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

############# Check the MCV dataset

# Load the audio file
audio_folder_file_path = 'Dataset/mozillaorg/common-voice/versions/2/cv-valid-test/'
audio_csv_file_path = 'Dataset/mozillaorg/common-voice/versions/2/cv-valid-test.csv'

data = pd.read_csv(audio_csv_file_path)
data.drop(columns=["up_votes", "down_votes", "age", "gender", "accent", "duration"], inplace=True)

score = 0

for i in range(len(data)):
    print(i)
    matching_score = 0
    file = data["filename"].iloc[i]
    y = data["text"].iloc[i]

    # Load the MP3 file with torchaudio directly
    waveform, sample_rate = torchaudio.load(audio_folder_file_path+file)

    # Resample if necessary to match DeepSpeech sample rate 16000Hz
    if sample_rate != ds.sampleRate():
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=ds.sampleRate())
        waveform = transform(waveform)

    # Normalize the waveform using TensorFlow functions
    waveform = waveform / tf.reduce_max(tf.abs(waveform))  # Normalize to [-1, 1]

    # Convert waveform to int16 for DeepSpeech
    waveform_int16 = (waveform.numpy()[0] * 32767).astype(np.int16)

    # Now pass waveform_int16 to the stt function
    original_transcription = ds.stt(waveform_int16)

    if original_transcription == y:
        score += 1
    else:
        split_original_transcription = original_transcription.split(" ")
        split_y = y.split(" ")

        for _ in split_y:
            if _ in split_original_transcription:
                matching_score += 1

        matching_score = matching_score/len(split_original_transcription)

    score += matching_score


print(score)
score /= len(data)
print(score)
