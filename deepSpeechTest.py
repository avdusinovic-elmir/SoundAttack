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

###Single file test

# Load the audio file
audio_file_path = 'Dataset/mozillaorg/common-voice/versions/2/cv-valid-test/cv-valid-test/sample-000003.mp3'

# if os.path.exists(audio_file_path):
#     print("File exists")
# else:
#     print("File does not exist")

#--------------------MP3 files------------------------------------------
# Define a function to convert MP3 to WAV using audioread
# Export the audio as WAV
# wav_file_path = 'sample_converted.wav'
# Load the MP3 file with torchaudio directly
waveform, sample_rate = torchaudio.load(audio_file_path)
original_wav = waveform
original_sample_rate = sample_rate

# # Save as a WAV file using soundfile
# sf.write(wav_file_path, audio_data, sample_rate)

# # Load and preprocess clean audio
# waveform, sample_rate = torchaudio.load(wav_file_path)
target_text = "where are you from"  # Target transcription for the adversarial sample

# Resample if necessary to match DeepSpeech sample rate 16000Hz
if sample_rate != ds.sampleRate():
    print("Sample rate changed")
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=ds.sampleRate())
    waveform = transform(waveform)

# Normalize the waveform using TensorFlow functions
normalization = tf.reduce_max(tf.abs(waveform))
waveform = waveform / normalization  # Normalize to [-1, 1]
# Convert waveform to int16 for DeepSpeech
waveform_int16 = (waveform.numpy()[0] * 32767).astype(np.int16)
# print(waveform_int16.shape)
# print(waveform_int16)
# print(str(min(waveform_int16))+"/"+str(max(waveform_int16)))

# Now pass waveform_int16 to the stt function
original_transcription = ds.stt(waveform_int16)
print("Orignal transcription: ", original_transcription)

alg = EA(target=target_text)
print(max(waveform_int16))
print(min(waveform_int16))
result, fitness, ctc_loss = alg.attack_speech(org=waveform_int16, adv=target_text, model=ds, epochs=100)
result_text = ds.stt(waveform_int16 + result)
print("Result:", result_text)
print("Fitness:", fitness)
print("CTC Loss:", ctc_loss)

def save_audio_with_noise(noise, sr, file_path, n):
    new_wav = original_wav

    # Convert to PyTorch tensor if not already
    if not isinstance(noise, torch.Tensor):
        noise = torch.tensor(noise, dtype=torch.float32)

    transform2 = torchaudio.transforms.Resample(orig_freq=ds.sampleRate(), new_freq=sr)
    noise2 = transform2(noise)

    # if waveform_float32.dim() == 1:
    #     waveform_float32 = waveform_float32.unsqueeze(0)

    print(new_wav.shape)
    print(noise2.shape)
    noise2 /= max(waveform_int16)
    new_wav = new_wav + noise2
    # Save the waveform as a WAV file
    torchaudio.save(file_path, new_wav, sr)


save_audio_with_noise(result, original_sample_rate, "adv_audio.mp3", normalization)
# save_audio_with_noise(result, original_sample_rate, "adv_audio.wav", normalization)

waveform, sample_rate = torchaudio.load("adv_audio.mp3")

# Resample if necessary to match DeepSpeech sample rate 16000Hz
if sample_rate != ds.sampleRate():
    print("Sample rate changed")
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=ds.sampleRate())
    waveform = transform(waveform)

# Normalize the waveform using TensorFlow functions
normalization = tf.reduce_max(tf.abs(waveform))
waveform = waveform / normalization  # Normalize to [-1, 1]
# Convert waveform to int16 for DeepSpeech
waveform_int16 = (waveform.numpy()[0] * 32767).astype(np.int16)
# print(waveform_int16.shape)
# print(waveform_int16)
# print(str(min(waveform_int16))+"/"+str(max(waveform_int16)))

# Now pass waveform_int16 to the stt function
original_transcription = ds.stt(waveform_int16)
print("Orignal transcription: ", original_transcription)


# def save_audio_from_int16(wf_int16, sr, file_path, n):
#     # Convert to PyTorch tensor if not already
#     if not isinstance(wf_int16, torch.Tensor):
#         wf_int16 = torch.tensor(wf_int16, dtype=torch.int16)
#
#     # Normalize from int16 to float32 in the range [-1.0, 1.0]
#     waveform_float32 = wf_int16.to(torch.float32) / 32768.0
#
#     normalization = float(n)
#     waveform_float32 = waveform_float32 * normalization
#
#     if waveform_float32.dim() == 1:
#         waveform_float32 = waveform_float32.unsqueeze(0)
#
#     # waveform_float32 = waveform_float32 * normalization
#
#
#     transform2 = torchaudio.transforms.Resample(orig_freq=ds.sampleRate(), new_freq=sr)
#     waveform_float32 = transform2(waveform_float32)
#
#     # Save the waveform as a WAV file
#     torchaudio.save(file_path, waveform_float32, sr)
#
#
# save_audio_from_int16(waveform_int16, sample_rate, "adv_audio.mp3", normalization)
# save_audio_from_int16(waveform_int16, sample_rate, "adv_audio.wav", normalization)


def difference_spectrogram(mp3_path, wav_path):
    # Load MP3 file
    y_mp3, sr_mp3 = librosa.load(mp3_path, sr=None)
    spectrogram_mp3 = librosa.stft(y_mp3)
    spectrogram_db_mp3 = librosa.amplitude_to_db(np.abs(spectrogram_mp3), ref=np.max)

    # Load WAV file
    y_wav, sr_wav = librosa.load(wav_path, sr=None)
    spectrogram_wav = librosa.stft(y_wav)
    spectrogram_db_wav = librosa.amplitude_to_db(np.abs(spectrogram_wav), ref=np.max)

    # Ensure both spectrograms are aligned by time and frequency
    min_time = min(spectrogram_db_mp3.shape[1], spectrogram_db_wav.shape[1])
    spectrogram_db_mp3 = spectrogram_db_mp3[:, :min_time]
    spectrogram_db_wav = spectrogram_db_wav[:, :min_time]

    # Calculate absolute differences
    difference = np.abs(spectrogram_db_mp3 - spectrogram_db_wav)

    # Plot the differences
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(difference, sr=sr_mp3, x_axis='time', y_axis='log', cmap='coolwarm')
    plt.colorbar(format="%+2.0f dB", label="Difference Intensity")
    plt.title("Difference Spectrogram Between Original and Adversarial")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


difference_spectrogram(mp3_path=audio_file_path, wav_path="adv_audio.mp3")

# Now open the WAV file with wave
# w = wave.open(wav_file_path, 'r')
# rate = w.getframerate()
#
#
# frames = w.getnframes()
# buffer = w.readframes(frames)
# data16 = np.frombuffer(buffer, dtype=np.int16)
# print(data16)
# --------------------MP3 files------------------------------------------

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

# # Load the audio file
# audio_folder_file_path = "Dataset/mozillaorg/common-voice/versions/2/cv-valid-test/"
# audio_csv_file_path = "Dataset/mozillaorg/common-voice/versions/2/cv-valid-test.csv"
# matching_cases_file = open("matchingCases.txt", "w")
#
# data = pd.read_csv(audio_csv_file_path)
# data.drop(columns=["up_votes", "down_votes", "age", "gender", "accent", "duration"], inplace=True)
#
# score = 0
#
# for i in range(len(data)):
#     print(i)
#     matching_score = 0
#     file = data["filename"].iloc[i]
#     y = data["text"].iloc[i]
#
#     # Load the MP3 file with torchaudio directly
#     waveform, sample_rate = torchaudio.load(audio_folder_file_path+file)
#
#     # Resample if necessary to match DeepSpeech sample rate 16000Hz
#     if sample_rate != ds.sampleRate():
#         transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=ds.sampleRate())
#         waveform = transform(waveform)
#
#     # Normalize the waveform using TensorFlow functions
#     waveform = waveform / tf.reduce_max(tf.abs(waveform))  # Normalize to [-1, 1]
#
#     # Convert waveform to int16 for DeepSpeech
#     waveform_int16 = (waveform.numpy()[0] * 32767).astype(np.int16)
#
#     # Now pass waveform_int16 to the stt function
#     original_transcription = ds.stt(waveform_int16)
#
#     if original_transcription == y:
#         score += 1
#         ####### Save matching cases in text file #######
#         matching_cases_file.write(str(i)+"\n")
#         ################################################
#     # else:
#     #     split_original_transcription = original_transcription.split(" ")
#     #     split_y = y.split(" ")
#     #
#     #     for _ in split_y:
#     #         if _ in split_original_transcription:
#     #             matching_score += 1
#     #
#     #     matching_score = matching_score/len(split_original_transcription)
#
#     score += matching_score
#
# matching_cases_file.close()
# print(score)
# score /= len(data)
# print(score)
