import wave
import soundfile as sf
import numpy as np
import torch
import librosa
import pandas as pd
import torchaudio

# Load the audio file
audio_folder_file_path = 'Dataset/mozillaorg/common-voice/versions/2/'
audio_csv_file_path = 'Dataset/mozillaorg/common-voice/versions/2/cv-valid-train.csv'

data = pd.read_csv(audio_csv_file_path)
data.drop(columns=["up_votes", "down_votes", "age", "gender", "accent", "duration"], inplace=True)

train_dataset = audio_folder_file_path + 'cv-valid-train/cv-valid-train/'
test_dataset = audio_folder_file_path + 'cv-valid-test/cv-valid-test/'


#--------------------MP3 files------------------------------------------
# # Define a function to convert MP3 to WAV using audioread
# # Export the audio as WAV
# wav_file_path = 'sample_converted.wav'
# # Load the MP3 file with librosa
# audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
#
# # Save as a WAV file using soundfile
# sf.write(wav_file_path, audio_data, sample_rate)
#
# # Now open the WAV file with wave
# w = wave.open(wav_file_path, 'r')
# rate = w.getframerate()
#
# frames = w.getnframes()
# buffer = w.readframes(frames)
# data16 = np.frombuffer(buffer, dtype=np.int16)
#
# x = data16.copy()

# text = model.stt(x)
# print(text)