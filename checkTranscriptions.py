import os
import random

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch

# Load Pretrained Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

commands = ["yes", "up", "stop", "right", "on", "off", "no", "left", "go", "down"]

for command in commands:
    folder = "Dataset/synthetic-speech-commands-dataset/versions/4/augmented_dataset/augmented_dataset/"+command
    dir_list = os.listdir(folder)
    size = len(dir_list)
    print(command)
    counter = 0
    list_files = []
    while counter < 10:
        # Load Audio File
        # random = random.choice(dir_list)
        # audio_file = folder+command+"/"+str(i+1)+".wav"
        random_file = random.choice(dir_list)
        audio_file = folder+"/"+random_file
        if not os.path.isfile(audio_file):
            continue
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
        if transcription.lower() == command.lower() and audio_file not in list_files:
            print(f"Transcription {command}_{random_file}:", transcription)
            list_files.append(audio_file)
            counter += 1
            print(speech_array.shape)
