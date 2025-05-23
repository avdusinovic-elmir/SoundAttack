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
    folder = "Dataset/original_audio/"+command+"/original"
    dir_list = os.listdir(folder)
    size = len(dir_list)
    print(command)
    counter = 0
    for file in dir_list:
        # Load Audio File
        # random = random.choice(dir_list)
        # audio_file = folder+command+"/"+str(i+1)+".wav"
        audio_file = folder+"/"+file
        if not os.path.isfile(audio_file):
            continue
        speech_array, sampling_rate = torchaudio.load(audio_file)
        speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
        speech_array = speech_array.squeeze().numpy()

        speech_array_tensor = torch.tensor(speech_array, dtype=torch.float32).unsqueeze(0)
        speech_array_tensor = speech_array_tensor.squeeze().numpy()

        preprocessed_audio = processor(speech_array_tensor, sampling_rate=16000, return_tensors="pt",
                                       padding=True).input_values

        # Process Input Features
        # input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values

        # Perform Inference
        with torch.no_grad():
            logits = model(preprocessed_audio).logits

        # Decode Transcription and Probabilities
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        if transcription.lower() == command.lower():
            counter += 1
        else:
            print(f"{audio_file} has transcription {transcription}")
    print(counter)
