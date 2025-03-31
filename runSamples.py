import os

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

from EA_reduce import EA

commands = ["yes", "up", "stop", "right", "on", "off", "no", "left", "go", "down"]

for command in commands:
    folder = "audio/"+command+"/original/"
    dir_list = os.listdir(folder)
    for i in range(len(dir_list)-1):
        # ATTACK
        audio_file = folder+command+"_"+str(i+1)+".wav"
        for target in commands:
            if target == command:
                continue

            target_text = target  # Target transcription for the adversarial sample
            population = 100
            elits = 15
            epochs = 1000
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

            attackEA = EA(target=target_text, pop=population, elits=elits, mutatation_range=mutatation_range,
                          epsilon=epsilon,
                          start=start, end=end)

            result, noise, fitness, ctc_loss, final_epoch, perceptual_loss = attackEA.attack_speech(org=speech_array_tensor, adv=target_text,
                                                                      epochs=epochs)
            result = result.squeeze().numpy()
            result = processor(result, sampling_rate=16000, return_tensors="pt", padding=True).input_values
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
            result_array = speech_array + noise  # result.squeeze(0).numpy()
            # noise = noise.squeeze(0).numpy()

            folder_path = "audio/"+command+"/adversarial/"+target+"/"

            plt.plot(result_array, label='Adversarial Audio')
            plt.plot(speech_array, label='Clean Audio')
            plt.plot(noise, label='Adversarial Noise')
            plt.title(f"{result_text}_{perceptual_loss}_{ctc_loss}")
            plt.legend()
            plt.savefig(f"{folder_path}{result_text}_{i+1}_{final_epoch}.png")
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
            sf.write(f"{folder_path}{result_text}_{i+1}_{final_epoch}.wav", speech_array + noise, sampling_rate)
