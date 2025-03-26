from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# from EA_wav3vec import EA
from EA_range_32 import EA

mutation_list = [0.5, 0.25, 0.1, 0.05, 0.01]
epsilon_list = [0.5, 0.25, 0.1, 0.05, 0.01]
folder_path = "gridSearch/epoch300/"

for m in mutation_list:
    for e in epsilon_list:
        # ATTACK
        audio_file = "YES.wav"
        target_text = "NO"  # Target transcription for the adversarial sample
        population = 75
        elits = 10
        epochs = 300
        mutatation_range = m
        epsilon = e
        start = 7836 * 2
        end = 12408 * 2

        speech_array, sampling_rate = torchaudio.load(audio_file)
        temp = speech_array
        speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=32000)(speech_array)
        speech_array = speech_array.squeeze().numpy()
        temp = temp.squeeze().numpy()
        print("speech_array", speech_array.shape)
        print("speech_array", speech_array)

        speech_array_tensor = torch.tensor(speech_array, dtype=torch.float32).unsqueeze(0)
        print("speech_array", speech_array_tensor.shape)
        print("speech_array", speech_array_tensor)

        # plt.plot(speech_array, label='Clean Audio')
        # plt.legend()
        # plt.show()

        # Process Input Features
        # preprocessed_audio = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        # print(preprocessed_audio.shape)
        # print("preprocessed_audio", preprocessed_audio)

        attackEA = EA(target=target_text, pop=population, elits=elits, mutatation_range=mutatation_range,
                      epsilon=epsilon,
                      start=start, end=end)

        result, noise, fitness, ctc_loss, final_epoch = attackEA.attack_speech(org=speech_array_tensor, adv=target_text,
                                                                  epochs=epochs)
        result = torchaudio.transforms.Resample(orig_freq=32000, new_freq=16000)(result)
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

        plt.plot(result_array, label='Adversarial Audio')
        plt.plot(speech_array, label='Clean Audio')
        plt.plot(noise, label='Adversarial Noise')
        plt.title(f"s_audio_{m}_{e}_{ctc_loss}")
        plt.legend()
        plt.savefig(f"{folder_path}s_audio_{m}_{e}_{final_epoch}.png")
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

        new_audio = temp+noise
        resulting_sound = torchaudio.transforms.Resample(orig_freq=32000, new_freq=16000)(new_audio)
        sf.write(f"{folder_path}{result_text}_audio_{m}_{e}_{final_epoch}.wav", resulting_sound, sampling_rate)
