import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio.transforms as T
import torchaudio.functional as F
from Levenshtein import distance

model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

import time
import soundfile as sf
from Loss.ctcLoss import *

vocab = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
         'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
         't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, ' ': 27}


# from semantic_text_similarity.models import WebBertSimilarity
class Individual:
    def __init__(self, solution, fitness, ctc_fitness):
        self.solution = solution
        self.fitness = fitness
        self.ctc_fitness = ctc_fitness


class EA:
    def __init__(self, target, pop, elits, mutatation_range, epsilon, start, end):
        self.pop = pop
        self.target = target
        self.elits = elits
        self.mutation_range = mutatation_range
        self.epsilon = epsilon
        self.start = start
        self.end = end

    # Step 1: Generate Population

    def preprocess_audio(self, org_audio):
        # speech_array, sampling_rate = torchaudio.load(org_audio)
        # org_audio = torchaudio.transforms.Resample(orig_freq=org_audio, new_freq=16000)(speech_array)
        org_audio = org_audio.squeeze().numpy()

        # Process Input Features
        preprocessed_audio = processor(org_audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values

        return preprocessed_audio

    def transcript_audio(self, audio_file):
        # input_values = self.preprocess_audio(audio_file)
        # Perform Inference
        # ADD HERE PREPROCESS
        audio_file = self.preprocess_audio(audio_file)
        with torch.no_grad():
            logits = model(audio_file).logits

        # Decode Transcription and Probabilities
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        # print("Transcription:", transcription)
        return transcription

    def levenshtein_loss(self, y_targ, y_pred):
        return distance(y_targ, y_pred) / max(len(y_targ), len(y_pred))  # Normalize

    """
    Perceptual Loss	Meaning
        ≈ 0.0 - 0.5	Imperceptible noise, almost identical
        0.5 - 2.0	Slightly audible difference, but not obvious
        2.0 - 5.0	Clearly audible noise, some distortion
        5.0 - 10.0	Noticeable, likely unpleasant changes (your case: 7.83)
        > 10.0	Strong noise, likely unrecognizable distortion
    """

    def perceptual_loss_combined(self, original_audio, adversarial_noise):
        mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,  # Increase FFT size to capture more frequencies
            win_length=400,  # Standard for 16kHz speech
            hop_length=160,  # 10ms step (160 samples for 16kHz)
            n_mels=80  # Reduce from 128 to 80 for stability
        )

        # Compute Mel spectrograms
        original_mel = mel_transform(original_audio)
        adversarial_mel = mel_transform(adversarial_noise)

        # Compute log-Mel loss
        log_original = torch.log(original_mel + 1e-6)
        log_adversarial = torch.log(adversarial_mel + 1e-6)
        mel_loss = torch.nn.functional.mse_loss(log_original, log_adversarial)

        # Compute waveform loss (to capture transient noises)
        waveform_loss = torch.nn.functional.mse_loss(original_audio, adversarial_noise)

        # Combine losses (adjust weight if needed)
        total_loss = 0.7 * mel_loss + 0.3 * waveform_loss

        return total_loss.item()

    def generate_population(self, original, pop, start, end):
        population = []
        size = len(original[0])

        for _ in range(pop):
            # Step 1: Initialize array with zeros
            new_solution = np.zeros(size, dtype=np.float32)

            # Step 2: Assign random values only within the specified index range
            new_solution[start:end] = np.random.uniform(
                -self.epsilon, self.epsilon, end - start
            ).astype(np.float32)

            # Step 5: Store in population
            population.append(Individual(solution=new_solution, fitness=None, ctc_fitness=None))

        return population

    # Step 2: Sort
    def sort_population(self, original, population):
        fitts = []
        for indv in population:
            # original = self.preprocess_audio(original)
            combination = original + indv.solution
            # print(indv.solution)
            # Wav2Vec model
            text = self.transcript_audio(combination)
            print(text)

            # indv.fitness = self.fix_fitness(self.cosine_similarity_loss(self.target, text))
            indv.fitness = self.levenshtein_loss(self.target, text)
            print("indvFITNESS: ", indv.fitness)

            # indv.ctc_fitness = self.ctc_loss(original, indv.solution, self.target)
            indv.ctc_fitness = self.perceptual_loss_combined(original, combination)
            # print("CTC loss function: ", ctc_loss_numpy(original+indv.solution, self.target))
            print("indvFITNESS_CTC: ", indv.ctc_fitness)

            fitts.append(indv.ctc_fitness)

        # population.sort(key=lambda x: (x.fitness, x.ctc_fitness))
        # population.sort(key=lambda x: x.fitness)
        if population[0].fitness == 0.0:
            # print("HULLLLLLUUUULLLUUUUUUUUUUUUUUU")
            population.sort(key=lambda x: (x.fitness, x.ctc_fitness))
        else:
            population.sort(key=lambda x: x.fitness)

        fitts.sort()
        # print("FITTSSS: ", fitts)

        return population, fitts

    # Step 3: Selection
    def selection(self, population):
        for i in range(self.elits):
            population.pop()
        return population

    # Step 4: Repopulation/Crossover
    def crossover(self, population, start, end):
        for i in range(self.elits):
            parent1 = population[i].solution
            parent2 = population[i + 1].solution

            mask = np.random.randint(0, 2, size=parent1.shape).astype(bool)
            child = np.where(mask, parent1, parent2)
            new_ind = Individual(solution=child, fitness=None, ctc_fitness=None)
            population.append(new_ind)
            # if population[0].fitness == 0.0:
            #     ind1 = population[0]
            #     ind2 = population[i + 1]
            #     midpoint = start + (end - start)
            #     f_left = ind1.solution[:midpoint]
            #     f_right = ind1.solution[midpoint:]
            #     c_left = ind2.solution[:midpoint]
            #     c_right = ind2.solution[midpoint:]
            #     combined1 = np.concatenate((f_left, c_right))
            #     combined2 = np.concatenate((c_left, f_right))
            #     new_ind1 = Individual(solution=combined1, fitness=None, ctc_fitness=None)
            #     new_ind2 = Individual(solution=combined2, fitness=None, ctc_fitness=None)
            #     population.append(new_ind1)
            #     population.append(new_ind2)
            #     if i == 4:
            #         print("Population: ",len(population))
            #         break
            # else:
            #     ind1 = population[i]
            #     ind2 = population[i + 1]
            #
            #     midpoint = start + (end - start)
            #     first_half = ind1.solution[:midpoint]
            #     second_half = ind2.solution[midpoint:]
            #     combined = np.concatenate((first_half, second_half))
            #
            #     new_ind = Individual(solution=combined, fitness=None, ctc_fitness=None)
            #     # print("NEW IND: ", new_ind)
            #     population.append(new_ind)

        print("Population: ", len(population))
        return population

    def mutation(self, population, start, end):
        ranges = [-self.mutation_range, 0, self.mutation_range]

        for indv in population[self.elits:]:  # Skip elite individuals
            if random.random() > 0.1:
                size = len(indv.solution)
                # Generate noise with lower amplitude
                random_array = np.zeros(size, dtype=np.float32)
                random_array[start:end] = np.random.uniform(low=-self.mutation_range, high=self.mutation_range,
                                                            size=end - start)
                # Add noise to solution
                indv.solution += random_array

        return population

    def fine_tune(self, adversarial_audio, loss, epochs, original, start, end):
        size = len(adversarial_audio)

        for _ in range(epochs):
            # Generate small perturbation
            random_array = np.zeros(size, dtype=np.float32)
            random_array += adversarial_audio
            candidate = random_array
            perturbation = np.zeros(size, dtype=np.float32)
            perturbation[start:end] = np.random.uniform(low=-0.005, high=0.005,
                                                        size=end - start)
            # perturbation = np.random.uniform(-self.epsilon, self.epsilon, size=candidate.shape)

            # Optional: Apply envelope or bandpass
            # perturbation *= np.abs(current_noise)  # Optional masking
            # filtered = bandpass(perturbation)     # Optional filtering

            candidate += perturbation

            # Evaluate new audio
            new_audio = original + candidate
            new_transcript = self.transcript_audio(new_audio)
            new_loss = self.perceptual_loss_combined(original, new_audio)

            # Accept only if transcription is preserved AND loss is improved
            if new_transcript.strip().lower() == self.target.lower() and new_loss < loss:
                adversarial_audio = candidate
                loss = new_loss
                print(f"Updated loss: {loss:.4f}")

        print(self.transcript_audio(original+adversarial_audio))
        print(self.perceptual_loss_combined(original, original+adversarial_audio))
        return adversarial_audio, loss

    # Step 6: Generate, evaluate Population
    def attack_speech(self, org, adv, epochs):
        population = self.generate_population(org, self.pop, self.start, self.end)
        final_epoch = 0
        print(self.perceptual_loss_combined(org, org))
        for _ in range(epochs):

            print("Epoch:" + str(_))
            final_epoch = _

            b0 = time.time()
            population, fitts = self.sort_population(org, population)
            e0 = time.time()
            # print("SORT_POPULATION: ", e0 - b0)

            # Stop if fitness of one individual is 0
            b1 = time.time()
            population = self.selection(population)
            # print("POPULATION SIZE after selection: ", len(population))
            e1 = time.time()

            b2 = time.time()
            population = self.crossover(population, self.start, self.end)
            # print("POPULATION SIZE after crossover: ", len(population))
            e2 = time.time()
            # print("POPU aft Cross: ", population)
            b3 = time.time()
            population = self.mutation(population, self.start, self.end)
            # print("POPULATION SIZE after mutation: ", len(population))
            e3 = time.time()
            if population[0].fitness == 0.0:
                population[0].solution, population[0].ctc_fitness = self.fine_tune(population[0].solution, population[0].ctc_fitness, epochs, org, self.start, self.end)

            print(self.transcript_audio(org + population[0].solution), self.target)
            print(fitts[0])


            if self.transcript_audio(org + population[0].solution) == self.target and population[0].ctc_fitness <= 2.0:
                print("We reached our destination! OLLAAAAAA")
                break

        return (org + population[0].solution, population[0].solution, population[0].fitness, population[0].ctc_fitness,
                final_epoch)
