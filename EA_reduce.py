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
    def __init__(self, solution, fitness, ctc_fitness, perceptual_fitness):
        self.solution = solution
        self.fitness = fitness
        self.ctc_fitness = ctc_fitness
        self.perceptual_fitness = perceptual_fitness


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

    def ctc_loss(self, original_audio, adversarial_noise, target_text):
        # adversarial_audio = np.clip(original_audio + adversarial_noise, -1.0, 1.0)
        adversarial_audio = original_audio + adversarial_noise

        # convert numpy array to PyTorch tensor:
        audio_tensor = torch.tensor(adversarial_audio, dtype=torch.float32)
        audio_tensor = audio_tensor.unsqueeze((0))
        #Process audio input
        inputs = processor(audio_tensor, sampling_rate=16000, return_tensors='pt', padding=True)
        input_values = inputs.input_values.view(1, 1, -1).squeeze(0)


        # Get logits from the Wav2Vec2 model
        with torch.no_grad():
            logits = model(input_values).logits  # Shape: (batch, time, vocab_size)

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Convert target text to token IDs
        target_encoded = processor.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True).input_ids

        # Define sequence lengths
        input_lengths = torch.full(size=(log_probs.shape[0],), fill_value=log_probs.shape[1], dtype=torch.long)
        target_lengths = torch.full(size=(target_encoded.shape[0],), fill_value=target_encoded.shape[1], dtype=torch.long)

        # Compute CTC Loss
        ctc_loss_fn = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id, reduction='mean')
        loss = ctc_loss_fn(log_probs.permute(1, 0, 2), target_encoded, input_lengths, target_lengths)
        # print(loss.item())
        return loss.item()  # Lower loss means a better adversarial attack

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

            # new_solution = np.clip(new_solution, -0.7, 0.7)

            # #Apply psychoacoustic masking (make noise follow speech envelope)
            envelope = torch.abs(torch.tensor(original.clone().detach(), dtype=torch.float32))  # Ensure tensor format
            new_solution *= envelope.squeeze().numpy()  # Fix: Remove extra dimension before multiplication

            # Apply high-frequency bandpass filtering (hide noise in higher frequencies)
            noise_tensor = torch.tensor(new_solution, dtype=torch.float32)
            filtered_noise = F.bandpass_biquad(noise_tensor, 16000, central_freq=6000, Q=0.707)
            filtered_noise = np.clip(filtered_noise, -0.7, 0.7)
            # Step 5: Store in population
            population.append(Individual(solution=filtered_noise.numpy(), fitness=None, ctc_fitness=None, perceptual_fitness=None))

            # population.append(Individual(solution=new_solution, fitness=None, ctc_fitness=None))

        return population

    # Step 2: Sort
    def sort_population(self, original, population, epoch, max_epochs):
        fitts = []
        for indv in population:
            # original = self.preprocess_audio(original)
            combination = original + indv.solution
            # print(indv.solution)
            # Wav2Vec model
            text = self.transcript_audio(combination)
            # print(text)

            # indv.fitness = self.fix_fitness(self.cosine_similarity_loss(self.target, text))
            indv.fitness = self.levenshtein_loss(self.target, text)
            # print("indvFITNESS: ", indv.fitness)

            # indv.ctc_fitness = self.ctc_loss(original, indv.solution, self.target)
            indv.perceptual_fitness = self.perceptual_loss_combined(original, combination)
            # print("CTC loss function: ", ctc_loss_numpy(original+indv.solution, self.target))
            # print("Perceptual: ", indv.perceptual_fitness)
            indv.ctc_fitness = self.ctc_loss(original, indv.solution, self.target)
            # print("CTC: ", indv.ctc_fitness)
            # threshold_fitness = max(combination)

            # one_fitness = indv.fitness*10 - indv.ctc_fitness
            # if one_fitness > 0.0:
            #     indv.fitness = one_fitness
            # indv.fitness = one_fitness

            # coefficients to determine which fitnesses take the most effect
            coeff1 = 20
            coeff2 = 1/5
            coeff3 = 1/4

            # perceptual fitness should not exceed a certain threshold
            # if it does increase its fitness to a large value so that it gets popped
            # think about this again regarding the fine tuning at the end maybe some wriggle room allowed
            # if indv.ctc_fitness > 5.0:
            #     indv.fitness = 100
            # else:
            #     indv.fitness = coeff1 * indv.fitness + coeff2 * ctc_loss + coeff3 * indv.ctc_fitness
            # indv.fitness = coeff1 * indv.fitness + coeff2 * ctc_loss + coeff3 * indv.ctc_fitness

            # if indv.fitness == 0.0:
            #     indv.fitness = indv.ctc_fitness
            # else:
            #     indv.fitness = coeff1 * indv.fitness + ctc_loss/indv.ctc_fitness

            # alpha = 1.0
            # beta = 0.2 + 0.8 * (epoch / max_epochs)  # Slowly increase perceptual weight
            #
            # if indv.ctc_fitness <= 4:
            #     indv.fitness = alpha * ctc_loss + beta * indv.ctc_fitness
            # else:
            #     indv.fitness = alpha * ctc_loss + beta * indv.ctc_fitness + (
            #                 indv.ctc_fitness - 4) ** 2  # Heavy penalty above 4

            # After calculating losses:
            # if indv.ctc_fitness > 10:
            #     indv.fitness = 1e6  # discard extreme noise
            # elif indv.ctc_fitness > 4:
            #     indv.fitness = ctc_loss + 0.3 * 4 + (indv.ctc_fitness - 4) ** 2  # punish above 4
            # else:
            #     indv.fitness = ctc_loss + 0.3 * indv.ctc_fitness

            # if indv.fitness > 0.0:
            #     indv.fitness = self.ctc_loss(original, indv.solution, self.target)
            fitts.append(indv.ctc_fitness)

        # population.sort(key=lambda x: (x.fitness, x.ctc_fitness))
        # population.sort(key=lambda x: x.fitness)
        if population[0].fitness == 0.0:
            # print("HULLLLLLUUUULLLUUUUUUUUUUUUUUU")
            population.sort(key=lambda x: (x.fitness, x.perceptual_fitness, x.ctc_fitness))
        else:
            population.sort(key=lambda x: (x.fitness, -x.perceptual_fitness))

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
            new_ind = Individual(solution=child, fitness=None, ctc_fitness=None, perceptual_fitness=None)
            population.append(new_ind)
        print("Population: ", len(population))
        return population

    def mutation(self, population, start, end):
        ranges = [-self.mutation_range, 0, self.mutation_range]

        for indv in population[self.elits:]:  # Skip elite individuals
            random_array = np.zeros(len(indv.solution), dtype=np.float32)
            for i in range(start, end):
                if random.random() > 0.1:
                    random_array[i] += np.random.uniform(-self.mutation_range, self.mutation_range)

            # random_array = np.clip(random_array, -0.7, 0.7)
            # # Add noise to solution
            # indv.solution += random_array

            # Apply psychoacoustic masking (blend noise with speech envelope)
            envelope = np.abs(indv.solution)
            random_array *= envelope
            # random_array *= (0.5 * envelope + 0.5)

            # Apply high-frequency bandpass filter (hide noise from human perception)
            noise_tensor = torch.tensor(random_array, dtype=torch.float32)
            filtered_noise = F.bandpass_biquad(noise_tensor, 16000, central_freq=6000, Q=0.707)
            # filtered_noise = np.clip(filtered_noise, -0.7, 0.7)
            # Add noise to solution
            indv.solution += filtered_noise.numpy()
            indv.solution = np.clip(indv.solution, -0.7, 0.7)

            # if random.random() < 0.0005:
            #     size = len(indv.solution)
            #     # Generate noise with lower amplitude
            #     random_array = np.zeros(size, dtype=np.float32)
            #     random_array[start:end] = np.random.uniform(low=-self.mutation_range, high=self.mutation_range,
            #                                                 size=end - start)
            #
            #     random_array = np.clip(random_array, -1, 1)
            #     # Add noise to solution
            #     indv.solution += random_array
            # indv.solution = np.clip(indv.solution, -0.6, 0.6)
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
            perturbation *= np.abs(candidate)  # Optional masking
            # filtered = bandpass(perturbation)     # Optional filtering

            # Apply high-frequency bandpass filter (hide noise from human perception)
            noise_tensor = torch.tensor(perturbation, dtype=torch.float32)
            filtered_noise = F.bandpass_biquad(noise_tensor, 16000, central_freq=6000, Q=0.707)
            # filtered_noise = np.clip(filtered_noise, -0.7, 0.7)
            # Add noise to solution
            candidate += filtered_noise.numpy()

            # candidate += perturbation
            candidate = np.clip(candidate, -0.7, 0.7)

            # Evaluate new audio
            new_audio = original + candidate
            new_transcript = self.transcript_audio(new_audio)
            new_loss = self.perceptual_loss_combined(original, new_audio)

            # Accept only if transcription is preserved AND loss is improved
            if new_transcript.strip().lower() == self.target.lower() and new_loss < loss:
                adversarial_audio = candidate
                loss = new_loss
                # print(f"Updated loss: {loss:.4f}")

        # print(self.transcript_audio(original+adversarial_audio))
        # print(self.perceptual_loss_combined(original, original+adversarial_audio))
        return adversarial_audio, loss

    # Step 6: Generate, evaluate Population
    def attack_speech(self, org, adv, epochs):
        population = self.generate_population(org, self.pop, self.start, self.end)
        final_epoch = 0
        # print(self.perceptual_loss_combined(org, org))
        for _ in range(epochs):

            print("Epoch:" + str(_))
            final_epoch = _

            b0 = time.time()
            population, fitts = self.sort_population(org, population, _, epochs)
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
                for i in range(self.elits):
                    population[i].solution, population[i].perceptual_fitness = self.fine_tune(population[i].solution,
                                                                                   population[i].perceptual_fitness, 300,
                                                                                   org, self.start, self.end)

            print(self.transcript_audio(org + population[0].solution), self.target)
            # print(fitts[0])
            print(population[0].perceptual_fitness)

            if self.transcript_audio(org + population[0].solution) == self.target and population[0].perceptual_fitness <= 2.0:
                print("We reached our destination! OLLAAAAAA")
                break

        return (org + population[0].solution, population[0].solution, population[0].fitness, population[0].ctc_fitness,
                final_epoch, population[0].perceptual_fitness)
