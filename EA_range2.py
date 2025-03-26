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

    # web_model = WebBertSimilarity(device='cpu', batch_size=10)  # defaults to GPU prediction

    # clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10)  # defaults to GPU prediction
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

    def semantic_similarity(self, model2, target, test):
        similarity = model2.predict([(target, test)])
        return similarity[0]

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
        target_lengths = torch.full(size=(target_encoded.shape[0],), fill_value=target_encoded.shape[1],
                                    dtype=torch.long)

        # Compute CTC Loss
        ctc_loss_fn = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id, reduction='mean')
        loss = ctc_loss_fn(log_probs.permute(1, 0, 2), target_encoded, input_lengths, target_lengths)
        # print(loss.item())
        return loss.item()  # Lower loss means a better adversarial attack

    def levenshtein_loss(self, y_targ, y_pred):
        return distance(y_targ, y_pred) / max(len(y_targ), len(y_pred))  # Normalize

    def cosine_similarity_loss(self, y_targ, y_pred):
        # Tokenize using the vocabulary mapping
        y_targ_sequence = np.array([vocab.get(c.lower(), 27) for c in y_targ])  # Use .lower() for consistency
        y_pred_sequence = np.array([vocab.get(c.lower(), 27) for c in y_pred])

        # print(f"y_targ_sequence: {y_targ_sequence}")
        # print(f"y_pred_sequence: {y_pred_sequence}")

        # Make sequences the same length by padding the shorter one with zeros
        max_len = max(len(y_targ_sequence), len(y_pred_sequence))
        y_targ_sequence = np.pad(y_targ_sequence, (0, max_len - len(y_targ_sequence)), mode='constant')
        y_pred_sequence = np.pad(y_pred_sequence, (0, max_len - len(y_pred_sequence)), mode='constant')

        # print(f"Padded y_targ_sequence: {y_targ_sequence}")
        # print(f"Padded y_pred_sequence: {y_pred_sequence}")

        # Compute dot product
        dot_product = np.dot(y_targ_sequence, y_pred_sequence)
        # print(f"Dot Product: {dot_product}")

        # Compute norms
        norm1 = np.linalg.norm(y_targ_sequence)
        norm2 = np.linalg.norm(y_pred_sequence)
        # print(f"Norm 1: {norm1}, Norm 2: {norm2}")

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum loss (no similarity)

        # Compute cosine similarity with clipping for stability
        similarity = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
        # print(f"Cosine Similarity: {similarity}")

        # Convert to loss (ensure small values don't round to zero)
        loss = np.maximum(1 - similarity, 1e-10)  # Ensures min loss > 0
        # print(f"Final Loss: {loss}")

        return loss

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

            # # 🔹 Step 3: Apply psychoacoustic masking (make noise follow speech envelope)
            # envelope = torch.abs(torch.tensor(original, dtype=torch.float32))  # Ensure tensor format
            # new_solution *= envelope.squeeze().numpy()  # Fix: Remove extra dimension before multiplication

            # 🔹 Step 4: Apply high-frequency bandpass filtering (hide noise in higher frequencies)
            noise_tensor = torch.tensor(new_solution, dtype=torch.float32)
            filtered_noise = F.bandpass_biquad(noise_tensor, 16000, central_freq=6000, Q=0.707)

            # Step 5: Store in population
            population.append(Individual(solution=filtered_noise.numpy(), fitness=None, ctc_fitness=None))

        return population

        # Step 2: Sort

    def fix_fitness(self, fitness):
        return 0.0 if fitness <= 1e-10 else fitness

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
            #         print(len(population))
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

            # paper version
            # ind1 = population[i]
            # ind2 = population[i + 1]
            # combined = np.zeros(len(ind1.solution), dtype=np.float32)
            # for j in range(len(ind1.solution)):
            #     if np.random.random() < 0.5:
            #         combined[j] = ind1.solution[j]
            #     else:
            #         combined[j] = ind2.solution[j]
            #
            # new_ind = Individual(solution=combined, fitness=None, ctc_fitness=None)
            # # print("NEW IND: ", new_ind)
            # population.append(new_ind)
        return population

    def mutation(self, population, start, end):
        ranges = [-self.mutation_range, 0, self.mutation_range]

        for indv in population[self.elits:]:  # Skip elite individuals
            if random.random() > 0.1:
                size = len(indv.solution)

                # # Directly apply mutation without masking
                # random_array = np.zeros(size, dtype=np.float32)
                # random_array[start:end] = np.random.uniform(low=-self.mutation_range, high=self.mutation_range, size=end - start)
                # indv.solution += random_array

                # Generate noise with lower amplitude
                random_array = np.zeros(size, dtype=np.float32)
                random_array[start:end] = np.random.uniform(low=-self.mutation_range, high=self.mutation_range,
                                                            size=end - start)

                # # Apply psychoacoustic masking (blend noise with speech envelope)
                # envelope = np.abs(indv.solution)
                # random_array *= envelope

                # Apply high-frequency bandpass filter (hide noise from human perception)
                noise_tensor = torch.tensor(random_array, dtype=torch.float32)
                filtered_noise = F.bandpass_biquad(noise_tensor, 16000, central_freq=6000, Q=0.707)

                # Add noise to solution
                indv.solution += filtered_noise.numpy()

        return population

    # Step 6: Generate, evaluate Population
    def attack_speech(self, org, adv, epochs):
        population = self.generate_population(org, self.pop, self.start, self.end)
        final_epoch = 0
        saved = 1
        current_fitness = 0
        counter = 0
        print(self.perceptual_loss_combined(org, org))
        for _ in range(epochs):

            print("Epoch:" + str(_))
            final_epoch = _

            b0 = time.time()
            population, fitts = self.sort_population(org, population)
            e0 = time.time()
            # print("SORT_POPULATION: ", e0 - b0)

            # print("Epochs: ", _ + 1, " Fitness_best: ", population[0].ctc_fitness,
            #       " Sentence: ", self.transcript_audio(org + population[0].solution))
            # print(" Fitness_worst: ", population[self.pop - 1].ctc_fitness,
            #       " Sentence: ", self.transcript_audio(org + population[-1].solution))
            # print(" Fitness_midd: ", population[-self.pop // 2].ctc_fitness,
            #       " Sentence: ", self.transcript_audio(org + population[-15].solution))
            # for i in range(50):
            #     print(i, " Fitness: ", population[i].fitness, " Sentence: ", model.stt(org+population[i].solution))

            # Stop if fitness of one individual is 0
            if population[0].fitness != 0.0 and population[0].ctc_fitness > 2.0:
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

            # print("SELECTION: ", e1-b1)
            # print("CROSSOVER: ", e2-b2)
            # print("MUTATION: ", e3-b3)
            print(self.transcript_audio(org + population[0].solution), self.target)
            print(fitts[0])

            if current_fitness > fitts[0]:
                current_fitness = fitts[0]
                counter = 0

            counter += 1
            # if self.transcript_audio(org + population[0].solution) == self.target and saved:
            #     print("We reached our destination! OLLAAAAAA")
            #     sf.write("first_advers_audio.wav", org + population[0].solution, 16000)
            #     saved = 0

            if self.transcript_audio(org + population[0].solution) == self.target and (
                    population[0].ctc_fitness <= 2.0 or counter >= 30):
                print("We reached our destination! OLLAAAAAA")
                break

        return (org + population[0].solution, population[0].solution, population[0].fitness, population[0].ctc_fitness,
                final_epoch)
