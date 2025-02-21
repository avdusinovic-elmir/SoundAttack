import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

import time

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
    def __init__(self,  target, pop, elits, mutatation_range, epsilon):
        self.pop = pop
        self.target = target
        self.elits = elits
        self.mutation_range = mutatation_range
        self.epsilon = epsilon

    # Step 1: Generate Population

    # web_model = WebBertSimilarity(device='cpu', batch_size=10)  # defaults to GPU prediction

    # clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10)  # defaults to GPU prediction
    def preprocess_audio(self, org_audio):
        speech_array, sampling_rate = torchaudio.load(org_audio)
        speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
        speech_array = speech_array.squeeze().numpy()

        # Process Input Features
        preprocessed_audio = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values

        return preprocessed_audio

    def transcript_audio(self, audio_file):
        # input_values = self.preprocess_audio(audio_file)
        # Perform Inference
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
        target_lengths = torch.full(size=(target_encoded.shape[0],), fill_value=target_encoded.shape[1], dtype=torch.long)

        # Compute CTC Loss
        ctc_loss_fn = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id, reduction='mean')
        loss = ctc_loss_fn(log_probs.permute(1, 0, 2), target_encoded, input_lengths, target_lengths)
        # print(loss.item())
        return loss.item()  # Lower loss means a better adversarial attack


    def cosine_similarity_loss(self, y_targ, y_pred):
        # Tokenize and pad y_targ_transcription
        # print("y_targ:", y_targ)
        y_targ_sequence = [vocab.get(c, 26) for c in y_targ]
        y_targ_padded = np.pad(y_targ_sequence, (0, 50 - len(y_targ_sequence)), mode='constant')
        # print("y_targ_padded: ", y_targ_padded)

        # Tokenize and pad y_pred_transcription
        y_pred =  y_pred
        y_pred_sequence = [vocab.get(c, 26) for c in y_pred]
        y_pred_padded = np.pad(y_pred_sequence, (0, 50 - len(y_pred_sequence)), mode='constant')
        # print("y_pred_padded: ", y_pred_padded)

        y_pred_n = np.array(y_pred_padded)
        y_targ_n = np.array(y_targ_padded)

        # Compute dot product
        dot_product = np.dot(y_pred_n, y_targ_n)

        # Compute magnitudes (norms)
        norm1 = np.linalg.norm(y_pred_n)
        norm2 = np.linalg.norm(y_targ_n)

        # Step 4: Compute cosine similarity
        similarity = dot_product / (norm1 * norm2)
        loss = 1 - similarity
        # print("SIMILARITY: ", loss)
        return loss

    def generate_population(self, original, pop):
        population = []
        size = len(original[0])
        # print(size)
        # print('what:', original.shape)
        half_size = size//2
        for i in range(pop):
            new_solution = np.random.uniform(-self.epsilon, self.epsilon, size).astype(np.float32)
            population.append(Individual(solution=new_solution, fitness=None, ctc_fitness=None))
            # new_solution = np.zeros(size, dtype=np.float32)  # Initialize all zeros
            # new_solution[half_size:] = np.random.uniform(-self.epsilon, self.epsilon, half_size).astype(np.float32)
            # population.append(Individual(solution=new_solution, fitness=None, ctc_fitness=None))
        # print(population)
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

            # indv.ctc_fitness = self.cosine_similarity_loss(self.target, text)


            indv.ctc_fitness = self.ctc_loss(original, indv.solution, self.target)
            print("indvFITNESS: ", indv.ctc_fitness)

            # indv.ctc_fitness = ctc_loss_numpy(combination, target_text=self.target)

            # penalty = 0
            # for _ in range(len(indv.solution)):
            #     if indv.solution[_] > 900:
            #         penalty += 1
            #
            # penalty = (penalty/len(indv.solution))
            # indv.fitness += penalty
            # Semantic Similarity
            # indv.fitness = self.semantic_similarity(self.web_model, self.target, text)

            # indv.fitness = sequence_alignment_score(target, text)
            fitts.append(indv.ctc_fitness)

        # Sort the population by fitness
        population.sort(key=lambda x: x.ctc_fitness, reverse=False)
        # population.sort(key=lambda x: (-x.fitness, x.ctc_fitness))
        fitts.sort()
        print("FITTSSS: ", fitts)

        return population, fitts

    # Step 3: Selection
    def selection(self, population):
        for i in range(self.elits):
            population.pop()
        return population

    # Step 4: Repopulation/Crossover
    def crossover(self, population):
        for i in range(self.elits):
            ind1 = population[i]
            ind2 = population[i + 1]
            midpoint = len(ind1.solution) // 2
            first_half = ind1.solution[:midpoint]
            second_half = ind2.solution[midpoint:]
            combined = np.concatenate((first_half, second_half))
            new_ind = Individual(solution=combined, fitness=None, ctc_fitness=None)
            # print("NEW IND: ", new_ind)
            population.append(new_ind)
        return population

    # def crossover(self, population):
    #     for i in range(self.elits):
    #         ind1 = population[i]
    #         ind2 = population[i + 1]
    #
    #         # Initialize new individual with zeros for the first 8000 elements
    #         new_solution = np.zeros(len(ind1.solution), dtype=np.float32)
    #
    #         # Set the next 4000 elements from ind1 (8000 to 12000)
    #         new_solution[11100:12500] = ind1.solution[11100:12500]
    #
    #         # Set the last 4000 elements from ind2 (12000 to 16000)
    #         new_solution[12500:16000] = ind2.solution[12500:16000]
    #
    #         # Create a new individual with the combined solution
    #         new_ind = Individual(solution=new_solution, fitness=None, ctc_fitness=None)
    #         # Add the new individual to the population
    #         population.append(new_ind)
    #
    #     return population

    # Step 5: Mutation
    # def mutation(self, population):
    #     for indv in population:
    #         for i in range(len(indv.solution)):
    #             if random.random() < 0.99:
    #                 add = random.randint(-9200, 9200)
    #                 indv.solution[i] += add
    #     return population

    def mutation(self, population):
        ranges = [-self.mutation_range, 0, self.mutation_range]

        for indv in population[self.elits:]:
            size = len(indv.solution)
            random_array = np.random.choice(ranges, size=size)
            indv.solution += random_array

        return population

    # def mutation(self, population):
    #     ranges = [-self.mutation_range, 0, self.mutation_range]
    #
    #     for indv in population[self.elits:]:
    #         size = len(indv.solution)
    #         range = 16000-11100
    #         random_array = np.random.choice(ranges, size=range)
    #         indv.solution[11100:] += random_array
    #
    #     return population

    # Step 6: Generate, evaluate Population
    def attack_speech(self, org, adv, epochs):
        population = self.generate_population(org, self.pop)
        for _ in range(epochs):

            print("Epoch:" + str(_))

            b0 = time.time()
            population, fitts = self.sort_population(org, population)
            e0 = time.time()
            print("SORT_POPULATION: ", e0 - b0)

            print("Epochs: ", _ + 1, " Fitness_best: ", population[0].ctc_fitness,
                  " Sentence: ", self.transcript_audio(org + population[0].solution))
            print(" Fitness_worst: ", population[self.pop - 1].ctc_fitness,
                  " Sentence: ", self.transcript_audio(org + population[-1].solution))
            print(" Fitness_midd: ", population[-self.pop // 2].ctc_fitness,
                  " Sentence: ", self.transcript_audio(org + population[-15].solution))
            # for i in range(50):
            #     print(i, " Fitness: ", population[i].fitness, " Sentence: ", model.stt(org+population[i].solution))
            b1 = time.time()
            population = self.selection(population)
            print("POPULATION SIZE after selection: ", len(population))
            e1 = time.time()

            b2 = time.time()
            population = self.crossover(population)
            print("POPULATION SIZE after crossover: ", len(population))
            e2 = time.time()
            # print("POPU aft Cross: ", population)
            b3 = time.time()
            population = self.mutation(population)
            print("POPULATION SIZE after mutation: ", len(population))
            e3 = time.time()

            # print("SELECTION: ", e1-b1)
            # print("CROSSOVER: ", e2-b2)
            # print("MUTATION: ", e3-b3)
            print(self.transcript_audio(org + population[0].solution), self.target)

            if self.transcript_audio(org + population[0].solution) == self.target:
                print("We reached our destination! OLLAAAAAA")
                break

        return org+population[0].solution, population[0].solution, population[0].ctc_fitness, population[0].ctc_fitness


