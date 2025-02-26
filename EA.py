import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import time

from Loss.ctcLoss import *

vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
                      'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18,
                      't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, ' ': 26}


# from semantic_text_similarity.models import WebBertSimilarity
class Individual:
    def __init__(self, solution, fitness, ctc_fitness):
        self.solution = solution
        self.fitness = fitness
        self.ctc_fitness = ctc_fitness


class EA:
    def __init__(self, target, elits):
        self.target = target
        self.elits = elits

    # Step 1: Generate Population

    # web_model = WebBertSimilarity(device='cpu', batch_size=10)  # defaults to GPU prediction

    # clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10)  # defaults to GPU prediction

    def semantic_similarity(self, model2, target, test):
        similarity = model2.predict([(target, test)])
        return similarity[0]

    def cosine_similarity_loss(self, y_targ, y_pred):
        # Tokenize and pad y_targ_transcription
        y_targ_sequence = [vocab.get(c, 26) for c in y_targ]
        # print(len(y_targ_sequence))
        y_targ_padded = np.pad(y_targ_sequence, (0, 18 - len(y_targ_sequence)), mode='constant')

        # Tokenize and pad y_pred_transcription
        y_pred_sequence = [vocab.get(c, 26) for c in y_pred]
        y_pred_padded = np.pad(y_pred_sequence, (0, 18 - len(y_pred_sequence)), mode='constant')
        # print("Y_PRED: ", y_pred_padded)
        # print("Y_ADV: ", y_pred_padded)

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

        return loss

    def generate_population(self, original, n=30):
        population = []
        size = len(original)
        # print('what:', len(original))
        for i in range(n):
            new_solution = np.random.randint(-500, 500, size, dtype=np.int16)
            # print(new_solution)
            # print("max: ", max(new_solution), "min: ", min(new_solution))
            population.append(Individual(solution=new_solution, fitness=None, ctc_fitness=None))

        # print(population)
        return population

        # Step 2: Sort

    def sort_population(self, original, population, model):
        fitts = []
        for indv in population:
            combination = original + indv.solution
            # print(combination)
            # Deepspeech model
            text = model.stt(combination)
            # print("TEXT: ", text)
            # print("self.Tareget: ", self.target)
            # if text != "and you know it":
            #     print(text)
            # Cosine Similarity of Word Embeddings
            # vectorizer = CountVectorizer().fit([self.target, text])
            # vectors = vectorizer.transform([self.target, text]).toarray()
            # print("VECTORS: ", vectors)
            # indv.fitness = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
            indv.fitness = self.cosine_similarity_loss(self.target, text)
            # print("indvFITNESS: ", indv.fitness)
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
            fitts.append(indv.fitness)

        # Sort the population by fitness
        population.sort(key=lambda x: x.fitness, reverse=False)
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
            population.append(new_ind)
        return population

    # Step 5: Mutation
    # def mutation(self, population):
    #     for indv in population:
    #         for i in range(len(indv.solution)):
    #             if random.random() < 0.99:
    #                 add = random.randint(-9200, 9200)
    #                 indv.solution[i] += add
    #     return population

    def mutation(self, population):
        ranges = [-200, 0, 200]

        for indv in population[self.elits:]:
            size = len(indv.solution)
            random_array = np.random.choice(ranges, size=size)
            indv.solution += random_array


        return population

    # Step 6: Generate, evaluate Population
    def attack_speech(self, org, adv, model, epochs=100):
        population = self.generate_population(org)
        for _ in range(epochs):


            print("Epoch:" + str(_))

            b0 = time.time()
            population, fitts = self.sort_population(org, population, model=model)
            e0 = time.time()
            print("SORT_POPULATION: ", e0-b0)

            print("Epochs: ", _+1,  " Fitness_best: ", population[0].fitness,
                  " Sentence: ", model.stt(org+population[0].solution))
            print(" Fitness_worst: ", population[29].fitness,
                  " Sentence: ", model.stt(org+population[-1].solution))
            print(" Fitness_midd: ", population[-15].fitness,
                  " Sentence: ", model.stt(org+population[-15].solution))
            # for i in range(50):
            #     print(i, " Fitness: ", population[i].fitness, " Sentence: ", model.stt(org+population[i].solution))
            b1 = time.time()
            population = self.selection(population)
            e1 = time.time()

            b2 = time.time()
            population = self.crossover(population)
            e2 = time.time()

            b3 = time.time()
            population = self.mutation(population)
            e3 = time.time()

            # print("SELECTION: ", e1-b1)
            # print("CROSSOVER: ", e2-b2)
            # print("MUTATION: ", e3-b3)

        return population[0].solution, population[0].fitness, population[0].ctc_fitness
