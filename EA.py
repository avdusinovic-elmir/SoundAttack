import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Loss.ctcLoss import *


# from semantic_text_similarity.models import WebBertSimilarity
class Individual:
    def __init__(self, solution, fitness, ctc_fitness):
        self.solution = solution
        self.fitness = fitness
        self.ctc_fitness = ctc_fitness


class EA:
    def __init__(self, target):
        self.target = target

    # Step 1: Generate Population

    # web_model = WebBertSimilarity(device='cpu', batch_size=10)  # defaults to GPU prediction

    # clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10)  # defaults to GPU prediction

    def semantic_similarity(self, model2, target, test):
        similarity = model2.predict([(target, test)])
        return similarity[0]

    def generate_population(self, original, n=50):
        population = []
        size = len(original)
        print(len(original))
        for _ in range(n):
            new_solution = np.random.randint(-200, 200, size, dtype=np.int16)
            # print(len(new_solution))
            population.append(Individual(solution=new_solution, fitness=None, ctc_fitness=None))

        return population

        # Step 2: Sort

    def sort_population(self, original, population, model):
        fitts = []
        for indv in population:
            combination = original + indv.solution
            # Deepspeech model
            text = model.stt(combination)
            # if text != "and you know it":
            #     print(text)
            # Cosine Similarity of Word Embeddings
            vectorizer = CountVectorizer().fit([self.target, text])
            vectors = vectorizer.transform([self.target, text]).toarray()
            indv.fitness = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
            indv.ctc_fitness = ctc_loss_numpy(combination, target_text=self.target)

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
        # population.sort(key=lambda x: x.fitness, reverse=True)
        population.sort(key=lambda x: (-x.fitness, x.ctc_fitness))
        fitts.sort()

        return population, fitts

    # Step 3: Selection
    def selection(self, population):
        for i in range(10):
            population.pop()
        return population

    # Step 4: Repopulation/Crossover
    def crossover(self, population):
        for i in range(10):
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
    def mutation(self, population):
        for indv in population:
            for i in range(len(indv.solution)):
                if indv.solution[i] >= 900:
                    continue
                if random.random() < 0.5:
                    add = random.randint(-100, 100)
                    indv.solution[i] += add
        return population

    # Step 6: Generate, evaluate Population
    def attack_speech(self, org, adv, model, epochs=100):
        population = self.generate_population(org)
        for _ in range(epochs):
            # print("Epochs: ", _+1,  " Fitness_best: ", population[0].fitness,
            #       " Sentence: ", model.stt(org+population[0].solution))
            # print(" Fitness_worst: ", population[-1].fitness,
            #       " Sentence: ", model.stt(org+population[-1].solution))
            # print(" Fitness_midd: ", population[-15].fitness,
            #       " Sentence: ", model.stt(org+population[-15].solution))

            print("Epoch:" + str(_))
            population, fitts = self.sort_population(org, population, model=model)
            population = self.selection(population)
            population = self.crossover(population)
            population = self.mutation(population)
            print('FITTS: ', fitts)

        return population[0].solution, population[0].fitness, population[0].ctc_fitness
