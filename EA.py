import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from semantic_text_similarity.models import WebBertSimilarity
class Individual:
    def __init__(self, solution, fitness):
        self.solution = solution
        self.fitness = fitness

class EA:
    def __init__(self, target):
        self.target = target
    # Step 1: Generate Population

    # web_model = WebBertSimilarity(device='cpu', batch_size=10)  # defaults to GPU prediction

    # clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10)  # defaults to GPU prediction

    def semantic_similarity(model2, target, test):
        similarity = model2.predict([(target, test)])
        return similarity[0]

    def generate_population(self, original, n=20):
        population = []
        size = len(original)
        print(len(original))
        for _ in range(n):
            new_solution = np.random.randint(-100, 100, size, dtype=np.int16)
            print(len(new_solution))
            population.append(Individual(solution=new_solution, fitness=None))
        return population

        # Step 2: Sort
    def sort_population(self, original, population, model):
        fitts = []
        for indv in population:
            combination = original + indv.solution
            # Deepspeech model
            text = model.stt(combination)
            # Cosine Similarity of Word Embeddings
            vectorizer = CountVectorizer().fit([self.target, text])
            vectors = vectorizer.transform([self.target, text]).toarray()
            indv.fitness = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

            # Semantic Similarity
            # indv.fitness = self.semantic_similarity(self.web_model, self.target, text)

            # indv.fitness = sequence_alignment_score(target, text)
            fitts.append(indv.fitness)

        # Sort the population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        return population, fitts


    # Step 3: Selection
    def selection(self, population):
        for i in range(40, 50):
            population.pop()
        return population

    # Step 4: Repopulation/Crossover
    def crossover(self, population):
        for i in range(10):
            ind1 = population[i]
            ind2 = population[i+1]
            midpoint = len(ind1.solution)//2
            first_half = ind1.solution[:midpoint]
            second_half = ind2.solution[midpoint:]
            combined = np.concatenate((first_half, second_half))
            new_ind = Individual(solution=combined, fitness=None)
            population.append(new_ind)
        return population


    # Step 5: Mutation
    def mutation(self, population):
        for indv in population:
            for i in range(len(indv.solution)):
                if random.random() < 0.5:
                    add = random.randint(1, 10)
                    if random.random() < 0.5:
                        add = -add
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

            print(len(population))
            population, fitts = self.sort_population(org, population, model=model)
            population = self.selection(population)
            population = self.crossover(population)
            population = self.mutation(population)
            print('FITTS: ', fitts)

        return population[0].solution, population[0].fitness