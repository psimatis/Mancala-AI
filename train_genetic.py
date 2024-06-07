import random
import mancala

SIMULATIONS = 10

def genetic_algorithm(generations=20, population_size=10, mutation_rate=0.1):
    def create_individual():
        return [random.uniform(0, 1) for _ in range(5)]

    def mutate(individual):
        if random.random() < mutation_rate:
            idx = random.randint(0, 4)
            individual[idx] = random.uniform(0, 1)

    def crossover(parent1, parent2):
        idx = random.randint(0, 4)
        return parent1[:idx] + parent2[idx:]

    def fitness(strategy):
        wins = 0
        for _ in range(SIMULATIONS):
            game = mancala.Game({1: ['AI', strategy], 2: ['random', None]})
            winner = game.game_loop()
            if winner == 1:
                wins += 1
        return wins

    population = [create_individual() for _ in range(population_size)]
    for generation in range(generations):
        population = sorted(population, key=fitness, reverse=True)
        next_population = population[:2]  # Elitism: pick the best two individuals
        while len(next_population) < population_size:
            parent1, parent2 = random.choices(population[:10], k=2)
            offspring = crossover(parent1, parent2)
            mutate(offspring)
            next_population.append(offspring)
        population = next_population
        best_individual = population[0]
        # print(f'Generation {generation}, Best Fitness: {fitness(best_individual)}')

    return population[0]