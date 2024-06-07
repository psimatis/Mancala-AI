import random
import mancala

SIMULATIONS = 30

def genetic_algorithm(generations=30, population_size=20, mutation_rate=0.1):
    def create_individual():
        return [random.uniform(0, 1) for _ in range(5)]

    def mutate(individual):
        if random.random() < mutation_rate:
            idx = random.randint(0, 4)
            individual[idx] = random.uniform(0, 1)

    def crossover(parent1, parent2):
        idx = random.randint(0, 4)
        return parent1[:idx] + parent2[idx:]

    def fitness(strategy, opponents):
        wins = 0
        for opponent in opponents:
            for _ in range(SIMULATIONS):
                game = mancala.Game({1: ['AI', strategy], 2: ['AI', opponent]})
                winner = game.game_loop()
                if winner == 1:
                    wins += 1
        return wins

    history = {}
    population = [create_individual() for _ in range(population_size)]
    for generation in range(generations):
        fitness_scores = []
        for individual in population:
            opponents = random.sample(population, min(len(population), 5))  # Select a few random opponents from the population
            score = fitness(individual, opponents)
            fitness_scores.append((score, individual))
        fitness_scores.sort(reverse=True, key=lambda x: x[0])
        population = [individual for _, individual in fitness_scores]

        next_population = population[:10]  # Elitism: pick the best two individuals
        while len(next_population) < population_size:
            parent1, parent2 = random.choices(population[:10], k=2)
            offspring = crossover(parent1, parent2)
            mutate(offspring)
            next_population.append(offspring)
        population = next_population
        best_individual = population[0]
        best_fitness = fitness(best_individual, population)
        print(f'Generation {generation}, Best Fitness: {best_fitness}, Individual: {best_individual}')
        history[generation] = {'Best Fitness': best_fitness, 'Individual': best_individual}

    return population[0], history