import random
import matplotlib.pyplot as plt
import mancala

def create_individual():
    return [random.uniform(0, 1) for _ in range(6)]

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx = random.randint(0, 5)
        individual[idx] = random.uniform(0, 1)

def crossover(parent1, parent2):
    idx = random.randint(0, 5)
    return parent1[:idx] + parent2[idx:]

def run_simulation(simulations, strategy, opponent):
    wins = 0
    for _ in range(simulations):
        game = mancala.Game({1: ['AI', strategy], 2: [opponent[0], opponent[1]]})
        result = game.game_loop()
        if result == 1:
            wins += 1
        elif result == 2:
            wins -= 1
    return wins

def fitness_tournament(strategy, opponents, simulations):
    wins = 0
    for opponent in opponents:
        wins += run_simulation(simulations, strategy, ('AI', opponent))
    return wins

def fitness_random(strategy, simulations):
    return run_simulation(simulations, strategy, ('random', None))

def evolve_population(population, fitness_func, mutation_rate, elitisism, simulations, top):
    fitness_scores = [(fitness_func(individual, simulations), individual) for individual in population]
    fitness_scores.sort(reverse=True, key=lambda x: x[0])
    population = [individual for _, individual in fitness_scores]

    next_population = population[:elitisism]  
    while len(next_population) < len(population):
        parent1, parent2 = random.choices(population[:top], k=2)
        offspring = crossover(parent1, parent2)
        mutate(offspring, mutation_rate)
        next_population.append(offspring)
    return next_population, fitness_scores[0]

def plot_training(history):
    plt.figure(figsize=(10, 5))
    plt.plot([history[g]['Best Fitness'] for g in history], marker='o', linestyle='-', color='b')
    plt.title('Best Fitness Per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.show()

def run_genetic_algorithm(generations=10, population_size=100, mutation_rate=0.1, simulations=100, elitisism=2, tournament=0, top=10, verbose=True):
    if verbose:
        training_type = 'tournament' if tournament else 'random'
        print(f'Training {training_type}-based genetic algorithm')
    history = {}
    population = [create_individual() for _ in range(population_size)]
    for generation in range(generations):
        if tournament:
            fitness_func = lambda individual, sims: fitness_tournament(individual, random.sample(population, tournament), sims)
        else:
            fitness_func = fitness_random
        population, (best_fitness, best_individual) = evolve_population(population, fitness_func, mutation_rate, elitisism, simulations, top)
        if verbose: print(f'Generation {generation}, Best Fitness: {best_fitness}, Individual: {best_individual}')
        history[generation] = {'Best Fitness': best_fitness, 'Individual': best_individual}
    if verbose == True:
        print(f'Best strategy: ', population[0])
        plot_training(history)
    return population[0], history