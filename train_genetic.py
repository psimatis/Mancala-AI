import random
import mancala

def create_individual():
    return [random.uniform(0, 1) for _ in range(5)]

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx = random.randint(0, 4)
        individual[idx] = random.uniform(0, 1)

def crossover(parent1, parent2):
    idx = random.randint(0, 4)
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

def genetic_algorithm(generations=30, population_size=20, mutation_rate=0.1, simulations=30, elitisism=2, tournament=1, top=10):
    history = {}
    population = [create_individual() for _ in range(population_size)]
    
    for generation in range(generations):
        if tournament:
            fitness_func = lambda individual, sims: fitness_tournament(individual, random.sample(population, tournament), sims)
        else:
            fitness_func = fitness_random
        
        population, (best_fitness, best_individual) = evolve_population(population, fitness_func, mutation_rate, elitisism, simulations, top)

        print(f'Generation {generation}, Best Fitness: {best_fitness}, Individual: {best_individual}')
        history[generation] = {'Best Fitness': best_fitness, 'Individual': best_individual}

    return population[0], history