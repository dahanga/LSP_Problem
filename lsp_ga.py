# tiny genetic algorithm by moshe sipper, www.moshesipper.com
import random
import networkx as nx

GANP = 0
GAIP = 1
GABPP = 2
GA_TYPE = GANP

POP_SIZE = 10  # population size
# GENOME_SIZE     = 5     # number of bits per individual in the population
GENERATIONS = 100  # maximal number of generations to run GA
# TOURNAMENT_SIZE = 3     # size of tournament for tournament selection
# PROB_MUTATION   = 0.1   # bitwise probability of mutation
ELITE = 5  # percentage of the population that carry automatically
MAX_PATH_COST = 10

G = nx.complete_graph(10)


def init_population():  # population comprises bitstrings
    def choose():
        neighbors = check_G.neighbors(node)
        print(list(neighbors))
        if not list(neighbors):
            print("empty neighbors", list(neighbors))
            return -1
        degrees_sum = sum([degree[1] for degree in list(check_G.degree(neighbors))])
        weights = [(degree[1] / degrees_sum) for degree in list(check_G.degree(neighbors))]
        print("neighbors", list(neighbors))
        print("weights", weights)
        return random.choices(population=list(neighbors), weights=weights, cum_weights=None, k=1)

    population = []
    for _ in range(POP_SIZE):
        check_G = G.copy()
        path = []
        start_node = random.choice(list(G.nodes))
        print(start_node)
        path += [start_node]
        node = start_node
        while True:
            new_node = choose()
            if not new_node == -1:
                path += [new_node]
                check_G.remove_node(node)
                node = new_node
            else:
                break
        node = start_node
        while True:
            new_node = choose()
            if not new_node == -1:
                path = [new_node] + path
                check_G.remove_node(node)
                node = new_node
            else:
                break
        population += [path]
    return population


def fitness(individual):  # fitness = weighted cost of the path
    return sum([G[individual[i]][individual[i + 1]]['weight'] for i in range(len(individual) - 1)])


def non_intersecting_crossover(parent1, parent2):
    candidates = []
    if not set(parent1).isdisjoint(parent2):
        return candidates
    i, j = 1, 1
    for node1 in parent1:
        if i == 1 or i == len(parent1) - 1:
            continue
        for node2 in parent2:
            if j == 1 or j == len(parent2) - 1:
                continue
            if G.has_edge(node1, node2):
                child1 = parent1[0:i + 1] + parent2[j:]
                child2 = parent1[0:i + 1] + parent2[j::-1]
                child3 = parent1[:i - 1:-1] + parent2[j:]
                child4 = parent1[:i - 1:-1] + parent2[j::-1]
                candidates += [child1, child2, child3, child4]
    return candidates


def intersecting_crossover(parent1, parent2):
    pass


def both_pairs_crossover(parent1, parent2):
    return non_intersecting_crossover(parent1, parent2) + intersecting_crossover(parent1, parent2)


def print_population(population):
    fitnesses = [fitness(population[i]) for i in range(POP_SIZE)]
    print(list(zip(population, fitnesses)))


def take_elite(population):
    take_quantity = int(POP_SIZE * ELITE / 100)
    elite = population[-take_quantity:]
    rest = population[:-take_quantity]
    print("elite ", elite)
    print("rest", rest)
    return elite, rest


def roulette_selection(population):
    def select_one():
        fitness_sum = sum([fitness(individual) for individual in population])
        pick = random.uniform(0, fitness_sum)
        current = 0
        for individual in population:
            current += fitness(individual)
            if current > pick:
                population.remove(individual)
                return individual

    selected = []
    n = int(POP_SIZE * (100 - ELITE) / 100)
    for i in range(n):
        selected += [select_one()]
         # += [x]
        # population.remove(x)
    return selected


crossovers = {
    GANP: non_intersecting_crossover,
    GAIP: intersecting_crossover,
    GABPP: both_pairs_crossover
}


def generate_candidates(population):
    candidates = []
    for parent1 in population:
        for parent2 in population:
            if parent1 == parent2:
                continue
            candidates += crossovers[GA_TYPE](parent1, parent2)
    return candidates


def GA():
    random.seed()  # initialize internal state of random number generator
    population = init_population()  # generation 0

    for gen in range(GENERATIONS):
        print("Generation ", gen)
        print_population(population)
        if max([fitness(population[i]) for i in range(POP_SIZE)]) == MAX_PATH_COST:
            print("max fitness")
            break
        candidates = sorted(generate_candidates(population), key=fitness)
        nextgen_population, candidates = take_elite(candidates)
        nextgen_population += roulette_selection(candidates)
        population = nextgen_population


# begin GA run
GA()
