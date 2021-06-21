# tiny genetic algorithm by moshe sipper, www.moshesipper.com
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time as t

from LSP_with_target import weird_path_search, count_nodes_bcc, reachable_nodes_heuristic

START_NODE = 3333
TARGET_NODE = 2222

CROSSES_PERFORMED = 0

POP_MAX = []
POP_AVG = []

GANP = 0
GAIP = 1
GABPP = 2
GA_TYPE = GANP

DO_MUTATION = False
P_MUTATION = 0.1
NODE_MUTATION = 0.1

N_NODES = 500
DENSITY = 0.005

time_cutoff = 1000

POP_SIZE = 150  # population size
# GENOME_SIZE     = 5     # number of bits per individual in the population
GENERATIONS = 500  # maximal number of generations to run GA
# TOURNAMENT_SIZE = 3     # size of tournament for tournament selection
# PROB_MUTATION   = 0.1   # bitwise probability of mutation
ELITE = 5  # percentage of the population that carry automatically
MAX_PATH_COST = 200000000

G = nx.complete_graph(10)


def has_duplicates(values):
    return len(values) != len(set(values))


def display_g(path1, path2):
    graph = G.copy()
    pos = nx.random_layout(graph)
    if not path1:
        nx.draw(graph, pos, with_labels=True)
    else:
        for (u, v) in graph.edges():
            graph[u][v]['color'] = "black"
        x1 = path1[0]
        for i in range(1, len(path1)):
            x2 = path1[i]
            graph[x1][x2]['color'] = "red"
            x1 = x2
        x1 = path2[0]
        for i in range(1, len(path2)):
            x2 = path2[i]
            graph[x1][x2]['color'] = "green"
            x1 = x2
        colors = nx.get_edge_attributes(graph, 'color').values()
        nx.draw(graph, pos, with_labels=True, edge_color=colors)
    plt.show()


def mutate(individual):
    for i in range(len(individual) - 2, 0, -1):
        nodes = [node for node in G.neighbors(individual[i]) if node not in individual]
        # try:
        #     nodes.remove(individual[i - 1])
        #     nodes.remove(individual[i + 1])
        # except Exception as e:
        #     print(i)
        #     print(nodes)
        #     print(individual)
        #     print(individual[i - 1], individual[i], individual[i + 1])
        #     print(list(G.neighbors(individual[i - 1])))
        #     raise e
        if nodes and random.uniform(0, 100) > NODE_MUTATION * 100:
            return individual[:i+1] + [nodes[0]]
    return individual


def pop_mutation(population):
    to_mutate, rest = map(list, np.split(population,
                                         [int(P_MUTATION * len(population))]))
    mutated = [mutate(x) for x in to_mutate]
    rest.extend(mutated)
    return rest


def init_population():  # population comprises bitstrings
    global G

    def choose():
        # print("node: ", node)
        neighbors = start_neighbors if node == start_node else check_G.neighbors(node)
        neighbors_lst = list(neighbors)
        # print("is empty?", neighbors_lst)
        # print("is empty again?", neighbors_lst)
        if not neighbors_lst:
            # print("empty neighbors", neighbors_lst)
            return -1
        degrees_lst = list(check_G.degree(neighbors_lst))
        degrees_sum = sum([degree[1] for degree in degrees_lst])
        weights = [(degree[1] / degrees_sum) for degree in degrees_lst]
        # print("neighbors", neighbors_lst)
        # print("weights", weights)
        return random.choices(population=neighbors_lst, weights=weights, cum_weights=None, k=1)[0]

    population = []
    for _ in range(POP_SIZE):
        check_G = G.copy()
        path = []
        start_node = random.choice(list(G.nodes))
        # print(start_node)
        path += [start_node]
        node = start_node
        start_neighbors = check_G.neighbors(node)
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
    # return sum([G[individual[i]][individual[i + 1]]['weight'] for i in range(len(individual) - 1)])
    return len(individual)


def non_intersecting_crossover(parent1, parent2):
    global CROSSES_PERFORMED
    # print(len(parent1), len(parent2))
    candidates = []
    # print("*********start crossoever******")
    # print(parent1, "\n", parent2)
    if not set(parent1).isdisjoint(parent2):
        # print(1)
        # = print(set(parent2).intersection(parent2))
        return candidates
    # 4print("performing crossover")
    # display_g(parent1, parent2)
    i = 1
    for node1 in parent1[1:-2]:
        neighbors_lst = list(G.neighbors(node1))
        j = 1
        # print(neighbors_lst)
        for node2 in parent2[1:-2]:
            if node2 in neighbors_lst:
                # print("cross cross")
                child1 = parent1[:i + 1] + parent2[j:]
                child2 = list(reversed(parent1[i:])) + parent2[j:]
                child3 = parent2[:j + 1] + parent1[i:]
                child4 = parent2[:j + 1] + parent1[i:]
                candidates += [child1, child2, child3, child4]
                CROSSES_PERFORMED += 1
                # print("=====================================================")
                # print(node1, node2)
                # print(parent1[:i + 1], parent2[j:])
                # print(list(reversed(parent1[i:])), parent2[j:])
                # print(parent2[:j + 1], parent1[i:])
                # print(parent2[:j + 1], parent1[i:])
                # print("=====================================================")
                # print(i, j, len(parent1), len(parent2))
            j += 1
        i += 1
    return candidates


def intersecting_crossover(parent1, parent2):
    pass


def both_pairs_crossover(parent1, parent2):
    return non_intersecting_crossover(parent1, parent2) + intersecting_crossover(parent1, parent2)


def print_population(population):
    global POP_AVG
    global POP_MAX
    # pops = [str(population[i]) for i in range(POP_SIZE)]
    m = max([fitness(population[i]) for i in range(POP_SIZE)])
    a = sum([fitness(population[i]) for i in range(POP_SIZE)]) / len(population)
    POP_MAX += [m]
    POP_AVG += [a]


def take_elite(population):
    take_quantity = int(POP_SIZE * ELITE / 100) + 1
    elite = population[-take_quantity:]
    rest = population[:-take_quantity]
    # print("elite ", elite)
    # print("rest", rest)
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
        return individual

    selected = []
    n = int(POP_SIZE * (100 - ELITE) / 100)
    # print(n)
    for i in range(n):
        pick = select_one()
        if not pick:
            print(i)
            raise Exception("wtf")
        selected += [pick]
        # += [x]
        # population.remove(x)
    # print("--", len(selected))
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
    # print(len(candidates))
    return candidates + pop_mutation(population)


def get_random_graph(num_of_nodes, prob_of_edge):
    new_graph = nx.fast_gnp_random_graph(num_of_nodes, prob_of_edge, directed=False)
    for (u, v) in new_graph.edges():
        new_graph[u][v]['weight'] = random.randint(0, 10)
    for node in new_graph.nodes:
        new_graph.nodes[node]["constraint_nodes"] = [node]
    return new_graph


def GA():
    global G
    global POP_SIZE
    global CROSSES_PERFORMED

    random.seed()  # initialize internal state of random number generator

    G = get_random_graph(N_NODES, DENSITY)
    time = t.time()
    population = init_population()  # generation 0

    for gen in range(GENERATIONS):
        if t.time() - time > time_cutoff:
            break
        print("Generation ", gen)
        # print_population(population)
        if max([fitness(population[i]) for i in range(POP_SIZE)]) == MAX_PATH_COST:
            print("max fitness")
            break
        candidates = sorted(generate_candidates(population), key=fitness)
        nextgen_population, candidates = take_elite(candidates)
        nextgen_population += roulette_selection(candidates)
        print_population(nextgen_population)
        print("crosses - ", CROSSES_PERFORMED)
        CROSSES_PERFORMED = 0
        population = nextgen_population

    # display data
    plt.scatter(range(len(POP_AVG)), list(POP_AVG), color="yellow")
    plt.scatter(range(len(POP_MAX)), list(POP_MAX), color="green")
    plt.show()


def run_h_search(hs):
    global START_NODE
    global TARGET_NODE

    G.add_node(START_NODE)
    G.add_node(TARGET_NODE)

    for node in G.nodes:
        if node != START_NODE and node != TARGET_NODE:
            G.add_edge(node, START_NODE)
            G.add_edge(node, TARGET_NODE)
            G.add_edge(START_NODE, node)
            G.add_edge(TARGET_NODE, node)

    G.nodes[START_NODE]["constraint_nodes"] = [START_NODE]
    G.nodes[TARGET_NODE]["constraint_nodes"] = [TARGET_NODE]

    for h in hs:
        path = weird_path_search(h, G)
        print("path length: ", len(path))


# begin GA run
GA()
# run_h_search([count_nodes_bcc, reachable_nodes_heuristic])
