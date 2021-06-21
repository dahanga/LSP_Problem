import networkx as nx
import time as t
import matplotlib.pyplot as plt

DEFAULT_WEIGHT = 1
G = nx.Graph()
PROBLEM_GRAPH = nx.Graph()
values = []
expansions = []

N = -1

TIMEOUT = 100
CUTOFF = 10000
FAILURE = -1
EXPANSION_TIME = 1

START_NODE = 3333
TARGET_NODE = 2222

# STATE = (CURRENT NODE, PATH, AVAILABLE NODES)
CURRENT_NODE = 0
PATH = 1
AVAILABLE_NODES = 2

counter = 0


def diff(li1, li2):
    return list(set(li1) - set(li2))


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def expand(state):
    global counter
    global PROBLEM_GRAPH
    ret = []
    current_v = state[CURRENT_NODE]
    neighbors = G.neighbors(current_v)
    availables = state[AVAILABLE_NODES]
    path = state[PATH]
    # if counter % 100 == 0:
    #     print(counter, end=" ")
    counter += 1
    for v in intersection(neighbors, availables):
        new_path = path + (v,)
        new_availables = tuple(diff(availables, G.nodes[current_v]["constraint_nodes"]))
        new_state = (v, new_path, new_availables)
        if new_state not in list(PROBLEM_GRAPH.nodes):
            PROBLEM_GRAPH.add_node(new_state, parent=state, g=-1)
            ret.append(new_state)
        PROBLEM_GRAPH.add_edge(new_state, state)
    return ret


def checkIfWorthOpen(closed, node, heuristic):
    return node not in closed


def limited_AStar(start_state, f, goal_check):
    global values
    global expansions
    start_time = t.time()
    expansion = 0
    OPEN = [start_state]
    CLOSED = []
    node = FAILURE
    values = []
    expansions = []
    while True:
        if expansion % 10 == 0:
            print(expansion, end=' ')
        if not OPEN:
            end_time = t.time() - start_time
            return node, expansion, end_time
        OPEN = sorted(OPEN, key=f)
        node = OPEN.pop(0)
        values += [f(node)]
        expansions += [expansion]
        if (t.time() - start_time > TIMEOUT) or (0 < CUTOFF == expansion) or goal_check(node):
            end_time = t.time() - start_time
            return node, expansion, end_time
        if checkIfWorthOpen(CLOSED, node, f):
            CLOSED.append(node)
            nodes = expand(node)
            OPEN += nodes
            expansion += 1


# def g(state):
#     parent = PROBLEM_GRAPH.nodes[state]["parent"]
#     if parent == 0:
#         PROBLEM_GRAPH.nodes[state]["g"] = 0
#         return 0
#     g_parent = PROBLEM_GRAPH.nodes[state]["parent"]["g"]
#     if g_parent == -1:
#         g_parent = g(parent)
#     g_value = 1 + g_parent
#     PROBLEM_GRAPH.nodes[state]["g"] = g_value
#     return g_value


def g(state):
    path = state[PATH]
    g_value = len(path)
    return g_value


def available_nodes_heuristic(state):
    left_nodes_num = len(state[AVAILABLE_NODES])
    return left_nodes_num


def reachable_nodes_heuristic(state):
    node = state[CURRENT_NODE]
    availables = state[AVAILABLE_NODES] + (node,)
    nested = G.subgraph(availables)
    reachables = nx.descendants(nested, source=node)
    return len(reachables) if TARGET_NODE in reachables else N


def longest_shortest_path(state):
    node = state[CURRENT_NODE]
    availables = state[AVAILABLE_NODES] + (node,)
    nested = G.subgraph(availables)
    try:
        path = nx.shortest_path(nested, source=node, target=TARGET_NODE)
        return N - len(path)
    except:
        return N + 1


def is_legit_shimony_pair(graph, in_node, out_node, x1, x2):
    def find_disjoint_paths(graph, segment):
        x1 = segment[0]
        x2 = segment[1]
        return list(nx.edge_disjoint_paths(graph, x1, x2))

    def paths_disjoint(p1, p2):
        for x in p1:
            for y in p2:
                if x == y:
                    return False
        return True

    def can_combine_paths(m, paths, combined):
        if m > 2:
            return True
        ret = False
        for path in paths[m]:
            if paths_disjoint(path[1:], combined):
                ret = can_combine_paths(m+1, paths, combined + path[1:])
            if ret:
                break
        return ret

    segments = [[in_node, x1], [x1, x2], [x2, out_node]]
    paths = []
    for seg in segments:
        paths += [find_disjoint_paths(graph, seg)]

    return can_combine_paths(0, paths, [in_node])


def shimony_pairs(graph, in_node, out_node):
    counter = 1 # first pair is in and out nodes
    for x1 in graph.nodes:
        if x1 == in_node or x1 == out_node:
            continue
        for x2 in graph.nodes:
            if x2 == in_node or x2 == out_node or x2 == x1:
                continue
            if is_legit_shimony_pair(graph, in_node, out_node, x1, x2):
                counter += 1
    return counter


def bcc_thingy(state):
    current_node = state[CURRENT_NODE]
    availables = state[AVAILABLE_NODES] + (current_node,)
    nested = G.subgraph(availables)
    reachables = nx.descendants(nested, source=current_node)
    reachables.add(current_node)
    reach_nested = nested.subgraph(reachables)
    bcc_comps = list(nx.biconnected_components(reach_nested))
    path = []
    if TARGET_NODE in reachables:
        path = nx.shortest_path(reach_nested, source=current_node, target=TARGET_NODE)
        if len(path) < 2:
            return -1, -1, -1, -1, -1, -1
    else:
        return -1, -1, -1, -1, -1, -1

    bcc_dict = {}
    for node in reachables:
        bcc_dict[node] = []
    comp_index = 0
    for comp in bcc_comps:
        for node in comp:
            # mapping nodes to biconnected comp
            bcc_dict[node] += [comp_index]
        comp_index += 1

    relevant_comps = []
    relevant_comps_index = []
    added_comp = -1
    # getting only relevant components
    for i in range(len(path)-1):
        current_comp = -1
        for comp in bcc_dict[path[i]]:
           for comp2 in  bcc_dict[path[i+1]]:
               if comp == comp2:
                   current_comp = comp
        if current_comp == -1:
            raise ConnectionError()
        # print(f"node - {path[i]}\tcomp - {bcc_dict[path[i]]}\t\tcurrent comp - {current_comp}")
        if current_comp != added_comp:
            relevant_comps += [bcc_comps[current_comp]]
            relevant_comps_index += [current_comp]
            added_comp = current_comp

    return reachables, bcc_dict, relevant_comps, relevant_comps_index, reach_nested, current_node


def count_nodes_bcc(state):
    _, _, relevant_comps, _, _, _ = bcc_thingy(state)
    if relevant_comps == -1:
        return len(G.nodes) ** 2  # if theres no path
    ret = 0
    for comp in relevant_comps:
        ret += len(comp)
    return ret


def shimony_pairs_bcc(state):
    reachables, bcc_dict, relevant_comps, relevant_comps_index, reach_nested, current_node = bcc_thingy(state)
    if relevant_comps == -1:
        return len(G.nodes) ** 2  # no path
    cut_node_dict = {}
    for node in reachables:
        comps = bcc_dict[node]
        # if node in more than 1 component, its a cut node
        if len(comps) > 1:
            for c1, c2 in [(a, b) for idx, a in enumerate(comps) for b in comps[idx + 1:]]:
                cut_node_dict[(c1, c2)] = node
                cut_node_dict[(c2, c1)] = node

    n = len(relevant_comps)
    in_pairs = 0
    for i in range(n):
        comp = relevant_comps[i]
        # getting cut nodes
        if i == 0:
            in_node = current_node
        else:
            in_node = cut_node_dict[(relevant_comps_index[i-1], relevant_comps_index[i])]
        if i == n-1:
            out_node = TARGET_NODE
        else:
            out_node = cut_node_dict[(relevant_comps_index[i], relevant_comps_index[i+1])]
        in_pairs += shimony_pairs(reach_nested.subgraph(comp), in_node, out_node)

    inter_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            # for every two nodes in different there's a path from start to target that visits them
            inter_pairs += len(relevant_comps[i]) * len(relevant_comps[j])

    return inter_pairs + in_pairs


def component_degree(comp, graph):
    graph = graph.subgraph(comp)
    return graph.number_of_edges()



def shimony_pairs_bcc_aprox(state):
    reachables, bcc_dict, relevant_comps, relevant_comps_index, reach_nested, current_node = bcc_thingy(state)
    if relevant_comps == -1:
        return len(G.nodes) ** 2  # no path
    cut_node_dict = {}

    n = len(relevant_comps)
    nodes_num = len(reachables)
    comp_degree_coeff = 1 / (nodes_num ** 2)
    in_pairs = 0
    nodes_per_comp = map(lambda comp: (comp_degree_coeff * component_degree(comp, reach_nested)) ** 2, relevant_comps)
    in_pairs = sum(nodes_per_comp)
    inter_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            # for every two nodes in different there's a path from start to target that visits them
            inter_pairs += len(relevant_comps[i]) * len(relevant_comps[j])

    return inter_pairs


def function(state, heuristic):
    res = g(state) + heuristic(state)
    return res


def goal_check_path(state):
    left = len(state[PATH])
    return left == len(G.nodes)


def weird_path_search(heuristic, g):
    global G
    G = g
    start_available = tuple(diff(list(G.nodes), G.nodes[START_NODE]["constraint_nodes"]))
    start_path = (START_NODE,)
    start_node = START_NODE
    start_state = (start_node, start_path, start_available)
    f = (lambda x: function(x, heuristic))
    PROBLEM_GRAPH.add_node(start_state, parent=0)
    (res, expansions, time) = limited_AStar(start_state, f, goal_check_path)
    print('-----------------------------------------------------------------')
    return res[PATH] if res != FAILURE else res


def parseVertex(row):
    global START_NODE
    global N
    N += 1
    parts = row.split()
    v_str = parts[0][2:]
    v = int(v_str)
    constraint_nodes = []
    if len(parts) > 1 and parts[1][0] == 'C':
        for node in parts[1][1:].split(','):
            constraint_nodes += [int(node)]
    G.add_node(v, constraint_nodes=constraint_nodes)
    if START_NODE == -1 or START_NODE > v:
        START_NODE = v


def parseEdge(row):
    parts = row.split()
    v1 = int(parts[1])
    v2 = int(parts[2])
    w = DEFAULT_WEIGHT
    G.add_edge(v1, v2, weight=w, color="blue")


def display_g(path):
    graph = G.copy()
    pos = nx.kamada_kawai_layout(graph)
    if not path:
        nx.draw(graph, pos, with_labels=True)
    else:
        x1 = path[0]
        for i in range(1, len(path)):
            x2 = path[i]
            graph[x1][x2]['color'] = "red"
            x1 = x2
        colors = nx.get_edge_attributes(graph, 'color').values()
        nx.draw(graph, pos, with_labels=True, edge_color=colors)
    plt.show()


def run():
    with open('setup.txt', 'r') as fp:
        for line in fp:
            if len(line) == 1:
                continue
            if line[1] == 'E':
                parseEdge(line)
            if line[1] == 'V':
                parseVertex(line)
        path = weird_path_search(count_nodes_bcc)
        print(path)
        display_g(path)
        return path


def get_random_graph(num_of_nodes, prob_of_edge):
    global TARGET_NODE
    new_graph = []
    path = []
    while True:
        new_graph = nx.fast_gnp_random_graph(num_of_nodes, prob_of_edge)
        try:
            path = nx.shortest_path(new_graph, source=0, target=1)
            break
        except:
            continue
    print(path)
    for node in new_graph.nodes:
        new_graph.nodes[node]["constraint_nodes"] = [node]
    TARGET_NODE = 1
    return new_graph


def test_heuristics(heuristic_name_func_pairs, runs, num_of_nodes, prob_of_edge, cutoff):
    global CUTOFF
    global G
    global START_NODE
    global PROBLEM_GRAPH
    CUTOFF = cutoff
    total_runtime = [0] * len(heuristic_name_func_pairs)
    total_path_length = [0] * len(heuristic_name_func_pairs)
    for run_index in range(runs):
        test_graph = get_random_graph(num_of_nodes, prob_of_edge)
        print(f"**** RUN NUMBER {run_index}")
        h_index = 0
        for name, h in heuristic_name_func_pairs:
            start_time = t.time()
            G = test_graph.copy()
            START_NODE = 0
            PROBLEM_GRAPH = nx.Graph()
            display_g([])
            path = weird_path_search(h)
            end_time = t.time()
            print(f"heuristic: {name}, \tpath length: {len(path)}, \truntime: {end_time - start_time}")
            total_runtime[h_index] += end_time - start_time
            total_path_length[h_index] += len(path)
            h_index += 1
        print()
    print("*** RESULTS:")
    for h_index in range(len(heuristic_name_func_pairs)):
        name = heuristic_name_func_pairs[h_index][0]
        mean_runtime = total_runtime[h_index] / runs
        mean_path_length = total_path_length[h_index] / runs
        print(f"heuristic: {name}, \tmean path length: {mean_path_length}, \tmean runtime: {mean_runtime}")


# heuristics = [
#     ["reachables", reachable_nodes_heuristic],
#     ["bcc nodes", count_nodes_bcc],
#     ["shimony pairs heuristics approx", shimony_pairs_bcc_aprox]
# ]
# test_heuristics(heuristics, 1, 100, 0.03, 500)
# plt.scatter(expansions, values)
# plt.show()