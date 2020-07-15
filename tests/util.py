import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
import itertools
from junctiontree import computation as comp
from junctiontree import construction as cons

def compute_num_combinations(n, r=2):
    '''Compute n choose r. (Function available in Python 3.8 but not requiring this version)

    :param n: number of objects
    :param r: number of objects in each combination
    :return: number of combinations of r items from collection of n items
    '''

    return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))


def build_graph(factors, full=False):
    '''Builds an adjacency matrix representation for a graph. Nodes in factors
    are connected by edges (non-zero matrix entry) in the graph.

    :param factors: list of factors from which to build a graph
    :param full: create the full (not just upper triangular) matrix
    :return: node_list: a list which maps nodes to index in adjacency matrix
    :return: adj_matrix: a 2-D numpy array representing adjacency matrix
    '''

    sorted_nodes = sorted({ node for factor in factors for node in factor })

    node_lookup = { node : i for i, node in enumerate(sorted_nodes) }

    node_count = len(sorted_nodes)
    adj_matrix = np.full((node_count, node_count), False)


    for factor in factors:
        for i, n1 in enumerate(factor):
            n1_idx = node_lookup[n1]
            for n2 in factor[i+1:]:
                n2_idx = node_lookup[n2]
                # add an edge between nodes
                adj_matrix[n1_idx, n2_idx] = True
                if full:
                    adj_matrix[n2_idx, n1_idx] = True

    return sorted_nodes, adj_matrix


def find_base_cycle(mst, start_node, end_node):
    '''Uses a depth-first traversal to find the cycle created by adding edge

    :param mst: the minimum spanning tree
    :param start_node: one of the nodes in the cycle generating edge used as start node of depth-first traversal
    :param end_node: the other node from the cycle generating edge where depth-first traversal should end
    :return: list of nodes representing cycle in augmented MST
    '''

    # need to know the path from start node to end node provided by tracing path
    # through predecessors starting at end node and ending at start node

    nodes, predecessors = depth_first_order(mst, start_node, directed=False)

    cycle = []

    c_node = end_node

    while c_node != start_node:
        cycle.append(c_node)
        c_node = predecessors[c_node]

    cycle.append(c_node)

    return cycle


def create_cycle_basis(adj_matrix):
    '''Create a cycle basis from an adjacency matrix representation of a graph, G. A cycle basis is
    formed by first finding a minimum spanning tree (MST) of G. Adding any edge in G that is not
    in the MST results in the formation of a single cycle. Each cycle formed in this manner is
    part of the cycle basis. This function returns all cycles in the cycle basis using the
    index of the node in G as its representation in the cycle.

    :param adj_matrix: adjacency matrix representation of the graph
    :return: the list of cycles (list of nodes) representing the cycle basis
    '''


    mst = minimum_spanning_tree(adj_matrix)

    cycle_edges = np.transpose(np.nonzero(adj_matrix - mst)) # provides non-zeros by index

    # a single unique cycle is created when cycle edge added to min spanning tree
    basis = [
                find_base_cycle(mst, edge[0], edge[1])
                for edge in cycle_edges
    ]

    return basis


def find_cycles(factors, num):
    '''Generate a list of all cycles from a factor graph with edge count greater than or equal to num.

    :param factors: a list of factors (variable lists) representing node connections
    :param num: the minimum number of edges in each cycle
    :return: a list of cycles meeting minimum edge requirement
    '''

    node_list, adj_mat = build_graph(factors)

    cb = create_cycle_basis(adj_mat)

    cb_edges = [zip(vars,(vars[1:] + vars[:1])) for vars in cb]

    graph_edges = [set(edge) for edge in np.transpose(np.nonzero(adj_mat))]

    # http://dspace.mit.edu/bitstream/handle/1721.1/68106/FTL_R_1982_07.pdf
    bit_seqs = np.zeros((len(cb_edges), len(graph_edges)), dtype=np.bool)

    # populate edge membership arrays for each cycle basis
    for i in range(0,len(cb_edges)):
        edge_list = [set(edge) for edge in cb_edges[i]]
        for j in range(0,len(graph_edges)):
            if graph_edges[j] in edge_list:
                bit_seqs[i][j] = 1

    cycles = [np.array(graph_edges)[np.nonzero(cycle)[0]]
                for cycle in gibbs_elem_cycles(bit_seqs) if sum(cycle) >= num]


    # replace indices with variables for edges and cycles representation
    graph_edges = [
                    set(
                        [
                            node_list[tuple(edge)[0]],
                            node_list[tuple(edge)[1]]
                        ]
                    )
                    for edge in graph_edges
    ]

    cycles = [
                [
                    set(
                        [
                            node_list[tuple(edge)[0]],
                            node_list[tuple(edge)[1]]
                        ]
                    )
                    for edge in cycle
                ]
                for cycle in cycles
    ]

    return graph_edges, cycles

def gibbs_elem_cycles(fcs):
    '''Generate all elementary cycles based on the set of fundamental cycles of an undirected graph.

    Norman E. Gibbs. 1969. A Cycle Generation Algorithm for Finite
        Undirected Linear Graphs. J. ACM 16, 4 (October 1969), 564-568.
        DOI=http://dx.doi.org/10.1145/321541.321545

    :param fcs: list of fundamental cycles (represented as list of nodes) of an undirected graph
    :return: list of elementary cycles (represented as list of nodes)
    '''

    s = [fcs[0]]
    q = [fcs[0]]
    r = []
    r_star = []
    i = 1
    while i < fcs.shape[0]:
        for t in q:
            if np.any(np.logical_and(t, fcs[i])):
                # append t ring_sum fcs[0] to r
                r.append(np.logical_xor(t,fcs[i]).tolist())
            else:
                # append t ring_sum fcs[0] to r_star
                r_star.append(np.logical_xor(t,fcs[i]).tolist())

        for u,v in itertools.combinations(r, 2):
            # check both ways u subset of v or v subset of u
            if np.array_equal(np.logical_and(u, v), u):
                if v in r: # may have already been removed
                    r.remove(v)
                if v not in r_star:
                    r_star.append(v)
            elif np.array_equal(np.logical_and(v, u), v):
                if u in r: # may have already been removed
                    r.remove(u)
                if u not in r_star:
                    r_star.append(u)

        #s U r U fcs[i]
        s = [list(st) for st in set(map(tuple, itertools.chain(s,r,[fcs[i]])))]

        #q U r U r_star U fcs[i]
        q = [list(st) for st in set(map(tuple, itertools.chain(q,r,r_star,[fcs[i]])))]

        r = []
        r_star = []
        i+=1

    return s


def assert_triangulated(factors, triangulation):
    '''Asserts that an undirected graph represented by list of factors is triangulated by addition of edges
    specified in triangulation

    An undirected graph is triangulated iff every cycle of length four or
    greater contains an edge that connects two nonadjacent nodes in the
    cycle. (Huang and Darwiche, 1996)

    :param t1: list of factors defining factor graph
    :param t2: a list of edges added to underlying undirected graph to make graph triangulated
    '''

    graph_edges, cycles = find_cycles(factors, 4)

    for cycle in cycles:
        cycle_vars = set([var for edge in cycle for var in edge])

        # at least one chord of cycle should be in triangulation or part of
        # original graph

        assert sum(
                    [
                        1 for edge in triangulation + graph_edges
                        if set(edge) not in cycle and set(edge).issubset(cycle_vars)
                    ]
        ) > 0


def assert_junction_trees_equal(t1, t2):
    '''Assert two junction trees are equal (both trees contain same edges and node indices)

    :param t1: a junction tree
    :param t2: a second junction tree
    '''

    pairs1 = set([tuple(sorted(p)) for p in cons.generate_potential_pairs(t1)])
    pairs2 = set([tuple(sorted(p)) for p in cons.generate_potential_pairs(t2)])
    assert pairs1 == pairs2


def assert_potentials_equal(p1, p2):
    '''Asserts that list of potentials has same number and all are equal

    :param p1: a list of factor graph potentials
    :param p2: a second list of factor graph potentials
    '''


    # Same number of potentials
    assert len(p1) == len(p2)

    if len(p1):
        # Check equality of arrays
        np.testing.assert_allclose(p1[0], p2[0])
        # recursively check remaining potentials
        assert_potentials_equal(p1[1:], p2[1:])
