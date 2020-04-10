import numpy as np

from junctiontree import beliefpropagation as bp
import unittest
#import networkx as nx
import itertools
import heapq
import copy
import junctiontree.junctiontree as jt
from junctiontree.sum_product import SumProduct
import math


# Tests here using pytest


def assert_junction_tree_equal(t1, t2):
    """Test equality of two junction trees

    Both trees contain same edges and node indices

    """

    pairs1 = set([tuple(sorted(p)) for p in bp.generate_potential_pairs(t1)])
    pairs2 = set([tuple(sorted(p)) for p in bp.generate_potential_pairs(t2)])
    assert pairs1 == pairs2


def assert_factor_graph_equal(fg1, fg2):
    # assert that the variable maps are equal
    assert fg1[0] == fg2[0]
    # ensure that factor lists are the same
    assert len(fg1[1]) == len(fg2[1])
    assert np.all([a1 == a2 for a1,a2 in zip(fg1[1],fg2[1])])
    assert len(fg1[2]) == len(fg2[2])
    assert np.all([np.allclose(a1, a2) for a1,a2 in zip(fg1[2],fg2[2])])

def assert_triangulated(factors, triangulation):
    '''
        An undirected graph is triangulated iff every cycle of length four or
        greater contains an edge that connects two nonadjacent nodes in the
        cycle. (Huang and Darwiche, 1996)

        Only one such edge is required.

        Triangulation is a list of edges added to underlying undirected graph to
        make graph triangulated
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

def compute_num_combinations(n, r=2):
    '''
    Compute n choose r. (Function available in Python 3.8 but not requiring this version)

    :param n: number of objects
    :param r: number of objects in each combination
    :return: number of combinations of r items from collection of n items
    '''

    return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))


def test_build_graph():

    factors = [['A','B','C'], ['C','D','E'], ['B','F']]

    # sum combinations to calculate total edge count

    num_edges = sum(
                    [
                        compute_num_combinations(len(factor), 2)
                        for factor in factors
                    ]
    )

    node_list, adj_matrix = build_graph(factors)

    assert len(node_list) == 6
    assert node_list == ['A', 'B', 'C', 'D', 'E', 'F']
    assert adj_matrix.sum() == num_edges

def build_graph(factors):
    '''
    Builds an adjacency matrix representation for a graph. Nodes in factors
    are connected by edges (non-zero matrix entry) in the graph.

    :param factors: list of factors from which to build a graph
    :return: node_list: a list which maps nodes to index in adjacency matrix
    :return: adj_matrix: a 2-D numpy array representing upper triangular adjacency matrix
    '''

    sorted_nodes = sorted({ node for factor in factors for node in factor })

    node_lookup = { node: i for i, node in enumerate(sorted_nodes) }

    node_count = len(sorted_nodes)
    adj_matrix = np.full((node_count, node_count), False)


    for factor in factors:
        for i, n1 in enumerate(factor):
            n1_idx = node_lookup[n1]
            for n2 in factor[i+1:]:
                n2_idx = node_lookup[n2]
                # add an edge between nodes
                adj_matrix[n1_idx, n2_idx] = True

    return sorted_nodes, adj_matrix

def create_cycle_basis(adj_matrix):
    '''
    Create a cycle basis from an adjancency matrix. A cycle basis

    :param adj_matrix:
    :return:
    '''
    pass

def find_cycles(factors, num):
    G = build_graph(factors)

    #cb = nx.cycle_basis(G)
    cb = create_cycle_basis()
    cb_edges = [zip(nodes,(nodes[1:]+nodes[:1])) for nodes in cb]
    graph_edges = [set(edge) for edge in G.edges()]
    # generate a list of all cycles greater than or equal to num
    # http://dspace.mit.edu/bitstream/handle/1721.1/68106/FTL_R_1982_07.pdf
    bit_seqs = np.zeros((len(cb_edges), len(graph_edges)), dtype=np.bool)
    # populate edge membership arrays for each cycle basis
    for i in range(0,len(cb_edges)):
        edge_list = [set(edge) for edge in cb_edges[i]]
        for j in range(0,len(graph_edges)):
            if graph_edges[j] in edge_list:
                bit_seqs[i][j] = 1

    cycles = [np.array(graph_edges)[[np.nonzero(cycle)[0]]]
                for cycle in gibbs_elem_cycles(bit_seqs) if sum(cycle) >= num]

    return list(G.edges()), cycles

def gibbs_elem_cycles(fcs):
    '''
        Generate all elementary cycles based on the set of fundamental cycles
        of a undirected graph.

        Norman E. Gibbs. 1969. A Cycle Generation Algorithm for Finite
            Undirected Linear Graphs. J. ACM 16, 4 (October 1969), 564-568.
            DOI=http://dx.doi.org/10.1145/321541.321545
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
                #r.append(np.logical_xor(t,fcs[i]).astype(int).tolist())
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



def assert_potentials_equal(p1, p2):
    """Test equality of two potentials

    """

    # Same number of potentials
    assert len(p1) == len(p2)

    if len(p1):
        # Check equality of arrays
        np.testing.assert_allclose(p1[0], p2[0])
        # recursively check remaining potentials
        assert_potentials_equal(p1[1:], p2[1:])

def get_arrays_and_keys(tree, node_list, potentials):
    """Get all arrays and their keys as a flat list

    Output: [array1, keys1, ..., arrayN, keysN]

    """
    return list([potentials[tree[0]],node_list[tree[0]]]) + sum(
        [
            get_arrays_and_keys(child_tree, node_list, potentials)
            for child_tree in tree[1:]
        ],
        []
    )

def brute_force_sum_product(tree, node_list, potentials):
    """Compute brute force sum-product with einsum """

    # Function to compute the sum-product with brute force einsum
    arrays_keys = get_arrays_and_keys(tree, node_list, potentials)
    f = lambda output_keys: bp.sum_product.einsum(*(arrays_keys + [output_keys]))

    def __run(tree, node_list, p, f, res=[]):
        res.append(f(node_list[tree[0]]))
        for child_tree in tree[1:]:
            __run(child_tree, node_list, p, f, res)
        return res

    return __run(tree, node_list, potentials, f)

def assert_sum_product(junction_tree, node_order, potentials):
    """ Test hugin vs brute force sum-product """

    # node_order represents the order nodes are traversed
    # in get_arrays_and_keys function
    tree = junction_tree.tree
    node_list = junction_tree.clique_tree.maxcliques + junction_tree.separators
    assert_potentials_equal(
        brute_force_sum_product(
                            tree,
                            [node_list[idx] for idx in node_order],
                            [potentials[idx] for idx in node_order]
        ),
        bp.hugin(
                tree,
                node_list,
                potentials,
                bp.sum_product
        )
    )


def assert_junction_tree_consistent(tree, potentials):
    '''
        For each clique/sepset pair in tree, check consistency

        Function checks that junction tree is consistent with respect to the
        provided potentials
    '''

    node_list = tree.get_node_list()
    assert np.all(
                    [
                        potentials_consistent(
                                            potentials[c_ix1],
                                            node_list[c_ix1],
                                            potentials[c_ix2],
                                            node_list[c_ix2]
                        )
                        for c_ix1, c_ix2 in bp.generate_potential_pairs(tree.get_struct())
                ]
            )

def potentials_consistent(pot1, vars1, pot2, vars2):
    '''
        Ensure that summing over clique potentials for variables not present in
        sepset generates a potential equal to sepset potential (definition of
        consistent)
    '''
    c_pot, c_vars, s_pot, s_vars = (
                                    pot1,
                                    vars1,
                                    pot2,
                                    vars2
    ) if len(vars1) > len(vars2) else (
                                        pot2,
                                        vars2,
                                        pot1,
                                        vars1

    )

    return np.allclose(
                bp.sum_product.einsum(
                    c_pot,
                    c_vars,
                    np.intersect1d(c_vars, s_vars).tolist()
                ),
                s_pot
            )




class TestHUGINFunctionality(unittest.TestCase):
    '''
    examples taken from here:
    https://www.cs.ru.nl/~peterl/BN/examplesproofs.pdf
    http://www.inf.ed.ac.uk/teaching/courses/pmr/docs/jta_ex.pdf
    background here:
    http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/171216.pdf
    comparison between HUGIN and Shafer-Shenoy
    https://arxiv.org/pdf/1302.1553.pdf
    some python code for junction tree algorithm
    http://leo.ugr.es/pgm2012/proceedings/eproceedings/evers_a_framework.pdf
    https://github.com/eBay/bayesian-belief-networks

    var -> key_id:
    V1 -> 0
    V2 -> 1
    V3 -> 2

    factor -> factor_id:
    f(V2) -> 0 [f(1)]
    f(V1,V2) -> 1 [f(0,1)]
    f(V3,V2) -> 2 [f(2,1)]

    clique -> clique_id:
    (V1,V2) -> 0
    (V2) -> 1
    (V2,V3) -> 2

    [0, [0,1], (1, [1], [2, [1,2]]),]

    '''

    #ensure that likelihoods can be used in inference with and without evidence

    def test_marginalize_variable(self):
        '''
            given consistent clique potentials, calculate the marginal probability of
            a variable in the clique
            use example from Huang and Darwiche (H&D)

             a   b   d  |  phi_ABD(abd)
            --------------------------
            on  on  on  |   0.225
            on  on  off |   0.025
            on  off on  |   0.125
            on  off off |   0.125
            off on  on  |   0.180
            off on  off |   0.020
            off off on  |   0.150
            off off off |   0.150

            >>> ABD = np.ndarray(shape=(2,2,2))
            >>> ABD[1,1,1] = 0.225
            >>> ABD[1,1,0] = 0.025
            >>> ABD[1,0,1] = 0.125
            >>> ABD[1,0,0] = 0.125
            >>> ABD[0,1,1] = 0.180
            >>> ABD[0,1,0] = 0.020
            >>> ABD[0,0,1] = 0.150
            >>> ABD[0,0,0] = 0.150
        '''

        phiABD=np.array([
                            [
                                [ 0.15 ,  0.15 ],
                                [ 0.02 ,  0.18 ]
                            ],
                            [
                                [ 0.125,  0.125],
                                [ 0.025,  0.225]
                            ]
                        ])
        # https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
        # marginal probability of A, P(A)
        np.testing.assert_allclose(bp.sum_product.einsum(phiABD, [0,1,2], [0]), np.array([0.500, 0.500]))
        # marginal probability of D, P(D)
        np.testing.assert_allclose(np.array([0.32,0.68]), np.array([0.320, 0.680]))


    def test_pass_message(self):
        r"""
            Example taken from here: https://www.cs.ru.nl/~peterl/BN/examplesproofs.pdf
            Example will be processed under the assumption that potentials have been
            properly initialized outside of this test

            Variables: V1, V2, V3
            \phi_{V1} = [V2] # parents of V1
            \phi_{V2} = [] # parents of V2
            \phi_{V3} = [V2] # parents of V3
            F_{V1} = [V1, V2]
            F_{V2} = [V2]
            F_{V3} = [V2, V3]

            P(v1|v2) = 0.2
            P(v1|~v2) = 0.6
            P(~v1|v2) = 0.8
            P(~v1|~v2) = 0.4
            P(v3|v2) = 0.5
            P(v3|~v2) = 0.7
            P(~v3|v2) = 0.5
            P(~v3|~v2) = 0.3
            P(v2) = 0.9
            P(~v2) = 0.1


            V1  V2  |   \phi_{V1V2} (P(V1|V2))
            ------------------------
            0   0   |   0.4
            0   1   |   0.8
            1   0   |   0.6
            1   1   |   0.2


            V2  |   \phi_{V2} (1)
            -----------------
            0   |   1
            1   |   1

            V2  V3  |   \phi_{V2V3} (P(V3|V2)P(V2))
            -------------------------
            0   0   |   0.3 * 0.1 = 0.03
            0   1   |   0.7 * 0.1 = 0.07
            1   0   |   0.5 * 0.9 = 0.45
            1   1   |   0.5 * 0.9 = 0.45
        """

        phi12 = np.array([
                            [0.4, 0.8],
                            [0.6, 0.2]
                        ])

        phi2 = np.array([1, 1])
        phi23 = np.array([
                            [0.03, 0.07],
                            [0.45, 0.45]
                        ])

        phi2n = bp.sum_product.project(phi12, [0,1], [1])
        np.testing.assert_allclose(phi2n, np.array([1,1]))
        phi23 = bp.sum_product.absorb(phi23, [0,1], phi2, phi2n, [1])
        np.testing.assert_allclose(phi23, np.array([
                                        [0.03,0.07],
                                        [0.45,0.45]
                                    ]))

        phi2nn = bp.sum_product.project(phi23, [0,1], [0])
        np.testing.assert_allclose(phi2nn, np.array([0.1, 0.9]))
        phi12 = bp.sum_product.absorb(phi12, [0,1], phi2n, phi2nn, [1])
        np.testing.assert_allclose(phi12, np.array([
                                        [0.04,0.72],
                                        [0.06,0.18]
                                    ]))


    def test_collect_messages(self):
        # constructor for junction tree taking a list based definition
        # will have a function that can convert factor graph into JT
        jt = [
                0,
                (
                    1,
                    [
                        2,
                    ]
                )

            ]

        node_list = [["V1","V2"],["V2"],["V2", "V3"]]

        phi = []
        phi.append(
                    np.array(
                                [
                                    [0.4, 0.8],
                                    [0.6, 0.2]
                                ]
                            )
                    )

        phi.append(np.array([1, 1]))
        phi.append(
                    np.array(
                                [
                                    [0.03, 0.07],
                                    [0.45, 0.45]
                                ]
                            )
                    )

        phiN = bp.collect(
                            jt,
                            node_list,
                            phi,
                            [0]*len(phi),
                            bp.sum_product
        )

        np.testing.assert_allclose(
            phiN[2],
            np.array(
                [
                    [0.03,0.07],
                    [0.45,0.45]
                ]
            )
        )


    def test_distribute_messages(self):
        jt = [
                0,
                (
                    1,
                    [
                        2,
                    ]
                )
            ]
        node_list = [["V1","V2"],["V2"],["V2", "V3"]]
        phi = []
        phi.append(
                    np.array(
                                [
                                    [0.4, 0.8],
                                    [0.6, 0.2]
                                ]
                            )
                    )

        phi.append(np.array([1, 1]))
        phi.append(
                    np.array(
                                [
                                    [0.03, 0.07],
                                    [0.45, 0.45]
                                ]
                            )
                    )

        phiN = bp.collect(
                            jt,
                            node_list,
                            phi,
                            [0]*len(phi),
                            bp.sum_product
        )

        phiN2 = bp.distribute(
                                jt,
                                node_list,
                                phiN,
                                [0]*len(phiN),
                                bp.sum_product
        )

        np.testing.assert_allclose(
            phiN2[0],
            np.array(
                [
                    [0.04,0.72],
                    [0.06,0.18]
                ]
            )
        )

        np.testing.assert_allclose(
            phiN[1],
            np.array([0.1,0.9])
        )

        np.testing.assert_allclose(
            phiN[2],
            np.array(
                [
                    [0.03,0.07],
                    [0.45,0.45]
                ]
            )
        )


    def test_one_scalar_node(self):
        assert_sum_product(
                jt.JunctionTree(
                            [
                                0,
                            ],
                            [],
                            jt.CliqueGraph(
                                maxcliques=[[]],
                                factor_to_maxclique=[],
                                factor_graph=jt.FactorGraph(
                                    factors=[],
                                    sizes={}
                                )
                            )
                ),
                [0],
                [
                    np.random.randn(),
                ]
        )

    def test_one_matrix_node(self):
        assert_sum_product(
                jt.JunctionTree(
                            [
                                0,
                            ],
                            [],
                            jt.CliqueGraph(
                                maxcliques=[[3, 5]],
                                factor_to_maxclique=[0],
                                factor_graph=jt.FactorGraph(
                                    factors=[[3,5]],
                                    sizes={
                                        3:2,
                                        5:3
                                    }
                                )
                            )
                ),
                [0],
                [
                    np.random.randn(2, 3),
                ]
        )

    def test_one_child_node_with_all_variables_shared(self):
        assert_sum_product(
            jt.JunctionTree(
                        [
                            0,
                            (
                                2,
                                [
                                    1,
                                ]
                            )
                        ],
                        [[5, 3]],
                        jt.CliqueGraph(
                            maxcliques=[[3, 5],[5, 3]],
                            factor_to_maxclique=[0, 1],
                            factor_graph=jt.FactorGraph(
                                factors=[[3], [5]],
                                sizes={
                                    3:2,
                                    5:3
                                }
                            )
                        )
            ),
            [0,2,1],
            [
                np.random.randn(2, 3),
                np.random.randn(3, 2),
                np.ones((3, 2)),
            ]
        )

    def test_one_child_node_with_one_common_variable(self):
        assert_sum_product(
                    jt.JunctionTree(
                                [
                                    0,
                                    (
                                        2,
                                        [
                                            1,
                                        ]
                                    )
                                ],
                                [[5]],
                                jt.CliqueGraph(
                                    maxcliques=[[3, 5],[5, 9]],
                                    factor_to_maxclique=[0, 1],
                                    factor_graph=jt.FactorGraph(
                                        factors=[[3,5],[5,9]],
                                        sizes={
                                            3:2,
                                            5:3,
                                            9:4
                                        }
                                    )
                                )
                    ),
                    [0,2,1],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 4),
                        np.ones((3,)),
                    ]
        )

    def test_one_child_node_with_no_common_variable(self):
        assert_sum_product(
                    jt.JunctionTree(
                                [
                                    0,
                                    (
                                        2,
                                        [
                                            1,
                                        ]
                                    )
                                ],
                                [[]],
                                jt.CliqueGraph(
                                    maxcliques=[[3],[9]],
                                    factor_to_maxclique=[0, 1],
                                    factor_graph=jt.FactorGraph(
                                        factors=[[3],[9]],
                                        sizes={
                                            3:2,
                                            9:3
                                        }
                                    )
                                )
                    ),
                    [0,2,1],
                    [
                        np.random.randn(2),
                        np.random.randn(3),
                        np.ones(()),
                    ]
        )

    def test_one_grand_child_node_with_no_variable_shared_with_grand_parent(self):
        assert_sum_product(
                    jt.JunctionTree(
                                [
                                    0,
                                    (
                                        3,
                                        [
                                            1,
                                            (
                                                4,
                                                [
                                                    2,
                                                ]
                                            )
                                        ]
                                    )
                                ],
                                [[5],[9]],
                                jt.CliqueGraph(
                                    maxcliques=[[3, 5],[5, 9],[9, 1]],
                                    factor_to_maxclique=[0,1,2],
                                    factor_graph=jt.FactorGraph(
                                        factors=[[3, 5],[5, 9],[9, 1]],
                                        sizes={
                                            1:5,
                                            3:2,
                                            5:3,
                                            9:4,
                                        }
                                    )
                                )
                    ),
                    [0,2,4,1,3],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 4),
                        np.random.randn(4, 5),
                        np.ones((3,)),
                        np.ones((4,)),
                    ]
        )

    def test_one_grand_child_node_with_variable_shared_with_grand_parent(self):
        assert_sum_product(
                    jt.JunctionTree(
                                [
                                    0,
                                    (
                                        3,
                                        [
                                            1,
                                            (
                                                4,
                                                [
                                                    2,
                                                ]
                                            )
                                        ]
                                    )
                                ],
                                [[5],[5]],
                                jt.CliqueGraph(
                                    maxcliques=[[3, 5],[5, 9],[1, 5]],
                                    factor_to_maxclique=[0,1,2],
                                    factor_graph=jt.FactorGraph(
                                        factors=[[3, 5],[5, 9],[1, 5]],
                                        sizes={
                                            1:6,
                                            3:2,
                                            5:3,
                                            9:4
                                        },
                                    )
                                )
                    ),
                    [0,2,4,1,3],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 4),
                        np.random.randn(6, 3),
                        np.ones((3,)),
                        np.ones((3,)),
                    ]
        )

    def test_two_children_with_no_variable_shared(self):
        assert_sum_product(
                    jt.JunctionTree(
                                 [
                                    0,
                                    (
                                        3,
                                        [
                                            1,
                                        ]
                                    ),
                                    (
                                        4,
                                        [
                                            2,
                                        ]
                                    )
                                ],
                                [[5],[3]],
                                jt.CliqueGraph(
                                    maxcliques=[[3, 5],[5, 9],[3, 1]],
                                    factor_to_maxclique=[0,1,2],
                                    factor_graph=jt.FactorGraph(
                                        factors=[[3, 5],[5, 9],[3, 1]],
                                        sizes={
                                            3:2,
                                            5:3,
                                            9:4,
                                            1:5
                                        },
                                    )
                                )
                    ),
                    [0,2,4,1,3],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 4),
                        np.random.randn(2, 5),
                        np.ones((3,)),
                        np.ones((2,)),
                    ]
        )

    def test_two_child_with_shared_variable(self):
        assert_sum_product(
                    jt.JunctionTree(
                                 [
                                    0,
                                    (
                                        3,
                                        [
                                            1,
                                        ]
                                    ),
                                    (
                                        4,
                                        [
                                            2,
                                        ]
                                    )
                                ],
                                [[5],[5]],
                                jt.CliqueGraph(
                                    maxcliques=[[3, 5],[5, 9],[5]],
                                    factor_to_maxclique=[0,1,2],
                                    factor_graph=jt.FactorGraph(
                                        factors=[[3, 5],[5, 9],[5]],
                                        sizes={
                                            3:2,
                                            5:3,
                                            9:4,
                                        },
                                    )
                                )
                    ),
                    [0,2,4,1,3],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 4),
                        np.random.randn(3),
                        np.ones((3,)),
                        np.ones((3,)),

                    ]
        )

    def test_two_children_with_3D_tensors(self):
        assert_sum_product(
                    jt.JunctionTree(
                                 [
                                    0,
                                    (
                                        3,
                                        [
                                            1,
                                        ]
                                    ),
                                    (
                                        4,
                                        [
                                            2,
                                        ]
                                    )
                                ],
                                [[5, 7],[5]],
                                jt.CliqueGraph(
                                    maxcliques=[[3, 5, 7],[5, 7, 9],[5, 1]],
                                    factor_to_maxclique=[0,1,2],
                                    factor_graph=jt.FactorGraph(
                                        factors=[[3, 5, 7],[5, 7, 9],[5, 1]],
                                        sizes={
                                            1:6,
                                            3:2,
                                            5:3,
                                            7:4,
                                            9:5
                                        },
                                    )
                                )
                    ),
                    [0,2,4,1,3],
                    [
                        np.random.randn(2, 3, 4),
                        np.random.randn(3, 4, 5),
                        np.random.randn(3, 6),
                        np.ones((3, 4)),
                        np.ones((3,)),
                    ]
        )

    def test_evidence_shrinking(self):
        A = np.random.rand(3,4,2) # vars: a,b,c
        a = [0]*3
        a[2]=1
        b = [0]*4
        b[2]=1
        c=[0]*2
        c[0]=1

        # update potential A based on observing a=2
        A_updated = bp.sum_product.einsum(A, [0,1,2], a, [0], [0,1,2])

        # shrinking from evidence
        # set the axis representing a (ax=0) to the value of a
        A_updated_es = A_updated[2,:,:]
        assert A_updated_es.shape == (4,2)

        # imagine we have another potential sharing vars b and c
        B = np.random.rand(4,2) # vars: a,c
        B_updated = bp.sum_product.einsum(A_updated, [0,1,2], B, [1,2], [1,2])

        B_updated_es = bp.sum_product.einsum(A_updated_es, [1,2], B, [1,2], [1,2])

        np.testing.assert_allclose(
                                B_updated,
                                B_updated_es
        )

        # what happens if the only shared variables between potentials is
        # the single variable in potential

        C = np.random.rand(3)
        C_updated = bp.sum_product.einsum(C, [0], a, [0], [0])
        C_updated_es = C_updated[2]

        np.testing.assert_allclose(
                        bp.sum_product.einsum(A_updated, [0,1,2], C_updated, [0], []),
                        bp.sum_product.einsum(A_updated_es, [1,2], C_updated_es, [], [])
        )
        np.testing.assert_allclose(
                        bp.sum_product.einsum(A_updated, [0,1,2], C_updated, [0], [1,2]),
                        bp.sum_product.einsum(A_updated_es, [1,2], C_updated_es, [], [1,2])
        )

        # evidence shrinking can be incorporated by removing axis corresponding to observed
        # variable

class TestJunctionTreeConstruction(unittest.TestCase):
    # just view the factors with respect to the variables which are the inputs
    # to the factor and proceed with algorithms from there
    # Each factor sharing a variable will have an edge connecting the factors
    # The shared variable will be on that edge
    # The nodes in our factor graph are either factors (sets of variables) or
    # variables (implicit in the fg formalization).

    # The relevant material from Aji and McEliece seems to indicate
    # that we should only need to construct the triangulation graph from the
    # factor arguments (local domains). Each of the variables for the factor
    # graph will be represented in the resulting junction tree.

    def test_can_locate_clique_containing_variable(self):
        tree = [0, (1, [2, ])]
        node_list = [[0,1],[1], [1,2]]
        clique, _vars = bp.get_clique(tree, node_list, 2)
        assert clique == 2

    def test_convert_factor_graph_to_undirected_graph(self):
        factors = [
                    ["A"],
                    ["A", "C"],
                    ["B", "C", "D"],
                    ["A", "D"]
                ]

        edges = bp.factors_to_undirected_graph(factors)
        assert frozenset(("A","C")) in edges
        assert frozenset(("A","D")) in edges
        assert frozenset(("B","C")) in edges
        assert frozenset(("B","D")) in edges
        assert frozenset(("C","D")) in edges


    def test_store_nodes_in_heap(self):
        heap = []
        # push some nodes onto heap (NUM_EDGES_ADDED, CLUSTER_WEIGHT, FACTOR_ID)
        heapq.heappush(heap, (3, 4, 0))
        heapq.heappush(heap, (1, 3, 1))
        heapq.heappush(heap, (5, 3, 2))
        heapq.heappush(heap, (4, 6, 3))

        # check that heappop returns the element with the lowest value in the tuple
        assert heapq.heappop(heap) == (1, 3, 1)

        # add value back to heap
        heapq.heappush(heap, (1, 3, 1))
        # add two new tuples that have the same first value but smaller second
        # value
        heapq.heappush(heap, (1, 2, 4))
        heapq.heappush(heap, (1, 1, 5))

        # ensure that tie is broken by second element
        assert heapq.heappop(heap) == (1, 1, 5)
        # ensure that updated heap returns second smalles element with tie-
        # breaker
        assert heapq.heappop(heap) == (1, 2, 4)


    def test_node_heap_construction(self):
        _vars = {
                    "A": 2,
                    "B": 4,
                    "C": 3,
                    "D": 5
                }

        factors = [
                    ["A"], # weight: 2
                    ["A", "C"], # weight: 6
                    ["B", "C", "D"], # weight: 60
                    ["A", "D"] # weight: 10
                ]
        edges = bp.factors_to_undirected_graph(factors)
        hp, ef = bp.initialize_triangulation_heap(
                                                _vars,
                                                edges
        )

        assert len(hp) == 4
        '''
            Entries:
            [0, 30, "A"] # A has 2 neighbors (all vars connected)
            [0, 60, "B"] # B has 2 neighbors (all vars connected)
            [1, 120, "C"] # C has 3 neighbors (A-B edge added)
            [1, 120, "D"] # C has 3 neighbors (A-B edge added)
        '''

        assert heapq.heappop(hp) == [0, 30, "A"]
        assert heapq.heappop(hp) == [0, 60, "B"]
        assert heapq.heappop(hp) == [1, 120, "C"]
        assert heapq.heappop(hp) == [1, 120, "D"]


    def test_heap_update_after_node_removal(self):
        _vars = {
                    "A": 2,
                    "B": 4,
                    "C": 3,
                    "D": 5
                }

        factors = [
                    ["A"],
                    ["A", "C"],
                    ["B", "C", "D"],
                    ["A", "D"]
                ]

        edges = bp.factors_to_undirected_graph(factors)
        heap, entry_finder = bp.initialize_triangulation_heap(
                                                        _vars,
                                                        edges,
        )

        item, heap, entry_finder, rem_vars = bp.remove_next(
                                                        heap,
                                                        entry_finder,
                                                        list(_vars.keys()),
                                                        _vars,
                                                        edges
        )

        assert item == [0, 30, "A"]


        '''
            Entries:
            [0, 60, "B"] # B has 2 neighbors (all nodes connected)
            [0, 60, "C"] # C has 2 neighbors (all nodes connected)
            [0, 60, "D"] # D has 2 neighbors (all nodes connected)
        '''
        chk_heap = [entry for entry in heapq.nsmallest(len(heap), heap) if entry[2] != ""]
        assert len(chk_heap) == 3
        assert chk_heap[0] == [0, 60, "B"]
        assert chk_heap[1] == [0, 60, "C"]
        assert chk_heap[2] == [0, 60, "D"]

        item, heap, entry_finder, rem_vars = bp.remove_next(
                                                        heap,
                                                        entry_finder,
                                                        rem_vars,
                                                        _vars,
                                                        edges
        )

        assert item == [0, 60, "B"]

        '''
            Entries:
            [0, 15, "C"] # C has 1 neighbor (already connected)
            [0, 15, "D"] # D has 1 neighbor (already connected)
        '''

        chk_heap = [entry for entry in heapq.nsmallest(len(heap), heap) if entry[2] != ""]
        assert len(chk_heap) == 2
        assert chk_heap[0] == [0, 15, "C"]
        assert chk_heap[1] == [0, 15, "D"]

        item, heap, entry_finder, rem_vars = bp.remove_next(
                                                        heap,
                                                        entry_finder,
                                                        rem_vars,
                                                        _vars,
                                                        edges,
        )

        assert item == [0, 15, "C"]

        '''
            Entries:
            [0, 5, "D"] # D has 0 neighbors (no connections possible)
        '''

        chk_heap = [entry for entry in heapq.nsmallest(len(heap), heap) if entry[2] != ""]
        assert len(chk_heap) == 1
        assert chk_heap[0] == [0, 5, "D"]

        item, heap, entry_finder, factors = bp.remove_next(
                                                        heap,
                                                        entry_finder,
                                                        rem_vars,
                                                        _vars,
                                                        edges
        )
        assert item == [0, 5, "D"]


    def test_gibbs_algo_implementation(self):
        '''
            Example 1 as described in:
            http://dspace.mit.edu/bitstream/handle/1721.1/68106/FTL_R_1982_07.pdf
            (pp. 15-17)
        '''
        edges = [
                    ("A","B"),
                    ("A","E"),
                    ("A","E"),
                    ("B","E"),
                    ("B","C"),
                    ("C","E"),
                    ("C","D"),
                    ("D","E")
        ]
        fcs = np.array(
                    [
                        [1,1,0,1,0,0,0,0],
                        [0,1,1,1,0,0,0,0],
                        [0,0,0,1,1,1,0,0],
                        [0,0,0,0,0,1,1,1]
                    ]
        )
        ecs = gibbs_elem_cycles(fcs)
        assert len(ecs) == 10
        ecs_str = ["".join(map(str, [1 if c_i else 0 for c_i in c])) for c in ecs]
        test_str = ["".join(map(str, c))
                        for c in [
                            [1,1,0,1,0,0,0,0], # fcs[0]
                            [1,0,1,0,0,0,0,0], # fcs[0] xor fcs[1]
                            [0,1,1,1,0,0,0,0], # fcs[1]
                            [1,1,0,0,1,1,0,0], # fcs[0] xor fcs[2]
                            [0,1,1,0,1,1,0,0], # fcs[1] xor fcs[2]
                            [0,0,0,1,1,1,0,0], # fcs[2]
                            [1,1,0,0,1,0,1,1], # fcs[0] xor fcs[2] xor fcs[3]
                            [0,1,1,0,1,0,1,1], # fcs[1] xor fcs[2] xor fcs[3]
                            [0,0,0,1,1,0,1,1], # fcs[2] xor fcs[3]
                            [0,0,0,0,0,1,1,1] # fcs[3]
                        ]
                    ]
        assert set(ecs_str) == set(test_str)


    def test_assert_triangulated(self):
        factors = [
                    [0,1],
                    [1,2],
                    [2,3,4],
                    [0,4]
                ]
        tri0 = []
        self.assertRaises(AssertionError, assert_triangulated, factors, tri0)

        tri1 = [(0,2)]
        assert_triangulated(factors, tri1)

    def test_triangulate_factor_graph(self):
        _vars = {
                    "A": 2,
                    "B": 4,
                    "C": 3,
                    "D": 5,
                    "E": 2
                }
        factors = [
                    ["A", "B"],
                    ["B", "C"],
                    ["C", "D", "E"],
                    ["A", "E"]
                ]
        values = [
                    np.random.randn(2, 4),
                    np.random.randn(4, 3),
                    np.random.randn(3, 5, 2),
                    np.random.randn(2, 2),
                ]
        fg = [_vars, factors, values]
        tri, ics, max_cliques, factor_to_maxclique = bp.find_triangulation(fg[1], fg[0])

        # triangulation should consist of 1 edge
        assert len(tri) == 1
        assert len(tri[0]) == 2
        # to triangulate we have a few options
        assert set(tri[0]) in [set(("A","C")),set(("A","D")),set(("B","D")),set(("B","E"))]
        # ensure factors to cliques mapping is correct
        for factor_ix, clique_ix in enumerate(factor_to_maxclique):
            assert np.all([factor_key in max_cliques[clique_ix] for factor_key in factors[factor_ix]])
        assert_triangulated(fg[1], tri)

    def test_triangulate_factor_graph2(self):
        _vars = {
            "A": 2,
            "B": 2,
            "C": 2,
            "D": 2,
            "E": 2,
            "F": 2,
            "G": 2,
            "H": 2,
        }

        factors = [
                    ["A", "B"], #0
                    ["A", "C"], #1
                    ["B", "D"], #2
                    ["C", "E"], #3
                    ["C", "G"], #4
                    ["G", "E", "H"], #5
                    ["D", "E", "F"]  #6
        ]

        tri, ics, max_cliques, _ = bp.find_triangulation(factors, _vars)
        assert_triangulated(factors, tri)


        assert len(max_cliques) == 6


    def test_triangulate_factor_graph3(self):
        '''
            Example taken from here:

            https://courses.cs.washington.edu/courses/cse515/11sp/class7-exactinfalgos.pdf
        '''

        _vars = {
            "C": 2,
            "D": 2,
            "I": 2,
            "G": 2,
            "S": 2,
            "L": 2,
            "J": 2,
            "H": 2,
        }

        factors = [
                    ["C","D"], #0
                    ["D","I","G"], #1
                    ["I","S"], #2
                    ["G","H","J"], #3
                    ["G","L"], #4
                    ["S","L","J"], #5
        ]

        tri, ics, max_cliques, _ = bp.find_triangulation(factors, _vars)
        assert_triangulated(factors, tri)
        cliques = bp.identify_cliques(ics)

        assert len(max_cliques) == 5

        assert ["C","D"] in cliques
        assert ["D","G","I"] in cliques
        assert ["G","I","S"] in cliques
        assert ["G","J","L","S"] in cliques
        assert ["G","H","J"] in cliques


    def test_triangulate_factor_graph_with_duplicate_factors(self):
        tri, ics, max_cliques, factor_to_maxclique = bp.find_triangulation([ ["x", "y"], ["x", "y"] ], {"x":2, "y":3})
        assert None not in factor_to_maxclique

    def test_can_use_integer_keys(self):
        x = 0
        y = 1
        assert type(jt.create_junction_tree([ [x], [x, y] ], {x: 10, y: 20})) == jt.JunctionTree

    def test_identify_cliques(self):
        """
            test_identify_cliques

            Example taken from section 4.4.3 (Huang and Darwiche, 1996)

        """

        # Factors based on moralized graph from example

        factors = [
                    ["A"], #0
                    ["A", "B"], #1
                    ["A", "C"], #2
                    ["B", "D"], #3
                    ["C", "E", "G"], #4
                    ["G", "E", "H"], #5
                    ["D", "E", "F"]  #6
        ]

        tri = [
            ["D","E"], #0
            ["D"], #1
            ["E"], #2
            [], #3
            [], #4
            [], #5
            [] #6
        ]

        cliques = bp.identify_cliques([f+t for f,t in zip(factors,tri)])

        assert len(cliques) == 6

        assert ["E","G","H"] in cliques
        assert ["C","E","G"] in cliques
        assert ["D","E","F"] in cliques
        assert ["A","C","E"] in cliques
        assert ["A","B","D"] in cliques
        assert ["A","D","E"] in cliques

    def test_join_trees_with_single_cliques(self):
        tree1 = [0,]
        sepset = [2,]
        tree2 = [1,]

        output = bp.merge_trees(
                            tree1,
                            tree1[0],
                            tree2,
                            tree2[0],
                            sepset[0]
        )

        merged_tree = [
                0,
                (
                    2,
                    [
                        1,
                    ]
                )
            ]

        assert merged_tree == output
        assert_junction_tree_equal(output, merged_tree)

    def test_join_tree_with_single_clique_to_multiclique_tree(self):
        tree1 = [0,]
        sepset = [3,]
        tree2 = [4, (5, [6, ])]

        output = bp.merge_trees(
                            tree1,
                            tree1[0],
                            tree2,
                            tree2[0],
                            sepset[0]
        )
        merged_tree = [
                    0,
                    (
                        3,
                        [
                            4,
                            (
                                5,
                                [
                                    6,
                                ]
                            )
                        ]
                    )
        ]

        assert_junction_tree_equal(output, merged_tree)


    def test_join_tree_with_multiple_cliques_to_tree_with_multiple_cliques(self):
        tree1 = [0, (1, [2, ])]
        sepset = [3, ]
        tree2 = [4, (5, [6, ])]

        output = bp.merge_trees(
                            tree1,
                            tree1[0],
                            tree2,
                            tree2[0],
                            sepset[0]
        )
        merged_tree = [
                    0,
                    (
                        1,
                        [
                            2,
                        ]
                    ),
                    (
                        3,
                        [
                            4,
                            (
                                5,
                                [
                                    6,
                                ]
                            )
                        ]
                    )
        ]

        assert_junction_tree_equal(output, merged_tree)

    def test_change_root(self):
        tree1 = [
                    4,
                    (
                        5,
                        [
                            0,
                            (
                                1,
                                [
                                    2,
                                ]
                            )
                        ]
                    )
                ]

        assert bp.change_root(tree1, 3) == []

        output = bp.change_root(tree1, 4)

        assert output == tree1

        output = bp.change_root(copy.deepcopy(tree1), 0)

        tree2 = [
                    0,
                    (
                        1,
                        [
                            2,
                        ]
                    ),
                    (
                        5,
                        [
                            4,
                        ]
                    )
                ]

        assert output == tree2
        assert_junction_tree_equal(tree1, output)


        output = bp.change_root(copy.deepcopy(tree1), 2)

        tree3 = [
                    2,
                    (
                        1,
                        [
                            0,
                            (
                                5,
                                [
                                    4,
                                ]
                            )
                        ]
                    )
                ]


        assert output == tree3
        assert_junction_tree_equal(tree1, output)

        tree4 = [
                    4,
                    (
                        5,
                        [
                            0,
                            (
                                1,
                                [
                                    2,
                                ]
                            ),
                            (
                                3,
                                [
                                    6,
                                ]
                            )
                        ]
                    ),
                    (
                        7,
                        [
                            8,
                        ]
                    )
                ]


        output = bp.change_root(tree4, 2)



        tree5 = [
                    2,
                    (
                        1,
                        [
                            0,
                            (
                                5,
                                [
                                    4,
                                    (
                                        7,
                                        [
                                            8,
                                        ]
                                    )
                                ]
                            ),
                            (
                                3,
                                [
                                    6,
                                ]
                            )
                        ]
                    )
                ]


        assert_junction_tree_equal(output,tree5)
        assert_junction_tree_equal(tree4, output)

    def test_join_trees_with_multiple_cliques_with_first_nested(self):
        tree1 = [4, (5, [0, (1, [2, ])])]
        sepset = [3,]
        tree2 = [8, (9, [10, ])]

        output = bp.merge_trees(
                            tree1,
                            0,
                            tree2,
                            8,
                            sepset[0]
        )
        merged_tree = [
                    4,
                    (
                        5,
                        [
                            0,
                            (
                                1,
                                [
                                    2,
                                ]
                            ),
                            (
                                3,
                                [
                                    8,
                                    (
                                        9,
                                        [
                                            10,
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]

        assert_junction_tree_equal(output, merged_tree)


    def test_join_trees_with_multiple_cliques_with_second_nested(self):
        tree1 = [0, (1, [2, ])]
        sepset = [3, ]
        tree2 = [6,  (7,  [8,  (9,  [10, ])])]

        output = bp.merge_trees(
                            tree1,
                            0,
                            tree2,
                            8,
                            sepset[0]
        )
        merged_tree = [
                    0,
                    (
                        1,
                        [
                            2,
                        ]
                    ),
                    (
                        3,
                        [
                            8,
                            (
                                9,
                                [
                                    10,
                                ]
                            ),
                            (
                                7,
                                [
                                    6,
                                ]
                            )
                        ]
                    )
        ]

        assert_junction_tree_equal(output, merged_tree)

    def test_join_trees_with_multiple_cliques_with_both_nested(self):
        tree1 = [4, (5, [0, (1,  [2,])])]
        sepset = [3, ]
        tree2 = [6, (7, [8, (9, [10, ])])]

        output = bp.merge_trees(
                            tree1,
                            0,
                            tree2,
                            8,
                            sepset[0]
        )

        merged_tree = [
                    4,
                    (
                        5,
                        [
                            0,
                            (
                                1,
                                [
                                    2,
                                ]
                            ),
                            (
                                3,
                                [
                                    8,
                                    (
                                        9,
                                        [
                                            10,
                                        ]
                                    ),
                                    (
                                        7,
                                        [
                                            6,
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]


        assert_junction_tree_equal(output, merged_tree)



class TestJunctionTreeInference(unittest.TestCase):
    def setUp(self):
        self.tree = [
                        0,
                        (
                            6,
                            [
                                1,
                            ]
                        ),
                        (
                            7,
                            [
                                2,
                            ]
                        ),
                        (
                            8,
                            [
                                3,
                                (
                                    9,
                                    [
                                        4,
                                        (
                                            10,
                                            [
                                                5,
                                            ]
                                        )
                                    ]
                                )
                            ]

                        )
                    ]

        self.node_list = [
                            ["A","D","E"],
                            ["A","B","D"],
                            ["D","E","F"],
                            ["A","C","E"],
                            ["C","E","G"],
                            ["E","G","H"],
                            ["A","D"],
                            ["D","E"],
                            ["A","E"],
                            ["C","E"],
                            ["E","G"],
        ]

        self.key_sizes = {
                            "A": 2,
                            "B": 2,
                            "C": 2,
                            "D": 2,
                            "E": 2,
                            "F": 2,
                            "G": 2,
                            "H": 2
                        }

        self.factors = [
                    ["A"],
                    ["A", "B"],
                    ["A", "C"],
                    ["B", "D"],
                    ["C", "E"],
                    ["C", "G"],
                    ["D", "E", "F"],
                    ["E", "G", "H"],

        ]

        self.values = [
                        np.array([0.5,0.5]),
                        np.array(
                                    [
                                        [0.6,0.4],
                                        [0.5,0.5]
                                    ]
                                ),
                        np.array(
                                    [
                                        [0.8,0.2],
                                        [0.3,0.7]
                                    ]
                                ),
                        np.array(
                                    [
                                        [0.5,0.5],
                                        [0.1,0.9]
                                    ]
                                ),
                        np.array(
                                    [
                                        [0.4,0.6],
                                        [0.7,0.3]
                                    ]
                                ),
                        np.array(
                                    [
                                        [0.9,0.1],
                                        [0.8,0.2]
                                    ]
                                ),
                        np.array(
                                    [
                                        [
                                            [0.01,0.99],
                                            [0.99,0.01]
                                        ],
                                        [
                                            [0.99,0.01],
                                            [0.99,0.01]
                                        ]
                                    ]
                                ),
                        np.array(
                                    [
                                        [
                                            [0.05,0.95],
                                            [0.05,0.95]
                                        ],
                                        [
                                            [0.05,0.95],
                                            [0.95,0.05]
                                        ]
                                    ]
                                )
                ]


    def test_transformation(self):
        tree = jt.create_junction_tree(self.factors, self.key_sizes)
        prop_values = tree.propagate(self.values)

        # check that marginal values are correct
        np.testing.assert_allclose(
                                prop_values[0],
                                np.array([0.500,0.500])
        )
        np.testing.assert_allclose(
                                np.sum(prop_values[1], axis=0),
                                np.array([0.550,0.450])
        )
        np.testing.assert_allclose(
                                np.sum(prop_values[2], axis=0),
                                np.array([0.550,0.450])
        )
        np.testing.assert_allclose(
                                np.sum(prop_values[3], axis=0),
                                np.array([0.320,0.680])
        )
        np.testing.assert_allclose(
                                np.sum(prop_values[4], axis=0),
                                np.array([0.535,0.465])
        )
        np.testing.assert_allclose(
                                np.sum(prop_values[5], axis=0),
                                np.array([0.855,0.145])
        )

        np.testing.assert_allclose(
                                bp.sum_product.einsum(
                                    tree.propagate(self.values)[6],
                                    [0, 1, 2],
                                    [2]
                                ),
                                np.array([0.824,0.176]),
                                atol=0.01
        )
        np.testing.assert_allclose(
                                bp.sum_product.einsum(
                                    tree.propagate(self.values)[7],
                                    [0, 1, 2],
                                    [2]
                                ),
                                np.array([ 0.104,  0.896]),
                                atol=0.01
        )

    def test_initialize_potentials(self):
        j_tree = jt.JunctionTree(
                    self.tree,
                    self.node_list[6:],
                    jt.CliqueGraph(
                        maxcliques=self.node_list[:6],
                        factor_to_maxclique=[0, 1, 3, 1, 3, 4, 2, 5],
                        factor_graph=jt.FactorGraph(
                            factors=self.factors,
                            sizes=self.key_sizes
                        )
                    )
        )

        init_phi  = j_tree.clique_tree.evaluate(self.values)

        assert_potentials_equal(
                        init_phi[3], # cluster ACE
                        np.array(
                            [
                                [
                                    [0.32,0.48],
                                    [0.14,0.06]
                                ],
                                [
                                    [0.12,0.18],
                                    [0.49,0.21]
                                ]
                            ]
                        )
        )

    def test_global_propagation(self):
        tree = jt.create_junction_tree(self.factors, self.key_sizes)

        prop_values = tree.propagate(self.values)
        # P(A)
        assert_potentials_equal(
                                prop_values[0],
                                np.array([0.500,0.500])
        )

        # P(D)
        assert_potentials_equal(
                                np.sum(prop_values[3], axis=0),
                                np.array([0.32,0.68])
        )

    def test_global_propagation_with_observations(self):
        #http://mensxmachina.org/files/software/demos/jtreedemo.html
        key_sizes = {
                        "cloudy": 2,
                        "sprinkler": 2,
                        "rain": 2,
                        "wet_grass": 2
                    }

        factors = [
                    ["cloudy"],
                    ["cloudy", "sprinkler"],
                    ["cloudy", "rain"],
                    ["rain", "sprinkler", "wet_grass"]
        ]

        values = [
                    np.array([0.5,0.5]),
                    np.array(
                                [
                                    [0.5,0.5],
                                    [0.9,0.1]
                                ]
                            ),
                    np.array(
                                [
                                    [0.8,0.2],
                                    [0.2,0.8]
                                ]
                            ),
                    np.array(
                                [
                                    [
                                        [1,0],
                                        [0.1,0.9]
                                    ],
                                    [
                                        [0.1,0.9],
                                        [0.01,0.99]
                                    ]
                                ]
                    )
        ]

        tree = jt.create_junction_tree(factors, key_sizes)

        # grass is wet
        tree.clique_tree.factor_graph.sizes["wet_grass"] = 1
        cond_values = copy.deepcopy(values)
        cond_values[3] = cond_values[3][:,:,1:]

        prop_values = tree.propagate(cond_values)
        marginal = np.sum(prop_values[1], axis=0)

        np.testing.assert_allclose(
                                marginal/np.sum(marginal),
                                np.array([0.57024,0.42976]),
                                atol=0.01
        )

        # grass is wet and it is raining
        tree.clique_tree.factor_graph.sizes["rain"] = 1
        cond_values[3] = cond_values[3][1:,:,:]
        cond_values[2] = cond_values[2][:,1:]
        prop_values = tree.propagate(cond_values)

        marginal = np.sum(prop_values[1], axis=0)

        np.testing.assert_allclose(
                                marginal/np.sum(marginal),
                                np.array([0.8055,0.1945]),
                                atol=0.01
        )





    def test_inference(self):
        #http://pages.cs.wisc.edu/~dpage/cs731/lecture5.ppt
        key_sizes = {
                        "A": 2,
                        "B": 2,
                        "C": 2,
                        "D": 2,
                        "E": 2,
                        "F": 2
                    }
        factors = [
                    ["A"],
                    ["B","A"],
                    ["C","A"],
                    ["B","D"],
                    ["C","E"],
                    ["D","E","F"]
        ]
        values = [
                    np.array([0.9,0.1]),
                    np.array(
                                [
                                    [0.1,0.9],
                                    [0.9,0.1]
                                ]
                    ),
                    np.array(
                                [
                                    [0.8,0.3],
                                    [0.2,0.7]
                                ]
                    ),
                    np.array(
                                [
                                    [0.3,0.7],
                                    [0.6,0.4]
                                ]
                    ),
                    np.array(
                                [
                                    [0.6,0.4],
                                    [0.5,0.5]
                                ]
                    ),
                    np.array(
                                [
                                    [
                                        [0.2,0.8],
                                        [0.6,0.4]
                                    ],
                                    [
                                        [0.5,0.5],
                                        [0.9,0.1]
                                    ]
                                ]
                    )
        ]

        struct = [
                    0,
                    (
                        4,
                        [
                            1,
                        ]
                    ),
                    (
                        5,
                        [
                            2,
                            (
                                6,
                                [
                                    3,
                                ]
                            )
                        ]
                    )
                ]

        node_list = [
                        ["C","D","E"],
                        ["D","E","F"],
                        ["B","C","D"],
                        ["A","B","C"],
                        ["D","E"],
                        ["C","D"],
                        ["B","C"],
        ]

        tree = jt.create_junction_tree(factors, key_sizes)

        prop_values = tree.propagate(values)

        # P(C)
        np.testing.assert_allclose(
                            np.sum(prop_values[2], axis=1),
                            np.array([0.75,0.25])
        )
        # P(A)
        np.testing.assert_allclose(
                            np.sum(prop_values[1], axis=0),
                            np.array([0.9,0.1])
        )

        # P(B)
        np.testing.assert_allclose(
                            np.sum(prop_values[1], axis=1),
                            np.array([0.18,0.82])
        )

        # P(D)
        np.testing.assert_allclose(
                            np.sum(prop_values[3], axis=0),
                            np.array([0.546,0.454])
        )

        # P(E)
        np.testing.assert_allclose(
                            np.sum(prop_values[4], axis=0),
                            np.array([0.575,0.425])
        )

        # P(F)
        np.testing.assert_allclose(
                            bp.sum_product.einsum(
                                prop_values[5],
                                [0,1,2],
                                [2]
                            ),
                            np.array([0.507,0.493]),
                            atol=0.001
        )


class TestJTTraversal(unittest.TestCase):
    def setUp(self):
        self.tree = [
                        0,
                        (
                            1,
                            [
                                2,
                            ]
                        ),
                        (
                            3,
                            [
                                4,
                                (
                                    5,
                                    [
                                        6,
                                    ]
                                )
                            ]
                        )
            ]

    def test_bf_traverse(self):
        assert list(bp.bf_traverse(self.tree)) == [
                                                    0,
                                                    1,
                                                    3,
                                                    2,
                                                    4,
                                                    5,
                                                    6,
                                                ]

    def test_df_traverse(self):
        assert list(bp.df_traverse(self.tree)) == [
                                                    0,
                                                    1,
                                                    2,
                                                    3,
                                                    4,
                                                    5,
                                                    6,
                                                ]


    def test_get_clique_keys(self):
        node_list = [
                        [0, 2, 4],
                        [0, 2],
                        [0, 1, 2],
                        [4],
                        [3, 4],
                        [3],
                        [1, 2, 3]

        ]
        assert bp.get_clique_keys(node_list, 0) == [0, 2, 4]
        assert bp.get_clique_keys(node_list, 1) == [0, 2]
        assert bp.get_clique_keys(node_list, 2) == [0, 1, 2]
        assert bp.get_clique_keys(node_list, 3) == [4]
        assert bp.get_clique_keys(node_list, 4) == [3, 4]
        assert bp.get_clique_keys(node_list, 5) == [3]
        assert bp.get_clique_keys(node_list, 6) == [1, 2, 3]
        assert bp.get_clique_keys(node_list, 7) == None

    def test_generate_potential_pairs(self):
        tree = [
                0,
                (
                    1,
                    [
                        2,
                    ]
                ),
                (
                    3,
                    [
                        4,
                    ]
                ),
                (
                    5,
                    [
                        6,
                        (
                            7,
                            [
                                8,
                                (
                                    9,
                                    [
                                        10,
                                    ]
                                )
                            ]
                        )
                    ]

                )
            ]

        assert bp.generate_potential_pairs(tree) == [
                                                        (0, 1),
                                                        (0, 3),
                                                        (0, 5),
                                                        (1, 2),
                                                        (3, 4),
                                                        (5, 6),
                                                        (6, 7),
                                                        (7, 8),
                                                        (8, 9),
                                                        (9, 10)
        ]

class TestMisc(unittest.TestCase):

    def test_index_error(self):
        key_sizes = {'a': 10, 'b': 6, 'c': 3, 'd': 10, 'e': 27, 'f': 2, 'g': 3, 'h': 3, 'i': 3, 'j': 3, 'k': 5, 'l': 6}

        factors = [
            ['f', 'k', 'a'],
            ['f', 'g', 'b'],
            ['c'],
            ['f', 'k', 'd'],
            ['g', 'e'],
            ['c', 'g', 'f'],
            ['c', 'g'],
            ['f', 'h'],
            ['c', 'g', 'i'],
            ['l', 'j'],
            ['g', 'j', 'k'],
            ['g', 'l']
        ]

        values = [

        np.random.random((key_sizes['f'], key_sizes['k'], key_sizes['a'])),
        np.random.random((key_sizes['f'], key_sizes['g'], key_sizes['b'])),
        np.random.random((key_sizes['c'])),
        np.random.random((key_sizes['f'], key_sizes['k'], key_sizes['d'])),
        np.random.random((key_sizes['g'], key_sizes['e'])),
        np.random.random((key_sizes['c'], key_sizes['g'], key_sizes['f'])),
        np.random.random((key_sizes['c'], key_sizes['g'])),
        np.random.random((key_sizes['f'], key_sizes['h'])),
        np.random.random((key_sizes['c'], key_sizes['g'], key_sizes['i'])),
        np.random.random((key_sizes['l'], key_sizes['j'])),
        np.random.random((key_sizes['g'], key_sizes['j'], key_sizes['k'])),
        np.random.random((key_sizes['g'], key_sizes['l']))

        ]

        tree = jt.create_junction_tree(factors, key_sizes)
        prop_values = tree.propagate(values)
