import numpy as np

import bp
import unittest
import networkx as nx
import itertools
import heapq
import copy
from junction_tree import JunctionTree
from sum_product import SumProduct
import math


# Tests here using pytest


def assert_junction_tree_equal(t1, t2):
    """Test equality of two junction trees

    Both trees contain same edges and node indices

    """

    pairs1 = set([tuple(sorted(p)) for p in bp.generate_potential_pairs(t1)])
    pairs2 = set([tuple(sorted(p)) for p in bp.generate_potential_pairs(t2)])
    assert pairs1 == pairs2

def assert_junction_tree_equal2(t1, t2):
    """Test equality of two junction trees

    Both trees contain same edges and cliques have same keys

    """


    def __build_dict(tree):
        # dict is: clique_keys -> set of each tuple of neighbor_keys
        d = {}

        stack = [tree]
        d[frozenset((tree[1]))] = set()
        while stack:
            tree = stack.pop()
            for child in reversed(tree[2:]):
                d[frozenset((tree[1]))].add(frozenset((child[1])))
                # child clique entry initialized
                d[frozenset((child[2][1]))] = set([frozenset((child[1]))])
                # separator entered in dictionary
                d[frozenset((child[1]))] = set([frozenset((tree[1])),frozenset((child[2][1]))])
                stack.append(child[2])

        return d

    d1 = __build_dict(t1)
    d2 = __build_dict(t2)

    assert d1 == d2


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


def build_graph(factors):
    G=nx.Graph()

    for factor in factors:
        factor_set = set(factor)
        for v1 in factor:
            for v2 in factor_set - set([v1]):
                G.add_edge(v1,v2)

    return G

def find_cycles(factors, num):
    G = build_graph(factors)

    cb = nx.cycle_basis(G)
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

def get_arrays_and_keys(tree, potentials):
    """Get all arrays and their keys as a flat list

    Output: [array1, keys1, ..., arrayN, keysN]

    """
    return list([potentials[tree[0]],tree[1]]) + sum(
        [
            get_arrays_and_keys(child_tree, potentials)
            for child_tree in tree[2:]
        ],
        []
    )

def brute_force_sum_product(tree, potentials):
    """Compute brute force sum-product with einsum """

    # Function to compute the sum-product with brute force einsum
    arrays_keys = get_arrays_and_keys(tree, potentials)
    f = lambda output_keys: bp.sum_product.einsum(*(arrays_keys + [output_keys]))

    def __run(tree, p, f, res=[]):
        res.append(f(tree[1]))
        for child_tree in tree[2:]:
            __run(child_tree, p, f, res)
        return res

    return __run(tree, potentials, f)

def assert_sum_product(junction_tree, potentials):
    """ Test hugin vs brute force sum-product """

    tree = junction_tree.get_struct()
    assert_potentials_equal(
        brute_force_sum_product(tree, potentials),
        bp.hugin(
                tree,
                junction_tree.get_label_order(),
                potentials,
                bp.sum_product,
        )
    )

def assert_junction_tree_consistent(tree, potentials):
    '''
        For each clique/sepset pair in tree, check consistency

        Function checks that junction tree is consistent with respect to the
        provided potentials
    '''

    assert np.all(
                    [
                        potentials_consistent(
                                            potentials[c_ix1],
                                            c_vars1,
                                            potentials[c_ix2],
                                            c_vars2
                        )
                        for c_ix1,
                            c_vars1,
                            c_ix2,
                            c_vars2 in bp.generate_potential_pairs2(tree.get_struct())
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
        """
            Example taken from here: https://www.cs.ru.nl/~peterl/BN/examplesproofs.pdf
            Example will be processed under the assumption that potentials have been
            properly initialized outside of this test

            Variables: V1, V2, V3
            \pi_{V1} = [V2] # parents of V1
            \pi_{V2} = [] # parents of V2
            \pi_{V3} = [V2] # parents of V3
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
                0, ["V1","V2"],
                (
                    1, ["V2"],
                    [
                        2, ["V2", "V3"]
                    ]
                )

            ]

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
                            {"V1": 0, "V2": 1, "V3": 2},
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
                0, ["V1","V2"],
                (
                    1, ["V2"],
                    [
                        2, ["V2", "V3"]
                    ]
                )
            ]
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
                            {"V1": 0, "V2": 1, "V3": 2},
                            phi,
                            [0]*len(phi),
                            bp.sum_product
        )

        phiN2 = bp.distribute(
                                jt,
                                {"V1": 0, "V2": 1, "V3": 2},
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
                JunctionTree(
                        {},
                        [
                            0, []
                        ]
                ),
                [
                    np.random.randn(),
                ]
        )

    def test_one_matrix_node(self):
        assert_sum_product(
                JunctionTree(
                        {
                            3:2,
                            5:3
                        },
                        [
                            0, [3, 5]
                        ],
                ),
                [
                    np.random.randn(2, 3),
                ]
        )

    def test_one_child_node_with_all_variables_shared(self):
        assert_sum_product(
            JunctionTree(
                            {
                                3:2,
                                5:3
                            },
                            [
                                0, [3, 5],
                                (
                                    1, [5, 3],
                                    [
                                        2, [5, 3],
                                    ]
                                )
                            ],
            ),
            [
                np.random.randn(2, 3),
                np.ones((3, 2)),
                np.random.randn(3, 2),
            ]
        )

    def test_one_child_node_with_one_common_variable(self):
        assert_sum_product(
                    JunctionTree(
                            {
                                3:2,
                                5:3,
                                9:4
                            },
                            [
                                0, [3, 5],
                                (
                                    1, [5],
                                    [
                                        2, [5, 9]
                                    ]
                                )
                            ]
                    ),
                    [
                        np.random.randn(2, 3),
                        np.ones((3,)),
                        np.random.randn(3, 4),
                    ]
        )

    def test_one_child_node_with_no_common_variable(self):
        assert_sum_product(
                    JunctionTree(
                        {
                            3:2,
                            9:3
                        },
                        [
                            0, [3],
                            (
                                1, [],
                                [
                                    2, [9]
                                ]
                            )
                        ]
                    ),
                    [
                        np.random.randn(2),
                        np.ones(()),
                        np.random.randn(3),
                    ]
        )

    def test_one_grand_child_node_with_no_variable_shared_with_grand_parent(self):
        assert_sum_product(
                    JunctionTree(
                            {
                                1:5,
                                3:2,
                                5:3,
                                9:4,
                            },
                            [
                                0, [3, 5],
                                (
                                    1, [5],
                                    [
                                        2, [5, 9],
                                        (
                                            3, [9],
                                            [
                                                4, [9, 1]
                                            ]
                                        )
                                    ]
                                )
                            ]
                    ),
                    [
                        np.random.randn(2, 3),
                        np.ones((3,)),
                        np.random.randn(3, 4),
                        np.ones((4,)),
                        np.random.randn(4, 5),
                    ]
        )

    def test_one_grand_child_node_with_variable_shared_with_grand_parent(self):
        assert_sum_product(
                    JunctionTree(
                            {
                                1:6,
                                3:2,
                                5:3,
                                9:4
                            },
                            [
                                0, [3, 5],
                                (
                                    1, [5],
                                    [
                                        2, [5, 9],
                                        (
                                            3, [5],
                                            [
                                                4, [1, 5]
                                            ]
                                        )
                                    ]
                                )
                            ]
                    ),
                    [
                        np.random.randn(2, 3),
                        np.ones((3,)),
                        np.random.randn(3, 4),
                        np.ones((3,)),
                        np.random.randn(6, 3),
                    ]
        )

    def test_two_children_with_no_variable_shared(self):
        assert_sum_product(
                    JunctionTree(
                            {
                                3:2,
                                5:3,
                                9:4,
                                1:5
                            },
                            [
                                0, [3, 5],
                                (
                                    1, [5],
                                    [
                                        2, [5, 9],
                                    ]
                                ),
                                (
                                    3, [3],
                                    [
                                        4, [3, 1]
                                    ]
                                )
                            ]
                    ),
                    [
                        np.random.randn(2, 3),
                        np.ones((3,)),
                        np.random.randn(3, 4),
                        np.ones((2,)),
                        np.random.randn(2, 5),
                    ]
        )

    def test_two_child_with_shared_variable(self):
        assert_sum_product(
                    JunctionTree(
                            {
                                3:2,
                                5:3,
                                9:4,
                            },
                            [
                                0, [3, 5],
                                (
                                    1, [5],
                                    [
                                        2, [5, 9],
                                    ]
                                ),
                                (
                                    3, [5],
                                    [
                                        4, [5]
                                    ]
                                )
                            ]
                    ),
                    [
                        np.random.randn(2, 3),
                        np.ones((3,)),
                        np.random.randn(3, 4),
                        np.ones((3,)),
                        np.random.randn(3),

                    ]
        )

    def test_two_children_with_3D_tensors(self):
        assert_sum_product(
                    JunctionTree(
                            {
                                1:6,
                                3:2,
                                5:3,
                                7:4,
                                9:5
                            },
                            [
                                0, [3, 5, 7],
                                (
                                    1, [5, 7],
                                    [
                                        2, [5, 7, 9],
                                    ]
                                ),
                                (
                                    3, [5],
                                    [
                                        4, [5, 1]
                                    ]
                                )
                            ]
                    ),
                    [
                        np.random.randn(2, 3, 4),
                        np.ones((3, 4)),
                        np.random.randn(3, 4, 5),
                        np.ones((3,)),
                        np.random.randn(3, 6),
                    ]
        )

    def test_can_observe_evidence_from_one_trial(self):
        jt = JunctionTree(
                            {
                                0: 4,
                                1: 8,
                                2: 5,
                                3: 3,
                                4: 6
                            },
                            [
                                0, [0,2,4],
                                (
                                    1, [0,2],
                                    [
                                        2, [0,1,2]
                                    ]
                                ),
                                (
                                    3, [4],
                                    [
                                        4, [3,4],
                                        (
                                            5, [3],
                                            [
                                                6, [1,2,3]
                                            ]
                                        )
                                    ]
                                )
                            ]
                    )

        # define arbitrary join tree potentials
        phi = [
                np.random.randn(4,5,6),
                np.random.randn(4,5),
                np.random.randn(4,8,5),
                np.random.randn(6),
                np.random.randn(3, 6),
                np.random.randn(3),
                np.random.randn(8,5,3)
        ]

        data = {0: 1, 2: 3, 4: 0}

        likelihood, phi0, shrink_mapping = jt.observe(phi, data)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))

        assert len(shrink_mapping) == len(phi0)

        # check dimensions and keys potentials after observing evidence
        assert shrink_mapping[0] == ((1,3,0),[])
        assert type(phi0[0][shrink_mapping[0][0]]) == np.float64
        assert shrink_mapping[1] == ((1,3),[])
        assert type(phi0[1][shrink_mapping[1][0]]) == np.float64
        assert shrink_mapping[2] == ((1,slice(None),3),[1])
        assert phi0[2][shrink_mapping[2][0]].shape == (8,)
        assert shrink_mapping[3] == ((0,),[])
        assert type(phi0[3][shrink_mapping[3][0]]) == np.float64
        assert shrink_mapping[4] == ((slice(None),0),[3])
        assert phi0[4][shrink_mapping[4][0]].shape == (3,)
        assert shrink_mapping[5] == ((slice(None),),[3])
        assert phi0[5][shrink_mapping[5][0]].shape == (3,)
        assert shrink_mapping[6] == ((slice(None),3,slice(None)),[1,3])
        assert phi0[6][shrink_mapping[6][0]].shape == (8,3)

        phi1 = bp.hugin(
                    jt.get_struct(),
                    jt.get_label_order(),
                    phi0,
                    bp.sum_product,
        )
        assert_junction_tree_consistent(jt, phi1)

        # test that a potential containing observed variable is altered properly after observing data
        # (Eventually this will be based on familial content of clique or a more optimal clique but
        # for now just find one but for now just check the first clique containing the variable)

        for var, val in data.items():
            clique, _vars = bp.get_clique(jt.get_struct(), var)
            pot = phi1[clique]
            assert pot.shape == phi[clique].shape
            var_ix = _vars.index(var)
            # check that potential properly updated
            mask = [val == dim for dim in range(pot.shape[var_ix])]
            # values along var axis not equal to val will have a 0 value
            assert np.all(np.compress(np.invert(mask), pot, axis=var_ix)) == 0
            # ensure that var axis equal to val is equivalent in both original and new potentials
            np.testing.assert_array_equal(
                                            np.compress(mask, pot, axis=var_ix),
                                            np.compress(mask, phi[clique], axis=var_ix)
                                        )

        # test that no change made to potential values for unobserved variables
        for var in jt.get_key_sizes():
            if var not in data.keys():
                # we have not observed a value for this var
                for clique_ix, _vars in bp.get_cliques(jt.get_struct(), var):
                    pot = phi1[clique_ix]
                    # get the vals for the observed axes and set unobserved to -1
                    test_arr = np.array([data[v] if v in data else -1 for v in _vars])
                    # retain indices in array which all observed axes set to observed val
                    # if none observed this should just evaluate to all indices of the potential
                    test_indices = np.array([a_ix for a_ix in np.ndindex(*pot.shape) if np.sum(test_arr == np.array(a_ix)) == (test_arr > -1).sum()]).transpose()
                    flat_indices = np.ravel_multi_index(test_indices, pot.shape)
                    # elements at these indices should not have changed by observations
                    np.testing.assert_array_equal(
                                                    np.take(pot, flat_indices),
                                                    np.take(phi[clique_ix], flat_indices)
                                                )


    def test_can_observe_dynamic_evidence_using_global_update_single_variable(self):
        jt = JunctionTree(
                            {
                                0: 4,
                                1: 8,
                                2: 5,
                                3: 3,
                                4: 6
                            },
                            [
                                0, [0,2,4],
                                (
                                    1, [0,2],
                                    [
                                        2, [0,1,2]
                                    ]
                                ),
                                (
                                    3, [4],
                                    [
                                        4, [3,4],
                                        (
                                            5, [3],
                                            [
                                                6, [1,2,3]
                                            ]
                                        )
                                    ]
                                )
                            ]
                    )

        # define arbitrary join tree potentials
        phi = [
                np.random.randn(4, 5, 6),
                np.random.randn(4,5),
                np.random.randn(4,8,5),
                np.random.randn(6),
                np.random.randn(3, 6),
                np.random.randn(3),
                np.random.randn(8,5,3)
        ]

        data0 = {0: 1, 2: 3, 4: 0}

        likelihood, phi0, shrink_mapping = jt.observe(phi, data0)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))

        assert len(shrink_mapping) == len(phi0)

        # check dimensions and keys potentials after observing evidence
        assert shrink_mapping[0] == ((1,3,0),[])
        assert type(phi0[0][shrink_mapping[0][0]]) == np.float64
        assert shrink_mapping[1] == ((1,3),[])
        assert type(phi0[1][shrink_mapping[1][0]]) == np.float64
        assert shrink_mapping[2] == ((1,slice(None),3),[1])
        assert phi0[2][shrink_mapping[2][0]].shape == (8,)
        assert shrink_mapping[3] == ((0,),[])
        assert type(phi0[3][shrink_mapping[3][0]]) == np.float64
        assert shrink_mapping[4] == ((slice(None),0),[3])
        assert phi0[4][shrink_mapping[4][0]].shape == (3,)
        assert shrink_mapping[5] == ((slice(None),),[3])
        assert phi0[5][shrink_mapping[5][0]].shape == (3,)
        assert shrink_mapping[6] == ((slice(None),3,slice(None)),[1,3])
        assert phi0[6][shrink_mapping[6][0]].shape == (8,3)

        phi1 = bp.hugin(
                    jt.get_struct(),
                    jt.get_label_order(),
                    phi0,
                    bp.sum_product
        )
        assert_junction_tree_consistent(jt, phi1)

        data = {0: 1, 1: 2, 2: 3, 4: 0}

        likelihood, phi2, shrink_mapping = jt.observe(phi1, data)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([0,0,1,0,0,0,0,0]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))

        assert len(shrink_mapping) == len(phi2)

        # check dimensions and keys potentials after observing evidence
        assert shrink_mapping[0] == ((1,3,0),[])
        assert type(phi2[0][shrink_mapping[0][0]]) == np.float64
        assert shrink_mapping[1] == ((1,3),[])
        assert type(phi2[1][shrink_mapping[1][0]]) == np.float64
        assert shrink_mapping[2] == ((1,2,3),[])
        assert type(phi2[2][shrink_mapping[2][0]]) == np.float64
        assert shrink_mapping[3] == ((0,),[])
        assert type(phi2[3][shrink_mapping[3][0]]) == np.float64
        assert shrink_mapping[4] == ((slice(None),0),[3])
        assert phi2[4][shrink_mapping[4][0]].shape == (3,)
        assert shrink_mapping[5] == ((slice(None),),[3])
        assert phi2[5][shrink_mapping[5][0]].shape == (3,)
        assert shrink_mapping[6] == ((2,3,slice(None)),[3])
        assert phi2[6][shrink_mapping[6][0]].shape == (3,)


        phi3 = bp.hugin(
                    jt.get_struct(),
                    jt.get_label_order(),
                    phi2,
                    bp.sum_product
        )
        assert_junction_tree_consistent(jt, phi3)

        # test that a potential containing observed variable is altered properly after observing data
        # (Eventually this will be based on familial content of clique or a more optimal clique but
        # for now just find one but for now just check the first clique containing the variable)

        for var, val in data.items():
            clique, _vars = bp.get_clique(jt.get_struct(), var)
            pot = phi3[clique]
            assert pot.shape == phi[clique].shape
            var_ix = _vars.index(var)
            # check that potential properly updated
            mask = [val == dim for dim in range(pot.shape[var_ix])]
            # values along var axis not equal to val will have a 0 value
            assert np.all(np.compress(np.invert(mask), pot, axis=var_ix)) == 0
            # ensure that var axis equal to val is equivalent in both original and new potentials
            np.testing.assert_array_equal(
                                            np.compress(mask, pot, axis=var_ix),
                                            np.compress(mask, phi[clique], axis=var_ix)
                                        )

        # test that no change made to potential values for unobserved variables
        for var in jt.get_key_sizes():
            if var not in data.keys():
                # we have not observed a value for this var
                for clique_ix, _vars in bp.get_cliques(jt.get_struct(), var):
                    pot = phi3[clique_ix]
                    # get the vals for the observed axes and set unobserved to -1
                    test_arr = np.array([data[v] if v in data else -1 for v in _vars])
                    # retain indices in array which all observed axes set to observed val
                    # if none observed this should just evaluate to all indices of the potential
                    test_indices = np.array([a_ix for a_ix in np.ndindex(*pot.shape) if np.sum(test_arr == np.array(a_ix)) == (test_arr > -1).sum()]).transpose()
                    flat_indices = np.ravel_multi_index(test_indices, pot.shape)
                    # elements at these indices should not have changed by observations
                    np.testing.assert_array_equal(
                                                    np.take(pot, flat_indices),
                                                    np.take(phi[clique], flat_indices)
                                                )

    def test_can_observe_dynamic_evidence_using_global_update_multi_variable(self):
        jt = JunctionTree(
                            {
                                0: 4,
                                1: 8,
                                2: 5,
                                3: 3,
                                4: 6
                            },
                            [
                                0, [0,2,4],
                                (
                                    1, [0,2],
                                    [
                                        2, [0,1,2]
                                    ]
                                ),
                                (
                                    3, [4],
                                    [
                                        4, [3,4],
                                        (
                                            5, [3],
                                            [
                                                6, [1,2,3]
                                            ]
                                        )
                                    ]
                                )
                            ]
                    )

        # define arbitrary join tree potentials
        phi = [
                np.random.randn(4, 5, 6),
                np.random.randn(4,5),
                np.random.randn(4,8,5),
                np.random.randn(6),
                np.random.randn(3, 6),
                np.random.randn(3),
                np.random.randn(8,5,3)
        ]

        data0 = {0: 1, 2: 3, 4: 0}

        likelihood, phi0, shrink_mapping = jt.observe(phi, data0)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))

        assert len(shrink_mapping) == len(phi0)

        # check dimensions and keys potentials after observing evidence
        assert shrink_mapping[0] == ((1,3,0),[])
        assert type(phi0[0][shrink_mapping[0][0]]) == np.float64
        assert shrink_mapping[1] == ((1,3),[])
        assert type(phi0[1][shrink_mapping[1][0]]) == np.float64
        assert shrink_mapping[2] == ((1,slice(None),3),[1])
        assert phi0[2][shrink_mapping[2][0]].shape == (8,)
        assert shrink_mapping[3] == ((0,),[])
        assert type(phi0[3][shrink_mapping[3][0]]) == np.float64
        assert shrink_mapping[4] == ((slice(None),0),[3])
        assert phi0[4][shrink_mapping[4][0]].shape == (3,)
        assert shrink_mapping[5] == ((slice(None),),[3])
        assert phi0[5][shrink_mapping[5][0]].shape == (3,)
        assert shrink_mapping[6] == ((slice(None),3,slice(None)),[1,3])
        assert phi0[6][shrink_mapping[6][0]].shape == (8,3)

        phi1 = bp.hugin(
                    jt.get_struct(),
                    jt.get_label_order(),
                    phi0,
                    bp.sum_product
        )
        assert_junction_tree_consistent(jt, phi1)

        data = {0: 1, 1: 2, 2: 3, 3: 2, 4: 0}


        likelihood, phi2, shrink_mapping = jt.observe(phi1, data)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([0,0,1,0,0,0,0,0]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([0,0,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))

        assert len(shrink_mapping) == len(phi2)

        # check dimensions and keys potentials after observing evidence
        assert shrink_mapping[0] == ((1,3,0),[])
        assert type(phi2[0][shrink_mapping[0][0]]) == np.float64
        assert shrink_mapping[1] == ((1,3),[])
        assert type(phi2[1][shrink_mapping[1][0]]) == np.float64
        assert shrink_mapping[2] == ((1,2,3),[])
        assert type(phi2[2][shrink_mapping[2][0]]) == np.float64
        assert shrink_mapping[3] == ((0,),[])
        assert type(phi2[3][shrink_mapping[3][0]]) == np.float64
        assert shrink_mapping[4] == ((2,0),[])
        assert type(phi2[4][shrink_mapping[4][0]]) == np.float64
        assert shrink_mapping[5] == ((2,),[])
        assert type(phi2[5][shrink_mapping[5][0]]) == np.float64
        assert shrink_mapping[6] == ((2,3,2),[])
        assert type(phi2[6][shrink_mapping[6][0]]) == np.float64

        phi3 = bp.hugin(
                    jt.get_struct(),
                    jt.get_label_order(),
                    phi2,
                    bp.sum_product
        )
        assert_junction_tree_consistent(jt, phi3)

        # test that a potential containing observed variable is altered properly after observing data
        # (Eventually this will be based on familial content of clique or a more optimal clique but
        # for now just find one but for now just check the first clique containing the variable)

        for var, val in data.items():
            clique, _vars = bp.get_clique(jt.get_struct(), var)
            pot = phi3[clique]
            assert pot.shape == phi[clique].shape
            var_ix = _vars.index(var)
            # check that potential properly updated
            mask = [val == dim for dim in range(pot.shape[var_ix])]
            # values along var axis not equal to val will have a 0 value
            assert np.all(np.compress(np.invert(mask), pot, axis=var_ix)) == 0
            # ensure that var axis equal to val is equivalent in both original and new potentials
            np.testing.assert_array_equal(
                                            np.compress(mask, pot, axis=var_ix),
                                            np.compress(mask, phi[clique], axis=var_ix)
                                        )

        # test that no change made to potential values for unobserved variables
        for var in jt.get_key_sizes():
            if var not in data.keys():
                # we have not observed a value for this var
                for clique_ix, _vars in bp.get_cliques(jt.get_struct(), var):
                    pot = phi3[clique_ix]
                    # get the vals for the observed axes and set unobserved to -1
                    test_arr = np.array([data[v] if v in data else -1 for v in _vars])
                    # retain indices in array which all observed axes set to observed val
                    # if none observed this should just evaluate to all indices of the potential
                    test_indices = np.array([a_ix for a_ix in np.ndindex(*pot.shape) if np.sum(test_arr == np.array(a_ix)) == (test_arr > -1).sum()]).transpose()
                    flat_indices = np.ravel_multi_index(test_indices, pot.shape)
                    # elements at these indices should not have changed by observations
                    np.testing.assert_array_equal(
                                                    np.take(pot, flat_indices),
                                                    np.take(phi[clique], flat_indices)
                                                )

    '''
        skipping this test for now as explicitly handling retraction might be
        unnecessary
    def test_can_observe_dynamic_evidence_using_global_retraction(self):
        # dim(0): 4, dim(1): 8, dim(2): 5, dim(3): 3, dim(4): 6
        jt = JunctionTree(
                            {
                                0: 4,
                                1: 8,
                                2: 5,
                                3: 3,
                                4: 6
                            },
                            [
                                0, [0,2,4],
                                (
                                    1, [0,2],
                                    [
                                        2, [0,1,2]
                                    ]
                                ),
                                (
                                    3, [4],
                                    [
                                        4, [3,4],
                                        (
                                            5, [3],
                                            [
                                                6, [1,2,3]
                                            ]
                                        )
                                    ]
                                )
                            ]
                    )


        # define arbitrary initial join tree potentials
        phi = [
                np.random.randn(4,5,6),
                np.random.randn(4,5),
                np.random.randn(4,8,5),
                np.random.randn(6),
                np.random.randn(3,6),
                np.random.randn(3),
                np.random.randn(8,5,3)
        ]
        data0 = {0: 1, 2: 3, 4: 0}

        likelihood, phi0, shrink_mapping = jt.observe(phi, data0)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))

        phi1 = bp.hugin(
                    jt.get_struct(),
                    jt.get_label_order(),
                    phi0,
                    bp.sum_product
        )
        assert_junction_tree_consistent(jt, phi1)

        data = {0: 2, 2: 3, 4: 0}

        likelihood, phi2, shrink_mapping = jt.observe(phi1, data, "retract")
        np.testing.assert_array_equal(likelihood[0], np.array([0,0,1,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))

        phi3 = bp.hugin(
                    jt.get_struct(),
                    jt.get_label_order(),
                    phi2,
                    bp.sum_product
        )
        assert_junction_tree_consistent(jt, phi3)

        # test that a potential containing observed variable is altered properly after observing data
        # (Eventually this will be based on familial content of clique or a more optimal clique but
        # for now just find one but for now just check the first clique containing the variable)

        for var, val in data.items():
            clique, _vars = bp.get_clique(jt.get_struct(), var)
            pot = phi3[clique]
            assert pot.shape == phi[clique].shape
            var_ix = _vars.index(var)
            # check that potential properly updated
            mask = [val == dim for dim in range(pot.shape[var_ix])]
            # values along var axis not equal to val will have a 0 value
            assert np.all(np.compress(np.invert(mask), pot, axis=var_ix)) == 0
            # ensure that var axis equal to val is equivalent in both original and new potentials
            np.testing.assert_array_equal(
                                            np.compress(mask, pot, axis=var_ix),
                                            np.compress(mask, phi[clique], axis=var_ix)
                                        )

        # test that no change made to potential values for unobserved variables
        for var in jt.get_key_sizes():
            if var not in data.keys():
                # we have not observed a value for this var
                for clique_ix, _vars in bp.get_cliques(jt.get_struct(), var):
                    pot = phi3[clique_ix]
                    # get the vals for the observed axes and set unobserved to -1
                    test_arr = np.array([data[v] if v in data else -1 for v in _vars])
                    # retain indices in array which all observed axes set to observed val
                    # if none observed this should just evaluate to all indices of the potential
                    test_indices = np.array([a_ix for a_ix in np.ndindex(*pot.shape) if np.sum(test_arr == np.array(a_ix)) == (test_arr > -1).sum()]).transpose()
                    flat_indices = np.ravel_multi_index(test_indices, pot.shape)
                    # elements at these indices should not have changed by observations
                    np.testing.assert_array_equal(
                                                    np.take(pot, flat_indices),
                                                    np.take(phi[clique], flat_indices)
                                                )

        '''

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
        tree = [0, [0,1], (1, [1], [2, [1,2]])]
        clique, _vars = bp.get_clique(tree, 2)
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
        tri, ics, max_cliques = bp.find_triangulation(fg[1], fg[0])

        # triangulation should consist of 1 edge
        assert len(tri) == 1
        assert len(tri[0]) == 2
        # to triangulate we have a few options
        assert set(tri[0]) in [set(("A","C")),set(("A","D")),set(("B","D")),set(("B","E"))]

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

        tri, ics, max_cliques = bp.find_triangulation(factors, _vars)
        assert_triangulated(factors, tri)
        #cliques = bp.identify_cliques(ics)


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

        tri, ics, max_cliques = bp.find_triangulation(factors, _vars)
        assert_triangulated(factors, tri)
        cliques = bp.identify_cliques(ics)

        assert len(max_cliques) == 5

        assert ["C","D"] in cliques
        assert ["D","G","I"] in cliques
        assert ["G","I","S"] in cliques
        assert ["G","J","L","S"] in cliques
        assert ["G","H","J"] in cliques


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

    def test_index_vars(self):
        var_lookup = {
                "A": 0,
                "B": 1,
                "C": 2,
                "D": 4,
                "E": 8,
                "F": 9,
                "G": 10
        }
        in_tree = [
                    2, ["B","C"],
                    (
                        1, ["C"],
                        [
                            0, ["A","C","D"],
                            (
                                5, ["A"],
                                [
                                    4,["A","E"],
                                    (
                                        7, ["E"],
                                        [
                                            8, ["E","G"]
                                        ]
                                    )
                                ]
                            ),
                            (
                                3, ["D"],
                                [
                                    6, ["D","F"]
                                ]
                            )
                        ]
                    )
                ]

        out_tree = JunctionTree.map_keys(in_tree, var_lookup)



        test_tree = [
                    2, [1,2],
                    (
                        1, [2],
                        [
                            0, [0,2,4],
                            (
                                5, [0],
                                [
                                    4,[0,8],
                                    (
                                        7, [8],
                                        [
                                            8, [8, 10]
                                        ]
                                    )
                                ]
                            ),
                            (
                                3, [4],
                                [
                                    6, [4,9]
                                ]
                            )
                        ]
                    )
                ]

        assert_junction_tree_equal2(out_tree, test_tree)

    def test_unindex_vars(self):
        var_lookup = {
                "A": 0,
                "B": 1,
                "C": 2,
                "D": 4,
                "E": 8,
                "F": 9,
                "G": 10
        }

        in_tree = [
                    2, [1,2],
                    (
                        1, [2],
                        [
                            0, [0,2,4],
                            (
                                5, [0],
                                [
                                    4,[0,8],
                                    (
                                        7, [8],
                                        [
                                            8, [8, 10]
                                        ]
                                    )
                                ]
                            ),
                            (
                                3, [4],
                                [
                                    6, [4,9]
                                ]
                            )
                        ]
                    )
                ]

        out_tree = JunctionTree.map_keys(in_tree, {v:k for k, v in var_lookup.items()})

        test_tree = [
                    2, ["B","C"],
                    (
                        1, ["C"],
                        [
                            0, ["A","C","D"],
                            (
                                5, ["A"],
                                [
                                    4,["A","E"],
                                    (
                                        7, ["E"],
                                        [
                                            8, ["E","G"]
                                        ]
                                    )
                                ]
                            ),
                            (
                                3, ["D"],
                                [
                                    6, ["D","F"]
                                ]
                            )
                        ]
                    )
                ]

        assert_junction_tree_equal2(out_tree, test_tree)

    def test_join_cliques_into_junction_tree(self):
        """
            test_join_cliques_into_junction_tree

            Example taken from section 4.4.3 (Huang and Darwiche, 1996)

        """

        key_sizes = {
                        "A": 2,
                        "B": 2,
                        "C": 2,
                        "D": 2,
                        "E": 2,
                        "F": 2,
                        "G": 2,
                        "H": 2
        }

        factors = [
                    ["A"], # 0
                    ["B"], # 1
                    ["C"], # 2
                    ["D"], # 3
                    ["E"], # 4
                    ["F"], # 5
                    ["G"], # 6
                    ["H"]  # 7
        ]

        cliques = [
                    ["A","B","D"],#[0,1,3]
                    ["A","C","E"],#[0,2,4]
                    ["A","D","E"],#[0,3,4]
                    ["C","E","G"],#[2,4,6]
                    ["D","E","F"],#[3,4,5]
                    ["E","G","H"],#[4,6,7]
                ]

        tree, sepsets = bp.construct_junction_tree2(cliques, key_sizes)

        jt0 = JunctionTree(key_sizes, tree)
        label_dict = jt0.get_label_order()

        def __sepsets_using_tree_index(jt, sepsets):
            label_dict = jt0.get_label_order()
            r_sepsets = []
            for ss in sepsets:
                l = []
                for key_label in sorted(ss):
                    l.append(label_dict[key_label])
                r_sepsets.append(tuple(l))
            return r_sepsets

        assert set(__sepsets_using_tree_index(jt0, sepsets)) == set([(0,3), (3,4), (0,4), (2,4), (4,6)])

        # expected junction tree

        jt1 = [
                0, [0,3,4],
                (
                    1, [0,3],
                    [
                        2, [0,1,3]
                    ]
                ),
                (
                    3, [3,4],
                    [
                        4,[3,4,5]
                    ]
                ),
                (
                    5, [0,4],
                    [
                        6, [0,2,4],
                        (
                            7, [2,4],
                            [
                                8, [2,4,6],
                                (
                                    9, [4,6],
                                    [
                                        10, [4,6,7]
                                    ]
                                )
                            ]
                        )
                    ]

                )
            ]

        assert_junction_tree_equal2(jt0.get_struct(), jt1)

    def test_junction_tree_structure_as_indices_into_node_list(self):
        key_sizes = {
                        "A": 2,
                        "B": 2,
                        "C": 2,
                        "D": 2,
                        "E": 2,
                        "F": 2,
                        "G": 2,
                        "H": 2
        }

        factors = [
                    ["A"], # 0
                    ["B"], # 1
                    ["C"], # 2
                    ["D"], # 3
                    ["E"], # 4
                    ["F"], # 5
                    ["G"], # 6
                    ["H"]  # 7
        ]

        cliques = [
                    ["A","B","D"],#[0,1,3]
                    ["A","C","E"],#[0,2,4]
                    ["A","D","E"],#[0,3,4]
                    ["C","E","G"],#[2,4,6]
                    ["D","E","F"],#[3,4,5]
                    ["E","G","H"],#[4,6,7]
                ]

        tree1, sepsets = bp.construct_junction_tree2(cliques, key_sizes)
        traversed_tree = list(bp.bf_traverse2(tree1))

        tree2, sepsets = bp.construct_junction_tree(cliques, key_sizes)
        node_list = cliques + sepsets

        # check that both tree structures have same underlying nodes
        assert set([tuple(set(val)) for i, val in enumerate(traversed_tree) if (i+1) % 2 == 0]) == set([tuple(set(node)) for node in node_list])

        # check that relationship between nodes is preserved in node list representation of tree
        assert set([tuple(set([tuple(pairs[1]), tuple(pairs[3])])) for pairs in bp.generate_potential_pairs2(tree1)]) == set([tuple(set([tuple(node_list[pair[0]]), tuple(node_list[pair[1]])])) for pair in bp.generate_potential_pairs(tree2)])

class TestJunctionTreeInference(unittest.TestCase):
    def setUp(self):
        self.tree = [
                        0, ["A","D","E"],
                        (
                            1, ["A","D"],
                            [
                                2, ["A","B","D"]
                            ]
                        ),
                        (
                            3, ["D","E"],
                            [
                                4,["D","E","F"]
                            ]
                        ),
                        (
                            5, ["A","E"],
                            [
                                6, ["A","C","E"],
                                (
                                    7, ["C","E"],
                                    [
                                        8, ["C","E","G"],
                                        (
                                            9, ["E","G"],
                                            [
                                                10, ["E","G","H"]
                                            ]
                                        )
                                    ]
                                )
                            ]

                        )
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

        self.fg = [self.key_sizes,self.factors,self.values]


    def test_transformation(self):
        jt0 = JunctionTree(self.key_sizes, self.tree)
        init_phi = JunctionTree.init_potentials(jt0, self.factors, self.values)
        phi0 = jt0.propagate(init_phi)
        jt, init_phi = JunctionTree.from_factor_graph(self.fg)
        phi = jt.propagate(init_phi)

        # check that marginal values are same with different tree structures
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["A"]),
                                jt0.marginalize(phi0, ["A"])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["B"]),
                                jt0.marginalize(phi0, ["B"])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["C"]),
                                jt0.marginalize(phi0, ["C"])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["D"]),
                                jt0.marginalize(phi0, ["D"])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["E"]),
                                jt0.marginalize(phi0, ["E"])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["G"]),
                                jt0.marginalize(phi0, ["G"])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["F"]),
                                jt0.marginalize(phi0, ["F"])
        )

        np.testing.assert_allclose(
                                jt.marginalize(phi, ["H"]),
                                jt0.marginalize(phi0, ["H"])
        )

        # check that marginal values are correct
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["A"]),
                                np.array([0.500,0.500])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["B"]),
                                np.array([0.550,0.450])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["C"]),
                                np.array([0.550,0.450])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["D"]),
                                np.array([0.320,0.680])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["E"]),
                                np.array([0.535,0.465])
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["G"]),
                                np.array([0.855,0.145])
        )

        np.testing.assert_allclose(
                                jt.marginalize(phi, ["F"]),
                                np.array([0.824,0.176]),
                                atol=0.01
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["H"]),
                                np.array([ 0.104,  0.896]),
                                atol=0.01
        )

    def test_initialize_potentials(self):
        jt = JunctionTree(self.key_sizes,self.tree)
        init_phi = JunctionTree.init_potentials(jt, self.fg[1], self.fg[2])
        assert_potentials_equal(
                                    init_phi[6:8], # clusters ACE and CE
                                    [
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
                                                ),
                                        np.array(
                                                    [
                                                        [1,1],
                                                        [1,1]
                                                    ]
                                                )
                                    ]
                                )

    def test_global_propagation(self):
        jt = JunctionTree(self.key_sizes,self.tree)
        init_phi = JunctionTree.init_potentials(jt, self.fg[1], self.fg[2])
        phi = jt.propagate(init_phi)
        assert_potentials_equal(
                                [phi[2]],
                                [
                                    np.array(
                                                [
                                                    [
                                                        [0.150,0.150],
                                                        [0.020,0.180]
                                                    ],
                                                    [
                                                        [0.125,0.125],
                                                        [0.025,0.225]
                                                    ]
                                                ]
                                            )
                                    ]
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

        fg = [key_sizes, factors, values]
        jt, init_phi = JunctionTree.from_factor_graph(fg)

        # grass is wet
        phi = jt.propagate(init_phi, in_place=False, data={"wet_grass":1})
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["sprinkler"], normalize=True),
                                np.array([0.57024,0.42976]),
                                atol=0.01
        )

        # grass is wet and it is raining
        # no need to calculate init_phi in place because init_phi not used again
        phi = jt.propagate(init_phi, data={"wet_grass":1, "rain": 1})
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["sprinkler"], normalize=True),
                                np.array([0.8055,0.1945]),
                                atol=0.01
        )
        np.testing.assert_allclose(
                                jt.marginalize(phi, ["rain"], normalize=True),
                                np.array([0,1]),
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

        tree = [
                    0, ["C","D","E"],
                    (
                        1, ["D","E"],
                        [
                            2, ["D","E","F"]
                        ]
                    ),
                    (
                        3, ["C","D"],
                        [
                            4, ["B","C","D"],
                            (
                                5, ["B","C"],
                                [
                                    6, ["A","B","C"]
                                ]
                            )
                        ]
                    )
                ]

        fg = [key_sizes,factors,values]
        jt = JunctionTree(key_sizes, tree)
        init_phi = JunctionTree.init_potentials(jt, fg[1], fg[2])

        phi = jt.propagate(init_phi)
        assert math.isclose(phi[6][0,0,0], 0.072, abs_tol=0.01)
        assert math.isclose(phi[6][0,0,1], 0.018, abs_tol=0.01)
        assert math.isclose(phi[6][0,1,0], 0.648, abs_tol=0.01)
        assert math.isclose(phi[6][0,1,1], 0.162, abs_tol=0.01)
        assert math.isclose(phi[6][1,0,0], 0.027, abs_tol=0.01)
        assert math.isclose(phi[6][1,0,1], 0.063, abs_tol=0.01)
        assert math.isclose(phi[6][1,1,0], 0.003, abs_tol=0.01)
        assert math.isclose(phi[6][1,1,1], 0.007, abs_tol=0.01)

        assert math.isclose(phi[4][0,0,0], 0.030, abs_tol=0.01)
        assert math.isclose(phi[4][0,0,1], 0.069, abs_tol=0.01)
        assert math.isclose(phi[4][0,1,0], 0.024, abs_tol=0.01)
        assert math.isclose(phi[4][0,1,1], 0.057, abs_tol=0.01)
        assert math.isclose(phi[4][1,0,0], 0.391, abs_tol=0.01)
        assert math.isclose(phi[4][1,0,1], 0.260, abs_tol=0.01)
        assert math.isclose(phi[4][1,1,0], 0.101, abs_tol=0.01)
        assert math.isclose(phi[4][1,1,1], 0.068, abs_tol=0.01)

        assert math.isclose(phi[2][0,0,0], 0.063, abs_tol=0.01)
        assert math.isclose(phi[2][0,0,1], 0.252, abs_tol=0.01)
        assert math.isclose(phi[2][0,1,0], 0.139, abs_tol=0.01)
        assert math.isclose(phi[2][0,1,1], 0.092, abs_tol=0.01)
        assert math.isclose(phi[2][1,0,0], 0.130, abs_tol=0.01)
        assert math.isclose(phi[2][1,0,1], 0.130, abs_tol=0.01)
        assert math.isclose(phi[2][1,1,0], 0.175, abs_tol=0.01)
        assert math.isclose(phi[2][1,1,1], 0.019, abs_tol=0.01)

        assert math.isclose(phi[0][0,0,0], 0.252, abs_tol=0.01)
        assert math.isclose(phi[0][0,0,1], 0.168, abs_tol=0.01)
        assert math.isclose(phi[0][0,1,0], 0.198, abs_tol=0.01)
        assert math.isclose(phi[0][0,1,1], 0.132, abs_tol=0.01)
        assert math.isclose(phi[0][1,0,0], 0.063, abs_tol=0.01)
        assert math.isclose(phi[0][1,0,1], 0.063, abs_tol=0.01)
        assert math.isclose(phi[0][1,1,0], 0.062, abs_tol=0.01)
        assert math.isclose(phi[0][1,1,1], 0.062, abs_tol=0.01)

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
