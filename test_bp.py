import numpy as np

import bp
from bp import JunctionTree
import unittest
import networkx as nx


# Tests here using pytest


def assert_junction_tree_equal(t1, t2):
    """Test equality of two junction trees

    Junction tree syntax:

    [array, keys, child_tree1, ..., child_treeN]

    Note this syntax supports separators in the trees because the separators
    just add levels.

    """
    # Equality of arrays
    np.testing.assert_allclose(t1[0], t2[0])
    # Equality of keys
    assert t1[1] == t2[1]

    # Same number of child trees
    assert len(t1) == len(t2)
    # Equality of child trees (recursively)
    for (child_t1, child_t2) in zip(t1[2:], t2[2:]):
        assert_junction_tree_equal(child_t1, child_t2)



def assert_factor_graph_equal(fg1, fg2):
    # assert that the variable maps are equal
    assert fg1[0] == fg2[0]
    # ensure that factor lists are the same
    assert len(fg1[1]) == len(fg[1])
    assert np.all([a1 == a2 for a1,a2 in zip(fg1[1],fg2[1])])
    assert len(fg1[2]) == len(fg2[2])
    assert np.all([np.testing.assert_allclose(a1, a2) for a1,a2 in zip(fg1[2],fg2[2])])

def assert_triangulated(factors, triangulation):
    '''
        An undirected graph is triangulated iff every cycle of length four or
        greater contains an edge that connects two nonadjacent nodes in the
        cycle. (Huang and Darwiche, 1996)
    '''

    cycles = __find_cycles(factors, 4)
    f_sets = [set(f) for f in factors]
    for cycle in cycles:
        s = set(cycle)
        assert any([s == f_set for f_set in sets if len(s) == len(f_set)])

def __find_cycles(factors, num):
    G=nx.Graph()
    G.add_nodes_from(range(len(factors)))

    for i in range(len(factors)):
        for j in range(i,len(factors)):
            # add edge to graph if any vars shared
            if not set(factors[i]).isdisjoint(factors[j]):
                G.add_edge(i,j)
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

    cycles = [np.array(graph_edges)[[np.nonzero(cycle)]]
                for cycle in __gibbs_elem_cycles(graph_edges,
                                                bit_seqs) if sum(cycle) >= num]
    return cycles

def __gibbs_elem_cycles(edges, fcs):
    '''
        Norman E. Gibbs. 1969. A Cycle Generation Algorithm for Finite
            Undirected Linear Graphs. J. ACM 16, 4 (October 1969), 564-568.
            DOI=http://dx.doi.org/10.1145/321541.321545
    '''
    s = [fcs[0]]
    q = [fcs[0]]
    r = []
    r_star = []
    i = 1
    while i <= bits.shape[0]:
        for t in q:
            if np.any(np.logical_and(t, fcs[i])):
                # append t ring_sum fcs[0] to r
                r.append(np.logical_xor(t,fcs[0]).astype(int).tolist())

            else:
                # append t ring_sum fcs[0] to r_star
                r_star.append(np.logical_xor(t,fcs[0]).astype(int).tolist())
        for u,v in itertools.combinations(r, 2):
            # check both ways u subset of v or v subset of u
            if np.array_equal(np.logical_and(u, v), u):
                r.remove(v)
                if v not in r_star:
                    r_star.append(v)
            elif np.array_equal(np.logical_and(v, u), v):
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


def get_arrays_and_keys(tree):
    """Get all arrays and their keys as a flat list

    Output: [array1, keys1, ..., arrayN, keysN]

    """
    return list(tree[:2]) + sum(
        [
            get_arrays_and_keys(child_tree)
            for child_tree in tree[2:]
        ],
        []
    )


def brute_force_sum_product(junction_tree):
    """Compute brute force sum-product with einsum """

    # Function to compute the sum-product with brute force einsum
    arrays_keys = get_arrays_and_keys(junction_tree)
    f = lambda output_keys: np.einsum(*(arrays_keys + [output_keys]))

    def __run(tree, f):
        return [
            f(tree[1]),
            tree[1],
            *[
                __run(child_tree, f)
                for child_tree in tree[2:]
            ]
        ]

    return __run(junction_tree, f)


def assert_sum_product(tree):
    """ Test hugin vs brute force sum-product """
    assert_junction_tree_equal(
        brute_force_sum_product(tree),
        bp.hugin(tree, bp.sum_product)
    )
    pass

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

def get_arrays_and_keys2(tree, potentials):
    """Get all arrays and their keys as a flat list

    Output: [array1, keys1, ..., arrayN, keysN]

    """
    return list([potentials[tree[0]],tree[1]]) + sum(
        [
            get_arrays_and_keys2(child_tree, potentials)
            for child_tree in tree[2:]
        ],
        []
    )

def brute_force_sum_product2(junction_tree, potentials):
    """Compute brute force sum-product with einsum """

    # Function to compute the sum-product with brute force einsum
    arrays_keys = get_arrays_and_keys2(junction_tree, potentials)
    f = lambda output_keys: np.einsum(*(arrays_keys + [output_keys]))

    def __run(tree, p, f, res=[]):
        res.append(f(tree[1]))
        for child_tree in tree[2:]:
            __run(child_tree, p, f, res)
        return res

    return __run(junction_tree, potentials, f)

def assert_sum_product2(tree, potentials):
    """ Test hugin vs brute force sum-product """
    assert_potentials_equal(
        brute_force_sum_product2(tree, potentials),
        bp.hugin(tree, potentials, bp.sum_product)
    )
    pass

def assert_junction_tree_consistent(tree, potentials):
    '''
        For each clique/sepset pair in tree, check consistency

        Function checks that junction tree is consistent with respect to the
        provided potentials
    '''

    assert np.all(
                [
                    potentials_consistent(c_pot, c_vars, s_pot, s_vars)
                        for c_pot,
                            c_vars,
                            s_pot,
                            s_vars in bp.generate_potential_pairs(
                                                        tree.get_struct(),
                                                        potentials
                                    )
                ]
            )

def potentials_consistent(c_pot, c_vars, s_pot, s_vars):
    '''
        Ensure that summing over clique potentials for variables not present in
        sepset generates a potential equal to sepset potential (definition of
        consistent)
    '''

    return np.allclose(
                bp.compute_marginal(
                    c_pot,
                    c_vars,
                    np.intersect1d(c_vars, s_vars).tolist()
                ),
                s_pot
            )


def test_hugin():
    """ Test hugin sum-product """

    # One scalar node
    assert_sum_product(
        [
            np.random.randn(),
            []
        ]
    )

    # One matrix node
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5]
        ]
    )

    # One child node with all variables shared
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3, 2)),
                [5, 3],
                [
                    np.random.randn(3, 2),
                    [5, 3],
                ]
            )
        ]
    )

    # One child node with one common variable
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 4),
                    [5, 9]
                ]
            )
        ]
    )

    # One child node with no common variable
    assert_sum_product(
        [
            np.random.randn(2),
            [3],
            (
                np.ones(()),
                [],
                [
                    np.random.randn(3),
                    [9]
                ]
            )
        ]
    )

    # One grand child node (not sharing with grand parent)
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 4),
                    [5, 9],
                    (
                        np.ones((4,)),
                        [9],
                        [
                            np.random.randn(4, 5),
                            [9, 1]
                        ]
                    )
                ]
            )
        ]
    )

    # One grand child node (sharing with grand parent)
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 4),
                    [5, 9],
                    (
                        np.ones((3,)),
                        [5],
                        [
                            np.random.randn(6, 3),
                            [1, 5]
                        ]
                    )
                ]
            )
        ]
    )

    # Two children (not sharing)
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 4),
                    [5, 9],
                ]
            ),
            (
                np.ones((2,)),
                [3],
                [
                    np.random.randn(2, 5),
                    [3, 1]
                ]
            )
        ]
    )

    # Two children (sharing)
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 4),
                    [5, 9],
                ]
            ),
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3),
                    [5]
                ]
            )
        ]
    )

    # Two children (with 3-D tensors)
    assert_sum_product(
        [
            np.random.randn(2, 3, 4),
            [3, 5, 7],
            (
                np.ones((3, 4)),
                [5, 7],
                [
                    np.random.randn(3, 4, 5),
                    [5, 7, 9],
                ]
            ),
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 6),
                    [5, 1]
                ]
            )
        ]
    )

    pass

    # One matrix node
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5]
        ]
    )

    # One child node with all variables shared
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3, 2)),
                [5, 3],
                [
                    np.random.randn(3, 2),
                    [5, 3],
                ]
            )
        ]
    )

    # One child node with one common variable
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 4),
                    [5, 9]
                ]
            )
        ]
    )

    # One child node with no common variable
    assert_sum_product(
        [
            np.random.randn(2),
            [3],
            (
                np.ones(()),
                [],
                [
                    np.random.randn(3),
                    [9]
                ]
            )
        ]
    )

    # One grand child node (not sharing with grand parent)
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 4),
                    [5, 9],
                    (
                        np.ones((4,)),
                        [9],
                        [
                            np.random.randn(4, 5),
                            [9, 1]
                        ]
                    )
                ]
            )
        ]
    )

    # One grand child node (sharing with grand parent)
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 4),
                    [5, 9],
                    (
                        np.ones((3,)),
                        [5],
                        [
                            np.random.randn(6, 3),
                            [1, 5]
                        ]
                    )
                ]
            )
        ]
    )

    # Two children (not sharing)
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 4),
                    [5, 9],
                ]
            ),
            (
                np.ones((2,)),
                [3],
                [
                    np.random.randn(2, 5),
                    [3, 1]
                ]
            )
        ]
    )

    # Two children (sharing)
    assert_sum_product(
        [
            np.random.randn(2, 3),
            [3, 5],
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 4),
                    [5, 9],
                ]
            ),
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3),
                    [5]
                ]
            )
        ]
    )

    # Two children (with 3-D tensors)
    assert_sum_product(
        [
            np.random.randn(2, 3, 4),
            [3, 5, 7],
            (
                np.ones((3, 4)),
                [5, 7],
                [
                    np.random.randn(3, 4, 5),
                    [5, 7, 9],
                ]
            ),
            (
                np.ones((3,)),
                [5],
                [
                    np.random.randn(3, 6),
                    [5, 1]
                ]
            )
        ]
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
        assert np.allclose(bp.compute_marginal(phiABD, [0,1,2], [0]), np.array([0.500, 0.500]))
        # marginal probability of D, P(D)
        assert np.allclose(np.array([0.32,0.68]), np.array([0.320, 0.680]))


    def test_pass_message(self):
        '''
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

        '''

        phi12 = np.array([
                            [0.4, 0.8],
                            [0.6, 0.2]
                        ])

        phi2 = np.array([1, 1])
        phi23 = np.array([
                            [0.03, 0.07],
                            [0.45, 0.45]
                        ])

        phi2n = bp.project(phi12, [1])
        np.allclose(phi2n, np.array([1,1])) == True
        phi23 = bp.absorb(phi23, phi2, phi2n)
        np.allclose(phi23, np.array([
                                        [0.03,0.07],
                                        [0.45,0.45]
                                    ])) == True

        phi2nn = bp.project(phi23, [0])
        np.allclose(phi2nn, np.array([0.9, 0.1])) == True
        phi12 = bp.absorb(phi12, phi2n, phi2nn)
        np.allclose(phi12, np.array([
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
        # jt.collect_messages(POTENTIALS, CLIQUE_INDEX=0)
        phiN = bp.collect(jt, phi, [0]*len(phi), 2)
        np.allclose(phiN[2], np.array([
                                        [0.03,0.07],
                                        [0.45,0.45]
                                    ])) == True


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


        # jt.distribute_messages(POTENTIALS, CLIQUE_INDEX=0)
        phiN = bp.distribute(jt, phi, [0]*len(phi), 2)
        np.allclose(phiN[2], np.array([
                                        [0.04,0.72],
                                        [0.06,0.18]
                                    ]))


    def test_one_scalar_node(self):
        assert_sum_product2(
            [
                0, []
            ],
            [
                np.random.randn(),
            ]
        )

    def test_one_matrix_node(self):
        assert_sum_product2(
            [
                0, [3, 5]
            ],
            [
                np.random.randn(2, 3),
            ]
        )

    def test_one_child_node_with_all_variables_shared(self):
        # is it possible to have two cliques with the exact same set of variables


        potentials =             [
                        np.random.randn(2, 3),
                        np.ones((3, 2)),
                        np.random.randn(3, 2),
                    ]

        assert_sum_product2(
            [
                0, [3, 5],
                (
                    1, [5, 3],
                    [
                        2, [5, 3],
                    ]
                )
            ],
            [
                np.random.randn(2, 3),
                np.ones((3, 2)),
                np.random.randn(3, 2),
            ]
        )

    def test_one_child_node_with_one_common_variable(self):
        assert_sum_product2(
            [
                0, [3, 5],
                (
                    1, [5],
                    [
                        2, [5, 9]
                    ]
                )
            ],
            [
                np.random.randn(2, 3),
                np.ones((3,)),
                np.random.randn(3, 4),
            ]
        )

    def test_one_child_node_with_no_common_variable(self):
        assert_sum_product2(
            [
                0, [3],
                (
                    1, [],
                    [
                        2, [9]
                    ]
                )
            ],
            [
                np.random.randn(2),
                np.ones(()),
                np.random.randn(3),
            ]
        )

    def test_one_grand_child_node_with_no_variable_shared_with_grand_parent(self):
        assert_sum_product2(
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
            ],
            [
                np.random.randn(2, 3),
                np.ones((3,)),
                np.random.randn(3, 4),
                np.ones((4,)),
                np.random.randn(4, 5),
            ]
        )

    def test_one_grand_child_node_with_variable_shared_with_grand_parent(self):
        assert_sum_product2(
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
            ],
            [
                np.random.randn(2, 3),
                np.ones((3,)),
                np.random.randn(3, 4),
                np.ones((3,)),
                np.random.randn(6, 3),
            ]
        )

    def test_two_children_with_no_variable_shared(self):
        assert_sum_product2(
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
            ],
            [
                np.random.randn(2, 3),
                np.ones((3,)),
                np.random.randn(3, 4),
                np.ones((2,)),
                np.random.randn(2, 5),
            ]
        )

    def test_two_child_with_shared_variable(self):
        assert_sum_product2(
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
            ],
            [
                np.random.randn(2, 3),
                np.ones((3,)),
                np.random.randn(3, 4),
                np.ones((3,)),
                np.random.randn(3),

            ]
        )

    def test_two_children_with_3D_tensors(self):
        assert_sum_product2(
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
            ],
            [
                np.random.randn(2, 3, 4),
                np.ones((3, 4)),
                np.random.randn(3, 4, 5),
                np.ones((3,)),
                np.random.randn(3, 6),
            ]
        )

    def test_can_observe_evidence_from_one_trial(self):
        # dim(0): 4, dim(1): 8, dim(2): 5, dim(3): 3, dim(4): 6
        jt = [
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

        likelihood, phiN = bp.observe(jt, phi, None, data)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        assert_junction_tree_consistent(jt, phiN)

        # test that a potential containing observed variable is altered properly after observing data
        # (Eventually this will be based on familial content of clique or a more optimal clique but
        # for now just find one but for now just check the first clique containing the variable)

        for var, val in data.items():
            clique, _vars = bp.get_clique(jt, var)
            pot = phiN[clique]
            assert pot.shape == phi[clique].shape
            var_idx = _vars.index(var)
            # check that potential properly updated
            mask = [val == dim for dim in range(pot.shape[var_idx])]
            # values along var axis not equal to val will have a 0 value
            assert np.all(np.compress(np.invert(mask), pot, axis=var_idx)) == 0
            # ensure that var axis equal to val is equivalent in both original and new potentials
            np.testing.assert_array_equal(
                                            np.compress(mask, pot, axis=var_idx),
                                            np.compress(mask, phi[clique], axis=var_idx)
                                        )

        # test that no change made to potential values for unobserved variables
        for var in bp.get_vars(tree):
            if var not in data.keys():
                # we have not observed a value for this var
                for clique, _vars in bp.get_cliques(tree, var).iteritems():

                    # get the vals for the observed axes and set unobserved to -1
                    test_arr = np.array([data[v] if v in data else -1 for v in _vars])
                    # retain indices in array which all observed axes set to observed val
                    # if none observed this should just evaluate to all indices of the potential
                    test_indices = np.array([a_idx for a_idx in np.ndindex(*pot.shape) if np.sum(test_arr == np.array(a_idx)) == (test_arr > -1).sum()]).transpose()
                    flat_indices = np.ravel_multi_index(test_indices, pot.shape)
                    # elements at these indices should not have changed by observations
                    np.testing.assert_array_equal(
                                                    np.take(pot, flat_indices),
                                                    np.take(phi[clique], flat_indices)
                                                )


    def test_can_observe_dynamic_evidence_using_global_update_single_variable(self):
        # dim(0): 4, dim(1): 8, dim(2): 5, dim(3): 3, dim(4): 6
        jt = [
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

        likelihood, phi0 = bp.observe(jt, phi, None, data0)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        assert_junction_tree_consistent(jt, phi0)

        data = {0: 1, 1: 2, 2: 3, 4: 0}

        likelihood, phiN = bp.observe(jt, phi0, likelihood, data, "update")
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([0,0,1,0,0,0,0,0]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        assert_junction_tree_consistent(jt, phiN)

        # test that a potential containing observed variable is altered properly after observing data
        # (Eventually this will be based on familial content of clique or a more optimal clique but
        # for now just find one but for now just check the first clique containing the variable)

        for var, val in data.items():
            clique, _vars = bp.get_clique(jt, var)
            pot = phiN[clique]
            assert pot.shape == phi[clique].shape
            var_idx = _vars.index(var)
            # check that potential properly updated
            mask = [val == dim for dim in range(pot.shape[var_idx])]
            # values along var axis not equal to val will have a 0 value
            assert np.all(np.compress(np.invert(mask), pot, axis=var_idx)) == 0
            # ensure that var axis equal to val is equivalent in both original and new potentials
            np.testing.assert_array_equal(
                                            np.compress(mask, pot, axis=var_idx),
                                            np.compress(mask, phi[clique], axis=var_idx)
                                        )

        # test that no change made to potential values for unobserved variables
        for var in bp.get_vars(tree):
            if var not in data.keys():
                # we have not observed a value for this var
                for clique, _vars in bp.get_cliques(tree, var).iteritems():

                    # get the vals for the observed axes and set unobserved to -1
                    test_arr = np.array([data[v] if v in data else -1 for v in _vars])
                    # retain indices in array which all observed axes set to observed val
                    # if none observed this should just evaluate to all indices of the potential
                    test_indices = np.array([a_idx for a_idx in np.ndindex(*pot.shape) if np.sum(test_arr == np.array(a_idx)) == (test_arr > -1).sum()]).transpose()
                    flat_indices = np.ravel_multi_index(test_indices, pot.shape)
                    # elements at these indices should not have changed by observations
                    np.testing.assert_array_equal(
                                                    np.take(pot, flat_indices),
                                                    np.take(phi[clique], flat_indices)
                                                )

    def test_can_observe_dynamic_evidence_using_global_update_multi_variable(self):
        # dim(0): 4, dim(1): 8, dim(2): 5, dim(3): 3, dim(4): 6
        jt = [
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

        likelihood, phi0 = bp.observe(jt, phi, None, data0)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        assert_junction_tree_consistent(jt, phi0)

        data = {0: 1, 1: 2, 2: 3, 3: 2, 4: 0}


        likelihood, phiN = bp.observe(jt, phi0, likelihood, data, "update")
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([0,0,1,0,0,0,0,0]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([0,0,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        assert_junction_tree_consistent(jt, phiN)

        # test that a potential containing observed variable is altered properly after observing data
        # (Eventually this will be based on familial content of clique or a more optimal clique but
        # for now just find one but for now just check the first clique containing the variable)

        for var, val in data.items():
            clique, _vars = bp.get_clique(jt, var)
            pot = phiN[clique]
            assert pot.shape == phi[clique].shape
            var_idx = _vars.index(var)
            # check that potential properly updated
            mask = [val == dim for dim in range(pot.shape[var_idx])]
            # values along var axis not equal to val will have a 0 value
            assert np.all(np.compress(np.invert(mask), pot, axis=var_idx)) == 0
            # ensure that var axis equal to val is equivalent in both original and new potentials
            np.testing.assert_array_equal(
                                            np.compress(mask, pot, axis=var_idx),
                                            np.compress(mask, phi[clique], axis=var_idx)
                                        )

        # test that no change made to potential values for unobserved variables
        for var in bp.get_vars(tree):
            if var not in data.keys():
                # we have not observed a value for this var
                for clique, _vars in bp.get_cliques(tree, var).iteritems():

                    # get the vals for the observed axes and set unobserved to -1
                    test_arr = np.array([data[v] if v in data else -1 for v in _vars])
                    # retain indices in array which all observed axes set to observed val
                    # if none observed this should just evaluate to all indices of the potential
                    test_indices = np.array([a_idx for a_idx in np.ndindex(*pot.shape) if np.sum(test_arr == np.array(a_idx)) == (test_arr > -1).sum()]).transpose()
                    flat_indices = np.ravel_multi_index(test_indices, pot.shape)
                    # elements at these indices should not have changed by observations
                    np.testing.assert_array_equal(
                                                    np.take(pot, flat_indices),
                                                    np.take(phi[clique], flat_indices)
                                                )

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


        # define arbitrary join tree potentials
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

        likelihood, phi0 = bp.observe(jt, phi, None, data0)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        self.fail("Implement generic traversal algorithm")
        assert_junction_tree_consistent(jt, phi0)

        data = {0: 2, 2: 3, 4: 0}

        likelihood, phiN = bp.observe(jt, phi0, likelihood, data, "retract")
        np.testing.assert_array_equal(likelihood[0], np.array([0,0,1,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        assert_junction_tree_consistent(jt, phiN)

        # test that a potential containing observed variable is altered properly after observing data
        # (Eventually this will be based on familial content of clique or a more optimal clique but
        # for now just find one but for now just check the first clique containing the variable)

        for var, val in data.items():
            clique, _vars = bp.get_clique(jt, var)
            pot = phiN[clique]
            assert pot.shape == phi[clique].shape
            var_idx = _vars.index(var)
            # check that potential properly updated
            mask = [val == dim for dim in range(pot.shape[var_idx])]
            # values along var axis not equal to val will have a 0 value
            assert np.all(np.compress(np.invert(mask), pot, axis=var_idx)) == 0
            # ensure that var axis equal to val is equivalent in both original and new potentials
            np.testing.assert_array_equal(
                                            np.compress(mask, pot, axis=var_idx),
                                            np.compress(mask, phi[clique], axis=var_idx)
                                        )

        # test that no change made to potential values for unobserved variables
        for var in bp.get_vars(tree):
            if var not in data.keys():
                # we have not observed a value for this var
                for clique, _vars in bp.get_cliques(tree, var).iteritems():

                    # get the vals for the observed axes and set unobserved to -1
                    test_arr = np.array([data[v] if v in data else -1 for v in _vars])
                    # retain indices in array which all observed axes set to observed val
                    # if none observed this should just evaluate to all indices of the potential
                    test_indices = np.array([a_idx for a_idx in np.ndindex(*pot.shape) if np.sum(test_arr == np.array(a_idx)) == (test_arr > -1).sum()]).transpose()
                    flat_indices = np.ravel_multi_index(test_indices, pot.shape)
                    # elements at these indices should not have changed by observations
                    np.testing.assert_array_equal(
                                                    np.take(pot, flat_indices),
                                                    np.take(phi[clique], flat_indices)
                                                )



    def test_marginalize_variable_with_evidence(self):
        '''
            Potentials to be used based on assignments in:
            http://www.inf.ed.ac.uk/teaching/courses/pmr/docs/jta_ex.pdf

            peace = 1
            war = 0
            yes = 1
            no = 0
            stay = 0
            run = 1
            decrease = 0
            no change = 1
            increase = 2

            P(L=1) = 0.4
            P(Q=1) = 0.6
            P(S=1|L=1) = 0.8
            P(S=0|L=1) = 0.2
            P(S=1|L=0) = 0.3
            P(S=0|L=0) = 0.7
            P(F=1|Q=1) = 0.8
            P(F=0|Q=1) = 0.2
            P(F=1|Q=0) = 0.1
            P(F=0|Q=1) = 0.9
            P(B=1|L=1,Q=1) = 0.2
            P(B=0|L=1,Q=1) = 0.8
            P(B=1|L=1,Q=0) = 1
            P(B=0|L=1,Q=0) = 0
            P(B=1|L=0,Q=1) = 1
            P(B=0|L=0,Q=1) = 0
            P(B=1|L=0,Q=0) = 1
            P(B=0|L=0,Q=0) = 0
            P(G=2|L=1,Q=1) = 0.3
            P(G=1|L=1,Q=1) = 0.6
            P(G=0|L=1,Q=1) = 0.1
            P(G=2|L=0,Q=1) = 0.1
            P(G=1|L=0,Q=1) = 0.2
            P(G=0|L=0,Q=1) = 0.7
            P(G=2|L=1,Q=0) = 0.8
            P(G=1|L=1,Q=0) = 0.1
            P(G=0|L=1,Q=0) = 0.1
            P(G=2|L=0,Q=0) = 0.2
            P(G=1|L=0,Q=0) = 0.2
            P(G=0|L=0,Q=0) = 0.6
            P(H=1|B=1,F=0) = 1
            P(H=0|B=1,F=0) = 0
            P(H=1|B=1,F=1) = 0.4
            P(H=0|B=1,F=1) = 0.6
            P(H=1|B=0,F=0) = 0.5
            P(H=0|B=0,F=0) = 0.5
            P(H=1|B=0,F=1) = 0.1
            P(H=0|B=0,F=1) = 0.9
            P(I=1|G=1,H=1) = 0
            P(I=0|G=1,H=1) = 1
            P(I=1|G=0,H=1) = 0
            P(I=0|G=0,H=1) = 1
            P(I=1|G=2,H=1) = 0
            P(I=1|G=2,H=1) = 1
            P(I=1|G=1,H=0) = 0.3
            P(I=0|G=1,H=0) = 0.7
            P(I=1|G=0,H=0) = 0.1
            P(I=0|G=0,H=0) = 0.9
            P(I=1|G=2,H=0) = 1
            P(I=1|G=2,H=0) = 0

                                    {F Q B H}
                                        |
            {S L} --- {L Q B G} --- {H B Q G} --- {I G H}

            S   L   |   \phi_{SL} (P(L)P(S|L))
            ----------------------------------
            0   0   |   0.6 x 0.7 = 0.42
            0   1   |   0.4 x 0.2 = 0.08
            1   0   |   0.6 x 0.3 = 0.18
            1   1   |   0.4 x 0.8 = 0.32

            L   Q   B   G   |   \phi_{LQBG} (P(B|L,Q)P(G|L,Q))
            ----------------------------------------------
            0   0   0   0   |   0 x 0.6 = 0
            0   0   0   1   |   0 x 0.2 = 0
            0   0   0   2   |   0 ...   = 0
            0   0   1   0   |   1 x 0.6 = 0.6
            0   0   1   1   |   1 x 0.2 = 0.2
            0   0   1   2   |   1 x 0.2 = 0.2
            0   1   0   0   |   0 ...   = 0
            0   1   0   1   |   0 ...   = 0
            0   1   0   2   |   0 ...   = 0
            0   1   1   0   |   1 x 0.7 = 0.7
            0   1   1   1   |   1 x 0.2 = 0.2
            0   1   1   2   |   1 x 0.1 = 0.1
            1   0   0   0   |   0 ...   = 0
            1   0   0   1   |   0 ...   = 0
            1   0   0   2   |   0 ...   = 0
            1   0   1   0   |   1 x 0.1 = 0.1
            1   0   1   1   |   1 x 0.1 = 0.1
            1   0   1   2   |   1 x 0.8 = 0.8
            1   1   0   0   |   0.8 x 0.1 = 0.08
            1   1   0   1   |   0.8 x 0.6 = 0.48
            1   1   0   2   |   0.8 x 0.3 = 0.32
            1   1   1   0   |   0.2 x 0.1 = 0.02
            1   1   1   1   |   0.2 x 0.6 = 0.12
            1   1   1   2   |   0.2 x 0.3 = 0.06

            F   Q   B   H   |   \phi_{LQBG} (P(F|Q)P(H|B,F))
            ------------------------------------------------
            0   0   0   0   |   0.9 x 0.5 = 0.45
            0   0   0   1   |   0.9 x 0.5 = 0.45
            0   0   1   0   |   0.9 x 0.5 = 0.45
            0   0   1   1   |   0.9 x 0.5 = 0.45
            0   1   0   0   |   0.2 x 0.5 = 0.10
            0   1   0   1   |   0.2 x 0.5 = 0.10
            0   1   1   0   |   0.2 x 0 = 0
            0   1   1   1   |   0.2 x 1 = 0.2
            1   0   0   0   |   0.1 x 0.9 = 0.09
            1   0   0   1   |   0.1 x 0.1 = 0.01
            1   0   1   0   |   0.1 x 0.6 = 0.06
            1   0   1   1   |   0.1 x 0.4 = 0.04
            1   1   0   0   |   0.8 x 0.9 = 0.72
            1   1   0   1   |   0.8 x 0.1 = 0.08
            1   1   1   0   |   0.8 x 0.6 = 0.48
            1   1   1   1   |   0.8 x 0.4 = 0.32

            H   B   Q   G   |   \phi_{HBQG} (P(Q))
            ------------------------------------------------
            0   0   0   0   |   0.4
            0   0   0   1   |   0.4
            0   0   0   2   |   0.4
            0   0   1   0   |   0.6
            0   0   1   1   |   0.6
            0   0   1   2   |   0.6
            0   1   0   0   |   0.4
            0   1   0   1   |   0.4
            0   1   0   2   |   0.4
            0   1   1   0   |   0.6
            0   1   1   1   |   0.6
            0   1   1   2   |   0.6
            1   0   0   0   |   0.4
            1   0   0   1   |   0.4
            1   0   0   2   |   0.4
            1   0   1   0   |   0.6
            1   0   1   1   |   0.6
            1   0   1   2   |   0.6
            1   1   0   0   |   0.4
            1   1   0   1   |   0.4
            1   1   0   2   |   0.4
            1   1   1   0   |   0.6
            1   1   1   1   |   0.6
            1   1   1   2   |   0.6

            I   G   H   |   \phi_{IGH} (P(I|H,G))
            ------------------------------------------------
            0   0   0   |   0.9
            0   0   1   |   1
            0   1   0   |   0.7
            0   1   1   |   1
            0   2   0   |   0
            0   2   1   |   1
            1   0   0   |   0.1
            1   0   1   |   0
            1   1   0   |   0.3
            1   1   1   |   0
            1   2   0   |   1
            1   2   1   |   0

        '''

        phi = []
        _vars = sorted("L","Q","S","F","G","B","I","H")

        #("S","L") -> 0

        phi.append(
                    np.array([
                            [0.42,0.08],
                            [0.18,0.32]
                        ])

                    )

        #("L","B","Q","G") -> 1
        phi.append(
                    np.array([
                                [
                                    [
                                        [0,0,0],
                                        [0.6,0.2,0.2],
                                    ],
                                    [
                                        [0,0,0],
                                        [0.7,0.2,0.1],
                                    ]
                                ],
                                [
                                    [
                                        [0,0,0],
                                        [0.1,0.1,0.8],
                                    ],
                                    [
                                        [0.08,0.48,0.32],
                                        [0.02,0.12,0.06],
                                    ]
                                ]
                            ])
                        )

        #("F","Q","B","H") -> 2
        phi.append(
                    np.array([
                                [
                                    [
                                        [0.45,0.45],
                                        [0.45,0.45],
                                    ],
                                    [
                                        [0.10,0.10],
                                        [0,0.2],
                                    ]
                                ],
                                [
                                    [
                                        [0.09,0.01],
                                        [0.06,0.04],
                                    ],
                                    [
                                        [0.72,0.08],
                                        [0.48,0.32],
                                    ]
                                ]

                        ])
                    )

        #("H","B","Q","G") -> 3
        phi.append(
                    np.array([
                                [
                                    [
                                        [0.4,0.4,0.4],
                                        [0.6,0.6,0.6],
                                    ],
                                    [
                                        [0.4,0.4,0.4],
                                        [0.6,0.6,0.6],
                                    ]
                                ],
                                [
                                    [
                                        [0.4,0.4,0.4],
                                        [0.6,0.6,0.6],
                                    ],
                                    [
                                        [0.4,0.4,0.4],
                                        [0.6,0.6,0.6],
                                    ]
                                ]

                        ])
                    )

        #("I","G","H") -> 4
        phi.append(
                    np.array([
                            [
                                [0.9,1],
                                [0.7,1],
                                [0,1],
                            ],
                            [

                                [0.1,0],
                                [0.3,0],
                                [1,0],
                            ]

                        ])
                    )

        #("L") -> 5
        phi.append(np.array([1,1]))

        #("B","H","Q") -> 6
        phi.append(
                    np.array([
                            [
                                [1,1],
                                [1,1],
                            ],
                            [
                                [1,1],
                                [1,1],
                            ]
                        ])
                    )

        #("Q", "B", "G") -> 7
        phi.append(
                    np.array([
                            [
                                [1,1,1],
                                [1,1,1],
                            ],
                            [
                                [1,1,1],
                                [1,1,1],
                            ]
                        ])
                    )

        #("G", "H") -> 8
        phi.append(
                    np.array([
                                [1,1],
                                [1,1],
                        ])
                    )

        # TODO: Need to think about internal mapping of variable names to variable index

        jt = [
                0, [_vars.index("H"), _vars.index("B"), _vars.index("Q"), _vars.index("G")],
                (
                    1, [_vars.index("B"), _vars.index("H"), _vars.index("Q")],
                    [
                        2, [_vars.index("F"), _vars.index("Q"), _vas.index("B"), _vars.index("H")],
                    ]
                ),
                (
                    3, [_vars.index("G"), _vars.index("H")],
                    [
                        4, [_vars.index("I"), _vars.index("G"), _vars.index("H")]
                    ]

                ),
                (
                    5, [_vars.index("Q"),_vars.index("B"),_vars.index("G")],
                    [
                        6, [_vars.index("L"), _vars.index("Q"), _vars.index("B"), _vars.index("G")],
                        (
                            7, [_vars.index("L")],
                            [
                                8, [_vars.index("S"), _vars.index("L")]
                            ]
                        )
                    ]
                )
            ]

        phiN = bp.collect(jt, phi, []*len(phi))
        # need to set evidence here: Q=0, G=0, F=1
        np.allclose(marginalize(jt, "H"), np.array([0.4, 0.6])) == True



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

    def test_generate_deep_copy_of_factor_graph_nodes(self):
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
        values = [
                    np.random.randn(2),
                    np.random.randn(2, 4),
                    np.random.randn(4, 3, 5),
                    np.random.randn(2, 5),
                ]

        fg = [_vars, factors, values]
        fg2 = bp.copy_factor_graph(fg)
        assert_factor_graph_equal(fg, fg2)

    def test_store_nodes_in_heap(self):
        heap = []
        # push some nodes onto heap (NUM_EDGES_ADDED, CLUSTER_WEIGHT, FACTOR_ID)
        heapq.heappush(heap, (3, 4, 0))
        heapq.heappush(heap, (1, 3, 1))
        heapq.heappush(heap, (5, 3, 2))
        heapq.heappush(heap, (4, 6, 3))

        # check that heappop returns the element with the lowest value in the tuple
        assert heapq.pop(heap) == (1, 3, 1)

        # add value back to heap
        heapq.heappush(heap, (1, 3, 1))
        # add two new tuples that have the same first value but smaller second
        # value
        heapq.heappush(heap, (1, 2, 4))
        heapq.heappush(heap, (1, 1, 5))

        # ensure that tie is broken by second element
        assert heapq.pop(heap) == (1, 1, 5)
        # ensure that updated heap returns second smalles element with tie-
        # breaker
        assert heapq.pop(heap) == (1, 2, 4)

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
        heap = bp.initialize_triangulation_heap(factors)
        assert len(heap) == 4
        '''
            Entries:
            (0, 120, 0) # factor 0 has 2 neighbors (all nodes connected)
            (1, 7200, 1) # factor 1 has 3 neighbors (0-2 edge added)
            (0, 3600, 2) # factor 2 has 2 neighbors (all nodes connected)
            (1, 7200, 3) # factor 3 has 3 neighbors (0-2 edge added)
        '''
        assert heap[0] == (0, 120, 0)
        assert heap[1] == (0, 3600, 2)
        assert heap[2] == (1, 7200, 1)
        assert heap[3] == (1, 7200, 3)


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
        heap = bp.initialize_triangulation_heap(factors)
        item, heap = bp.remove_next(heap)
        assert item == (0, 120, 0)

        '''
            factors_p = [
                            ["A", "C"], # weight: 6
                            ["B", "C", "D"], # weight: 60
                            ["A", "D"] # weight: 10
                    ]
            Entries:
            (0, 3600, 1) # factor 1 has 2 neighbors (all nodes connected)
            (0, 3600, 2) # factor 2 has 2 neighbors (all nodes connected)
            (0, 3600, 3) # factor 3 has 2 neighbors (all nodes connected)
        '''
        assert len(heap) == 3
        assert heap[0] == (0, 3600, 1)
        assert heap[1] == (0, 3600, 2)
        assert heap[2] == (0, 3600, 3)

        item, heap = bp.remove_next(heap)
        assert item == (0, 3600, 1)
        '''
            factors_p = [
                            ["B", "C", "D"], # weight: 60
                            ["A", "D"] # weight: 10
                    ]
            Entries:
            (0, 600, 2) # factor 2 has 1 neighbors (already connected)
            (0, 600, 3) # factor 3 has 1 neighbors (already connected)
        '''

        assert len(heap) == 2
        assert heap[0] == (0, 600, 2)
        assert heap[1] == (0, 600, 3)

        item, heap = bp.remove_next(heap)
        assert item == (0, 600, 2)
        '''
            factors_p = [
                            ["A", "D"] # weight: 10
                    ]
            Entries:
            (0, 10, 3) # factor 3 has 0 neighbors (no connections possible)
        '''

        assert len(heap) == 1
        assert heap[0] == (0, 10, 3)


        item, heap = bp.remove_next(heap)
        assert item == (0, 10, 3)


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
        fcs = [
                [1,1,0,1,0,0,0,0],
                [0,1,1,1,0,0,0,0],
                [0,0,0,1,1,1,0,0],
                [0,0,0,0,0,1,1,1]
        ]
        ecs = __gibs_elem_cycles(edges, fcs)
        assert len(ecs) == 10
        assert ecs[0] == [1,1,0,1,0,0,0,0]
        assert ecs[1] == [1,0,1,0,0,0,0,0]
        assert ecs[2] == [0,1,1,1,0,0,0,0]
        assert ecs[3] == [1,1,0,0,1,1,0,0]
        assert ecs[4] == [0,1,1,0,1,1,0,0]
        assert ecs[5] == [0,0,0,1,1,1,0,0]
        assert ecs[6] == [1,1,0,0,1,0,1,1]
        assert ecs[7] == [0,1,1,0,1,0,1,1]
        assert ecs[8] == [0,0,0,1,1,0,1,1]
        assert ecs[9] == [0,0,0,0,0,1,1,1]



    def test_assert_triangulated(self):
        '''
            f0----f1
            |      |
            |      |
            f2----f3
        '''
        factors = [
                    ["A", "B"],
                    ["B"],
                    ["B", "C"],
                    ["A", "C"]
                ]
        tri0 = []
        self.assertRaises(AssertionError, assert_triangulated(factors, tri0))

        tri1 = [[0,1,2,3]]
        assert_triangulated(factors, tri1)

    def test_triangulate_factor_graph(self):
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
        values = [
                    np.random.randn(2),
                    np.random.randn(2, 4),
                    np.random.randn(4, 3, 5),
                    np.random.randn(2, 5),
                ]
        fg = [_vars, factors, values]
        tri = bp.find_triangulation(fg, sizes)

        # only 2 cliques should be returned as they are maximal and complete
        assert len(tri) == 2

        assert tri[0] == [0,1,3]
        assert tri[1] == [1,2,3]

        assert_triangulated(fg[1], tri)

    def test_join_cliques_into_junction_tree(self):
        '''
            Example taken from section 4.4.3 (Huang and Darwiche, 1996)

            factors: 0 -> A, 1 -> B, 2 -> C, 3 -> D, 4 -> E, 5 -> F, 6 -> G,
                        7 -> H
        '''

        cliques = [
                    [0,1,3],#["A","B","D"]
                    [0,2,4],#["A","C","E"],
                    [0,3,4],#["A","D","E"],
                    [2,4,6],#["C","E","G"],
                    [3,4,5],#["D","E","F"],
                    [4,6,7],#["E","G","H"]
                ]
        jt0 = bp.construct_junction_tree(cliques)

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
                )
                (
                    5, [0,4],
                    [
                        6, [0,2,4]
                        (
                            7, [2,4],
                            [
                                8, [2,4,6]
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

        assert_junction_tree_equal(jt0, jt1)

class TestJunctionInference(unittest.TestCase):
    def setUp(self):
        self.jt = [
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
                    )
                    (
                        5, [0,4],
                        [
                            6, [0,2,4]
                            (
                                7, [2,4],
                                [
                                    8, [2,4,6]
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


        _vars = {
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
                    ["A"],
                    ["A","B"],
                    ["A","C"],
                    ["B","D"],
                    ["C","E"],
                    ["D","E","F"],
                    ["C","G"],
                    ["E","G","H"],
        ]

        values = [
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
                                    [0.9,0.1],
                                    [0.8,0.2]
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

        self.fg = [_vars,factors,values]


    def test_transformation(self):
        tree = bp.convert(self.fg)
        assert_junction_tree_equal(self.jt, tree)


    def test_initialize_potentials(self):
        # this initialization is important to get proper messages passed
        # discussed on page 111 of Bayesian Reasoning and Machine Learnging
        # discussed on page 723 of Machine Learning: A Probabilistic Perspective
        # when evaluating potentials based on factor values, we can ignore
        # the additional values added by the junction tree conversion because
        # factors only depend on the variable arguments (local domain) of the
        # factor (Aji and McEliece)
        phi = bp.init_tree(self.jt, self.fg)
        assert_potentials_equal(
                                    phi[6:8], # clusters ACE and CE
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
        init_phi = bp.init_tree(self.jt, self.fg)
        phi = bp.hugin(junction_tree, potentials, bp.sum_product)
        assert_potentials_equal([phi[2]], [
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

    def test_marginalization(self):
        init_phi = bp.init_tree(self.jt, self.fg)
        phi = bp.hugin(junction_tree, potentials, bp.sum_product)
        np.testing.assert_array_equal(
                                        bp.compute_marginal(phi, range(8), [0]),
                                        np.array([0.500,0.500])
                                    )
        np.testing.assert_array_equal(
                                        bp.compute_marginal(phi, range(8), [3]),
                                        np.array([0.320,0.680])
        )

class TestJTTraversal(unittest.TestCase):
    def setUp(self):
        self.tree = [
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

    def test_flatten_tree(self):
        assert list(bp.traverse(self.tree)) == [
                                                    0, [0, 2, 4],
                                                    1, [0, 2],
                                                    3, [4],
                                                    2, [0, 1, 2],
                                                    4, [3, 4],
                                                    5, [3],
                                                    6, [1, 2, 3]
                                                ]

    def test_get_clique_keys(self):
        assert bp.get_clique_keys(self.tree, 0) == [0, 2, 4]
        assert bp.get_clique_keys(self.tree, 1) == [0, 2]
        assert bp.get_clique_keys(self.tree, 2) == [0, 1, 2]
        assert bp.get_clique_keys(self.tree, 3) == [4]
        assert bp.get_clique_keys(self.tree, 4) == [3, 4]
        assert bp.get_clique_keys(self.tree, 5) == [3]
        assert bp.get_clique_keys(self.tree, 6) == [1, 2, 3]
        assert bp.get_clique_keys(self.tree, 7) == None
