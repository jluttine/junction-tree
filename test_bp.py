import numpy as np

import bp
from bp import JunctionTree
import unittest
import networkx as nx
import itertools
import heapq
import copy


# Tests here using pytest


def assert_junction_tree_equal(t1, t2):
    """Test equality of two junction trees

    Both trees contain same edges and cliques have same keys

    """


    def __build_dict(trees):
        # dict is: clique_keys -> set of each tuple of neighbor_keys
        d = {}
        for tree in trees:
            stack = [tree]
            d[tuple(tree[1])] = set()
            while stack:
                tree = stack.pop()
                for child in reversed(tree[2:]):
                    d[tuple(tree[1])].add(tuple(child[1]))
                    # child clique entry initialized
                    d[tuple(child[2][1])] = set([tuple(child[1])])
                    # separator entered in dictionary
                    d[tuple(child[1])] = set([tuple(tree[1]),tuple(child[2][1])])
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

    return G.edges(), cycles

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

def brute_force_sum_product(junction_tree, potentials):
    """Compute brute force sum-product with einsum """

    # Function to compute the sum-product with brute force einsum
    arrays_keys = get_arrays_and_keys(junction_tree.get_struct(), potentials)
    f = lambda output_keys: np.einsum(*(arrays_keys + [output_keys]))

    def __run(tree, p, f, res=[]):
        res.append(f(tree[1]))
        for child_tree in tree[2:]:
            __run(child_tree, p, f, res)
        return res

    return __run(junction_tree.get_struct(), potentials, f)

def assert_sum_product(tree, potentials):
    """ Test hugin vs brute force sum-product """
    assert_potentials_equal(
        brute_force_sum_product(tree, potentials),
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
                        potentials_consistent(
                                            potentials[c_ix1],
                                            c_vars1,
                                            potentials[c_ix2],
                                            c_vars2
                        )
                        for c_ix1,
                            c_vars1,
                            c_ix2,
                            c_vars2 in bp.generate_potential_pairs(tree.get_struct())
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
                bp.compute_marginal(
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

        phi2n = bp.sum_product.project(phi12, [0,1], [1])
        np.allclose(phi2n, np.array([1,1])) == True
        phi23 = bp.sum_product.absorb(phi23, [0,1], phi2, phi2n, [1])
        np.allclose(phi23, np.array([
                                        [0.03,0.07],
                                        [0.45,0.45]
                                    ])) == True

        phi2nn = bp.sum_product.project(phi23, [0,1], [0])
        np.allclose(phi2nn, np.array([0.9, 0.1])) == True
        phi12 = bp.sum_product.absorb(phi12, [0,1], phi2n, phi2nn, [0])
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

        phiN = bp.collect(
                            jt,
                            {"V1": 0, "V2": 1, "V3": 2},
                            phi,
                            [0]*len(phi),
                            bp.sum_product
        )
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


        phiN = bp.distribute(
                                jt,
                                {"V1": 0, "V2": 1, "V3": 2},
                                phi,
                                [0]*len(phi),
                                bp.sum_product
        )
        np.allclose(phiN[2], np.array([
                                        [0.04,0.72],
                                        [0.06,0.18]
                                    ]))


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
                        ],
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
                            ],

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
                            ],
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
                            ],
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
                                2:2,
                                3:3,
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
                            ],
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
                            ],
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

        likelihood, phi0 = bp.observe(jt, phi, data)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        phi1 = bp.hugin(jt, phi0, bp.sum_product)
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
        for var in jt.get_vars():
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

        likelihood, phi0 = bp.observe(jt, phi, data0)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        phi1 = bp.hugin(jt, phi0, bp.sum_product)
        assert_junction_tree_consistent(jt, phi1)

        data = {0: 1, 1: 2, 2: 3, 4: 0}

        likelihood, phi2 = bp.observe(jt, phi1, data)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([0,0,1,0,0,0,0,0]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        phi3 = bp.hugin(jt, phi2, bp.sum_product)
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
        for var in jt.get_vars():
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

        likelihood, phi0 = bp.observe(jt, phi, data0)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        phi1 = bp.hugin(jt, phi0, bp.sum_product)
        assert_junction_tree_consistent(jt, phi1)

        data = {0: 1, 1: 2, 2: 3, 3: 2, 4: 0}


        likelihood, phi2 = bp.observe(jt, phi1, data)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([0,0,1,0,0,0,0,0]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([0,0,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        phi3 = bp.hugin(jt, phi2, bp.sum_product)
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
        for var in jt.get_vars():
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

        likelihood, phi0 = bp.observe(jt, phi, data0)
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        phi1 = bp.hugin(jt, phi0, bp.sum_product)
        assert_junction_tree_consistent(jt, phi1)

        data = {0: 2, 2: 3, 4: 0}

        likelihood, phi2 = bp.observe(jt, phi1, data, "retract")
        np.testing.assert_array_equal(likelihood[0], np.array([0,0,1,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        phi3 = bp.hugin(jt, phi2, bp.sum_product)
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
        for var in jt.get_vars():
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

        phi0 = []
        _vars = sorted(["L","Q","S","F","G","B","I","H"])

        #("H","B","Q","G") -> 0
        phi0.append(
                    np.array(
                        [
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
                        ]
                    )
        )

        #("B","H","Q") -> 1
        phi0.append(
                    np.array(
                        [
                            [
                                [1,1],
                                [1,1],
                            ],
                            [
                                [1,1],
                                [1,1],
                            ]
                        ]
                    )
        )

        #("F","Q","B","H") -> 2
        phi0.append(
                    np.array(
                        [
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
                        ]
                    )
        )

        #("G", "H") -> 3
        phi0.append(
                    np.array(
                        [
                            [1,1],
                            [1,1],
                            [1,1]
                        ]
                    )
        )

        #("I","G","H") -> 4
        phi0.append(
                    np.array(
                        [
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

                        ]
                    )
        )

        #("Q", "B", "G") -> 5
        phi0.append(
                    np.array(
                        [
                            [
                                [1,1,1],
                                [1,1,1],
                            ],
                            [
                                [1,1,1],
                                [1,1,1],
                            ]
                        ]
                    )
        )

        #("L","Q","B","G") -> 6
        phi0.append(
                    np.array(
                        [
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
                        ]
                    )
        )

        #("L") -> 7
        phi0.append(np.array([1,1]))



        #("S","L") -> 8
        phi0.append(
                    np.array(
                        [
                            [0.42,0.08],
                            [0.18,0.32]
                        ]
                    )

        )


        # ['B', 'F', 'G', 'H', 'I', 'L', 'Q', 'S']
        # [0    1    2    3    4    5    6    7  ]

        jt = JunctionTree(  {
                                "L":2,
                                "Q":2,
                                "S":2,
                                "F":2,
                                "G":3,
                                "B":2,
                                "I":2,
                                "H":2
                            },
                            [
                                0, ["H","B","Q","G"],
                                (
                                    1, ["B", "H", "Q"],
                                    [
                                        2, ["F","Q","B","H"],
                                    ]
                                ),
                                (
                                    3, ["G","H"],
                                    [
                                        4, ["I", "G", "H"]
                                    ]

                                ),
                                (
                                    5, ["Q","B","G"],
                                    [
                                        6, ["L","Q","B","G"],
                                        (
                                            7, ["L"],
                                            [
                                                8, ["S","L"]
                                            ]
                                        )
                                    ]
                                )
                            ]
        )


        # need to set evidence here: Q=0, G=0, F=1
        data = {"Q":0, "G":0, "F":1}
        likelihood, phi1 = bp.observe(jt, phi0, data)

        phi2 = bp.collect(
                        jt.get_struct(),
                        jt.get_label_order(),
                        phi1,
                        [0]*len(phi1),
                        bp.sum_product
        )

        np.allclose(bp.marginalize(jt, phi2, "H"), np.array([0.4, 0.6])) == True



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
        tri, ics = bp.find_triangulation(fg[1], fg[0])

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

        tri, ics = bp.find_triangulation(factors, _vars)
        assert_triangulated(factors, tri)
        cliques = bp.identify_cliques(ics)


        assert len(cliques) == 6


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

        tri, ics = bp.find_triangulation(factors, _vars)
        assert_triangulated(factors, tri)
        cliques = bp.identify_cliques(ics)

        clique_sets = [set(c) for c in cliques]

        assert len(clique_sets) == 5

        assert set(["C","D"]) in clique_sets
        assert set(["G","I","D"]) in clique_sets
        assert set(["G","S","I"]) in clique_sets
        assert set(["G","J","S","L"]) in clique_sets
        assert set(["H","G","J"]) in clique_sets


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

        clique_sets = [set(c) for c in cliques]

        assert len(clique_sets) == 6

        assert set(["E","G","H"]) in clique_sets
        assert set(["C","E","G"]) in clique_sets
        assert set(["D","E","F"]) in clique_sets
        assert set(["A","C","E"]) in clique_sets
        assert set(["A","B","D"]) in clique_sets
        assert set(["A","D","E"]) in clique_sets

    def test_join_trees_with_single_cliques(self):
        tree1 = [0, [0,1,2]]
        sepset = [2, [2]]
        tree2 = [1, [2,3,4]]

        output = bp.merge_trees(
                            tree1,
                            tree1[0],
                            tree2,
                            tree2[0],
                            sepset[0],
                            sepset[1]
        )

        merged_tree = [
                0, [0,1,2],
                (
                    2, [2],
                    [
                        1, [2,3,4]
                    ]
                )
            ]


        assert_junction_tree_equal([output], [merged_tree])

    def test_join_tree_with_single_clique_to_multiclique_tree(self):
        tree1 = [0, [0,2,4]]
        sepset = [3, [4]]
        tree2 = [4, [3,4], (5, [3],[6, [1,2,3]])]

        output = bp.merge_trees(
                            tree1,
                            tree1[0],
                            tree2,
                            tree2[0],
                            sepset[0],
                            sepset[1]
        )
        merged_tree = [
                    0, [0,2,4],
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

        assert_junction_tree_equal([output], [merged_tree])


    def test_join_tree_with_multiple_cliques_to_tree_with_multiple_cliques(self):
        tree1 = [0, [0,2,4], (1, [0,2], [2, [0,1,2]])]
        sepset = [3, [4]]
        tree2 = [4, [3,4], (5, [3],[6, [1,2,3]])]

        output = bp.merge_trees(
                            tree1,
                            tree1[0],
                            tree2,
                            tree2[0],
                            sepset[0],
                            sepset[1]
        )
        merged_tree = [
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

        assert_junction_tree_equal([output], [merged_tree])

    def test_change_root(self):
        tree1 = [
                    4,[0,8],
                    (
                        5, [0],
                        [
                            0, [0,2,4],
                            (
                                1, [2],
                                [
                                    2, [1,2]
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
                    0, [0,2,4],
                    (
                        1, [2],
                        [
                            2, [1,2]
                        ]
                    ),
                    (
                        5, [0],
                        [
                            4, [0,8]
                        ]
                    )
                ]

        assert output == tree2
        assert_junction_tree_equal([tree1], [output])


        output = bp.change_root(copy.deepcopy(tree1), 2)

        tree3 = [
                    2, [1,2],
                    (
                        1, [2],
                        [
                            0, [0,2,4],
                            (
                                5, [0],
                                [
                                    4,[0,8]
                                ]
                            )
                        ]
                    )
                ]


        assert output == tree3
        assert_junction_tree_equal([tree1], [output])

        tree4 = [
                    4,[0,8],
                    (
                        5, [0],
                        [
                            0, [0,2,4],
                            (
                                1, [2],
                                [
                                    2, [1,2]
                                ]
                            ),
                            (
                                3, [4],
                                [
                                    6, [4,9]
                                ]
                            )
                        ]
                    ),
                    (
                        7, [8],
                        [
                            8, [8, 10]
                        ]
                    )
                ]


        output = bp.change_root(tree4, 2)

        # need to return re-rooted tree and attach nodes/separators higher up in
        # the tree to the tree that is being built.

        # each move up the treei is changing the root at the current level of the tree ??

        tree5 = [
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

        assert_junction_tree_equal([tree4], [output])

    def test_join_trees_with_multiple_cliques_with_first_nested(self):
        tree1 = [4,[0,8], (5, [0], [0, [0,2,4], (1, [2], [2, [1,2]])])]
        sepset = [3, [4]]
        tree2 = [8, [4,5,6], (9, [6], [10, [6,7]])]

        output = bp.merge_trees(
                            tree1,
                            0,
                            tree2,
                            8,
                            sepset[0],
                            sepset[1]
        )
        merged_tree = [
                    4, [0,8],
                    (
                        5, [0],
                        [
                            0, [0,2,4],
                            (
                                1, [2],
                                [
                                    2, [1,2]
                                ]
                            ),
                            (
                                3, [4],
                                [
                                    8, [4,5,6],
                                    (
                                        9, [6],
                                        [
                                            10, [6,7]
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]

        assert_junction_tree_equal([output], [merged_tree])


    def test_join_trees_with_multiple_cliques_with_second_nested(self):
        tree1 = [0, [0,2,4], (1, [0,2], [2, [0,1,2]])]
        sepset = [3, [4]]
        tree2 = [6, [3,5,8], (7, [5], [8, [4,5,6], (9, [6], [10, [6,7]])])]

        output = bp.merge_trees(
                            tree1,
                            0,
                            tree2,
                            8,
                            sepset[0],
                            sepset[1]
        )
        merged_tree = [
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
                            8, [4,5,6],
                            (
                                9, [6],
                                [
                                    10, [6,7]
                                ]
                            ),
                            (
                                7, [5],
                                [
                                    6, [3,5,8]
                                ]
                            )
                        ]
                    )
        ]

        assert_junction_tree_equal([output], [merged_tree])

    def test_join_trees_with_multiple_cliques_with_both_nested(self):
        tree1 = [4,[0,8], (5, [0], [0, [0,2,4], (1, [2], [2, [1,2]])])]
        sepset = [3, [4]]
        tree2 = [6, [3,5], (7, [5], [8, [4,5,6], (9, [6], [10, [6,7]])])]

        output = bp.merge_trees(
                            tree1,
                            0,
                            tree2,
                            8,
                            sepset[0],
                            sepset[1]
        )

        merged_tree = [
                    4, [0,8],
                    (
                        5, [0],
                        [
                            0, [0,2,4],
                            (
                                1, [2],
                                [
                                    2, [1,2]
                                ]
                            ),
                            (
                                3, [4],
                                [
                                    8, [4,5,6],
                                    (
                                        9, [6],
                                        [
                                            10, [6,7]
                                        ]
                                    ),
                                    (
                                        7, [5],
                                        [
                                            6, [3,5]
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]


        assert_junction_tree_equal([output], [merged_tree])

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

        out_tree = JunctionTree.map_vars(in_tree, var_lookup)

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

        assert_junction_tree_equal([out_tree], [test_tree])

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

        out_tree = JunctionTree.map_vars(in_tree, {v:k for k, v in var_lookup.items()})

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

        assert_junction_tree_equal([out_tree], [test_tree])

    def test_join_cliques_into_junction_tree(self):
        """
            test_join_cliques_into_junction_tree

            Example taken from section 4.4.3 (Huang and Darwiche, 1996)

        """

        var_sizes = {
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
        trees = bp.construct_junction_tree(cliques, var_sizes)

        assert len(trees) == 1

        jt0 = JunctionTree(var_sizes, trees)

        # expected junction tree

        jt1 = [
                [
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
            ]

        assert_junction_tree_equal(jt0.get_struct(), jt1)

class TestJunctionTreeInference(unittest.TestCase):
    def setUp(self):
        self.trees = [
                        [
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
                    ["A", "B"],
                    ["A", "C"],
                    ["B", "D"],
                    ["C", "E"],
                    ["C", "G"],
                    ["G", "E", "H"],
                    ["D", "E", "F"]
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

        self.fg = [_vars,factors,values]


    def test_transformation(self):
        jt = JunctionTree.from_factor_graph(self.fg)
        assert_junction_tree_equal(self.trees, jt.get_struct())


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

    def test_bf_traverse(self):
        assert list(bp.bf_traverse(self.tree)) == [
                                                    0, [0, 2, 4],
                                                    1, [0, 2],
                                                    3, [4],
                                                    2, [0, 1, 2],
                                                    4, [3, 4],
                                                    5, [3],
                                                    6, [1, 2, 3]
                                                ]

    def test_df_traverse(self):
        assert list(bp.df_traverse(self.tree)) == [
                                                    0, [0, 2, 4],
                                                    1, [0, 2],
                                                    2, [0, 1, 2],
                                                    3, [4],
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

    def test_generate_potential_pairs(self):
        tree = [
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


        assert bp.generate_potential_pairs(tree) == [
                                                            (0, [0,3,4], 1, [0,3]),
                                                            (0, [0,3,4], 3, [3,4]),
                                                            (0, [0,3,4], 5, [0,4]),
                                                            (1, [0,3], 2, [0,1,3]),
                                                            (3, [3,4], 4, [3,4,5]),
                                                            (5, [0,4], 6, [0,2,4]),
                                                            (6, [0,2,4], 7, [2,4]),
                                                            (7, [2,4], 8, [2,4,6]),
                                                            (8, [2,4,6], 9, [4,6]),
                                                            (9, [4,6], 10, [4,6,7])
        ]
