import numpy as np

import bp
import unittest


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

    pass


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
        bp.hugin2(tree, potentials, bp.sum_product)
    )
    pass


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
        assert np.allclose(bp.compute_marginal(phiABD, [0]), np.array([0.500, 0.500])) == True
        # marginal probability of D, P(D)
        assert np.allclose(np.array([0.32,0.68]), np.array([0.320, 0.680])) == True


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
                np.random.randn(4, 5, 6),
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
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1])
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        assert_consistent(jt, phiN)

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

        data = {0: 1, 1: 2, 2: 3, 4: 0}

        likelihood = {
                        np.array([0,1,0,0]),
                        np.array([1,1,1,1,1,1,1,1]),
                        np.array([0,0,0,1,0]),
                        np.array([1,1,1]),
                        np.array([1,0,0,0,0,0])
                    }

        likelihood, phiN = bp.observe(jt, phi, data, likelihood, "update")
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([0,0,1,0,0,0,0,0]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        assert_consistent(jt, phiN)

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

        data = {0: 1, 1: 2, 2: 3, 3: 2, 4: 0}

        likelihood = {
                        np.array([0,1,0,0]),
                        np.array([1,1,1,1,1,1,1,1]),
                        np.array([0,0,0,1,0]),
                        np.array([1,1,1]),
                        np.array([1,0,0,0,0,0])
                    }


        likelihood, phiN = bp.observe(jt, phi, likelihood, data, "update")
        np.testing.assert_array_equal(likelihood[0], np.array([0,1,0,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([0,0,1,0,0,0,0,0]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([0,0,1])
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        assert_consistent(jt, phiN)

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

        data = {0: 2, 2: 3, 4: 0}

        likelihood = {
                        np.array([0,1,0,0]),
                        np.array([1,1,1,1,1,1,1,1]),
                        np.array([0,0,0,1,0]),
                        np.array([1,1,1]),
                        np.array([1,0,0,0,0,0])
                    }

        likelihood, phiN = bp.observe(jt, phi, data, likelihood, "retract")
        np.testing.assert_array_equal(likelihood[0], np.array([0,0,1,0]))
        np.testing.assert_array_equal(likelihood[1], np.array([1,1,1,1,1,1,1,1]))
        np.testing.assert_array_equal(likelihood[2], np.array([0,0,0,1,0]))
        np.testing.assert_array_equal(likelihood[3], np.array([1,1,1]))
        np.testing.assert_array_equal(likelihood[4], np.array([1,0,0,0,0,0]))
        assert_consistent(jt, phiN)

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
    def test_can_locate_clique_containing_variable(self):
        tree = [0, [0,1], (1, [1], [2, [1,2]])]
        clique, _vars = bp.get_clique(tree, 2)
        assert clique == 2

    def test_assign_var_to_cluster(self):
        pass

    def test_initialize_potentials(self):
        # this initialization is important to get proper messages passed
        # discussed on page 111 of Bayesian Reasoning and Machine Learnging
        # discussed on page 723 of Machine Learning: A Probabilistic Perspective
        pass

    def test_convert_factor_graph_to_junction_tree(self):
        pass
