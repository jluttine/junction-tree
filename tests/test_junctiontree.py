import numpy as np
import junctiontree as jt
from .util import assert_sum_product, assert_potentials_equal
import unittest
from junctiontree import computation as comp
import copy

def test_one_scalar_node():
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


def test_one_matrix_node():
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


def test_one_child_node_with_all_variables_shared():
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


def test_one_child_node_with_one_common_variable():
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


def test_one_child_node_with_no_common_variable():
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


def test_one_grand_child_node_with_no_variable_shared_with_grand_parent():
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


def test_one_grand_child_node_with_variable_shared_with_grand_parent():
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


def test_two_children_with_no_variable_shared():
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


def test_two_children_with_shared_variable():
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


def test_two_children_with_3D_tensors():
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


def test_clique_graph():

    def _run(factors, sizes, maxcliques, factor_to_maxclique, function):
        g = jt.CliqueGraph(
            maxcliques=maxcliques,
            factor_to_maxclique=factor_to_maxclique,
            factor_graph=jt.FactorGraph(
                factors=factors,
                sizes=sizes
            )
        )

        xs = [
            np.random.randn(
                *[sizes[key] for key in factor]
            )
            for factor in factors
        ]

        ys = g.evaluate(xs)
        yhs = function(xs)

        assert len(ys) == len(yhs)

        for (y, yh) in zip(ys, yhs):
            np.testing.assert_allclose(y, yh)

        return

    _run(
        factors=[ ['a', 'b'], ['b', 'c'] ],
        sizes={
            'a': 2,
            'b': 3,
            'c': 4,
            'd': 5,
        },
        maxcliques=[ ['a', 'b'], ['b', 'c'] ],
        factor_to_maxclique=[0, 1],
        function=lambda xs: [
            xs[0],
            xs[1],
        ]
    )

    _run(
        factors=[ ['a', 'b'], ['a'] ],
        sizes={
            'a': 2,
            'b': 3,
            'c': 4,
            'd': 5,
        },
        maxcliques=[ ['a', 'b'] ],
        factor_to_maxclique=[0, 0],
        function=lambda xs: [
            np.einsum('a,ab->ab', xs[1], xs[0])
        ]
    )

    _run(
        factors=[ ['a', 'b'], ['b', 'c'], ['c', 'd'], ['a', 'd'] ],
        sizes={
            'a': 2,
            'b': 3,
            'c': 4,
            'd': 5,
        },
        maxcliques=[
            ['a', 'b', 'c'],
            ['a', 'c', 'd'],
        ],
        factor_to_maxclique=[0, 0, 1, 1],
        function=lambda xs: [
            np.einsum('ab,bc->abc', xs[0], xs[1]),
            np.einsum('cd,ad->acd', xs[2], xs[3]),
        ]
    )

    # Test a clique with a key that doesn't exist in its factors
    _run(
        factors=[ ['a', 'b'], ['b', 'c'], ['c', 'd'], ['a', 'e'] ],
        sizes={
            'a': 2,
            'b': 3,
            'c': 4,
            'd': 5,
            'e': 6,
        },
        maxcliques=[
            ['a', 'b', 'c'],
            ['a', 'c', 'd', 'e'],
            ['a', 'd', 'e'],
        ],
        factor_to_maxclique=[0, 0, 1, 2],
        function=lambda xs: [
            np.einsum('ab,bc->abc', xs[0], xs[1]),
            np.einsum('ae,cd->acde', [[1]], xs[2]),
            np.einsum('d,ae->ade', [1], xs[3]),
        ]
    )

    return


tree = [
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

node_list = [
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
    ["A"],
    ["A", "B"],
    ["A", "C"],
    ["B", "D"],
    ["C", "E"],
    ["C", "G"],
    ["D", "E", "F"],
    ["E", "G", "H"],

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


def test_transformation():
    _tree = jt.create_junction_tree(factors, key_sizes)
    prop_values = _tree.propagate(values)

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
                            comp.sum_product.einsum(
                                _tree.propagate(values)[6],
                                [0, 1, 2],
                                [2]
                            ),
                            np.array([0.824,0.176]),
                            atol=0.01
    )
    np.testing.assert_allclose(
                            comp.sum_product.einsum(
                                _tree.propagate(values)[7],
                                [0, 1, 2],
                                [2]
                            ),
                            np.array([ 0.104,  0.896]),
                            atol=0.01
    )


def test_initialize_potentials():
    j_tree = jt.JunctionTree(
                tree,
                node_list[6:],
                jt.CliqueGraph(
                    maxcliques=node_list[:6],
                    factor_to_maxclique=[0, 1, 3, 1, 3, 4, 2, 5],
                    factor_graph=jt.FactorGraph(
                        factors=factors,
                        sizes=key_sizes
                    )
                )
    )

    init_phi  = j_tree.clique_tree.evaluate(values)

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


def test_global_propagation():
    _tree = jt.create_junction_tree(factors, key_sizes)

    prop_values = _tree.propagate(values)
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


def test_global_propagation_with_observations():

    # Example from: http://mensxmachina.org/files/software/demos/jtreedemo.html

    _key_sizes = {
                    "cloudy": 2,
                    "sprinkler": 2,
                    "rain": 2,
                    "wet_grass": 2
                }

    _factors = [
                ["cloudy"],
                ["cloudy", "sprinkler"],
                ["cloudy", "rain"],
                ["rain", "sprinkler", "wet_grass"]
    ]

    _values = [
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

    tree = jt.create_junction_tree(_factors, _key_sizes)

    # grass is wet
    tree.clique_tree.factor_graph.sizes["wet_grass"] = 1
    cond_values = copy.deepcopy(_values)
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


def test_inference():

    # Example from http://pages.cs.wisc.edu/~dpage/cs731/lecture5.ppt

    _key_sizes = {
                    "A": 2,
                    "B": 2,
                    "C": 2,
                    "D": 2,
                    "E": 2,
                    "F": 2
                }
    _factors = [
                ["A"],
                ["B","A"],
                ["C","A"],
                ["B","D"],
                ["C","E"],
                ["D","E","F"]
    ]

    _values = [
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

    tree = jt.create_junction_tree(_factors, _key_sizes)

    prop_values = tree.propagate(_values)

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
                        comp.sum_product.einsum(
                            prop_values[5],
                            [0,1,2],
                            [2]
                        ),
                        np.array([0.507,0.493]),
                        atol=0.001
    )


def test_marginalize_variable():
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
    np.testing.assert_allclose(comp.sum_product.einsum(phiABD, [0,1,2], [0]), np.array([0.500, 0.500]))
    # marginal probability of D, P(D)
    np.testing.assert_allclose(np.array([0.32,0.68]), np.array([0.320, 0.680]))


def test_pass_message():
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

    phi2n = comp.sum_product.project(phi12, [0,1], [1])
    np.testing.assert_allclose(phi2n, np.array([1,1]))
    phi23 = comp.sum_product.absorb(phi23, [0,1], phi2, phi2n, [1])
    np.testing.assert_allclose(phi23, np.array([
                                    [0.03,0.07],
                                    [0.45,0.45]
                                ]))

    phi2nn = comp.sum_product.project(phi23, [0,1], [0])
    np.testing.assert_allclose(phi2nn, np.array([0.1, 0.9]))
    phi12 = comp.sum_product.absorb(phi12, [0,1], phi2n, phi2nn, [1])
    np.testing.assert_allclose(phi12, np.array([
                                    [0.04,0.72],
                                    [0.06,0.18]
                                ]))



@unittest.skip("Tests are not relevant yet")
def test_junction_tree():

    def _run(maxcliques, sizes, tree):
        g = jt.CliqueGraph(
            maxcliques=maxcliques,
            factor_to_maxclique=list(range(len(maxcliques))),
            factor_graph=jt.FactorGraph(
                factors=maxcliques,
                sizes=sizes,
            )
        )
        t = g.create_junction_tree()

        assert t == tree

        return

    raise NotImplementedError("These tests aren't relevant yet.")

    _run(
        maxcliques=[
            ['a', 'b'],
            ['c', 'd'],
            ['b', 'c'],
        ],
        sizes={
            'a': 2,
            'b': 3,
            'c': 4,
            'd': 5,
        },
        tree=[
            1,
            ['c', 'd'],
            (
                4,
                ['c'],
                [
                    2,
                    ['b', 'c'],
                    (
                        3,
                        ['b'],
                        [
                            0,
                            ['a', 'b'],
                        ],
                    )
                ]
            )
        ]
    )

    _run(
        maxcliques=[
            ['a', 'b'],
            ['b', 'c'],
        ],
        sizes={
            'a': 2,
            'b': 3,
            'c': 4,
        },
        tree=[
            0,
            ['a', 'b'],
            (
                2,
                ['b'],
                [
                    1,
                    ['b', 'c']
                ]
            )
        ]
    )

    return


