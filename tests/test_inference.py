import numpy as np
from junctiontree import beliefpropagation as bp
import copy
import junctiontree.junctiontree as jt
from .util import assert_potentials_equal


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
                            bp.sum_product.einsum(
                                _tree.propagate(values)[6],
                                [0, 1, 2],
                                [2]
                            ),
                            np.array([0.824,0.176]),
                            atol=0.01
    )
    np.testing.assert_allclose(
                            bp.sum_product.einsum(
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
                        bp.sum_product.einsum(
                            prop_values[5],
                            [0,1,2],
                            [2]
                        ),
                        np.array([0.507,0.493]),
                        atol=0.001
    )
