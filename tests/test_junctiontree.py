import numpy as np
import junctiontree as jt
from .util import assert_sum_product
import unittest


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
