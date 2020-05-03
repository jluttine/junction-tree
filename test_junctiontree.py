import numpy as np
import pytest

import junctiontree as jt


def test_create_junction_tree():

    # Just test that multiple-character names are supported
    g = jt.create_junction_tree(
        [["foo"], ["foo", "bar"]],
        {"foo": 3, "bar": 4},
    )

    assert g.clique_tree.factor_to_maxclique == [0, 0]

    return


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

    pytest.skip("These tests aren't relevant yet")

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
