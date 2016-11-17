import numpy as np

import bp


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
