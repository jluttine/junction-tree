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
    np.testing.assert_allclose(t1[0], t2[0])
    assert t1[1] == t2[1]

    assert len(t1) == len(t2)
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


def test_hugin():

    #
    # One scalar node
    #
    x = np.random.randn()
    x_keys = []
    tree = [
        x, x_keys
    ]
    assert_junction_tree_equal(
        bp.hugin(tree, bp.sum_product),
        brute_force_sum_product(tree)
    )

    #
    # One matrix node
    #
    x = np.random.randn(2, 3)
    x_keys = [3, 5]
    tree = [
        x, x_keys
    ]
    assert_junction_tree_equal(
        bp.hugin(tree, bp.sum_product),
        brute_force_sum_product(tree)
    )

    #
    # One child node with all variables shared
    #
    x = np.random.randn(2, 3)
    x_keys = [3, 5]
    y = np.random.randn(3, 2)
    y_keys = [5, 3]
    xy = np.ones((3, 2))
    xy_keys = [5, 3]
    tree = [
        x, x_keys,
        (
            xy, xy_keys,
            [
                y, y_keys
            ]
        )
    ]
    assert_junction_tree_equal(
        bp.hugin(tree, bp.sum_product),
        brute_force_sum_product(tree)
    )

    #
    # One child node with one common variable
    #
    x = np.random.randn(2, 3)
    x_keys = [3, 5]
    y = np.random.randn(3, 4)
    y_keys = [5, 9]
    xy = np.ones((3,))
    xy_keys = [5]
    tree = [
        x, x_keys,
        (
            xy, xy_keys,
            [
                y, y_keys
            ]
        )
    ]
    assert_junction_tree_equal(
        bp.hugin(tree, bp.sum_product),
        brute_force_sum_product(tree)
    )

    #
    # One child node with no common variable
    #
    x = np.random.randn(2)
    x_keys = [3]
    y = np.random.randn(3)
    y_keys = [9]
    xy = np.ones(())
    xy_keys = []
    tree = [
        x, x_keys,
        (
            xy, xy_keys,
            [
                y, y_keys
            ]
        )
    ]
    assert_junction_tree_equal(
        bp.hugin(tree, bp.sum_product),
        brute_force_sum_product(tree)
    )

    #
    # One grand child node
    #
    x = np.random.randn(2, 3)
    x_keys = [3, 5]
    xy = np.ones((3,))
    xy_keys = [5]
    y = np.random.randn(3, 4)
    y_keys = [5, 9]
    yz = np.ones((4,))
    yz_keys = [9]
    z = np.random.randn(4, 5)
    z_keys = [9, 1]
    tree = [
        x, x_keys,
        (
            xy, xy_keys,
            [
                y, y_keys,
                (
                    yz, yz_keys,
                    [
                        z, z_keys
                    ]
                )
            ]
        )
    ]
    assert_junction_tree_equal(
        bp.hugin(tree, bp.sum_product),
        brute_force_sum_product(tree)
    )

    #
    # Two children
    #
    x = np.random.randn(2, 3)
    x_keys = [3, 5]
    xy = np.ones((3,))
    xy_keys = [5]
    y = np.random.randn(3, 4)
    y_keys = [5, 9]
    xz = np.ones((2,))
    xz_keys = [3]
    z = np.random.randn(2, 5)
    z_keys = [3, 1]
    tree = [
        x, x_keys,
        (
            xy, xy_keys,
            [
                y, y_keys,
            ]
        ),
        (
            xz, xz_keys,
            [
                z, z_keys
            ]
        )
    ]
    assert_junction_tree_equal(
        bp.hugin(tree, bp.sum_product),
        brute_force_sum_product(tree)
    )

    pass
