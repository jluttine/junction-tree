from junctiontree import computation as comp
import numpy as np
import numbers
from .util import assert_potentials_equal

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


def get_factors_and_vars(factors, variables, evidence):
    '''Creates a flat list of all factor arrays and lists of their variables
    Output: [factor1, vars1, ..., factorN, varsN]

    :param factors: list of factors
    :param variables: list of variables included in each factor in factors list
    :param evidence: dictionary specifying (possible) observations for variables
    :param factors_and_vars: list of interleaved factors and corresponding variable lists
    '''


    return sum(
                [
                    [
                        # index array based on evidence value when evidence provided otherwise use full array
                        fac[
                            tuple(
                                    [
                                        slice(evidence.get(var, 0), evidence.get(var, fac.shape[i]) + 1)
                                        for i, var in enumerate(vars)
                                    ]
                            )
                        # workaround for scalar factors
                        ] if not (isinstance(fac, numbers.Number)) else fac,
                        vars
                    ]
                    for fac, vars in zip(factors, variables)
                ],
                []
    )

def get_arrays_and_keys(tree, node_list, potentials):
    """Get all arrays and their keys as a flat list
    Output: [array1, keys1, ..., arrayN, keysN]
    """
    return list([potentials[tree[0]],node_list[tree[0]]]) + sum(
        [
            get_arrays_and_keys(child_tree, node_list, potentials)
            for child_tree in tree[1:]
        ],
        []
    )


def brute_force_sum_product(tree, node_list, potentials):
    """Compute brute force sum-product with einsum """

    # Function to compute the sum-product with brute force einsum
    arrays_keys = get_arrays_and_keys(tree, node_list, potentials)
    f = lambda output_vars: np.einsum(*(arrays_keys + [output_vars]))

    def __run(tree, node_list, p, f, res=[]):
        res.append(f(node_list[tree[0]]))
        for child_tree in tree[1:]:
            __run(child_tree, node_list, p, f, res)
        return res

    return __run(tree, node_list, potentials, f)


def assert_sum_product(tree, node_order, potentials, variables):
    """ Test shafer-shenoy vs brute force sum-product """

    # node_order represents the order nodes are traversed
    # in get_factors_and_vars function

    assert_potentials_equal(
        brute_force_sum_product(
                            tree,
                            [variables[idx] for idx in node_order],
                            [potentials[idx] for idx in node_order]
        ),
        comp.compute_beliefs(tree, potentials, variables)
    )


def test_one_scalar_node():
    assert_sum_product(
                [
                    0,
                ],
                [0],
                [
                    np.random.randn(),
                ],
                [[]] # no variables for scalar
)

def test_one_matrix_node():
    assert_sum_product(
                    [
                        0,
                    ],
                    [0],
                    [
                        np.random.randn(2, 3),
                    ],
                    [
                        [3,5]
                    ]

    )

def test_one_child_node_with_all_variables_shared():
    assert_sum_product(
                    [
                        0,
                        (
                            2,
                            [
                                1,
                            ]
                        )
                    ],
                    [0,2,1],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 2),
                        np.ones((3, 2)),
                    ],
                    [
                        [3,5],
                        [5,3],
                        [5,3]
                    ]
    )

def test_one_child_node_with_one_common_variable():
    assert_sum_product(
                    [
                        0,
                        (
                            2,
                            [
                                1,
                            ]
                        )
                    ],
                    [0,2,1],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 4),
                        np.ones((3,)),
                    ],
                    [
                        [3,5],
                        [5,9],
                        [5]
                    ]
    )

def test_one_child_node_with_no_common_variable():
    assert_sum_product(
                    [
                        0,
                        (
                            2,
                            [
                                1,
                            ]
                        )
                    ],
                    [0,2,1],
                    [
                        np.random.randn(2),
                        np.random.randn(3),
                        np.ones(()),
                    ],
                    [
                        [3],
                        [9],
                        []
                    ]

    )

def test_one_grand_child_node_with_no_variable_shared_with_grand_parent():
    assert_sum_product(
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
                    [0,2,4,1,3],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 4),
                        np.random.randn(4, 5),
                        np.ones((3,)),
                        np.ones((4,)),
                    ],
                    [
                        [3, 5],
                        [5, 9],
                        [9, 1],
                        [5],
                        [9]
                    ]
    )

def test_one_grand_child_node_with_variable_shared_with_grand_parent():
    assert_sum_product(
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
                    [0,2,4,1,3],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 4),
                        np.random.randn(6, 3),
                        np.ones((3,)),
                        np.ones((3,)),
                    ],
                    [
                        [3, 5],
                        [5, 9],
                        [1, 5],
                        [5],
                        [5]
                    ]
    )

def test_two_children_with_no_variable_shared():
    assert_sum_product(
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
                    [0,2,4,1,3],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 4),
                        np.random.randn(2, 5),
                        np.ones((3,)),
                        np.ones((2,)),
                    ],
                    [
                        [3, 5],
                        [5, 9],
                        [3, 1],
                        [5],
                        [3]
                    ]
    )


def test_two_child_with_shared_variable():
    assert_sum_product(
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
                    [0,2,4,1,3],
                    [
                        np.random.randn(2, 3),
                        np.random.randn(3, 4),
                        np.random.randn(3),
                        np.ones((3,)),
                        np.ones((3,)),

                    ],
                    [
                        [3, 5],
                        [5, 9],
                        [5],
                        [5],
                        [5]
                    ]
    )

def test_two_children_with_3D_tensors():
    assert_sum_product(
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
                    [0,2,4,1,3],
                    [
                        np.random.randn(2, 3, 4),
                        np.random.randn(3, 4, 5),
                        np.random.randn(3, 6),
                        np.ones((3, 4)),
                        np.ones((3,)),
                    ],
                    [
                        [3,5,7],
                        [5,7,9],
                        [5,1],
                        [5,7],
                        [5]
                    ]

    )


def test_divide_matrix_product():
    # dividing messages from product when neighbor message is excluded
    # this avoids re-doing einsum calculations to accomplish the same
    # one full message product is calculated and messages are removed from the
    # product by performing the division operation

    potentials = [
                np.random.randn(2, 3, 6),
                np.random.randn(3, 4),
                np.random.randn(2, 5),
                np.ones((3,)),
                np.ones((2,)),
                np.ones((6,)),
                np.random.randn(4, 6)
    ]

    variables = [
                [3, 5, 7],
                [5, 9],
                [3, 1],
                [5],
                [3],
                [7],
                [2, 7]
    ]

    msg1 = np.einsum(potentials[1], variables[1], variables[3])
    msg2 = np.einsum(potentials[2], variables[2], variables[4])
    msg3 = np.einsum(potentials[6], variables[6], variables[5])

    msg_prod = np.einsum(msg1, variables[3], msg2, variables[4], msg3, variables[5], variables[0])

    msg_prod_x6 = np.einsum(msg1, variables[3], msg2, variables[4], [3,5])
    assert np.allclose(msg_prod_x6, np.divide(msg_prod, msg3[None, None, ...])[:,:,0])

    mask = np.in1d(variables[0], variables[6])
    exp_ix = np.full(msg_prod.ndim, None)
    exp_ix[mask] = slice(None)

    slice_ix = np.full(msg_prod.ndim, slice(None))
    slice_ix[mask] = 0

    np.divide(msg_prod, msg3[tuple(exp_ix)])[tuple(slice_ix)]
    assert np.allclose(msg_prod_x6, np.divide(msg_prod, msg3[tuple(exp_ix)])[tuple(slice_ix)])

    msg_prod_x1 = np.einsum(msg2, variables[4], msg3, variables[-2], [3,7])
    assert np.allclose(msg_prod_x1, np.divide(msg_prod, msg1[None, ..., None])[:,0,:])

    msg_prod_x2 = np.einsum(msg1, variables[3], msg3, variables[5], [5,7])
    assert np.allclose(msg_prod_x2, np.divide(msg_prod, msg2[..., None, None])[0,:,:])


def test_apply_evidence_to_potentials():
    potentials = [
                np.random.randn(2, 3, 6),
                np.random.randn(3, 4),
                np.random.randn(2, 5),
                np.ones((3,)),
                np.ones((2,)),
                np.ones((6,)),
                np.random.randn(4, 6)
    ]

    variables = [
                [3, 5, 7],
                [5, 9],
                [3, 1],
                [5],
                [3],
                [7],
                [2, 7]
    ]

    evidence = {3:0, 9:2}

    shrunken_potentials = comp.apply_evidence(potentials, variables, evidence)

    np.allclose(potentials[0][0, :, :], shrunken_potentials[0])
    np.allclose(potentials[1][:, 2], shrunken_potentials[1])
    np.allclose(potentials[2][0, 1], shrunken_potentials[2])
    np.allclose(potentials[3], shrunken_potentials[3])
    np.allclose(potentials[4][0], shrunken_potentials[4])
    np.allclose(potentials[5], shrunken_potentials[5])
    np.allclose(potentials[6], shrunken_potentials[6])


def test_evidence_shrinking():
    # evidence shrinking can be incorporated by removing axis
    # corresponding to observed variable
    A = np.random.rand(3,4,2) # vars: a,b,c
    a = [0]*3
    a[2] = 1
    b = [0]*4
    b[2] = 1
    c = [0]*2
    c[0] = 1

    # update potential A based on observing a=2
    A_updated = comp.sum_product.einsum(A, [0,1,2], a, [0], [0,1,2])

    # shrinking from evidence
    # set the axis representing a (ax=0) to the value of a
    A_updated_es = A_updated[2,:,:]
    assert A_updated_es.shape == (4,2)

    # imagine we have another potential sharing vars b and c
    B = np.random.rand(4,2) # vars: b,c
    B_updated = comp.sum_product.einsum(A_updated, [0,1,2], B, [1,2], [1,2])

    B_updated_es = comp.sum_product.einsum(A_updated_es, [1,2], B, [1,2], [1,2])

    # the result of the calculation should be the same regardless of if we use
    # the updated potentials from A_updated (without evidence shrinking)
    # or A_updated_es (with evidence shrinking)
    np.testing.assert_allclose(
                            B_updated,
                            B_updated_es
    )

    # what happens if the only shared variables between potentials is
    # the single variable in potential

    C = np.random.rand(3) # vars: a
    C_updated = comp.sum_product.einsum(C, [0], a, [0], [0])
    C_updated_es = C_updated[2]

    np.testing.assert_allclose(
                    comp.sum_product.einsum(A_updated, [0,1,2], C_updated, [0], []),
                    comp.sum_product.einsum(A_updated_es, [1,2], C_updated_es, [], [])
    )

    np.testing.assert_allclose(
                    comp.sum_product.einsum(A_updated, [0,1,2], C_updated, [0], [1,2]),
                    comp.sum_product.einsum(A_updated_es, [1,2], C_updated_es, [], [1,2])
    )
