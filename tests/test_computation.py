# Tests in test_hugin and test_inference will be streamlined and centralized here.
# There are only two types of computation happening in the junctiion tree:
# 1) xxxx the potentials
# 2) performing inference by computing marginals (with and without evidence)

from junctiontree import computation as comp
import numpy as np
import numbers

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
    factors_vars = get_arrays_and_keys(tree, node_list, potentials)
    f = lambda output_vars: np.einsum(*(factors_vars + [output_vars]))

    def __run(tree, node_list, p, f, res=[]):
        res.append(f(node_list[tree[0]]))
        for child_tree in tree[1:]:
            __run(child_tree, node_list, p, f, res)
        return res

    return __run(tree, node_list, potentials, f)


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


def test_ve_one_with_scalar_factor():
    factor = np.random.randn()
    factor_vars = []

    m_factor, m_factor_vars = comp.eliminate_variables([factor], [factor_vars], [])
    assert type(factor) == type(m_factor)
    assert factor == m_factor
    assert type(m_factor_vars) == list


def test_ve_with_one_factor():
    factor = np.random.randn(2, 3)
    factor_vars = [3, 5]

    m_factor, m_factor_vars = comp.eliminate_variables([factor], [factor_vars], [])
    np.testing.assert_allclose(m_factor, factor)
    assert type(factor) == type(m_factor)
    assert type(m_factor_vars) == list

    assert type(m_factor) == np.ndarray
    assert type(m_factor_vars) == list

    np.testing.assert_allclose(m_factor, factor)

    m_factor, m_factor_vars = comp.eliminate_variables([factor], [factor_vars], [5])

    assert m_factor.shape[0] == 2
    assert m_factor_vars == [3]
    np.testing.assert_allclose(np.sum(factor, axis=1), m_factor)

def test_ve_three_factors_with_all_variables_shared():
    factor1 = np.random.randn(2, 3)
    factor1_vars = [3, 5]
    factor2 = np.ones((3, 2))
    factor2_vars = [5, 3]
    factor3 = np.random.randn(3, 2)
    factor3_vars = [5, 3]
    factors = [factor1, factor2, factor3]
    variables = [factor1_vars, factor2_vars, factor3_vars]
    inputs = sum([[factor, vars] for factor, vars in zip(factors, variables)], [])

    output = comp.eliminate_variables(factors, variables, [])

    for in_item, out_item in zip(
                                output,
                                inputs
    ):
        np.testing.assert_allclose(in_item, out_item)


    m_factor, m_factor_vars = comp.eliminate_variables(factors, variables, [5])

    assert m_factor_vars == [3]
    np.testing.assert_allclose(
                            m_factor,
                            np.einsum(
                                    factor1,
                                    factor1_vars,
                                    factor2,
                                    factor2_vars,
                                    factor3,
                                    factor3_vars,
                                    [3]
                            )
    )

def test_ve_two_factors_with_some_variables_shared():
    factor1 = np.random.randn(2, 3)
    factor1_vars = [3, 5]
    factor2 = np.random.randn(3, 4)
    factor2_vars = [5, 7]
    factors = [factor1, factor2]
    variables = [factor1_vars, factor2_vars]

    m_factor, m_factor_vars = comp.eliminate_variables(factors, variables, [5])
    assert m_factor_vars == [3, 7]

    np.testing.assert_allclose(
                        m_factor,
                        np.einsum(
                                factor1,
                                factor1_vars,
                                factor2,
                                factor2_vars,
                                m_factor_vars
                        )
    )

    m_factor, m_factor_vars = comp.eliminate_variables(factors, variables, [3])

    assert m_factor_vars == [5, 7]
    assert m_factor.shape == (3, 4)

    np.testing.assert_allclose(
                            m_factor,
                            np.einsum(
                                    factor1,
                                    factor1_vars,
                                    factor2,
                                    factor2_vars,
                                    m_factor_vars
                            )
    )

    m_factor, m_factor_vars = comp.eliminate_variables(factors, variables, [3, 7])

    assert m_factor_vars == [5]
    assert m_factor.shape == (3,)

    np.testing.assert_allclose(
                            m_factor,
                            np.einsum(
                                    factor1,
                                    factor1_vars,
                                    factor2,
                                    factor2_vars,
                                    m_factor_vars
                            )
    )



def test_variable_elimination_against_stanford_grading_example():
    phi_d = np.array([0.6, 0.4])
    phi_d_vars = [0]

    phi_i = np.array([0.7, 0.3])
    phi_i_vars = [1]

    phi_lg = np.array(
            [[0.1, 0.4, 0.99],
            [0.9,0.6,0.01]]
    )
    phi_lg_vars = [2, 3]

    phi_idg = np.array(
        [[[0.3, 0.4, 0.3],
          [0.05, 0.25, 0.7]],
          [[0.9, 0.08, 0.02],
          [0.5, 0.3, 0.2]]]
    )
    phi_idg_vars = [1, 0, 3]

    phi_is = np.array(
        [[0.95, 0.05],
         [0.2, 0.8]]
    )
    phi_is_vars = [1, 4]

    factors = [phi_d, phi_i, phi_lg, phi_idg, phi_is]
    variables = [phi_d_vars, phi_i_vars, phi_lg_vars, phi_idg_vars, phi_is_vars]

    m_factor, m_factor_vars = comp.eliminate_variables(factors, variables, [0, 1, 4, 3])

    assert m_factor_vars == [2]
    assert m_factor.shape == (2, )

    np.testing.assert_allclose(
                        m_factor,
                        np.einsum(
                                phi_d,
                                phi_d_vars,
                                phi_i,
                                phi_i_vars,
                                phi_lg,
                                phi_lg_vars,
                                phi_idg,
                                phi_idg_vars,
                                phi_is,
                                phi_is_vars,
                                m_factor_vars
                        )
    )

def test_variable_elimination_with_evidence():
    factor1 = np.random.randn(2, 3)
    factor1_vars = [3, 5]
    factor2 = np.random.randn(3, 2, 4)
    factor2_vars = [5, 3, 2]
    factor3 = np.random.randn(3, 4, 5)
    factor3_vars = [5, 2, 1]
    factors = [factor1, factor2, factor3]
    variables = [factor1_vars, factor2_vars, factor3_vars]
    order = [3, 5]
    evidence = {3: 0, 5: 1}

    m_factor, m_factor_vars = comp.eliminate_variables(factors, variables, order, evidence)

    assert m_factor_vars == [1, 2]

    np.testing.assert_allclose(
                    m_factor,
                    np.einsum(
                            factor1[slice(0,1), slice(1,2)],
                            factor1_vars,
                            factor2[slice(1,2),slice(0,1),:],
                            factor2_vars,
                            factor3[slice(1,2),:,:],
                            factor3_vars,
                            m_factor_vars
                    )
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
    # product by doing the divisioin operation

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
    exp_ix[mask] = ...

    slice_ix = np.full(msg_prod.ndim, slice(None))
    slice_ix[mask] = 0

    np.divide(msg_prod, msg3[tuple(exp_ix)])[tuple(slice_ix)]
    assert np.allclose(msg_prod_x6, np.divide(msg_prod, msg3[tuple(exp_ix)])[tuple(slice_ix)])

    msg_prod_x1 = np.einsum(msg2, variables[4], msg3, variables[-2], [3,7])
    assert np.allclose(msg_prod_x1, np.divide(msg_prod, msg1[None, ..., None])[:,0,:])

    msg_prod_x2 = np.einsum(msg1, variables[3], msg3, variables[5], [5,7])
    assert np.allclose(msg_prod_x2, np.divide(msg_prod, msg2[..., None, None])[0,:,:])

