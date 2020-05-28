# Tests in test_hugin and test_inference will be streamlined and centralized here.
# There are only two types of computation happening in the junctiion tree:
# 1) xxxx the potentials
# 2) performing inference by computing marginals (with and without evidence)

from junctiontree import computation as comp
import numpy as np
from junctiontree.sum_product import SumProduct as sum_product

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


def assert_sum_product(tree, value):
    """ Test hugin vs brute force sum-product """
    assert_junction_tree_equal(
        brute_force_sum_product(tree),
        [value, *tree[1:]]
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
