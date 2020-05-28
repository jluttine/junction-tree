import numpy as np
from junctiontree import beliefpropagation as bp

'''
HUGIN Tests

examples:
https://www.cs.ru.nl/~peterl/BN/examplesproofs.pdf
http://www.inf.ed.ac.uk/teaching/courses/pmr/docs/jta_ex.pdf

background:
http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/171216.pdf
'''


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
    np.testing.assert_allclose(bp.sum_product.einsum(phiABD, [0,1,2], [0]), np.array([0.500, 0.500]))
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

    phi2n = bp.sum_product.project(phi12, [0,1], [1])
    np.testing.assert_allclose(phi2n, np.array([1,1]))
    phi23 = bp.sum_product.absorb(phi23, [0,1], phi2, phi2n, [1])
    np.testing.assert_allclose(phi23, np.array([
                                    [0.03,0.07],
                                    [0.45,0.45]
                                ]))

    phi2nn = bp.sum_product.project(phi23, [0,1], [0])
    np.testing.assert_allclose(phi2nn, np.array([0.1, 0.9]))
    phi12 = bp.sum_product.absorb(phi12, [0,1], phi2n, phi2nn, [1])
    np.testing.assert_allclose(phi12, np.array([
                                    [0.04,0.72],
                                    [0.06,0.18]
                                ]))


def test_collect_messages():
    # constructor for junction tree taking a list based definition
    # will have a function that can convert factor graph into JT
    jt = [
            0,
            (
                1,
                [
                    2,
                ]
            )

        ]

    node_list = [["V1","V2"],["V2"],["V2", "V3"]]

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

    phiN = bp.collect(
                        jt,
                        node_list,
                        phi,
                        [0]*len(phi),
                        bp.sum_product
    )

    np.testing.assert_allclose(
        phiN[2],
        np.array(
            [
                [0.03,0.07],
                [0.45,0.45]
            ]
        )
    )


def test_distribute_messages():
    jt = [
            0,
            (
                1,
                [
                    2,
                ]
            )
        ]
    node_list = [["V1","V2"],["V2"],["V2", "V3"]]
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

    phiN = bp.collect(
                        jt,
                        node_list,
                        phi,
                        [0]*len(phi),
                        bp.sum_product
    )

    phiN2 = bp.distribute(
                            jt,
                            node_list,
                            phiN,
                            [0]*len(phiN),
                            bp.sum_product
    )

    np.testing.assert_allclose(
        phiN2[0],
        np.array(
            [
                [0.04,0.72],
                [0.06,0.18]
            ]
        )
    )

    np.testing.assert_allclose(
        phiN[1],
        np.array([0.1,0.9])
    )

    np.testing.assert_allclose(
        phiN[2],
        np.array(
            [
                [0.03,0.07],
                [0.45,0.45]
            ]
        )
    )


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
    A_updated = bp.sum_product.einsum(A, [0,1,2], a, [0], [0,1,2])

    # shrinking from evidence
    # set the axis representing a (ax=0) to the value of a
    A_updated_es = A_updated[2,:,:]
    assert A_updated_es.shape == (4,2)

    # imagine we have another potential sharing vars b and c
    B = np.random.rand(4,2) # vars: b,c
    B_updated = bp.sum_product.einsum(A_updated, [0,1,2], B, [1,2], [1,2])

    B_updated_es = bp.sum_product.einsum(A_updated_es, [1,2], B, [1,2], [1,2])

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
    C_updated = bp.sum_product.einsum(C, [0], a, [0], [0])
    C_updated_es = C_updated[2]

    np.testing.assert_allclose(
                    bp.sum_product.einsum(A_updated, [0,1,2], C_updated, [0], []),
                    bp.sum_product.einsum(A_updated_es, [1,2], C_updated_es, [], [])
    )

    np.testing.assert_allclose(
                    bp.sum_product.einsum(A_updated, [0,1,2], C_updated, [0], [1,2]),
                    bp.sum_product.einsum(A_updated_es, [1,2], C_updated_es, [], [1,2])
    )

def test_evidence_shrinking_in_hugin():
    pass
