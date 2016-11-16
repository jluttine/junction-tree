"""
Belief propagation
"""

#[ state, variables, child_trees]

import numpy as np


def sum_product(tree, variables):

    common_child_variables = [
        [
            variable
            for variable in variables
            if variable in child_tree[1]
        ]
        for child_tree in tree[2:]
    ]

    xs = [
        sum_product(
            child_tree,
            child_variables
        )
        for (child_tree, child_variables) in zip(tree[2:], common_child_variables)
    ]

    xs_is = zip(xs, common_child_variables)
    args = [
        z
        for x_i in xs_is
        for z in x_i
    ] + [tree[0], tree[1], variables]

    return np.einsum(*args)



def initialize(tree):
    """ Given Junction tree, initialize separators to one """

    # Give tree in form:
    #
    [f, variables, [sep_var, [...]], [sep_var, [...]]]
    # Result in form:
    #
    # [f, variables, [separator, sep_var, [...]], [separator, sep_var, [...]], ...]
    return [

    ]


def update(clique_a, clique_b, separator):
    # See:
    # http://compbio.fmph.uniba.sk/vyuka/gm/old/2010-02/handouts/junction-tree.pdf

    # Sum variables in A that are not in B
    new_separator = np.einsum(
        clique_a, variables_a,
        variables_separator
    )

    # Compensate the updated separator in the clique
    new_clique_b = np.einsum(
        new_separator / separator, variables_separator,
        clique_b, variables_b,
        variables_b
    )

    return (new_clique_b, new_separator)


def collect_evidence(tree):
    return [
    ]


def distribute_evidence():
    pass


def hugin():
    pass


print("HELLO")

print(
    sum_product(
        [
            [1,2,3],
            [0],
            [
                [4, 5, 6],
                [1]
            ]

        ],
        [0],
    )
)
