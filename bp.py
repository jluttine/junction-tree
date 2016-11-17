"""Belief propagation, junction trees etc

Everything in this file is just rough sketching and ideas at the moment.
Probably many things will change when implementing and learning.


Starting point:

Graphs are given as factor graphs. A factor graph is given as a list of
factors, where each factor is a tuple of the shape of the array and keys that
tell which variable each axis of the array corresponds to.

[(shape1, axis_keys1), ..., (shapeN, axis_keysN)]

Here, shape is the shape of the array that will be inputted later.

len(shape) == len(axis_keys)

Thus, no moralization function should be needed.


Idea:

Junction tree construction constructs a function which can then be used
multiple times to create the junction tree for particular array values in the
factor graph. This allows one to compile the junction tree only once and then
use the result for varying values in the factors (but with the same graph
structure, obviously). That is, the junction tree construction depends only on
the factor graph structure and the array shapes, but not the values in the
arrays.

One junction tree algorithm reference:
http://compbio.fmph.uniba.sk/vyuka/gm/old/2010-02/handouts/junction-tree.pdf


General guidelines:

- Use functional approach:

  - pure functions (no side effects)

  - immutable data structures

  - prefer expressions over statements (e.g., if-else expression rather than
    if-else statement)

- Provide functions as a library so that users can use only some functions
  alone without needing to use the full stack

- Try to keep the implementation such that it would be possible to use, for
  instance, TensorFlow or Theano for the library. Not a thing to focus now
  but just something to keep in mind.

- Write tests comprehensively, preferably before the actual implementation

"""


import numpy as np


def triangulate(factor_graph):
    """
    Triangulate given factor graph.

    Input:

    Factor graph syntax is a list of tuples (shape, keys):
    [(shape1, keys1), ..., (shapeN, keysN)]

    Output: ???

    NOTE: A clique may contain multiple factors.
    """
    raise NotImplementedError()


def construct_junction_tree(tbd):
    """
    Input: ?

    Output: ?
    """
    raise NotImplementedError()


def get_maximum_weight_spanning_tree(tbd):
    """
    Input: ?

    Output: ?
    """
    raise NotImplementedError()


def eliminate_variables(junction_tree):
    """Eliminate all other variables except the root variables"""

    def __run(tree, variables):
        """Run variable elimination recursively

        Construct trees with nested lists as:

        [array, axis_keys, child_tree1, ..., child_treeN]

        where each child tree has the same syntax recursively.

        Axis keys are some unique identifiers used to determine which axis in
        different array correspond to each other and they are used directly as
        keys for numpy.einsum. It should hold that len(axis_keys) ==
        np.ndim(array). TODO: numpy.einsum supports only keys up to 32, thus in
        order to support arbitrary number keys in the whole tree, one should
        map the keys for the curren numpy.einsum to unique integers starting
        from 0.

        """

        common_child_variables = [
            [
                variable
                for variable in variables
                if variable in child_tree[1]
            ]
            for child_tree in tree[2:]
        ]

        xs = [
            __run(
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

    return __run(junction_tree, junction_tree[1])


def initialize(tree):
    """Given Junction tree, initialize separator arrays

    TODO/FIXME: Perhaps this should be part of the junction tree constructor
    function. That is, this shouldn't need to be run when array values change
    but only once for a graph!

    Input tree format:

    [array, keys, (separator1_keys, child_tree1), ..., (separatorN_keys, child_treeN)]

    Output tree format:

    [array, keys, (separator1_array, separator1_keys, child_tree1), ... (separatorN_array, separatorN_keys, child_treeN)]

    QUESTION: How to separate the graph structure and its junction tree from
    the array values in each factor? Perhaps use a list of arrays and then the
    tree just contains indices to find the correct array in that list?

    """
    raise NotImplementedError()


def collect(tree):
    """ Used by Hugin algorithm to collect messages """
    raise NotImplementedError()


def distribute():
    """ Used by Hugin algorithm to distribute messages """
    raise NotImplementedError()


def hugin(junction_tree, distributive_law):
    """Run hugin algorithm by using the given distributive law.

    Input tree format:

    [array, keys, (separator1_array, separator1_keys, child_tree1), ... (separatorN_array, separatorN_keys, child_treeN)]

    Output tree format is the same?

    See page 3:
    http://compbio.fmph.uniba.sk/vyuka/gm/old/2010-02/handouts/junction-tree.pdf
    """
    raise NotImplementedError()


class SumProduct():
    """ Sum-product distributive law """


    def __init__(self, einsum):
        # Perhaps support for different frameworks (TensorFlow, Theano) could
        # be provided by giving the necessary functions.
        self.einsum = einsum
        return


    def initialize(self, tbd):
        raise NotImplementedError()


    def update(self, clique_a, clique_b, separator):
        # See page 2:
        # http://compbio.fmph.uniba.sk/vyuka/gm/old/2010-02/handouts/junction-tree.pdf

        # Sum variables in A that are not in B
        new_separator = self.einsum(
            clique_a, variables_a,
            variables_separator
        )

        # Compensate the updated separator in the clique
        new_clique_b = self.einsum(
            new_separator / separator, variables_separator,
            clique_b, variables_b,
            variables_b
        )

        return (new_clique_b, new_separator) # may return unchanged clique_a
                                             # too if it helps elsewhere


# Sum-product distributive law for NumPy
sum_product = SumProduct(np.einsum)
