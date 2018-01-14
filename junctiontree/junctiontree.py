"""
Simple user interface for the Junction tree algorithm library
"""

import numpy as np

from . import beliefpropagation as bp
import attr


def create_junction_tree(factors, sizes):
    """Create a Junction tree for a given factor graph."""
    fg = FactorGraph(factors=factors, sizes=sizes)
    return fg.triangulate().create_junction_tree()


def argfind1(xs, cond):
    """Returns the index of the first element in xs satisfying cond."""
    return next(i for (i, x) in enumerate(xs) if cond(x))


def take(xs, inds):
    """Support multiple indices for a list."""
    return [xs[ind] for ind in inds]


def is_subset(a, b):
    """Test whether every element in a is also in b."""
    return set(a).issubset(set(b))


def einsum(xs, xs_keys, y_keys):
    """Thin wrapper for numpy.einsum with some extra support.

    Adds support for:

    - arbitrary keys

    - some keys only in outputs (they are created as new axes)

    """

    # Find keys
    input_keys = list(set([key for x_keys in xs_keys for key in x_keys]))
    all_keys = list(set(y_keys + input_keys))

    # Einsum doesn't support keys that are only in the output. Thus, if some
    # keys are only in outputs, add those keys as auxiliary axes to the first
    # input. This leads to new axes in the output for those keys.
    missing_input_keys = [
        key
        for key in all_keys
        if key not in input_keys
    ]
    xs[0] = np.reshape(
        xs[0],
        len(missing_input_keys) * (1,) + np.shape(xs[0])
    )
    xs_keys[0] = missing_input_keys + xs_keys[0]

    # Mapping from arbitrary keys to numpy.einsum acceptable keys
    keymap = {
        key: i
        for (i, key) in enumerate(all_keys)
    }

    # numpy.einsum argument list
    argsi = [
        arg
        for x_and_keys in zip(xs, xs_keys)
        for arg in (
                x_and_keys[0],
                [keymap[key] for key in x_and_keys[1]]
        )
    ] + [
        [keymap[key] for key in y_keys]
    ]
    return np.einsum(*argsi)


@attr.s(frozen=True)
class FactorGraph():
    """A graph containing a set of nodes that each contain a set of keys.

    Each key has a corresponding size associated to it.

    """

    # Axis keys in each factor
    #
    # TODO: Check that factors don't contain duplicate keys.
    #
    # TODO: Check that all keys in factors are in sizes dictionary.
    factors = attr.ib()

    # Size of each axis
    sizes = attr.ib()


    def triangulate(self):
        """Create a triangulated clique tree from a factor graph."""

        # Let's use the triangulation methods of undirected graphs.
        #
        # FIXME: Perhaps the triangulation could be done in such a way that the
        # factor mappings are found at the same time. It is not very efficient
        # to first use only undirected graph and then finding out how the
        # factors are connected to that result.
        #
        # Also, I didn't quite understand the purpose of the other (the first
        # and the second) outputs.
        (_, _, maxcliques) = bp.find_triangulation(
            self.factors,
            self.sizes
        )

        # FIXME: This should be done already in find_triangulation function as
        # a part of its process.
        #
        # Now, back to the world of factors:
        #
        # We need a mapping factorID -> cliqueID, where the IDs are just list
        # indices.
        #
        # Thus, for each factor, find the maximal clique they belong to.

        def _find_clique(factor):
            """Find the maximal clique that contains the given factor"""
            return argfind1(
                maxcliques,
                lambda clique: is_subset(factor, clique),
            )

        # TODO: This is the result we should obtain from find_triangulation:
        #
        # A list which gives each factor the maxclique ID they belong to. So,
        # the list length is equivalent to the length of the factors list.
        #
        # [maxclique_of_factor_1, ..., maxclique_of_factor_N]
        #
        # Note that the number of maxcliques is less or equivalent to the
        # number of factors. Thus, multiple factors can point to the same
        # maxclique ID.
        factor_to_maxclique = [
            _find_clique(factor)
            for factor in self.factors
        ]

        return CliqueGraph(
            maxcliques=maxcliques,
            factor_to_maxclique=factor_to_maxclique,
            factor_graph=self,
        )


@attr.s(frozen=False)
class CliqueGraph():
    """
    Clique graph for an underlying factor graph.
    """

    # Axis keys in each maximal clique
    maxcliques = attr.ib()

    # Maximal clique for each factor (multiple factors can belong to the same
    # maximal clique)
    factor_to_maxclique = attr.ib()


    # The underlying factor graph
    factor_graph = attr.ib()


    def create_junction_tree(self):
        """Create a Junction tree from a triangulated clique tree."""

        # TODO/FIXME: The Junction tree could perhaps be represented with only
        # indices in the tree data structure. These indices correspond to
        # elements in the clique graph `maxcliques` list. That is, each node in
        # the tree is a maxclique. This `maxclique` already has the information
        # about the keys it contains, so there's no need to include that
        # information in the tree. So, a tree is defined recursively as:
        #
        #   tree = (node_index, subtree1, ..., subtreeN)
        #
        # NOTE: node_index here refers to an index pointing to a node list
        # which is a concatenation of maxclique and separator lists. So in the
        # raw tree data structure, there is no distinction between separators
        # and maximum cliques, nodes are just indices to node_list. This
        # Junction tree class of course knows that there's always a separator
        # node between two maxclique nodes.


        # `construct_junction_tree` provides the list of separators, which are
        # in the order that the indices in the tree use.
        #
        # For instance, a list of two separators:
        #
        #   separators = [ ['a', 'b'], ['c'] ]
        #
        # An example tree with indices pointing to the node list:
        #
        # [
        #   1, # maxclique
        #   [
        #     4, # separator
        #     [
        #       0, # maxclique
        #     ]
        #   ],
        #   [
        #     3, # separator
        #     [
        #       2, # maxclique
        #     ]
        #   ]
        # ]
        #
        # node_list = maxcliques + separators
        #           = [ ['a', 'b', 'c', 'd'], [ 'a', 'b', 'e'], ['c', 'f'] ]
        #             + [ ['a', 'b'], ['c'] ]
        #
        # Note how the indices in the tree are pointing to this node_list.
        #
        # So, what we need is the above tree structure and separators list.

        (tree, separators) = bp.construct_junction_tree(
            self.maxcliques,
            self.factor_graph.sizes
        )

        return JunctionTree(
            tree=tree,
            separators=separators,
            clique_tree=self
        )


    def evaluate(self, xs):
        """Compute maximum clique values based on factor values."""

        # FIXME: This should be computed once at creation time because it
        # doesn't depend on xs. Computing it every time here adds overhead.
        maxclique_to_factors = [
            [
                i for (i, y) in enumerate(self.factor_to_maxclique)
                if y == maxclique
            ]
            for maxclique in range(len(self.maxcliques))
        ]

        return [
            einsum(
                take(xs, factors),
                take(self.factor_graph.factors, factors),
                maxclique
            )
            for (factors, maxclique) in zip(
                    maxclique_to_factors,
                    self.maxcliques
            )
        ]


    def marginalize(self, ys):
        """Marginalize results for maxcliques to results for factors

        This needs to be done because each maxclique may contain multiple
        factors and also some auxiliary variables. The results should be given
        for factors.

        Basically, for each factor, find the maxclique they belong to, and
        marginalize (e.g., sum) the axes that don't belong to that factor.

        Inputs
        ------

        ys : A list of arrays containing the result (e.g., consistent clique
             potentials) for each maxclique.

        Outputs
        -------
        xs : A list of arrays containing the result for each factor.

        """

        # This tells which maxclique to use for each factor
        self.factor_to_maxclique

        # This tells the keys in each factor
        self.factor_graph.factors

        # This tells the keys in each maxclique
        self.maxcliques

        # Now, use my custom einsum to marginalize, something like (didn't test
        # this out):
        #
        # FIXME: This is most likely slightly incorrect as I didn't test it.
        return [
            einsum(
                [ys[maxclique]],
                [self.maxcliques[maxclique]],
                factor_keys
            )
            for (factor_keys, maxclique) in zip(
                    self.factor_graph.factors,
                    self.factor_to_maxclique
            )
        ]


@attr.s(frozen=True)
class JunctionTree():
    """
    Junction tree for an underlying factor graph.
    """

    # Tree data structure
    #
    # (cliqueID, (separatorID, subtree), (separatorID, subtree), ...)
    tree = attr.ib()

    # Tuple of axis keys in each separator
    #
    # ( (key3, key1), (key2, key1), (key2) )
    separators = attr.ib()

    # The underlying triangulated clique graph
    clique_tree = attr.ib()


    def propagate(self, xs):
        """Run belief propagation on the Junction tree."""

        # Let's fix the distributive law for now, as there are no other
        # distributive laws implemented currently. Probably other distributive
        # laws will require some changes in other places that we haven't
        # thought about yet, that is, some code may implicitly assume
        # sum-product distributive law.
        distributive_law = bp.sum_product

        # Evaluate maximum cliques based on factor values
        maxclique_values = self.clique_tree.evaluate(xs)

        # Initialize separator values
        sizes = self.clique_tree.factor_graph.sizes
        separator_values= [
            np.ones(tuple(sizes[key] for key in separator))
            for separator in self.separators
        ]

        # Node list is a concatenation of maxcliques and separators
        values = maxclique_values + separator_values

        # FIXME: There is some argument missing and not sure if these arguments
        # match what the function expects.
        ys = bp.hugin(
            self.tree,
            sizes,
            values,
            distributive_law
        )

        # The return result should be marginalized to the factors. That is, the
        # output list and the arrays inside it have the same length and shapes
        # as xs. That marginalization function should be provided by
        # CliqueGraph.
        return self.clique_tree.marginalize(ys)
