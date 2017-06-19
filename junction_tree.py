import copy
import bp
import numpy as np

class JunctionTree(object):
    def __init__(self, key_sizes, trees=[]):
        self.key_sizes = key_sizes
        self.labels = {vl:i for i, vl in enumerate(sorted(key_sizes.keys()))}
        self.struct = []
        self.tree_cliques = []
        for tree in trees:
            clique_id_keys = list(bp.bf_traverse([tree]))
            self.tree_cliques.append([(clique_id_keys[i], clique_id_keys[i+1]) for i in range(0, len(clique_id_keys), 2)])
            self.struct.append(self.map_keys(tree, self.labels))

    def find_key(self, key_label):
        """
        Return index of key in tree keys

        Input:
        ------

        Key label as provided when junction tree constructed

        Output:
        -------

        Key index (or None if key not in tree)

        """
        try:
            key_ix = self.labels[key_label]
            return key_ix
        except ValueError:
            return None

    def get_key_ix(self, clique_ix, key_label):
        """
        Returns index of key in clique's set of keys

        Input:
        ------

        Clique ID

        Key label

        Output:
        -------

        Index of key in clique's keys

        """
        try:
            keys = bp.get_clique_keys(self.get_struct(), clique_ix)
            return keys.index(self.find_key(key_label))
        except (AttributeError, ValueError):
            return None

    def get_key_sizes(self):
        """
        Returns dictionary of key labels as keys
            and key size as values

        """
        return self.key_sizes

    def get_label_order(self):
        """
        Return dictionary of key label as key
            and index as value
        """
        return self.labels

    def get_labels(self):
        """
        Returns key labels in sorted order

        """
        labels = [None]*len(self.labels)
        for k,i in self.labels.items():
            labels[i] = k

        return labels

    def get_struct(self):
        """
        Return structure of junction tree

        """
        return self.struct


    @staticmethod
    def map_keys(tree, lookup):
        """
        Map keys of cliques to index values

        Input:
        ------

        Tree structure

        Lookup dictonary with key label as key and
            key index (in junction tree) as value

        Output:
        -------
        Return tree with re-indexed clique keys

        """
        cp_tree = copy.deepcopy(tree)

        def __run(tree, lookup):
            ix, keys = tree[0:2]
            for i, k in enumerate(keys):
                keys[i] = lookup[k]
            separators = tree[2:]

            for separator in separators:
                separator_ix, separator_keys, c_tree = separator
                for i, k in enumerate(separator_keys):
                    separator_keys[i] = lookup[k]
                __run(c_tree, lookup)

        __run(cp_tree, lookup)
        return cp_tree


    @staticmethod
    def from_factor_graph(factor_graph):
        """
        Construct a junction tree from factor graph

        Input:
        ------

        Factor graph as list of key sizes, list of
            factors (keys), and list of factor potentials

        Output:
        -------

        Resulting JunctionTree and initial potentials


        """
        key_sizes = factor_graph[0]
        factors = factor_graph[1]
        values = factor_graph[2]
        tri,induced_clusters = bp.find_triangulation(
                            key_sizes=factor_graph[0],
                            factors=factor_graph[1]
        )

        cliques = bp.identify_cliques(induced_clusters)
        trees = bp.construct_junction_tree(cliques, key_sizes)
        jt = JunctionTree(key_sizes, trees)
        phi = JunctionTree.init_potentials(jt, factors, values)
        return jt, phi

    @staticmethod
    def init_potentials(tree, factors, values):
        """
        Creates initial potentials based on factors

        Input:
        ------

        Tree structure of the junction tree

        List of factors (key lists)

        List of factor potentials

        Output:
        -------

        Initial potentials

        """
        clique_id_keys = list(bp.bf_traverse(tree.get_struct()))
        clique_lookup = {}
        potentials = [[]]*(int(len(clique_id_keys)/2))

        labels = tree.get_labels()
        key_sizes = tree.get_key_sizes()

        for i in range(0, len(clique_id_keys), 2):
            clique_ix = clique_id_keys[i]
            clique_keys = clique_id_keys[i+1]
            # initialize all potentials
            clique_lookup[clique_ix] = clique_keys
            potentials[clique_ix] = np.ones([key_sizes[labels[ix]] for ix in clique_keys])

        for i, factor in enumerate(factors):
            # convert factor to its indexed keys
            factor_keys = set([tree.find_key(key) for key in factor])
            # find clique to multiply factor into
            for clique_ix, clique_keys in clique_lookup.items():
                if factor_keys.issubset(clique_keys):
                    # multiply factor into clique
                    potentials[clique_ix] = np.einsum(
                                                values[i],
                                                list(factor_keys),
                                                potentials[clique_ix],
                                                clique_keys,
                                                clique_keys
                    )
                    break

        return(potentials)

    def observe(self, potentials, data):
        """
        Return updated clique potentials based on observed data

        Input:
        ------

        List of potentials

        Dictionary of key label as key and key assignment as value

        Output:
        -------

        List of likelihood potentials

        List of updated clique potentials

        """
        key_sizes = self.get_key_sizes()
        # set values of ll based on data argument
        ll = [
                    [1 if j == data[key_lbl] else 0 for j in range(0, key_sizes[key_lbl])]
                        if key_lbl in data else [1]*key_sizes[key_lbl]
                            for key_lbl in self.get_labels()
                ]

        # alter potentials based on likelihoods
        for key_lbl in data:
            # find clique that contains key
            for tree in self.get_struct():
                clique_ix, clique_keys = bp.get_clique_of_key(
                                                        tree,
                                                        self.find_key(key_lbl)
                )
                if clique_ix and clique_keys: break

            # multiply clique's potential by likelihood
            pot = potentials[clique_ix]
            key_ix = self.get_key_ix(clique_ix, key_lbl)
            # reshape likelihood potential to allow multiplication with pot
            ll_pot = np.array(ll[self.find_key(key_lbl)]).reshape([1 if i!=key_ix else s for i, s in enumerate(pot.shape)])
            potentials[clique_ix] = pot*ll_pot
        return (ll,potentials)

    def propagate(self, potentials, in_place=True, data=None):
        """
        Return consistent potentials

        Input:
        ------

        List of inconsistent potentials

        Boolean to do updates in place

        Dictionary of key label as key and key assignment as value

        Output:
        -------

        Updated list of (consistent) potentials and
            normalization constants for each tree

        """
        new_potentials = potentials if in_place else copy.deepcopy(potentials)
        if data:
            likelihood, new_potentials = self.observe(new_potentials, data=data)

        for i, tree in enumerate(self.get_struct()):
            new_potentials = bp.hugin(tree, self.get_label_order(), new_potentials, bp.sum_product)

        return new_potentials

    def marginalize(self, potentials, key_label, normalize=False):
        """
        Marginalize key from consistent potentials

        Input:
        ------

        List of consistent potentials

        Key to marginalize

        Normalize value?

        Output:
        -------

        Marginalized value of key (unnormalized by default)

        """
        if key_label not in self.key_sizes:
            raise ValueError("Key %s not in tree" % key_label)

        key_ix = self.find_key(key_label)
        for i, tree in enumerate(self.get_struct()):
            clique_ix, clique_keys = bp.get_clique_of_key(tree, key_ix)
            if clique_ix and clique_keys: break

        value = bp.compute_marginal(
                            potentials[clique_ix],
                            clique_keys,
                            key_ix
        )

        Z = 1.0 if not normalize else np.sum(
                                            [
                                                bp.compute_marginal(
                                                    potentials[clique_ix],
                                                    clique_keys,
                                                    key_ix
                                                ) for key_ix in clique_keys
                                            ]
                                        ) + value

        return value/Z
