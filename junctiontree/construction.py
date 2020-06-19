import numpy as np
import heapq
from itertools import chain, combinations
import copy

def get_clique(tree, key_list, key_label):
    '''Finds a single clique containing key with label key_label

    :param tree: the tree structure (a list) of the junction tree
    :param key_list: contains the keys indexed by clique id for all cliques in tree
    :param key_label: label used for key
    :return: a clique containing the key or None if no such clique exists in tree
    '''

    ix = tree[0]
    keys = key_list[ix]
    separators = tree[1:]
    if key_label in keys:
        return ix, keys
    if separators == (): # base case reached (leaf)
        return None

    for separator in separators:
        separator_ix, c_tree = separator
        separator_keys = key_list[separator_ix]

        if key_label in separator_keys:
            return separator_ix, separator_keys

        clique_info = get_clique(c_tree, key_list, key_label)

        if clique_info:
            return clique_info

    return None


def factors_to_undirected_graph(factors):
    '''Represent factor graph as undirected graph

    :param factors: list of factors
    :return: undirected graph as dictionary with edges as keys and the factor from
            which edge originates as values
    '''

    factor_edges = {}

    for factor_ix, factor in enumerate(factors):
        for ix, k1 in enumerate(factor):
            for k2 in factor[ix+1:]:

                factor_edges.setdefault( frozenset( (k1,k2) ), set() ).add(factor_ix)

    return factor_edges


def initialize_triangulation_heap(key_sizes, edges):
    '''Create heap used for graph triangulation

    :param key_sizes: dictionary with key (node) label as keys and variable size as values
    :param edges: list of pairs of keys (nodes) representing factor graph edges
    :return heap: heap with entry structure:

            [
                num edges added to triangulated graph by removal of key,
                induced cluster weight,
                tuple (key associated with first two elements, factor key added to
            ]
    :return entry_finder: dictionary with key label as key and reference to heap entry for key

    '''

    heap, entry_finder = update_heap(key_sizes.keys(), edges, key_sizes)

    return heap, entry_finder


def update_heap(remaining_keys, edges, key_sizes, heap=None, entry_finder=None):
    '''Update heap entries

    :param remaining_keys: list of keys (nodes) remaining in the heap
    :param edges: list of edges (pairs of keys (nodes) )
    :param key_sizes: dictionary of keys (key label is key, size is value)
    :param heap: heap to be updated (None if new heap is to be created)
    :param entry_finder: entry_finder dictionary with references to heap elements
    :return h: updated (or newly created) heap
    :return entry_finder: dictionary with updated references to heap elements
    '''

    h = heap if heap else []
    entry_finder = entry_finder if entry_finder else {}
    for key in remaining_keys:
        rem_neighbors = [(set(edge) - set([key])).pop()
                            for edge in edges if key in edge and len(set(remaining_keys).intersection(edge)) == 2]

        # determine how many of key's remaining neighbors need to be connected
        num_new_edges = sum(
                            [
                                frozenset((n1,n2)) not in edges
                                for i, n1 in enumerate(rem_neighbors)
                                    for n2 in rem_neighbors[i+1:]

                            ]
        )
        # weight of a cluster is the product of all key lengths in cluster
        weight = key_sizes[key] * np.prod([key_sizes[n] for n in rem_neighbors])
        entry = [num_new_edges, weight, key]
        heapq.heappush(h, entry)
        # invalidate previous entry if it exists
        prev = entry_finder.get(key, None)
        if prev:
            # set entry to be removed
            prev[2] = ""
        entry_finder[key] = entry

    return h, entry_finder


def factors_to_undirected_graph(factors):
    '''Represent factor graph as undirected graph

    :param factors: list of factors
    :return: undirected graph as dictionary with edges as keys and the factor from
            which edge originates as values
    '''

    factor_edges = {}

    for factor_ix, factor in enumerate(factors):
        for ix, k1 in enumerate(factor):
            for k2 in factor[ix+1:]:

                factor_edges.setdefault( frozenset( (k1,k2) ), set() ).add(factor_ix)

    return factor_edges


def initialize_triangulation_heap(key_sizes, edges):
    '''Create heap used for graph triangulation

    :param key_sizes: dictionary with key (node) label as keys and variable size as values
    :param edges: list of pairs of keys (nodes) representing factor graph edges
    :return heap: heap with entry structure:

            [
                num edges added to triangulated graph by removal of key,
                induced cluster weight,
                tuple (key associated with first two elements, factor key added to
            ]
    :return entry_finder: dictionary with key label as key and reference to heap entry for key

    '''

    heap, entry_finder = update_heap(key_sizes.keys(), edges, key_sizes)

    return heap, entry_finder


def remove_next(heap, entry_finder, remaining_keys, key_sizes, edges):
    '''Removes next entry from heap

    :param heap: heap structure containing remaining factors and weights
    :param entry_finder: dictionary with updated references to heap elements
    :param remaining_keys: list of keys (nodes) remaining in G'
    :param key_sizes: key (node) sizes
    :param edges: list of edge pairs in original graph G
    :return entry: the entry removed from the heap
    :return heap: heap structure with updated keys after factor removal
    :return entry_finder: dictionary with updated references to heap elements
    :return remaining_keys: list of keys without most recently removed key
    '''

    entry = (None, None, "")

    while entry[2] == "":
        entry = heapq.heappop(heap)

    # remove entry from entry_finder
    del entry_finder[entry[2]]

    # remove key from remaining keys list
    remaining_keys.remove(entry[2])


    heap, entry_finder = update_heap(
                                remaining_keys,
                                edges,
                                key_sizes,
                                heap,
                                entry_finder
    )

    return entry, heap, entry_finder, remaining_keys

def find_triangulation(factors, key_sizes):
    '''Triangulate given factor graph.

    TODO: Provide different algorithms.

    :param factors: list of factors where each factor is given as a list of keys the factor contains:

            [keys1, ..., keysN]

    :param key_sizes: dictionary of variables consisting of ( key (node), size (num states) ) pairs

            {
                key1: size1,
                ...
                keyM: sizeM
            }

    :return tri: list of edges added to triangulate the undirected graph
    :return induced_clusters: list of key (node) lists representing induced clusters from triangulation
    :return max_cliques: list of maximal cliques generated during triangulation process
    :return factor_to_maxclique: dictionary mapping each factor to the max_clique which contains the factor
    '''

    def generate_subsets(factors):
        '''For each factor, identify all factors that are subset of that factor

        :param factors: list of factors (list of keys) representing the factor graph
        :return: a dictionary with factor index as key and a list of the factor indices for which the factor
            is a superset as value
        '''

        subsets = {}
        for ix, f1 in enumerate(factors):
            subset_of_ix = max(
                    enumerate(
                        [
                            -1 if not set(f1) <= set(f2) else len(set(f2) - set(f1))
                            for f2 in factors
                        ]
                    ),
                    key=lambda t: t[1]
            )[0]
            subsets.setdefault(subset_of_ix, []).append(ix)

        return subsets

    def find_origin_factors(factor_ixs, subsets, factor_to_maxclique):
        '''Creates a list of original factors that contain an edge

        :param factor_ixs: the original factors containing the edge
        :param subsets: dictionary of factor id as key and factor id of subset factors as value
        :param factor_to_maxclique: list mapping factor by id to assigned maxclique
        :return: a list of the original factors of factor graph which contain the edge
        '''

        return list(
                    chain(
                        *[
                            # adding this factor id and factor ids of factors that are subsets
                            list(
                                set(
                                    subsets.get(factor_ix, []) + [factor_ix]
                                )
                            )
                            for factor_ix in factor_ixs
                            if factor_to_maxclique[factor_ix] is None
                        ]
                    )
        )

    def find_unconnected_neighbors(neighbors, edges):
        '''Create a list of tuples representing edges between unconnected neighbors

        :param neighbors: list of keys representing neighbors in a factor
        :param edges: view of keys (frozensets representing a graph edge)
        :return:
        '''

        return [
                (k1,k2)
                for k1,k2 in combinations(neighbors, 2)
                if frozenset((k1, k2)) not in edges
        ]

    def find_maxclique(cluster, max_cliques):
        '''Identifies the index of max clique which contains cluster of keys

        :param cluster: a list of keys
        :param max_cliques: list of list of keys (representing a max clique)
        :return: the id of the clique for which the cluster is a subset, -1 otherwise
        '''
        search_results = [
            clique_ix
            for clique_ix, clique in enumerate(max_cliques)
            if set(cluster) < set(clique)
        ]
        return -1 if len(search_results) == 0 else search_results[0]


    # NOTE: Only keys that have been used at least in one factor should be
    # used. Ignore those key sizes that are not in any factor. Perhaps this
    # could be fixed elsewhere. Just added a quick fix here to filter key
    # sizes.

    used_keys = list(
        set(key for factor in factors for key in factor)
    )
    key_sizes = {
        key: size
        for (key, size) in key_sizes.items()
        if key in used_keys
    }

    factor_edges = factors_to_undirected_graph(factors)

    if len(factor_edges) == 0:
        # no edges present in factor graph
        return ([], factors, factors, {i:i for i in range(len(factors))})

    tri = []
    max_cliques = []

    factor_to_maxclique = [None]*len(factors)

    subsets = generate_subsets(factors)

    heap, entry_finder = initialize_triangulation_heap(key_sizes, factor_edges)

    rem_keys = used_keys

    while len(rem_keys) > 0:
        entry, heap, entry_finder, rem_keys = remove_next(
                                                        heap,
                                                        entry_finder,
                                                        rem_keys,
                                                        key_sizes,
                                                        factor_edges
        )

        # key is the 3rd element in entry list
        key = entry[2]

        rem_neighbors = []
        origin_factors = []

        # find neighbors that are in remaining keys
        for r_key in rem_keys:
            edge_set = frozenset([key, r_key])

            if edge_set in factor_edges:
                rem_neighbors.append(r_key)
                origin_factors.extend(find_origin_factors(factor_edges[edge_set], subsets, factor_to_maxclique))

        if len(origin_factors) > 0:
            # implies that list of origin factors not yet accounted for in existing maxcliques

            new_edges = find_unconnected_neighbors(rem_neighbors, factor_edges.keys())

            # connect all unconnected neighbors of key
            factor_edges.update({frozenset(edge): set() for edge in new_edges})
            tri.extend(new_edges)

            # if possible, assign factor to maxclique which is either
            # the factor itself or a factor which it is a subset of

            new_cluster = rem_neighbors + [key]

            maxclique_ix = find_maxclique(new_cluster, max_cliques)

            # new maxclique discovered if maxclique == -1

            max_cliques.extend( [] if maxclique_ix != -1 else [sorted(new_cluster)] )
            maxclique_ix = maxclique_ix if maxclique_ix != -1 else len(max_cliques) - 1

            for factor_ix in set(origin_factors):
                factor_to_maxclique[factor_ix] = maxclique_ix

    return tri, max_cliques, factor_to_maxclique


def merge_trees(tree1, clique1_ix, tree2, clique2_ix, sepset_ix):
    '''Merge two trees into one separated by sepset

    :param tree1: tree structure (a list) containing clique_1
    :param clique1_ix: clique_id for clique_1
    :param tree2: tree structure (a list) containing clique_2
    :param clique2_ix: clique_id for clique_2
    :param sepset_ix: sepset id for the sepset to be inserted
    :return tree: tree structure (a list) containing clique_1, clique_2, and sepset
    '''

    t2 = copy.deepcopy(tree2)

    # combine tree2 (rooted by clique2) with sepset
    sepset_group = (sepset_ix, change_root(t2, clique2_ix))

    merged_tree = insert_sepset(tree1, clique1_ix, sepset_group)

    return merged_tree


def change_root(tree, clique_ix, child=[], sep=[]):
    '''Restructures tree so that clique becomes root

    :param tree: tree to be altered
    :param clique_ix: id of the clique that will become tree's root
    :param child: child tree to be added to new root of tree
    :param sep: separator connecting root to recursively constructed child tree
    :return: tree with clique_ix as root

    If clique_ix is already root of tree, original tree is returned.
    If clique_ix not in tree, empty list is returned.
    '''


    if tree[0] == clique_ix:
        if len(child) > 0:
            tree.append((sep[0],child))
        return tree

    return list(
        chain(
                *[
                    change_root(
                                child_sepset[1],
                                clique_ix,
                                tree[:c_ix+1] + tree[c_ix+2:] + [(sep[0],child)] if len(child) else tree[:c_ix+1] + tree[c_ix+2:],
                                [child_sepset[0]]
                    )
                    for c_ix, child_sepset in enumerate(tree[1:])
                ]
        )
    )


def insert_sepset(tree, clique_ix, sepset_group):
    '''Inserts sepset into tree as child of clique

    :param tree: tree structure (a list) in which to insert sepset
    :param clique_ix: clique id of the sepset's parent
    :param sepset_group: sepset group being inserted
    :return new_tree: tree structure with the sepset inserted as a child of clique
    '''

    return [tree[0]] + list(
                            chain(
                                *[
                                    [(child_sepset[0], insert_sepset(child_sepset[1], clique_ix, sepset_group))]
                                    for child_sepset in tree[1:]
                                ],
                                [] if tree[0] != clique_ix else [(sepset_group)]
                            )
    )

