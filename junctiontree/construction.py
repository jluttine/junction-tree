import numpy as np
import heapq
from itertools import chain, combinations
import copy

def get_clique(tree, node_list, var_label):
    '''Finds a single clique containing variable

    :param tree: the tree structure (a list) of the junction tree
    :param key_list: contains the node indexed by clique id for all cliques in tree
    :param var_label: the variable label of variable being searched for
    :return: a clique containing the variable or None if no such clique exists in tree
    '''

    ix = tree[0]
    node = node_list[ix]
    separators = tree[1:]
    if var_label in node:
        return ix, node

    if separators == (): # base case reached (leaf)
        return None

    for separator in separators:
        separator_ix, c_tree = separator
        separator_vars = node_list[separator_ix]

        if var_label in separator_vars:
            return separator_ix, separator_vars

        clique_info = get_clique(c_tree, node_list, var_label)

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


def initialize_triangulation_heap(var_sizes, edges):
    '''Create heap used for graph triangulation

    :param key_sizes: dictionary with variable label as keys and variable size as values
    :param edges: list of pairs of variables representing factor graph edges
    :return heap: heap with entry structure:

            [
                num edges added to triangulated graph by removal of variable,
                induced cluster weight,
                variable associated with first two elements
            ]
    :return entry_finder: dictionary with variable label as key and reference to heap entry for variable

    '''

    heap, entry_finder = update_heap(var_sizes.keys(), edges, var_sizes)

    return heap, entry_finder


def update_heap(remaining_vars, edges, var_sizes, heap=None, entry_finder=None):
    '''Update heap entries

    :param remaining_vars: list of variables remaining in the heap
    :param edges: list of edges (pairs of variables )
    :param var_sizes: dictionary of variables (variable label is key, size is value)
    :param heap: heap to be updated (None if new heap is to be created)
    :param entry_finder: entry_finder dictionary with references to heap elements
    :return h: updated (or newly created) heap
    :return entry_finder: dictionary with updated references to heap elements
    '''

    h = heap if heap else []
    entry_finder = entry_finder if entry_finder else {}
    for var in remaining_vars:
        rem_neighbors = [(set(edge) - set([var])).pop()
                            for edge in edges if var in edge and len(set(remaining_vars).intersection(edge)) == 2]

        # determine how many of var's remaining neighbors need to be connected
        num_new_edges = sum(
                            [
                                frozenset((n1,n2)) not in edges
                                for i, n1 in enumerate(rem_neighbors)
                                    for n2 in rem_neighbors[i+1:]

                            ]
        )
        # weight of a cluster is the product of all key lengths in cluster
        weight = var_sizes[var] * np.prod([var_sizes[n] for n in rem_neighbors])
        entry = [num_new_edges, weight, var]
        heapq.heappush(h, entry)
        # invalidate previous entry if it exists
        prev = entry_finder.get(var, None)
        if prev:
            # set entry to be removed
            prev[2] = ""

        entry_finder[var] = entry

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


def remove_next(heap, entry_finder, remaining_vars, var_sizes, edges):
    '''Removes next entry from heap

    :param heap: heap structure containing remaining factors and weights
    :param entry_finder: dictionary with updated references to heap elements
    :param remaining_vars: list of variables remaining in G'
    :param var_sizes: dictionary of variables (variable label is key, size is value)
    :param edges: list of edge pairs in original graph G
    :return entry: the entry removed from the heap
    :return heap: heap structure with updated entries after variable removal
    :return entry_finder: dictionary with updated references to heap elements
    :return remaining_vars: list of variables without most recently removed variable
    '''

    entry = (None, None, "")

    while entry[2] == "":
        entry = heapq.heappop(heap)

    # remove entry from entry_finder
    del entry_finder[entry[2]]

    # remove key from remaining keys list
    remaining_vars.remove(entry[2])


    heap, entry_finder = update_heap(
                                remaining_vars,
                                edges,
                                var_sizes,
                                heap,
                                entry_finder
    )

    return entry, heap, entry_finder, remaining_vars

def find_triangulation(factors, var_sizes):
    '''Triangulate given factor graph.

    TODO: Provide different algorithms.

    :param factors: list of factors where each factor is given as a list of variables the factor contains:

            [vars1, ..., varsN]

    :param var_sizes: dictionary of variables (variable label is key, size is value)

            {
                key1: size1,
                ...
                keyM: sizeM
            }

    :return tri: list of edges added to triangulate the undirected graph
    :return induced_clusters: list of variable lists representing induced clusters from triangulation
    :return max_cliques: list of maximal cliques generated during triangulation process
    :return factor_to_maxclique: dictionary mapping each factor to the max_clique which contains the factor
    '''

    def generate_subsets(factors):
        '''For each factor, identify all factors that are subset of that factor

        :param factors: list of factors (list of variables) representing the factor graph
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

        :param neighbors: list of variables representing neighbors in a factor
        :param edges: view of variables (frozensets representing a graph edge)
        :return:
        '''

        return [
                (k1,k2)
                for k1,k2 in combinations(neighbors, 2)
                if frozenset((k1, k2)) not in edges
        ]

    def find_maxclique(cluster, max_cliques):
        '''Identifies the index of max clique which contains cluster of variables

        :param cluster: a list of variables
        :param max_cliques: list of variable lists (representing a max clique)
        :return: the id of the clique for which the cluster is a subset, -1 otherwise
        '''
        search_results = [
            clique_ix
            for clique_ix, clique in enumerate(max_cliques)
            if set(cluster) < set(clique)
        ]
        return -1 if len(search_results) == 0 else search_results[0]


    # NOTE: Only variables that have been used at least in one factor should be
    # used. Ignore those variable sizes that are not in any factor. Perhaps this
    # could be fixed elsewhere. Just added a quick fix here to filter variable
    # sizes.

    used_vars = list(
        set(var for factor in factors for var in factor)
    )
    var_sizes = {
        var: size
        for (var, size) in var_sizes.items()
        if var in used_vars
    }

    factor_edges = factors_to_undirected_graph(factors)

    if len(factor_edges) == 0:
        # no edges present in factor graph
        return ([], factors, factors, {i:i for i in range(len(factors))})

    tri = []
    max_cliques = []

    factor_to_maxclique = [None]*len(factors)

    subsets = generate_subsets(factors)

    heap, entry_finder = initialize_triangulation_heap(var_sizes, factor_edges)

    rem_vars = used_vars

    while len(rem_vars) > 0:
        entry, heap, entry_finder, rem_vars = remove_next(
                                                        heap,
                                                        entry_finder,
                                                        rem_vars,
                                                        var_sizes,
                                                        factor_edges
        )

        # var is the 3rd element in entry list
        var = entry[2]

        rem_neighbors = []
        origin_factors = []

        # find neighbors that are in remaining keys
        for r_var in rem_vars:
            edge_set = frozenset([var, r_var])

            if edge_set in factor_edges:
                rem_neighbors.append(r_var)
                origin_factors.extend(find_origin_factors(factor_edges[edge_set], subsets, factor_to_maxclique))

        if len(origin_factors) > 0:
            # implies that list of origin factors not yet accounted for in existing maxcliques

            new_edges = find_unconnected_neighbors(rem_neighbors, factor_edges.keys())

            # connect all unconnected neighbors of key
            factor_edges.update({frozenset(edge): set() for edge in new_edges})
            tri.extend(new_edges)

            # if possible, assign factor to maxclique which is either
            # the factor itself or a factor which it is a subset of

            new_cluster = rem_neighbors + [var]

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


def generate_potential_pairs(tree):
    '''Returns a list of tuples consisting of clique id and child separator ids

    :param tree: tree structure in list format
    :return: list of clique id/child sep id tuples

    [
        (clique_id0, child0_sep_id0),
        (clique_id0, child1_sep_id0),
        (clique_id1, child0_sep_id1),
        ...
        (clique_idN, child(M-1)_sep_idN),
        (clique_idN, childM_sep_idN)
    ]
    '''

    return list(bf_traverse(tree, func=yield_clique_pairs))


def yield_id(tree):
    '''Yields id of tree's root

    :param tree: tree structure in list format
    '''

    yield tree[0]


def bf_traverse(tree, clique_ix=None, func=yield_id):
    '''Breadth-first search traversal with optional early termination

    :param tree: tree structure in list format
    :param clique_ix: clique id used to terminate traversal
    :param func: function controlling component of tree output

    Output: Depends on func argument. Default is list of clique ids

    [id1, ..., idN] (or [id1, ..., cid])
    '''

    queue = [tree]
    while queue:
        tree = queue.pop(0)
        yield from func(tree)
        if tree[0] == clique_ix:
            raise StopIteration
        queue.extend([child for child in tree[1:]])


def yield_clique_pairs(tree):
    '''Yields tuples of root clique id and sepset id

    :param tree: tree structure in list format
    '''

    for child in tree[1:]:
        yield (tree[0], child[0])


def get_clique_vars(clique_vars, clique_ix):
    '''Get variables of the clique with id clique_ix

    :param clique_vars: list of variables (maxclique + separators)
    :param clique_ix: clique id to find
    :return: list of variables in clique clique_ix (or None if clique_ix not in tree)
    '''

    return clique_vars[clique_ix] if len(clique_vars) > clique_ix else None


def df_traverse(tree, clique_ix=None, func=yield_id):
    '''Depth-first traversal with optional early termination

    :param tree: tree structure in list format
    :param clique_ix: clique id used to terminate traversal
    :param func: function controlling component of tree output

    Output: Depends on func argument. Default is list of clique ids

    [id1, ..., idN] (or [id1, ..., cid])
    '''

    stack = [tree]
    while stack:
        tree = stack.pop()
        yield from func(tree)
        if tree[0] == clique_ix:
            raise StopIteration
        stack.extend([child for child in reversed(tree[1:])])


def construct_junction_tree(cliques, var_sizes):
    '''Construct junction tree from input cliques

    :param cliques: a list of maximal cliques where each maximal clique is a list of variables it contains
    :param var_sizes: a dictionary of (var label, var size) pairs
    :return tree: a junction tree structure from the input cliques
    :return separators: a list of separators in the order in which they appear in the tree.

    Note: Empty separator sets indicate the presence of distinct unconnected trees in the structure
    '''

    trees = [[c_ix] for c_ix, clique in enumerate(cliques)]
    # set of candidate sepsets
    sepsets = list()
    for i, X in enumerate(cliques):
        for j, Y in enumerate(cliques[i+1:]):
            sepset = tuple(set(X).intersection(Y))
            sepsets.append((sepset, (i,j+i+1)))

    separator_dict = {}

    heap = build_sepset_heap(sepsets, cliques, var_sizes)
    num_selected = 0

    while num_selected < len(cliques) - 1:
        entry = heapq.heappop(heap)
        ss_ix = entry[2]
        (cliq1_ix, cliq2_ix) = sepsets[ss_ix][1]

        tree1, tree2 = None, None
        for tree in trees:
            # find tree (tree1) containing cliq1_ix
            tree1 = tree1 if tree1 else (tree if find_subtree(tree,cliq1_ix) else None)
            # find tree (tree2) containing cliq2_ix
            tree2 = tree2 if tree2 else (tree if find_subtree(tree,cliq2_ix) else None)

        if tree1 != tree2:
            ss_tree_ix = len(cliques) + num_selected
            # merge tree1 and tree2 into new_tree
            new_tree = merge_trees(
                                tree1,
                                cliq1_ix,
                                tree2,
                                cliq2_ix,
                                ss_tree_ix
            )
            separator_dict[ss_tree_ix] = sepsets[ss_ix][0]
            # insert new_tree into forest
            trees.append(new_tree)

            # remove tree1 and tree2 from forest
            trees.remove(tree1)
            trees.remove(tree2)
            num_selected += 1

    # trees list contains one tree which is the fully constructed tree
    return trees[0], [list(separator_dict[ix]) for ix in sorted(separator_dict.keys())]


def build_sepset_heap(sepsets, cliques, var_sizes):
    '''Build sepset heap to be used for building junction tree from cliques

    :param sepsets: set of candidate sepsets consisting of sets of factor ids and tuple
                    of clique ids which produce sepset
    :param cliques: list of cliques (represented by variable list)
    :param key_sizes: dictionary of variable label as key and variable size as value
    :return sepset_heap: heap of sepset entries
    '''

    heap = []

    for i, (ss, (cliq1_ix, cliq2_ix)) in enumerate(sepsets):
        mass = len(ss) + 0.001 # avoids division by zero if sepset empty
        weight1 = np.prod([var_sizes[var] for var in cliques[cliq1_ix]])
        weight2 = np.prod([var_sizes[var] for var in cliques[cliq2_ix]])
        # invert mass to use minheap
        entry = [1.0/mass, weight1 + weight2, i]
        heapq.heappush(heap, entry)

    return heap


def find_subtree(tree, clique_ix):
    '''Evaluates if subtree rooted by clique exists in tree

    :param tree: tree structure (a list) to search
    :param clique_ix: id of the clique serving as root of subtree
    :return tree_found: True if subtree rooted by clique_ix, False otherwise
    '''

    if tree[0] == clique_ix:
        return True
    elif len(tree) == 1:
        return False
    else:
        for child_tree in tree[1:]:
            if find_subtree(child_tree, clique_ix):
                return True

    return False


def generate_potential_pairs(tree):
    '''Returns a list of tuples consisting of clique id and child separator ids

    :param tree: tree structure in list format
    :return: list of clique id/child sep id tuples

    [
        (clique_id0, child0_sep_id0),
        (clique_id0, child1_sep_id0),
        (clique_id1, child0_sep_id1),
        ...
        (clique_idN, child(M-1)_sep_idN),
        (clique_idN, childM_sep_idN)
    ]
    '''

    return list(bf_traverse(tree, func=yield_clique_pairs))
