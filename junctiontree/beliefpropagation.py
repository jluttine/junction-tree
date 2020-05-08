import numpy as np
import heapq
import copy
from itertools import combinations

# FIXME: Cyclic import

from .sum_product import SumProduct


def factors_to_undirected_graph(factors):
    ''' Represent factor graph as undirected graph

    :param factors: list of factors
    :return: undirected graph as dictionary with edges as keys and the factor from
            which edge originates as values
    '''

    #edges = {}
    #adj_list = {}
    factor_edges = {}

    for factor_ix, factor in enumerate(factors):
        for ix, k1 in enumerate(factor):
            for k2 in factor[ix+1:]:

                #adj_list.setdefault(k1, set()).add(k2)
                #adj_list.setdefault(k2, set()).add(k1)

                factor_edges.setdefault( frozenset( (k1,k2) ), set() ).add(factor_ix)

    return factor_edges#, adj_list
    #return edges


def find_triangulation(factors, key_sizes):
    ''' Triangulate given factor graph.

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
        ''' For each factor, identify all factors that are subset of that factor

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
        ''' Creates a list of original factors that contain an edge

        :param factor_ixs: the original factors containing the edge
        :param subsets: dictionary of factor id as key and factor id of subset factors as value
        :param factor_to_maxclique: list mapping factor by id to assigned maxclique
        :return: a list of the original factors of factor graph which contain the edge
        '''

        return (
                sum(
                    [
                        sum( # adding this factor id and factor ids of factors that are subsets
                            [[factor_ix]],
                            subsets.get(factor_ix, [])
                        ) if (factor_to_maxclique[factor_ix] is None) else []
                        for factor_ix in factor_ixs
                    ],
                    []
                )
        )


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


    tri = []
    induced_clusters = []
    induced_cluster_to_maxclique = {}
    max_cliques = []

    factor_edges = factors_to_undirected_graph(factors)

    if len(factor_edges) == 0:
        # no edges present in factor graph
        return (
                [],
                [factor for factor in factors],
                [factor for factor in factors],
                {i:i for i in range(len(factors))}
        )


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

            # connect all unconnected neighbors of key
            new_edges = [
                (k1,k2)
                for k1,k2 in combinations(rem_neighbors, 2)
                if frozenset((k1, k2)) not in factor_edges
            ]

            factor_edges.update({frozenset(edge): set() for edge in new_edges})
            tri.extend(new_edges)

            new_ic_ix = len(induced_clusters)

            # assign factor to maxclique which is either
            # the factor itself or a factor which it is a subset of

            new_cluster = rem_neighbors + [key]

            clusters_maxclique = [
                induced_cluster_to_maxclique[ic_ix]
                for ic_ix, cluster in enumerate(induced_clusters)
                if set(new_cluster) < set(cluster)
            ]

            # new maxclique discovered if length of clusters_maxclique > 0

            max_cliques.extend( [] if len(clusters_maxclique) > 0 else [sorted(new_cluster)] )
            maxclique_ix = clusters_maxclique[0] if len(clusters_maxclique) > 0 else len(max_cliques) - 1
            induced_clusters.extend( [] if len(clusters_maxclique) > 0 else [new_cluster] )

            induced_cluster_to_maxclique[new_ic_ix] = maxclique_ix

            for factor_ix in set(origin_factors):
                factor_to_maxclique[factor_ix] = induced_cluster_to_maxclique[new_ic_ix]

    return tri, induced_clusters, max_cliques, factor_to_maxclique


def initialize_triangulation_heap(key_sizes, edges):
    ''' Create heap used for graph triangulation

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
    ''' Update heap entries

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


def remove_next(heap, entry_finder, remaining_keys, key_sizes, edges):
    ''' Removes next entry from heap

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

def identify_cliques(induced_clusters):
    ''' Generate maximal cliques from induced clusters

        A clique may be composed of multiple factors. See:
        http://www.stat.washington.edu/courses/stat535/fall11/Handouts/l5-decomposable.pdf

        :param induced_clusters: list of clusters generated when finding graph triangulation
        :return: list of maximal cliques where each maximal clique is a list of key indices it contains:

        [clique1, ..., cliqueK]

        That is, if there are N keys, each clique contains some subset of numbers from {0, ..., N-1}
        as a tuple/list.

    '''

    # only retain clusters that are not a subset of another cluster
    sets=[frozenset(c) for c in induced_clusters]
    cliques=[]
    for s1 in sets:
        if any(s1 < s2 for s2 in sets):
            continue
        else:
            cliques.append(sorted(s1))


    return cliques


def build_graph(factors, full=False):
    ''' Builds an adjacency matrix representation for a graph. Nodes in factors
    are connected by edges (non-zero matrix entry) in the graph.

    :param factors: list of factors from which to build a graph
    :param full: create the full (not just upper triangular) matrix
    :return: node_list: a list which maps nodes to index in adjacency matrix
    :return: adj_matrix: a 2-D numpy array representing adjacency matrix
    '''

    sorted_nodes = sorted({ node for factor in factors for node in factor })

    node_lookup = { node : i for i, node in enumerate(sorted_nodes) }

    node_count = len(sorted_nodes)
    adj_matrix = np.full((node_count, node_count), False)


    for factor in factors:
        for i, n1 in enumerate(factor):
            n1_idx = node_lookup[n1]
            for n2 in factor[i+1:]:
                n2_idx = node_lookup[n2]
                # add an edge between nodes
                adj_matrix[n1_idx, n2_idx] = True
                if full:
                    adj_matrix[n2_idx, n1_idx] = True

    return sorted_nodes, adj_matrix


def construct_junction_tree(cliques, key_sizes):
    ''' Construct junction tree from input cliques

    :param cliques: a list of maximal cliques where each maximal clique is a list of key indices it contains
    :param key_sizes: a dictionary of (key label, key size) pairs
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

    heap = build_sepset_heap(sepsets, cliques, key_sizes)
    num_selected = 0

    while num_selected < len(cliques) - 1:
        entry = heapq.heappop(heap)
        ss_ix = entry[2]
        (cliq1_ix, cliq2_ix) = sepsets[ss_ix][1]

        tree1, tree2 = None, None
        for tree in trees:
            # find tree (tree1) containing cliq1_ix
            tree1 = tree1 if tree1 else (tree if find_subtree(tree,cliq1_ix) != [] else None)
            # find tree (tree2) containing cliq2_ix
            tree2 = tree2 if tree2 else (tree if find_subtree(tree,cliq2_ix) != [] else None)

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


def build_sepset_heap(sepsets, cliques, key_sizes):
    '''Build sepset heap to be used for building junction tree from cliques

    :param sepsets: set of candidate sepsets consisting of sets of factor ids and tuple
                    of clique ids which produce sepset
    :param cliques: list of cliques (represented by list of keys)
    :param key_sizes: dictionary of key label as key and key size as value
    :return sepset_heap: heap of sepset entries
    '''

    heap = []

    for i, (ss, (cliq1_ix, cliq2_ix)) in enumerate(sepsets):
        mass = len(ss) + 0.001 # avoids division by zero if sepset empty
        weight1 = np.prod([key_sizes[key] for key in cliques[cliq1_ix]])
        weight2 = np.prod([key_sizes[key] for key in cliques[cliq2_ix]])
        # invert mass to use minheap
        entry = [1.0/mass, weight1 + weight2, i]
        heapq.heappush(heap, entry)

    return heap

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


def insert_sepset(tree, clique_ix, sepset_group):
    ''' Inserts sepset into tree as child of clique

    :param tree: tree structure (a list) in which to insert sepset
    :param clique_ix: clique id of the sepset's parent
    :param sepset_group: sepset group being inserted
    :return new_tree: tree structure with the sepset inserted as a child of clique
    '''

    return [tree[0]] + sum(
        [
            [(child_sepset[0], insert_sepset(child_sepset[1], clique_ix, sepset_group))]
            for child_sepset in tree[1:]
        ],
        [] if tree[0] != clique_ix else [(sepset_group)]
    )


def find_subtree(tree, clique_ix):
    ''' Find subtree rooted by clique

    :param tree: tree structure (a list) to search
    :param clique_ix: id of the clique serving as root of subtree
    :return sub_tree: a copy of the subtree rooted by clique_ix if present, an empty list otherwise
    '''

    #TODO: Try to return a reference to the subtree rather than
    #a newly allocated version

    return ([] if tree[0] != clique_ix else tree) + sum(
                            [
                                find_subtree(child_tree, clique_ix)
                                for child_tree in tree[1:]
                            ],
                            []
    )


def change_root(tree, clique_ix, child=[], sep=[]):
    ''' Restructures tree so that clique becomes root

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


    return  sum(
                [
                    change_root(
                                child_sepset[1],
                                clique_ix,
                                tree[:c_ix+1] + tree[c_ix+2:] + [(sep[0],child)] if len(child) else tree[:c_ix+1] + tree[c_ix+2:],
                                [child_sepset[0]]
                    )
                    for c_ix, child_sepset in enumerate(tree[1:])
                ],
                []
            )


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
        map the keys for the current numpy.einsum to unique integers starting
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

        return sum_product.einsum(*args)

    return __run(junction_tree, junction_tree[1])



def collect(tree, node_list, potentials, visited, distributive_law, shrink_mapping=None):
    """
    Used by Hugin algorithm to collect messages

    Input:
    ------

    The tree structure of the junction tree

    List of nodes in tree

    List of clique potentials

    List of boolean entries representing visited status of cliques

    Distributive law for performing sum product calculations

    Shrink mapping for cliques

    Output:
    -------

    Updated potentials for collect phase of propagation


    """
    sm = shrink_mapping
    clique_ix = tree[0]
    clique_keys = node_list[clique_ix]
    # set clique_index in visited to 1
    visited[clique_ix] = 1

    # loop over neighbors of root of tree
    for neighbor in tree[1:]:
        sep_ix, child = neighbor
        sep_keys = node_list[sep_ix]
        child_ix = child[0]
        child_keys = node_list[child_ix]
        # call collect on neighbor if not marked as visited
        if not visited[child_ix]:
            potentials = collect(
                            child,
                            node_list,
                            potentials,
                            visited,
                            distributive_law,
                            shrink_mapping
            )

            new_clique_pot, new_sep_pot = distributive_law.update(
                                        potentials[child_ix] if not sm else potentials[child_ix][sm[child_ix][0]],
                                        child_keys if not sm else sm[child_ix][1],
                                        potentials[clique_ix] if not sm else potentials[clique_ix][sm[clique_ix][0]],
                                        clique_keys if not sm else sm[clique_ix][1],
                                        potentials[sep_ix] if not sm else potentials[sep_ix][sm[sep_ix][0]],
                                        sep_keys if not sm else sm[sep_ix][1]
            )

            # ensure that values are assigned to proper positions in potentials list
            if sm:
                potentials[clique_ix][sm[clique_ix][0]] = new_clique_pot
                potentials[sep_ix][sm[sep_ix][0]] = new_sep_pot
            else:
                potentials[clique_ix] = new_clique_pot
                potentials[sep_ix] = new_sep_pot



    # return the updated potentials
    return potentials


def distribute(tree, node_list, potentials, visited, distributive_law, shrink_mapping=None):
    """
    Used by Hugin algorithm to distribute messages

    Input:
    ------

    The tree structure of the junction tree

    List of nodes in tree

    List of clique potentials

    List of boolean entries representing visited status of cliques

    Distributive law for performing sum product calculations

    Shrink mapping for cliques

    Output:
    -------

    List of updated potentials for distribute phase of propagation

    """
    sm = shrink_mapping
    # set clique_index in visited to 1
    clique_ix = tree[0]
    clique_keys = node_list[clique_ix]
    visited[clique_ix] = 1

    # loop over neighbors of root of tree
    for neighbor in tree[1:]:
        sep_ix, child = neighbor
        sep_keys = node_list[sep_ix]
        child_ix = child[0]
        child_keys = node_list[child_ix]
        # call distribute on neighbor if not marked as visited
        if not visited[child_ix]:
            new_clique_pot, new_sep_pot = distributive_law.update(
                                        potentials[clique_ix] if not sm else potentials[clique_ix][sm[clique_ix][0]],
                                        clique_keys if not sm else sm[clique_ix][1],
                                        potentials[child_ix] if not sm else potentials[child_ix][sm[child_ix][0]],
                                        child_keys if not sm else sm[child_ix][1],
                                        potentials[sep_ix] if not sm else potentials[sep_ix][sm[sep_ix][0]],
                                        sep_keys if not sm else sm[sep_ix][1]
            )

            # ensure that values are assigned to proper positions
            if sm:
                potentials[child_ix][sm[child_ix][0]] = new_clique_pot
                potentials[sep_ix][sm[sep_ix][0]] = new_sep_pot
            else:
                potentials[child_ix] = new_clique_pot
                potentials[sep_ix] = new_sep_pot

            potentials = distribute(
                                child,
                                node_list,
                                potentials,
                                visited,
                                distributive_law,
                                shrink_mapping
            )

    # return the updated potentials
    return potentials


def hugin(tree, node_list, potentials, distributive_law, shrink_mapping=None):
    """
    Run hugin algorithm by using the given distributive law.

    Input:
    ------

    The tree structure of the junction tree

    List of nodes in tree

    List of (inconsistent) clique potentials

    Distributive law for performing sum product calculations

    Shrink mapping for cliques


    Output:
    -------

    List of (consistent) clique potentials

    See page 3:
    http://compbio.fmph.uniba.sk/vyuka/gm/old/2010-02/handouts/junction-tree.pdf
    """
    # initialize visited array which has the same number of elements as potentials array
    visited = [0]*len(potentials)

    # call collect on root_index storing the result in new_potentials
    new_potentials = collect(
                        tree,
                        node_list,
                        potentials,
                        visited,
                        distributive_law,
                        shrink_mapping
    )

    # initialize visited array again
    visited = [0]*len(new_potentials)

    # return the result of a call to distribute on root index
    return distribute(
                    tree,
                    node_list,
                    new_potentials,
                    visited,
                    distributive_law,
                    shrink_mapping
    )

def get_clique(tree, key_list, key_label):
    ''' Finds a single clique containing key with label key_label

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


def compute_marginal(potential, clique_keys, key_ix):
    """
    Compute marginal over potential for key

    Input:
    ------

    Potential to use for marginalization

    List of keys included in clique

    Key to use for marginalization

    Output:
    -------

    Marginal value of key

    """

    if key_ix not in clique_keys:
        return 0.0


    return sum_product.einsum(
                        potential,
                        clique_keys,
                        [key_ix]
    )

def yield_id(tree):
    """
    Function making it possible to yield id of tree's root

    Input:
    ------

    The tree structure of the junction tree

    Output:
    -------

    Clique ID of tree root

    """
    yield tree[0]

def yield_id_and_keys(tree):
    """
    Function making it possible to yield id and keys of tree's root

    Input:
    ------

    The tree structure of the junction tree

    Output:
    -------

    Clique ID and clique keys of tree root

    """
    yield tree[0]
    yield tree[1]

def yield_clique_pairs(tree):
    """

    Function making it possible to yield clique id/keys and
        child separators

    Input:
    ------

    The tree structure of the junction tree

    Output:
    -------

    Tuples of root clique id and child separator ids

    """
    for child in tree[1:]:
        yield (tree[0], child[0])


def bf_traverse(tree, clique_ix=None, func=yield_id):
    """
    Breadth-first traversal of tree

    Early termination of search is performed if clique_id provided

    Input:
    ------

    Tree structure to traverse

    (Optional) Clique ID used to terminate traversal

    (Optional) Function controlling output

    Output:
    -------

    Depends on func argument. Default is list of clique
        ids

    [id1, ..., idN] (or [id1, ..., cid])
    """

    queue = [tree]
    while queue:
        tree = queue.pop(0)
        yield from func(tree)
        if tree[0] == clique_ix:
            raise StopIteration
        queue.extend([child for child in tree[1:]])


def df_traverse(tree, clique_ix=None, func=yield_id):
    """
    Depth-first traversal of tree

    Early termination of search is performed if clique_id provided

    Input:
    ------

    Tree structure to traverse

    (Optional) Clique ID used to terminate traversal

    (Optional) Function controlling output


    Output:
    -------

    Depends on func argument. Default is list of clique
        ids

    [id1, ..., idN] (or [id1, ..., cid])

    """

    stack = [tree]
    while stack:
        tree = stack.pop()
        yield from func(tree)
        if tree[0] == clique_ix:
            raise StopIteration
        stack.extend([child for child in reversed(tree[1:])])


def get_clique_keys(node_list, clique_ix):
    """
    Return keys for clique with ID clique_ix
        (if clique_ix not in tree return None)

    Input:
    ------

    List of nodes (maxcliques + separators)

    Clique ID to find

    Output:
    -------

    A list containing clique keys (or None)

    """

    return node_list[clique_ix] if len(node_list) > clique_ix else None


def get_cliques(tree, node_list, key):
    """
    Return the (M) cliques (clique id/clique keys pairs) which
        include key and all other keys in clique

    Input:
    ------

    Tree structure to traverse

    Key to find

    Output:
    -------

    List of clique ids and corresponding keys containing key

    [clique_wkey_id1, clique_wkey_keys1, ..., clique_wkey_idM, clique_wkey_keysM]
    """

    flist = list(bf_traverse(tree))
    return [
            (clique_ix, node_list[clique_ix])
                for clique_ix in flist if key in node_list[clique_ix]
    ]

def get_clique_of_key(tree, node_list, key):
    """
    Returns a clique ID/keys containing key (if exists)

    Input:
    ------

    Tree structure of the junction tree

    Key to find

    Output:
    -------

    First clique ID/clique keys which contains key (or (None,None) pair)

    """

    ix = tree[0]
    keys = node_list[ix]
    separators = tree[1:]

    if key in keys:
        return ix, keys
    if separators == (): # base case reached (leaf)
        return None, None

    for separator in separators:
        separator_ix, c_tree = separator
        separator_keys = node_list[separator_ix]
        if key in separator_keys:
            return separator_ix, separator_keys
        clique_ix, clique_keys = get_clique_of_key(c_tree, node_list, key)
        if clique_ix != None:
            return clique_ix, clique_keys

    return None, None

def generate_potential_pairs(tree):
    """
    Returns cliques and child separators

    Input:
    ------

    Tree structure of the junction tree

    Output:
    -------

    List of clique id/child sep id pairs

    [
        (clique_id0, child0_sep_id0),
        (clique_id0, child1_sep_id0),
        (clique_id1, child0_sep_id1),
        ...
        (clique_idN, child(M-1)_sep_idN),
        (clique_idN, childM_sep_idN)
    ]

    """
    return list(bf_traverse(tree, func=yield_clique_pairs))



# Sum-product distributive law for NumPy
sum_product = SumProduct(np.einsum)
'''
TODO: setting optimize to true allows einsum to benefit from speed up due to
contraction order optimization but at the cost of memory usage
need to evaluate tradeoff within library
'''
#sum_product = SumProduct(np.einsum,optimize=True)
