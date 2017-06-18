"""Belief propagation, junction trees etc

Everything in this file is just rough sketching and ideas at the moment.
Probably many things will change when implementing and learning.


Factor graphs:
--------------

A factor graph is given as a list of keys that tell which variables are in the
factor. (A key corresponds to a variable.)

[keys1, ..., keysN]  # a list of N factors

The index in the list can be used as an ID for the factor, that is, the first
factor in the list has ID 0 and the last factor has ID N-1.

A companion list (of numpy arrays) of the same length as the factor list is
provided as a representation for the factor values

[values1, ..., valuesN]

Also, the size of each of the M variables can be given as a dictionary:

{
    key1: size1,
    ...
    keyM: sizeM
}

Here, size is an integer telling the size of the variable. It is the same as
the length of the corresponding axis in the array later.

No moralization function should be needed.


Generic trees (recursive definition):
-------------------------------------

[index, keys, child_tree1, ..., child_treeN]

The index can, for instance, refer to the index of the factor?


Junction trees:
---------------


[index, keys, (separator1_keys, child_tree1), ..., (separatorN_keys, child_treeN)]


Junction trees:

[
    index, keys
    (
        separator1_index, separator1_keys
        child_tree1
    ),
    ...,
    (
        separatorN_index, separatorN_keys
        child_treeN
    )
]

Potentials in (junction) trees:
-------------------------------

A list/dictionary of arrays. The node IDs in the tree graphs map
to the arrays in this data structure in order to get the numeric
arrays in the execution phase. The numeric arrays are not needed
in the compilation phase.


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
import heapq
import copy

def factors_to_undirected_graph(factors):
    """
        Represent factor graph as undirected graph

        Inputs:
        -------

        List of factors

        Output:
        -------

        Undirected graph as dictionary of edges
    """

    edges = {}

    for factor in factors:
        factor_set = set(factor)
        for v1 in factor:
            for v2 in factor_set - set([v1]):
                edges.setdefault(frozenset((v1,v2)), None)

    return edges


def find_triangulation(factors, key_sizes):
    """Triangulate given factor graph.

    TODO: Provide different algorithms.

    Inputs:
    -------

    A list of factor where each factor is given as a
        list of keys the factor contains:

    [keys1, ..., keysN]

    Also, give the sizes of the variables as a dictionary:

    {
        key1: size1,
        ...
        keyM: sizeM
    }

    Output:
    -------

    A list of edges added to triangulate the undirected graph

    A list of key lists representing induced clusters from triangulation

    """

    tri = []
    induced_clusters = []
    edges = factors_to_undirected_graph(factors)
    heap, entry_finder = initialize_triangulation_heap(
                                            key_sizes,
                                            edges
    )
    rem_keys = list(key_sizes.keys())
    while len(rem_keys) > 0:
        item, heap, entry_finder, rem_keys = remove_next(
                                                        heap,
                                                        entry_finder,
                                                        rem_keys,
                                                        key_sizes,
                                                        edges
        )
        key = item[2]

        rem_neighbors = [(set(edge) - set(key)).pop()
                            for edge in edges if key in edge and len(set(rem_keys).intersection(edge)) == 1]

        induced_clusters.append(rem_neighbors + [key])
        # connect all unconnected neighbors of key
        for i, n1 in enumerate(rem_neighbors):
            for n2 in rem_neighbors[i+1:]:
                if frozenset((n1,n2)) not in edges:
                    edges[frozenset((n1,n2))] = None
                    tri.append((n1,n2))

    return tri, induced_clusters


def triangulate(triangulation, arrays):
    """
    Apply pre-computed triangulation

    Inputs:
    -------

    Triangulation returned by find_triangulation.

    List of key lists for the factors

    Output:
    -------

    List of key lists for the cliques.

    """
    raise NotImplementedError()


def initialize_triangulation_heap(key_sizes, edges):
    """
    Input:
    ------

    A dictionary with key label as keys and variable size as values

    A list of pairs of keys representing factor graph edges


    Output:
    -------
    A heap of entries where entry has structure:

    [
        num edges added to triangulated graph by removal of key,
        induced cluster weight,
        key associated with first two elements
    ]

    A dictionary with key label as key and reference
        to heap entry for key
    """

    heap, entry_finder = update_heap(key_sizes.keys(), edges, key_sizes)

    return heap, entry_finder


def update_heap(remaining_keys, edges, key_sizes, heap=None, entry_finder=None):
    """
    Input:
    ------

    list of keys remaining

    list of edges (key pairs)

    dictionary of keys (key label is key, size is value)

    heap to be updated (None if new heap is to be created)

    entry_finder dictionary with references to heap elements

    Output:
    -------

    updated (or newly created) heap

    entry_finder dictionary with updated references to heap elements
    """

    h = heap if heap else []
    entry_finder = entry_finder if entry_finder else {}
    for key in remaining_keys:
        rem_neighbors = [(set(edge) - set(key)).pop()
                            for edge in edges if key in edge and len(set(remaining_keys).intersection(edge)) == 2]

        # determine how many of key's remaining neighbors need to be connected
        num_new_edges = sum(
                            [
                                frozenset((n1,n2)) not in edges
                                for i, n1 in enumerate(rem_neighbors)
                                    for n2 in rem_neighbors[i+1:]

                            ]
        )
        # weight of a cluster is the product of all key lenghts in cluster
        weight = key_sizes[key]*np.prod([key_sizes[n] for n in rem_neighbors])
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
    """
    Input:
    ------

    heap containing remaining factors and weights

    entry_finder dictionary with updated references to heap elements

    list of keys remaining in G'

    key sizes

    list of edge pairs in original graph G

    Output:
    -------

    the entry removed from the heap

    heap with updated keys after factor removal

    entry_finder dictionary with updated references to heap elements

    list of keys without most recently removed key
    """

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
    """
    Input:
    ------

    A list of clusters generated when finding graph triangulation

    Output:
    -------

    A list of maximal cliques where each maximal clique is a list of
        key indices it contains:

    [clique1, ..., cliqueK]

    That is, if there are N keys, each clique contains some subset of
        numbers from {0, ..., N-1} as a tuple/list.

    Notes
    -----
    A clique may contain multiple factors.

    See:
    http://www.stat.washington.edu/courses/stat535/fall11/Handouts/l5-decomposable.pdf


    """

    # only retain clusters that are not a subset of another cluster
    sets=[frozenset(c) for c in induced_clusters]
    cliques=[]
    for s1 in sets:
        if any(s1 < s2 for s2 in sets):
            continue
        else:
            cliques.append(list(s1))


    return cliques

def construct_junction_tree(cliques, key_sizes):
    """
    Input:
    ------

    A list of maximal cliques where each maximal clique is a list of
        key indices it contains

    A dictionary of (key label, key size) pairs

    Output:
    -------

    A list of junction tree structures from the input cliques.
        In most cases, there should only be a single tree in the
        returned list
    """

    trees = [[c_ix, clique] for c_ix, clique in enumerate(cliques)]
    # set of candidate sepsets
    sepsets = list()
    for i, X in enumerate(cliques):
        for j, Y in enumerate(cliques[i+1:]):
            sepset = tuple(set(X).intersection(Y))
            if len(sepset) > 0:
                sepsets.append((sepset, (i,j+i+1)))


    heap = build_sepset_heap(sepsets, cliques, key_sizes)
    num_selected = 0
    while num_selected < len(cliques) - 1:
        entry = heapq.heappop(heap)
        ss_id = entry[2]
        (cliq1_ix, cliq2_ix) = sepsets[ss_id][1]

        tree1, tree2 = None, None
        for tree in trees:
            # find tree (tree1) containing cliq1_ix
            tree1 = tree1 if tree1 else (tree if find_subtree(tree,cliq1_ix) != [] else None)
            # find tree (tree2) containing cliq2_ix
            tree2 = tree2 if tree2 else (tree if find_subtree(tree,cliq2_ix) != [] else None)

        if tree1 != tree2:
            # merge tree1 and tree2 into new_tree
            new_tree = merge_trees(
                                tree1,
                                cliq1_ix,
                                tree2,
                                cliq2_ix,
                                len(cliques) + num_selected,
                                list(sepsets[ss_id][0])
            )

            # insert new_tree into forest
            trees.append(new_tree)

            # remove tree1 and tree2
            trees.remove(tree1)
            trees.remove(tree2)
            num_selected += 1

    return trees


def build_sepset_heap(sepsets, cliques, key_sizes):
    """
    Input:
    ------

    Set of candidate sepsets consisting of sets of factor ids and
        tuple of clique ids which produce sepset

    Cliques of the tree

    Dictionary of key label as key and key size as value

    Output:
    -------

    Heap of sepset entries

    """

    heap = []

    for i, (ss, (cliq1_ix, cliq2_ix)) in enumerate(sepsets):
        mass = len(set(ss))
        weight1 = np.prod([key_sizes[key] for key in cliques[cliq1_ix]])
        weight2 = np.prod([key_sizes[key] for key in cliques[cliq2_ix]])
        # invert mass to use minheap
        entry = [1.0/mass, weight1 + weight2, i]
        heapq.heappush(heap, entry)

    return heap

def merge_trees(tree1, clique1_ix, tree2, clique2_ix, sepset_ix, sepset):
    """
    Input:
    ------

    Tree structure (list) containing clique_1

    The clique id for clique_1

    Tree structure (list) containing clique_2

    The clique id for clique_2

    The sepset id for the sepset to be inserted

    The sepset (list of factor ids) to be inserted

    Output:
    -------

    A tree structure (list) containing clique_1, clique_2, and sepset

    """

    t2 = copy.deepcopy(tree2)

    # combine tree2 (rooted by clique2) with sepset
    sepset_group = (sepset_ix, sepset, change_root(t2, clique2_ix))

    # merged tree
    merged_tree = insert_sepset(tree1, clique1_ix, sepset_group)


    # return the merged trees
    return merged_tree

def insert_sepset(tree, clique_ix, sepset_group):
    """
    Input:
    ------

    Tree structure in which to insert sepset

    The clique id of the sepset's parent

    The sepset group being inserted

    Output:
    -------

    A new tree structure with the sepset inserted as a
        child of clique

    """


    return [tree[0],tree[1]] + sum(
        [
            [(child_sepset[0], child_sepset[1], insert_sepset(child_sepset[2], clique_ix, sepset_group))]
            for child_sepset in tree[2:]
        ],
        [] if tree[0] != clique_ix else [(sepset_group)]
    )

def find_subtree(tree, clique_ix):
    """
    Input:
    ------

    Tree (potentially) containing clique as root

    The id of the clique serving as root of subtree

    Output:
    -------

    A (new) tree rooted by clique_ix if clique_ix is in tree.
        Otherwise return an empty tree ([])


    TODO: Try to return a reference to the subtree rather than
    a newly allocated version
    """

    return ([] if tree[0] != clique_ix else tree) + sum(
        [
            find_subtree(child_tree, clique_ix)
            for child_tree in tree[2:]
        ],
        []
    )

def change_root(tree, clique_ix, child=[], sep=[]):
    """
    Input:
    ------

    Tree to be altered

    ID of the clique that will become tree's root

    Child tree to be added to new root of tree (constructed during recursion)

    Separator connecting root to recursively constructed child tree

    Output:
    -------

    Tree with clique_ix as root


    If clique_ix is already root of tree, tree is returned

    If clique_ix not in tree, empty list is returned
    """

    if tree[0] == clique_ix:
        if len(child) > 0:
            tree.append((sep[0],sep[1],child))
        return tree


    return  sum(
                [
                    change_root(
                                child_sepset[2],
                                clique_ix,
                                tree[:c_ix+2] + tree[c_ix+3:] + [(sep[0],sep[1],child)] if len(child) else tree[:c_ix+2] + tree[c_ix+3:],
                                [child_sepset[0],child_sepset[1]]
                    )
                    for c_ix, child_sepset in enumerate(tree[2:])
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



def collect(tree, key_labels, potentials, visited, distributive_law):
    """
    Used by Hugin algorithm to collect messages

    Input:
    ------

    Output:
    -------


    """
    clique_ix, clique_keys = tree[:2]
    # set clique_index in visited to 1
    visited[clique_ix] = 1

    # loop over neighbors of root of tree
    for neighbor in tree[2:]:
        sep_ix, sep_keys, child = neighbor
        # call collect on neighbor if not marked as visited
        if not visited[child[0]]:
            potentials = collect(
                            child,
                            key_labels,
                            potentials,
                            visited,
                            distributive_law
            )
            new_clique_pot, new_sep_pot = distributive_law.update(
                                        potentials[child[0]], child[1],
                                        potentials[clique_ix], clique_keys,
                                        potentials[sep_ix], sep_keys
            )
            potentials[clique_ix] = new_clique_pot
            potentials[sep_ix] = new_sep_pot

    # return the updated potentials
    return potentials


def distribute(tree, key_labels, potentials, visited, distributive_law):
    """
    Used by Hugin algorithm to distribute messages

    Input:
    ------

    Output:
    -------

    """

    # set clique_index in visited to 1
    clique_ix, clique_keys = tree[:2]
    visited[clique_ix] = 1

    # loop over neighbors of root of tree
    for neighbor in tree[2:]:
        sep_ix, sep_keys, child = neighbor
        # call distribute on neighbor if not marked as visited
        if not visited[child[0]]:
            new_clique_pot, new_sep_pot = distributive_law.update(
                                        potentials[clique_ix], clique_keys,
                                        potentials[child[0]], child[1],
                                        potentials[sep_ix], sep_keys
            )
            potentials[child[0]] = new_clique_pot
            potentials[sep_ix] = new_sep_pot
            potentials = distribute(
                                child,
                                key_labels,
                                potentials,
                                visited,
                                distributive_law
            )

    # return the updated potentials
    return potentials


def hugin(tree, key_labels, potentials, distributive_law):
    """
    Run hugin algorithm by using the given distributive law.

    Input:
    ------

    Output:
    -------

    See page 3:
    http://compbio.fmph.uniba.sk/vyuka/gm/old/2010-02/handouts/junction-tree.pdf
    """
    # initialize visited array which has the same number of elements as potentials array
    visited = [0]*len(potentials)

    # call collect on root_index storing the result in new_potentials
    new_potentials = collect(
                        tree,
                        key_labels,
                        potentials,
                        visited,
                        distributive_law
    )

    # initialize visited array again
    visited = [0]*len(potentials)

    # return the result of a call to distribute on root index
    return distribute(
                    tree,
                    key_labels,
                    potentials,
                    visited,
                    distributive_law
    )

def get_clique(tree, key_label):
    """

    Input:
    ------

    Output:
    -------


    """
    ix, keys = tree[0:2]
    separators = tree[2:]
    if key_label in keys:
        return ix, keys
    if separators == (): # base case reached (leaf)
        return None

    for separator in separators:
        separator_ix, separator_keys, c_tree = separator
        if key_label in separator_keys:
            return separator_ix, separator_keys
        clique_info = get_clique(c_tree, key_label)
        if clique_info:
            return clique_info

    return None


def compute_marginal(potential, clique_keys, key_ix):
    """

    Input:
    ------

    Output:
    -------


    """

    if key_ix not in clique_keys:
        return 0.0

    # map keys to get around variable count limitation in einsum
    m_keys = {k:i for i,k in enumerate(clique_keys)}


    return np.einsum(
                        potential,
                        [m_keys[k] for k in clique_keys],
                        [m_keys[key_ix]]
    )

def yield_id_and_keys(tree):
    """

    Input:
    ------

    Output:
    -------


    """
    yield tree[0]
    yield tree[1]

def yield_clique_pairs(tree):
    """

    Input:
    ------

    Output:
    -------


    """
    for child in tree[2:]:
        yield (tree[0], tree[1], child[0], child[1])


def bf_traverse(trees, clique_id=None, func=yield_id_and_keys):
    """
    Breadth-first traversal of tree

    Early termination of search is performed if clique_id provided

    Input:
    ------

    Output:
    -------

    [id1, keys1, ..., idN, keysN] (or [id1, keys1, ..., cid, ckeys])
    """

    for tree in trees:
        queue = [tree]
        while queue:
            tree = queue.pop(0)
            yield from func(tree)
            if tree[0] == clique_id:
                raise StopIteration
            queue.extend([child for child in tree[2:]])

def df_traverse(trees, clique_id=None, func=yield_id_and_keys):
    """Depth-first traversal of tree

    Early termination of search is performed if clique_id provided

    Input:
    ------

    Output:
    -------

    [id1, keys1, ..., idN, keysN] (or [id1, keys1, ..., cid, ckeys])
    """

    for tree in trees:
        stack = [tree]
        while stack:
            tree = stack.pop()
            yield from func(tree)
            if tree[0] == clique_id:
                raise StopIteration
            stack.extend([child for child in reversed(tree[2:])])

def get_clique_keys(tree, clique_ix):
    """
    Return keys for clique with clique_id
        (if clique_id not in tree return None)

    Input:
    ------

    Output:
    -------

    clique_id_keys (or None)

    """
    flist = list(bf_traverse(tree, clique_ix))
    return flist[-1] if flist[-2] == clique_ix else None

def get_cliques(tree, key):
    """
    Return the (M) cliques which include key and all other keys
        in clique

    Input:
    ------

    Output:
    -------

    [clique_wkey_id1, clique_wkey_keys1, ..., clique_wkey_idM, clique_wkey_keysM]
    """

    flist = list(bf_traverse(tree))
    return [
            (flist[i], flist[i+1])
                for i in range(0, len(flist), 2) if key in flist[i+1]
    ]

def get_clique_of_key(tree, v):
    """

    Input:
    ------

    Output:
    -------


    """
    ix, keys = tree[0:2]
    separators = tree[2:]
    if v in keys:
        return ix, keys
    if separators == (): # base case reached (leaf)
        return None, None

    for separator in separators:
        separator_ix, separator_keys, c_tree = separator
        if v in separator_keys:
            return separator_ix, separator_keys
        clique_ix, clique_keys = get_clique_of_key(c_tree, v)
        if clique_ix:
            return clique_ix, clique_keys

    return None, None


def generate_potential_pairs(tree):
    """

    Input:
    ------

    Output:
    -------


    """
    return list(bf_traverse(tree, func=yield_clique_pairs))

class SumProduct():
    """ Sum-product distributive law """


    def __init__(self, einsum):
        # Perhaps support for different frameworks (TensorFlow, Theano) could
        # be provided by giving the necessary functions.
        self.einsum = einsum
        return

    def project(self, clique_pot, clique_keys, sep_keys):
        """

        Input:
        ------

        Output:
        -------


        """
        # map keys to get around variable count limitation in einsum
        m_keys = {k:i for i,k in enumerate(set(clique_keys + sep_keys))}
        return self.einsum(
            clique_pot,
            [m_keys[k] for k in clique_keys],
            [m_keys[k] for k in sep_keys]
        )

    def absorb(self, clique_pot, clique_keys, sep_pot, new_sep_pot, sep_keys):
        """

        Input:
        ------

        Output:
        -------


        """
        if np.all(sep_pot) == 0:
            return np.zeros_like(clique_pot)

        # map keys to get around variable count limitation in einsum
        m_keys = {k:i for i,k in enumerate(set(clique_keys + sep_keys))}
        return self.einsum(
            new_sep_pot / sep_pot, [m_keys[k] for k in sep_keys],
            clique_pot, [m_keys[k] for k in clique_keys],
            [m_keys[k] for k in clique_keys]
        )

    def update(self, clique1_pot, clique1_keys, clique2_pot, clique2_keys, sep_pot, sep_keys):
        """

        Input:
        ------

        Output:
        -------


        """
        # See page 2:
        # http://compbio.fmph.uniba.sk/vyuka/gm/old/2010-02/handouts/junction-tree.pdf


        # Sum keys in A that are not in B
        new_sep_pot = self.project(
                                clique1_pot,
                                clique1_keys,
                                sep_keys
        )

        # Compensate the updated separator in the clique
        new_clique2_pot = self.absorb(
                                clique2_pot,
                                clique2_keys,
                                sep_pot,
                                new_sep_pot,
                                sep_keys
        )

        return (new_clique2_pot, new_sep_pot) # may return unchanged clique_a
                                             # too if it helps elsewhere


# Sum-product distributive law for NumPy
sum_product = SumProduct(np.einsum)


class JunctionTree(object):
    def __init__(self, key_sizes, trees=[]):
        self.key_sizes = key_sizes
        self.labels = {vl:i for i, vl in enumerate(sorted(key_sizes.keys()))}
        self.struct = []
        self.tree_cliques = []
        for tree in trees:
            clique_id_keys = list(bf_traverse([tree]))
            self.tree_cliques.append([clique_id_keys[i] for i in range(0, len(clique_id_keys), 2)])
            self.struct.append(self.map_keys(tree, self.labels))
        self.phi = []

    def find_key(self, key_label):
        """

        Input:
        ------

        Output:
        -------


        """
        try:
            key_ix = self.labels[key_label]
            return key_ix
        except ValueError:
            return None

    def get_key_ix(self, clique_ix, key_label):
        """

        Input:
        ------

        Output:
        -------


        """
        try:
            keys = get_clique_keys(self.get_struct(), clique_ix)
            return keys.index(self.find_key(key_label))
        except (AttributeError, ValueError):
            return None

    def get_key_sizes(self):
        """

        Input:
        ------

        Output:
        -------


        """
        return self.key_sizes

    def get_label_order(self):
        """

        Input:
        ------

        Output:
        -------


        """
        return self.labels

    def get_labels(self):
        """
        Returns key labels in sorted order

        Input:
        ------

        Output:
        -------

        """
        labels = [None]*len(self.labels)
        for k,i in self.labels.items():
            labels[i] = k

        return labels

    def get_struct(self):
        """

        Input:
        ------

        Output:
        -------


        """
        return self.struct

    def get_potentials(self):
        """

        Input:
        ------

        Output:
        -------


        """
        return self.phi

    @staticmethod
    def map_keys(tree, lookup):
        """

        Input:
        ------

        Output:
        -------


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

        Input:
        ------

        Output:
        -------


        """
        key_sizes = factor_graph[0]
        factors = factor_graph[1]
        values = factor_graph[2]
        tri,induced_clusters = find_triangulation(
                            key_sizes=factor_graph[0],
                            factors=factor_graph[1]
        )

        cliques = identify_cliques(induced_clusters)
        trees = construct_junction_tree(cliques, key_sizes)
        jt = JunctionTree(key_sizes, trees)
        phi = JunctionTree.init_potentials(jt, factors, values)
        return jt, phi

    @staticmethod
    def init_potentials(tree, factors, values):
        """

        Input:
        ------

        Output:
        -------


        """
        clique_id_keys = list(bf_traverse(tree.get_struct()))
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

        Input:
        ------

        Output:
        -------


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
                clique_ix, clique_keys = get_clique_of_key(
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

        Input:
        ------

        Output:
        -------


        """
        new_potentials = potentials if in_place else copy.deepcopy(potentials)
        if data:
            likelihood, new_potentials = observe(self, new_potentials, data)

        for i, tree in enumerate(self.get_struct()):
            new_potentials = hugin(tree, self.get_label_order(), new_potentials, sum_product)
            #for clique_ix in self.tree_cliques[i]:
            #    [clique_ix] = new_phi[clique_ix]
        return(new_potentials)

    def marginalize(self, potentials, key_label):
        """

        Input:
        ------

        Output:
        -------


        """
        if key_label not in self.key_sizes:
            raise ValueError("Key %s not in tree" % key_label)

        key_ix = self.find_key(key_label)
        for tree in self.get_struct():
            clique_ix, clique_keys = get_clique_of_key(tree, key_ix)
            if clique_ix and clique_keys: break

        return compute_marginal(potentials[clique_ix], clique_keys, key_ix)
