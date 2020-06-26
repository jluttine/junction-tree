from junctiontree import construction as cons
from .util import assert_triangulated, gibbs_elem_cycles, assert_junction_trees_equal
from junctiontree import junctiontree as jt
import heapq
import numpy as np
import copy
from .util import compute_num_combinations, find_base_cycle, create_cycle_basis, build_graph
from scipy.sparse.csgraph import minimum_spanning_tree


'''
Notes on factor graph representation:

* view the factors with respect to the contained variables which are the inputs
  to the factor and proceed with algorithms from there
* each factor sharing a variable will have an edge connecting the factors
* the shared variable will be on that edge
* the nodes in our factor graph are either factors (sets of variables) or
  variables (implicit in the fg formalization).
'''

def test_build_graph():

    factors = [['A','B','C'], ['C','D','E'], ['B','F']]

    # sum combinations to calculate total edge count

    num_edges = sum(
                    [
                        compute_num_combinations(len(factor), 2)
                        for factor in factors
                    ]
    )

    node_list, adj_matrix = build_graph(factors)

    assert len(node_list) == 6
    assert node_list == ['A', 'B', 'C', 'D', 'E', 'F']
    assert adj_matrix.sum() == num_edges


def test_find_base_cycle():
    adj_matrix = np.array(
        [[0, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]]
    )

    mst = minimum_spanning_tree(adj_matrix).tolil()

    assert mst.sum() <= adj_matrix.sum()

    cycle_edge = (3,5)

    assert find_base_cycle(mst, cycle_edge[0], cycle_edge[1]) == [5, 0, 3]


def test_create_cycle_basis():
    # check for the 6 cycles in cycle basis from these notes
    # http://www.math.cmu.edu/~mradclif/teaching/241F18/CycleBases.pdf

    adj_matrix = np.array(
        [[0, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]]
    )

    cycle_basis = create_cycle_basis(adj_matrix)

    assert len(cycle_basis) == 6

    exp_cycles = [
                    [5, 0, 3],
                    [3, 0, 2],
                    [4, 1, 0, 3],
                    [5, 0, 1, 4],
                    [4, 1, 0, 2],
                    [2, 0, 1]
    ]
    assert np.all(exp_cycle in cycle_basis for exp_cycle in exp_cycles)

def test_can_locate_clique_containing_variable():
    tree = [0, (1, [2, ])]
    node_list = [[0,1],[1], [1,2]]
    clique, _vars = cons.get_clique(tree, node_list, 2)
    assert clique == 2

def test_convert_factor_graph_to_undirected_graph():
    factors = [
                ["A"],
                ["A", "C"],
                ["B", "C", "D"],
                ["A", "D"]
            ]

    factor_edges = cons.factors_to_undirected_graph(factors)
    assert frozenset(("A","C")) in factor_edges
    assert frozenset(("A","D")) in factor_edges
    assert frozenset(("B","C")) in factor_edges
    assert frozenset(("B","D")) in factor_edges
    assert frozenset(("C","D")) in factor_edges


def test_store_nodes_in_heap():
    heap = []
    # push some nodes onto heap (NUM_EDGES_ADDED, CLUSTER_WEIGHT, FACTOR_ID)
    heapq.heappush(heap, (3, 4, 0))
    heapq.heappush(heap, (1, 3, 1))
    heapq.heappush(heap, (5, 3, 2))
    heapq.heappush(heap, (4, 6, 3))

    # check that heappop returns the element with the lowest value in the tuple
    assert heapq.heappop(heap) == (1, 3, 1)

    # add value back to heap
    heapq.heappush(heap, (1, 3, 1))
    # add two new tuples that have the same first value but smaller second
    # value
    heapq.heappush(heap, (1, 2, 4))
    heapq.heappush(heap, (1, 1, 5))

    # ensure that tie is broken by second element
    assert heapq.heappop(heap) == (1, 1, 5)
    # ensure that updated heap returns second smalles element with tie-
    # breaker
    assert heapq.heappop(heap) == (1, 2, 4)


def test_node_heap_construction():
    _vars = {
                "A": 2,
                "B": 4,
                "C": 3,
                "D": 5
            }

    factors = [
                ["A"], # weight: 2
                ["A", "C"], # weight: 6
                ["B", "C", "D"], # weight: 60
                ["A", "D"] # weight: 10
            ]
    factor_edges = cons.factors_to_undirected_graph(factors)
    hp, ef = cons.initialize_triangulation_heap(
                                            _vars,
                                            factor_edges
    )

    assert len(hp) == 4
    '''
        Entries:
        [0, 30, "A"] # A has 2 neighbors (all vars connected)
        [0, 60, "B"] # B has 2 neighbors (all vars connected)
        [1, 120, "C"] # C has 3 neighbors (A-B edge added)
        [1, 120, "D"] # C has 3 neighbors (A-B edge added)
    '''

    assert heapq.heappop(hp) == [0, 30, "A"]
    assert heapq.heappop(hp) == [0, 60, "B"]
    assert heapq.heappop(hp) == [1, 120, "C"]
    assert heapq.heappop(hp) == [1, 120, "D"]


def test_heap_update_after_node_removal():
    _vars = {
                "A": 2,
                "B": 4,
                "C": 3,
                "D": 5
            }

    factors = [
                ["A"],
                ["A", "C"],
                ["B", "C", "D"],
                ["A", "D"]
            ]

    factor_edges = cons.factors_to_undirected_graph(factors)
    heap, entry_finder = cons.initialize_triangulation_heap(
                                                    _vars,
                                                    factor_edges
    )

    item, heap, entry_finder, rem_vars = cons.remove_next(
                                                    heap,
                                                    entry_finder,
                                                    list(_vars.keys()),
                                                    _vars,
                                                    factor_edges
    )

    assert item == [0, 30, "A"]

    '''
        Entries:
        [0, 60, "B"] # B has 2 neighbors (all nodes connected)
        [0, 60, "C"] # C has 2 neighbors (all nodes connected)
        [0, 60, "D"] # D has 2 neighbors (all nodes connected)
    '''

    chk_heap = [entry for entry in heapq.nsmallest(len(heap), heap) if entry[2] != ""]
    assert len(chk_heap) == 3
    assert chk_heap[0] == [0, 60, "B"]
    assert chk_heap[1] == [0, 60, "C"]
    assert chk_heap[2] == [0, 60, "D"]

    item, heap, entry_finder, rem_vars = cons.remove_next(
                                                    heap,
                                                    entry_finder,
                                                    rem_vars,
                                                    _vars,
                                                    factor_edges
    )

    assert item == [0, 60, "B"]

    '''
        Entries:
        [0, 15, "C"] # C has 1 neighbor (already connected)
        [0, 15, "D"] # D has 1 neighbor (already connected)
    '''

    chk_heap = [entry for entry in heapq.nsmallest(len(heap), heap) if entry[2] != ""]
    assert len(chk_heap) == 2
    assert chk_heap[0] == [0, 15, "C"]
    assert chk_heap[1] == [0, 15, "D"]

    item, heap, entry_finder, rem_vars = cons.remove_next(
                                                    heap,
                                                    entry_finder,
                                                    rem_vars,
                                                    _vars,
                                                    factor_edges
    )

    assert item == [0, 15, "C"]

    '''
        Entries:
        [0, 5, "D"] # D has 0 neighbors (no connections possible)
    '''

    chk_heap = [entry for entry in heapq.nsmallest(len(heap), heap) if entry[2] != ""]
    assert len(chk_heap) == 1
    assert chk_heap[0] == [0, 5, "D"]

    item, heap, entry_finder, factors = cons.remove_next(
                                                    heap,
                                                    entry_finder,
                                                    rem_vars,
                                                    _vars,
                                                    factor_edges
    )
    assert item == [0, 5, "D"]


def test_gibbs_algo_implementation():
    '''
    Example 1 as described in:
    http://dspace.mit.edu/bitstream/handle/1721.1/68106/FTL_R_1982_07.pdf
    (pp. 15-17)
    '''
    edges = [
                ("A","B"),
                ("A","E"),
                ("A","E"),
                ("B","E"),
                ("B","C"),
                ("C","E"),
                ("C","D"),
                ("D","E")
    ]
    fcs = np.array(
                [
                    [1,1,0,1,0,0,0,0],
                    [0,1,1,1,0,0,0,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,0,0,0,1,1,1]
                ]
    )
    ecs = gibbs_elem_cycles(fcs)
    assert len(ecs) == 10
    ecs_str = ["".join(map(str, [1 if c_i else 0 for c_i in c])) for c in ecs]
    test_str = ["".join(map(str, c))
                    for c in [
                        [1,1,0,1,0,0,0,0], # fcs[0]
                        [1,0,1,0,0,0,0,0], # fcs[0] xor fcs[1]
                        [0,1,1,1,0,0,0,0], # fcs[1]
                        [1,1,0,0,1,1,0,0], # fcs[0] xor fcs[2]
                        [0,1,1,0,1,1,0,0], # fcs[1] xor fcs[2]
                        [0,0,0,1,1,1,0,0], # fcs[2]
                        [1,1,0,0,1,0,1,1], # fcs[0] xor fcs[2] xor fcs[3]
                        [0,1,1,0,1,0,1,1], # fcs[1] xor fcs[2] xor fcs[3]
                        [0,0,0,1,1,0,1,1], # fcs[2] xor fcs[3]
                        [0,0,0,0,0,1,1,1] # fcs[3]
                    ]
                ]
    assert set(ecs_str) == set(test_str)


def test_assert_triangulated():
    factors = [
                [0,1],
                [1,2],
                [2,3,4],
                [0,4]
            ]
    tri0 = []

    try:
        assert_triangulated(factors, tri0)
    except AssertionError:
        pass
    else:
        assert False, "invalid triangulation did not raise AssertionError"

    tri1 = [(0,2)]
    assert_triangulated(factors, tri1)


def test_triangulate_factor_graph1():
    _vars = {
                "A": 2,
                "B": 4,
                "C": 3,
                "D": 5,
                "E": 2
            }
    factors = [
                ["A", "B"],
                ["B", "C"],
                ["C", "D", "E"],
                ["A", "E"]
            ]
    values = [
                np.random.randn(2, 4),
                np.random.randn(4, 3),
                np.random.randn(3, 5, 2),
                np.random.randn(2, 2),
            ]
    fg = [_vars, factors, values]

    tri, max_cliques, factor_to_maxclique = cons.find_triangulation(fg[1], fg[0])

    # triangulation should consist of 1 edge
    assert len(tri) == 1
    assert len(tri[0]) == 2
    # to triangulate we have a few options
    assert set(tri[0]) in [set(("A","C")),set(("A","D")),set(("B","D")),set(("B","E"))]
    # ensure factors to cliques mapping is correct
    for factor_ix, clique_ix in enumerate(factor_to_maxclique):
        assert np.all([factor_key in max_cliques[clique_ix] for factor_key in factors[factor_ix]])
    assert_triangulated(fg[1], tri)


def test_triangulate_factor_graph2():
    _vars = {
        "A": 2,
        "B": 2,
        "C": 2,
        "D": 2,
        "E": 2,
        "F": 2,
        "G": 2,
        "H": 2,
    }

    factors = [
                ["A", "B"], #0
                ["A", "C"], #1
                ["B", "D"], #2
                ["C", "E"], #3
                ["C", "G"], #4
                ["G", "E", "H"], #5
                ["D", "E", "F"]  #6
    ]

    tri, max_cliques, _ = cons.find_triangulation(factors, _vars)
    assert_triangulated(factors, tri)

    assert len(max_cliques) == 6


def test_triangulate_factor_graph3():
    '''
    Example taken from here:

    https://courses.cs.washington.edu/courses/cse515/11sp/class7-exactinfalgos.pdf
    '''

    _vars = {
        "C": 2,
        "D": 2,
        "I": 2,
        "G": 2,
        "S": 2,
        "L": 2,
        "J": 2,
        "H": 2,
    }

    factors = [
                ["C","D"], #0
                ["D","I","G"], #1
                ["I","S"], #2
                ["G","H","J"], #3
                ["G","L"], #4
                ["S","L","J"], #5
    ]

    tri, max_cliques, _ = cons.find_triangulation(factors, _vars)
    assert_triangulated(factors, tri)

    assert len(max_cliques) == 5

    assert ["C","D"] in max_cliques
    assert ["D","G","I"] in max_cliques
    assert ["G","I","S"] in max_cliques
    assert ["G","J","L","S"] in max_cliques
    assert ["G","H","J"] in max_cliques


def test_triangulate_factor_graph_with_duplicate_factors():
    tri, max_cliques, factor_to_maxclique = cons.find_triangulation([ ["x", "y"], ["x", "y"] ], {"x":2, "y":3})
    assert None not in factor_to_maxclique

def test_can_use_integer_keys():
    x = 0
    y = 1
    assert type(jt.create_junction_tree([ [x], [x, y] ], {x: 10, y: 20})) == jt.JunctionTree


def test_join_trees_with_single_cliques():
    tree1 = [0,]
    sepset = [2,]
    tree2 = [1,]

    output = cons.merge_trees(
                        tree1,
                        tree1[0],
                        tree2,
                        tree2[0],
                        sepset[0]
    )

    merged_tree = [
            0,
            (
                2,
                [
                    1,
                ]
            )
        ]

    assert merged_tree == output
    assert_junction_trees_equal(output, merged_tree)

def test_join_tree_with_single_clique_to_multiclique_tree():
    tree1 = [0,]
    sepset = [3,]
    tree2 = [4, (5, [6, ])]

    output = cons.merge_trees(
                        tree1,
                        tree1[0],
                        tree2,
                        tree2[0],
                        sepset[0]
    )
    merged_tree = [
                0,
                (
                    3,
                    [
                        4,
                        (
                            5,
                            [
                                6,
                            ]
                        )
                    ]
                )
    ]

    assert_junction_trees_equal(output, merged_tree)


def test_join_tree_with_multiple_cliques_to_tree_with_multiple_cliques():
    tree1 = [0, (1, [2, ])]
    sepset = [3, ]
    tree2 = [4, (5, [6, ])]

    output = cons.merge_trees(
                        tree1,
                        tree1[0],
                        tree2,
                        tree2[0],
                        sepset[0]
    )
    merged_tree = [
                0,
                (
                    1,
                    [
                        2,
                    ]
                ),
                (
                    3,
                    [
                        4,
                        (
                            5,
                            [
                                6,
                            ]
                        )
                    ]
                )
    ]

    assert_junction_trees_equal(output, merged_tree)

def test_change_root():
    tree1 = [
                4,
                (
                    5,
                    [
                        0,
                        (
                            1,
                            [
                                2,
                            ]
                        )
                    ]
                )
            ]

    assert cons.change_root(tree1, 3) == []

    output = cons.change_root(tree1, 4)

    assert output == tree1

    output = cons.change_root(copy.deepcopy(tree1), 0)

    tree2 = [
                0,
                (
                    1,
                    [
                        2,
                    ]
                ),
                (
                    5,
                    [
                        4,
                    ]
                )
            ]

    assert output == tree2
    assert_junction_trees_equal(tree1, output)


    output = cons.change_root(copy.deepcopy(tree1), 2)

    tree3 = [
                2,
                (
                    1,
                    [
                        0,
                        (
                            5,
                            [
                                4,
                            ]
                        )
                    ]
                )
            ]


    assert output == tree3
    assert_junction_trees_equal(tree1, output)

    tree4 = [
                4,
                (
                    5,
                    [
                        0,
                        (
                            1,
                            [
                                2,
                            ]
                        ),
                        (
                            3,
                            [
                                6,
                            ]
                        )
                    ]
                ),
                (
                    7,
                    [
                        8,
                    ]
                )
            ]


    output = cons.change_root(tree4, 2)



    tree5 = [
                2,
                (
                    1,
                    [
                        0,
                        (
                            5,
                            [
                                4,
                                (
                                    7,
                                    [
                                        8,
                                    ]
                                )
                            ]
                        ),
                        (
                            3,
                            [
                                6,
                            ]
                        )
                    ]
                )
            ]


    assert_junction_trees_equal(output,tree5)
    assert_junction_trees_equal(tree4, output)

def test_join_trees_with_multiple_cliques_with_first_nested():
    tree1 = [4, (5, [0, (1, [2, ])])]
    sepset = [3,]
    tree2 = [8, (9, [10, ])]

    output = cons.merge_trees(
                        tree1,
                        0,
                        tree2,
                        8,
                        sepset[0]
    )
    merged_tree = [
                4,
                (
                    5,
                    [
                        0,
                        (
                            1,
                            [
                                2,
                            ]
                        ),
                        (
                            3,
                            [
                                8,
                                (
                                    9,
                                    [
                                        10,
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]

    assert_junction_trees_equal(output, merged_tree)


def test_join_trees_with_multiple_cliques_with_second_nested():
    tree1 = [0, (1, [2, ])]
    sepset = [3, ]
    tree2 = [6,  (7,  [8,  (9,  [10, ])])]

    output = cons.merge_trees(
                        tree1,
                        0,
                        tree2,
                        8,
                        sepset[0]
    )
    merged_tree = [
                0,
                (
                    1,
                    [
                        2,
                    ]
                ),
                (
                    3,
                    [
                        8,
                        (
                            9,
                            [
                                10,
                            ]
                        ),
                        (
                            7,
                            [
                                6,
                            ]
                        )
                    ]
                )
    ]

    assert_junction_trees_equal(output, merged_tree)

def test_join_trees_with_multiple_cliques_with_both_nested():
    tree1 = [4, (5, [0, (1,  [2,])])]
    sepset = [3, ]
    tree2 = [6, (7, [8, (9, [10, ])])]

    output = cons.merge_trees(
                        tree1,
                        0,
                        tree2,
                        8,
                        sepset[0]
    )

    merged_tree = [
                4,
                (
                    5,
                    [
                        0,
                        (
                            1,
                            [
                                2,
                            ]
                        ),
                        (
                            3,
                            [
                                8,
                                (
                                    9,
                                    [
                                        10,
                                    ]
                                ),
                                (
                                    7,
                                    [
                                        6,
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]


    assert_junction_trees_equal(output, merged_tree)


def test_traversal():
    tree = [
        0,
        (
            1,
            [
                2,
            ]
        ),
        (
            3,
            [
                4,
                (
                    5,
                    [
                        6,
                    ]
                )
            ]
        )
    ]


    assert list(cons.bf_traverse(tree)) == [0,1,3,2,4,5,6,]

    assert list(cons.df_traverse(tree)) == [0,1,2,3,4,5,6,]


def test_get_clique_vars():
    node_list = [
                    [0, 2, 4],
                    [0, 2],
                    [0, 1, 2],
                    [4],
                    [3, 4],
                    [3],
                    [1, 2, 3]

    ]
    assert cons.get_clique_vars(node_list, 0) == [0, 2, 4]
    assert cons.get_clique_vars(node_list, 1) == [0, 2]
    assert cons.get_clique_vars(node_list, 2) == [0, 1, 2]
    assert cons.get_clique_vars(node_list, 3) == [4]
    assert cons.get_clique_vars(node_list, 4) == [3, 4]
    assert cons.get_clique_vars(node_list, 5) == [3]
    assert cons.get_clique_vars(node_list, 6) == [1, 2, 3]
    assert cons.get_clique_vars(node_list, 7) == None

def test_generate_potential_pairs():
    tree = [
        0,
        (
            1,
            [
                2,
            ]
        ),
        (
            3,
            [
                4,
            ]
        ),
        (
            5,
            [
                6,
                (
                    7,
                    [
                        8,
                        (
                            9,
                            [
                                10,
                            ]
                        )
                    ]
                )
            ]

        )
    ]

    assert cons.generate_potential_pairs(tree) == [
                                        (0, 1),
                                        (0, 3),
                                        (0, 5),
                                        (1, 2),
                                        (3, 4),
                                        (5, 6),
                                        (6, 7),
                                        (7, 8),
                                        (8, 9),
                                        (9, 10)
    ]


