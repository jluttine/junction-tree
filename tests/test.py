import numpy as np

from junctiontree import beliefpropagation as bp
import unittest
import heapq
import copy
import junctiontree.junctiontree as jt
from .util import compute_num_combinations, build_graph, find_base_cycle, create_cycle_basis
from .util import assert_triangulated, gibbs_elem_cycles, assert_junction_trees_equal
from .util import assert_potentials_equal
from scipy.sparse.csgraph import minimum_spanning_tree


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


class TestJunctionTreeConstruction(unittest.TestCase):
    # just view the factors with respect to the variables which are the inputs
    # to the factor and proceed with algorithms from there
    # Each factor sharing a variable will have an edge connecting the factors
    # The shared variable will be on that edge
    # The nodes in our factor graph are either factors (sets of variables) or
    # variables (implicit in the fg formalization).

    # The relevant material from Aji and McEliece seems to indicate
    # that we should only need to construct the triangulation graph from the
    # factor arguments (local domains). Each of the variables for the factor
    # graph will be represented in the resulting junction tree.

    def test_can_locate_clique_containing_variable(self):
        tree = [0, (1, [2, ])]
        node_list = [[0,1],[1], [1,2]]
        clique, _vars = bp.get_clique(tree, node_list, 2)
        assert clique == 2

    def test_convert_factor_graph_to_undirected_graph(self):
        factors = [
                    ["A"],
                    ["A", "C"],
                    ["B", "C", "D"],
                    ["A", "D"]
                ]

        edges = bp.factors_to_undirected_graph(factors)
        assert frozenset(("A","C")) in edges
        assert frozenset(("A","D")) in edges
        assert frozenset(("B","C")) in edges
        assert frozenset(("B","D")) in edges
        assert frozenset(("C","D")) in edges


    def test_store_nodes_in_heap(self):
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


    def test_node_heap_construction(self):
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
        edges = bp.factors_to_undirected_graph(factors)
        hp, ef = bp.initialize_triangulation_heap(
                                                _vars,
                                                edges
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


    def test_heap_update_after_node_removal(self):
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

        edges = bp.factors_to_undirected_graph(factors)
        heap, entry_finder = bp.initialize_triangulation_heap(
                                                        _vars,
                                                        edges,
        )

        item, heap, entry_finder, rem_vars = bp.remove_next(
                                                        heap,
                                                        entry_finder,
                                                        list(_vars.keys()),
                                                        _vars,
                                                        edges
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

        item, heap, entry_finder, rem_vars = bp.remove_next(
                                                        heap,
                                                        entry_finder,
                                                        rem_vars,
                                                        _vars,
                                                        edges
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

        item, heap, entry_finder, rem_vars = bp.remove_next(
                                                        heap,
                                                        entry_finder,
                                                        rem_vars,
                                                        _vars,
                                                        edges,
        )

        assert item == [0, 15, "C"]

        '''
            Entries:
            [0, 5, "D"] # D has 0 neighbors (no connections possible)
        '''

        chk_heap = [entry for entry in heapq.nsmallest(len(heap), heap) if entry[2] != ""]
        assert len(chk_heap) == 1
        assert chk_heap[0] == [0, 5, "D"]

        item, heap, entry_finder, factors = bp.remove_next(
                                                        heap,
                                                        entry_finder,
                                                        rem_vars,
                                                        _vars,
                                                        edges
        )
        assert item == [0, 5, "D"]


    def test_gibbs_algo_implementation(self):
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


    def test_assert_triangulated(self):
        factors = [
                    [0,1],
                    [1,2],
                    [2,3,4],
                    [0,4]
                ]
        tri0 = []
        self.assertRaises(AssertionError, assert_triangulated, factors, tri0)

        tri1 = [(0,2)]
        assert_triangulated(factors, tri1)

    def test_triangulate_factor_graph1(self):
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
        tri, ics, max_cliques, factor_to_maxclique = bp.find_triangulation(fg[1], fg[0])

        # triangulation should consist of 1 edge
        assert len(tri) == 1
        assert len(tri[0]) == 2
        # to triangulate we have a few options
        assert set(tri[0]) in [set(("A","C")),set(("A","D")),set(("B","D")),set(("B","E"))]
        # ensure factors to cliques mapping is correct
        for factor_ix, clique_ix in enumerate(factor_to_maxclique):
            assert np.all([factor_key in max_cliques[clique_ix] for factor_key in factors[factor_ix]])
        assert_triangulated(fg[1], tri)

    def test_triangulate_factor_graph2(self):
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

        tri, ics, max_cliques, _ = bp.find_triangulation(factors, _vars)
        assert_triangulated(factors, tri)


        assert len(max_cliques) == 6


    def test_triangulate_factor_graph3(self):
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

        tri, ics, max_cliques, _ = bp.find_triangulation(factors, _vars)
        assert_triangulated(factors, tri)
        cliques = bp.identify_cliques(ics)

        assert len(max_cliques) == 5

        assert ["C","D"] in cliques
        assert ["D","G","I"] in cliques
        assert ["G","I","S"] in cliques
        assert ["G","J","L","S"] in cliques
        assert ["G","H","J"] in cliques


    def test_triangulate_factor_graph_with_duplicate_factors(self):
        tri, ics, max_cliques, factor_to_maxclique = bp.find_triangulation([ ["x", "y"], ["x", "y"] ], {"x":2, "y":3})
        assert None not in factor_to_maxclique

    def test_can_use_integer_keys(self):
        x = 0
        y = 1
        assert type(jt.create_junction_tree([ [x], [x, y] ], {x: 10, y: 20})) == jt.JunctionTree

    def test_identify_cliques(self):
        """
            test_identify_cliques

            Example taken from section 4.4.3 (Huang and Darwiche, 1996)

        """

        # Factors based on moralized graph from example

        factors = [
                    ["A"], #0
                    ["A", "B"], #1
                    ["A", "C"], #2
                    ["B", "D"], #3
                    ["C", "E", "G"], #4
                    ["G", "E", "H"], #5
                    ["D", "E", "F"]  #6
        ]

        tri = [
            ["D","E"], #0
            ["D"], #1
            ["E"], #2
            [], #3
            [], #4
            [], #5
            [] #6
        ]

        cliques = bp.identify_cliques([f+t for f,t in zip(factors,tri)])

        assert len(cliques) == 6

        assert ["E","G","H"] in cliques
        assert ["C","E","G"] in cliques
        assert ["D","E","F"] in cliques
        assert ["A","C","E"] in cliques
        assert ["A","B","D"] in cliques
        assert ["A","D","E"] in cliques

    def test_join_trees_with_single_cliques(self):
        tree1 = [0,]
        sepset = [2,]
        tree2 = [1,]

        output = bp.merge_trees(
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

    def test_join_tree_with_single_clique_to_multiclique_tree(self):
        tree1 = [0,]
        sepset = [3,]
        tree2 = [4, (5, [6, ])]

        output = bp.merge_trees(
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


    def test_join_tree_with_multiple_cliques_to_tree_with_multiple_cliques(self):
        tree1 = [0, (1, [2, ])]
        sepset = [3, ]
        tree2 = [4, (5, [6, ])]

        output = bp.merge_trees(
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

    def test_change_root(self):
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

        assert bp.change_root(tree1, 3) == []

        output = bp.change_root(tree1, 4)

        assert output == tree1

        output = bp.change_root(copy.deepcopy(tree1), 0)

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


        output = bp.change_root(copy.deepcopy(tree1), 2)

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


        output = bp.change_root(tree4, 2)



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

    def test_join_trees_with_multiple_cliques_with_first_nested(self):
        tree1 = [4, (5, [0, (1, [2, ])])]
        sepset = [3,]
        tree2 = [8, (9, [10, ])]

        output = bp.merge_trees(
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


    def test_join_trees_with_multiple_cliques_with_second_nested(self):
        tree1 = [0, (1, [2, ])]
        sepset = [3, ]
        tree2 = [6,  (7,  [8,  (9,  [10, ])])]

        output = bp.merge_trees(
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

    def test_join_trees_with_multiple_cliques_with_both_nested(self):
        tree1 = [4, (5, [0, (1,  [2,])])]
        sepset = [3, ]
        tree2 = [6, (7, [8, (9, [10, ])])]

        output = bp.merge_trees(
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



class TestJunctionTreeInference(unittest.TestCase):
    def setUp(self):
        self.tree = [
                        0,
                        (
                            6,
                            [
                                1,
                            ]
                        ),
                        (
                            7,
                            [
                                2,
                            ]
                        ),
                        (
                            8,
                            [
                                3,
                                (
                                    9,
                                    [
                                        4,
                                        (
                                            10,
                                            [
                                                5,
                                            ]
                                        )
                                    ]
                                )
                            ]

                        )
                    ]

        self.node_list = [
                            ["A","D","E"],
                            ["A","B","D"],
                            ["D","E","F"],
                            ["A","C","E"],
                            ["C","E","G"],
                            ["E","G","H"],
                            ["A","D"],
                            ["D","E"],
                            ["A","E"],
                            ["C","E"],
                            ["E","G"],
        ]

        self.key_sizes = {
                            "A": 2,
                            "B": 2,
                            "C": 2,
                            "D": 2,
                            "E": 2,
                            "F": 2,
                            "G": 2,
                            "H": 2
                        }

        self.factors = [
                    ["A"],
                    ["A", "B"],
                    ["A", "C"],
                    ["B", "D"],
                    ["C", "E"],
                    ["C", "G"],
                    ["D", "E", "F"],
                    ["E", "G", "H"],

        ]

        self.values = [
                        np.array([0.5,0.5]),
                        np.array(
                                    [
                                        [0.6,0.4],
                                        [0.5,0.5]
                                    ]
                                ),
                        np.array(
                                    [
                                        [0.8,0.2],
                                        [0.3,0.7]
                                    ]
                                ),
                        np.array(
                                    [
                                        [0.5,0.5],
                                        [0.1,0.9]
                                    ]
                                ),
                        np.array(
                                    [
                                        [0.4,0.6],
                                        [0.7,0.3]
                                    ]
                                ),
                        np.array(
                                    [
                                        [0.9,0.1],
                                        [0.8,0.2]
                                    ]
                                ),
                        np.array(
                                    [
                                        [
                                            [0.01,0.99],
                                            [0.99,0.01]
                                        ],
                                        [
                                            [0.99,0.01],
                                            [0.99,0.01]
                                        ]
                                    ]
                                ),
                        np.array(
                                    [
                                        [
                                            [0.05,0.95],
                                            [0.05,0.95]
                                        ],
                                        [
                                            [0.05,0.95],
                                            [0.95,0.05]
                                        ]
                                    ]
                                )
                ]


    def test_transformation(self):
        tree = jt.create_junction_tree(self.factors, self.key_sizes)
        prop_values = tree.propagate(self.values)

        # check that marginal values are correct
        np.testing.assert_allclose(
                                prop_values[0],
                                np.array([0.500,0.500])
        )
        np.testing.assert_allclose(
                                np.sum(prop_values[1], axis=0),
                                np.array([0.550,0.450])
        )
        np.testing.assert_allclose(
                                np.sum(prop_values[2], axis=0),
                                np.array([0.550,0.450])
        )
        np.testing.assert_allclose(
                                np.sum(prop_values[3], axis=0),
                                np.array([0.320,0.680])
        )
        np.testing.assert_allclose(
                                np.sum(prop_values[4], axis=0),
                                np.array([0.535,0.465])
        )
        np.testing.assert_allclose(
                                np.sum(prop_values[5], axis=0),
                                np.array([0.855,0.145])
        )

        np.testing.assert_allclose(
                                bp.sum_product.einsum(
                                    tree.propagate(self.values)[6],
                                    [0, 1, 2],
                                    [2]
                                ),
                                np.array([0.824,0.176]),
                                atol=0.01
        )
        np.testing.assert_allclose(
                                bp.sum_product.einsum(
                                    tree.propagate(self.values)[7],
                                    [0, 1, 2],
                                    [2]
                                ),
                                np.array([ 0.104,  0.896]),
                                atol=0.01
        )

    def test_initialize_potentials(self):
        j_tree = jt.JunctionTree(
                    self.tree,
                    self.node_list[6:],
                    jt.CliqueGraph(
                        maxcliques=self.node_list[:6],
                        factor_to_maxclique=[0, 1, 3, 1, 3, 4, 2, 5],
                        factor_graph=jt.FactorGraph(
                            factors=self.factors,
                            sizes=self.key_sizes
                        )
                    )
        )

        init_phi  = j_tree.clique_tree.evaluate(self.values)

        assert_potentials_equal(
                        init_phi[3], # cluster ACE
                        np.array(
                            [
                                [
                                    [0.32,0.48],
                                    [0.14,0.06]
                                ],
                                [
                                    [0.12,0.18],
                                    [0.49,0.21]
                                ]
                            ]
                        )
        )

    def test_global_propagation(self):
        tree = jt.create_junction_tree(self.factors, self.key_sizes)

        prop_values = tree.propagate(self.values)
        # P(A)
        assert_potentials_equal(
                                prop_values[0],
                                np.array([0.500,0.500])
        )

        # P(D)
        assert_potentials_equal(
                                np.sum(prop_values[3], axis=0),
                                np.array([0.32,0.68])
        )

    def test_global_propagation_with_observations(self):
        #http://mensxmachina.org/files/software/demos/jtreedemo.html
        key_sizes = {
                        "cloudy": 2,
                        "sprinkler": 2,
                        "rain": 2,
                        "wet_grass": 2
                    }

        factors = [
                    ["cloudy"],
                    ["cloudy", "sprinkler"],
                    ["cloudy", "rain"],
                    ["rain", "sprinkler", "wet_grass"]
        ]

        values = [
                    np.array([0.5,0.5]),
                    np.array(
                                [
                                    [0.5,0.5],
                                    [0.9,0.1]
                                ]
                            ),
                    np.array(
                                [
                                    [0.8,0.2],
                                    [0.2,0.8]
                                ]
                            ),
                    np.array(
                                [
                                    [
                                        [1,0],
                                        [0.1,0.9]
                                    ],
                                    [
                                        [0.1,0.9],
                                        [0.01,0.99]
                                    ]
                                ]
                    )
        ]

        tree = jt.create_junction_tree(factors, key_sizes)

        # grass is wet
        tree.clique_tree.factor_graph.sizes["wet_grass"] = 1
        cond_values = copy.deepcopy(values)
        cond_values[3] = cond_values[3][:,:,1:]

        prop_values = tree.propagate(cond_values)
        marginal = np.sum(prop_values[1], axis=0)

        np.testing.assert_allclose(
                                marginal/np.sum(marginal),
                                np.array([0.57024,0.42976]),
                                atol=0.01
        )

        # grass is wet and it is raining
        tree.clique_tree.factor_graph.sizes["rain"] = 1
        cond_values[3] = cond_values[3][1:,:,:]
        cond_values[2] = cond_values[2][:,1:]
        prop_values = tree.propagate(cond_values)

        marginal = np.sum(prop_values[1], axis=0)

        np.testing.assert_allclose(
                                marginal/np.sum(marginal),
                                np.array([0.8055,0.1945]),
                                atol=0.01
        )





    def test_inference(self):
        #http://pages.cs.wisc.edu/~dpage/cs731/lecture5.ppt
        key_sizes = {
                        "A": 2,
                        "B": 2,
                        "C": 2,
                        "D": 2,
                        "E": 2,
                        "F": 2
                    }
        factors = [
                    ["A"],
                    ["B","A"],
                    ["C","A"],
                    ["B","D"],
                    ["C","E"],
                    ["D","E","F"]
        ]
        values = [
                    np.array([0.9,0.1]),
                    np.array(
                                [
                                    [0.1,0.9],
                                    [0.9,0.1]
                                ]
                    ),
                    np.array(
                                [
                                    [0.8,0.3],
                                    [0.2,0.7]
                                ]
                    ),
                    np.array(
                                [
                                    [0.3,0.7],
                                    [0.6,0.4]
                                ]
                    ),
                    np.array(
                                [
                                    [0.6,0.4],
                                    [0.5,0.5]
                                ]
                    ),
                    np.array(
                                [
                                    [
                                        [0.2,0.8],
                                        [0.6,0.4]
                                    ],
                                    [
                                        [0.5,0.5],
                                        [0.9,0.1]
                                    ]
                                ]
                    )
        ]

        struct = [
                    0,
                    (
                        4,
                        [
                            1,
                        ]
                    ),
                    (
                        5,
                        [
                            2,
                            (
                                6,
                                [
                                    3,
                                ]
                            )
                        ]
                    )
                ]

        node_list = [
                        ["C","D","E"],
                        ["D","E","F"],
                        ["B","C","D"],
                        ["A","B","C"],
                        ["D","E"],
                        ["C","D"],
                        ["B","C"],
        ]

        tree = jt.create_junction_tree(factors, key_sizes)

        prop_values = tree.propagate(values)

        # P(C)
        np.testing.assert_allclose(
                            np.sum(prop_values[2], axis=1),
                            np.array([0.75,0.25])
        )
        # P(A)
        np.testing.assert_allclose(
                            np.sum(prop_values[1], axis=0),
                            np.array([0.9,0.1])
        )

        # P(B)
        np.testing.assert_allclose(
                            np.sum(prop_values[1], axis=1),
                            np.array([0.18,0.82])
        )

        # P(D)
        np.testing.assert_allclose(
                            np.sum(prop_values[3], axis=0),
                            np.array([0.546,0.454])
        )

        # P(E)
        np.testing.assert_allclose(
                            np.sum(prop_values[4], axis=0),
                            np.array([0.575,0.425])
        )

        # P(F)
        np.testing.assert_allclose(
                            bp.sum_product.einsum(
                                prop_values[5],
                                [0,1,2],
                                [2]
                            ),
                            np.array([0.507,0.493]),
                            atol=0.001
        )


class TestJTTraversal(unittest.TestCase):
    def setUp(self):
        self.tree = [
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

    def test_bf_traverse(self):
        assert list(bp.bf_traverse(self.tree)) == [
                                                    0,
                                                    1,
                                                    3,
                                                    2,
                                                    4,
                                                    5,
                                                    6,
                                                ]

    def test_df_traverse(self):
        assert list(bp.df_traverse(self.tree)) == [
                                                    0,
                                                    1,
                                                    2,
                                                    3,
                                                    4,
                                                    5,
                                                    6,
                                                ]


    def test_get_clique_keys(self):
        node_list = [
                        [0, 2, 4],
                        [0, 2],
                        [0, 1, 2],
                        [4],
                        [3, 4],
                        [3],
                        [1, 2, 3]

        ]
        assert bp.get_clique_keys(node_list, 0) == [0, 2, 4]
        assert bp.get_clique_keys(node_list, 1) == [0, 2]
        assert bp.get_clique_keys(node_list, 2) == [0, 1, 2]
        assert bp.get_clique_keys(node_list, 3) == [4]
        assert bp.get_clique_keys(node_list, 4) == [3, 4]
        assert bp.get_clique_keys(node_list, 5) == [3]
        assert bp.get_clique_keys(node_list, 6) == [1, 2, 3]
        assert bp.get_clique_keys(node_list, 7) == None

    def test_generate_potential_pairs(self):
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

        assert bp.generate_potential_pairs(tree) == [
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

class TestMisc(unittest.TestCase):

    def test_index_error(self):
        key_sizes = {'a': 10, 'b': 6, 'c': 3, 'd': 10, 'e': 27, 'f': 2, 'g': 3, 'h': 3, 'i': 3, 'j': 3, 'k': 5, 'l': 6}

        factors = [
            ['f', 'k', 'a'],
            ['f', 'g', 'b'],
            ['c'],
            ['f', 'k', 'd'],
            ['g', 'e'],
            ['c', 'g', 'f'],
            ['c', 'g'],
            ['f', 'h'],
            ['c', 'g', 'i'],
            ['l', 'j'],
            ['g', 'j', 'k'],
            ['g', 'l']
        ]

        values = [

        np.random.random((key_sizes['f'], key_sizes['k'], key_sizes['a'])),
        np.random.random((key_sizes['f'], key_sizes['g'], key_sizes['b'])),
        np.random.random((key_sizes['c'])),
        np.random.random((key_sizes['f'], key_sizes['k'], key_sizes['d'])),
        np.random.random((key_sizes['g'], key_sizes['e'])),
        np.random.random((key_sizes['c'], key_sizes['g'], key_sizes['f'])),
        np.random.random((key_sizes['c'], key_sizes['g'])),
        np.random.random((key_sizes['f'], key_sizes['h'])),
        np.random.random((key_sizes['c'], key_sizes['g'], key_sizes['i'])),
        np.random.random((key_sizes['l'], key_sizes['j'])),
        np.random.random((key_sizes['g'], key_sizes['j'], key_sizes['k'])),
        np.random.random((key_sizes['g'], key_sizes['l']))

        ]

        tree = jt.create_junction_tree(factors, key_sizes)
        prop_values = tree.propagate(values)
