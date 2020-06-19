from junctiontree import beliefpropagation as bp


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


    assert list(bp.bf_traverse(tree)) == [0,1,3,2,4,5,6,]

    assert list(bp.df_traverse(tree)) == [0,1,2,3,4,5,6,]


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
    assert bp.get_clique_vars(node_list, 0) == [0, 2, 4]
    assert bp.get_clique_vars(node_list, 1) == [0, 2]
    assert bp.get_clique_vars(node_list, 2) == [0, 1, 2]
    assert bp.get_clique_vars(node_list, 3) == [4]
    assert bp.get_clique_vars(node_list, 4) == [3, 4]
    assert bp.get_clique_vars(node_list, 5) == [3]
    assert bp.get_clique_vars(node_list, 6) == [1, 2, 3]
    assert bp.get_clique_vars(node_list, 7) == None

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
