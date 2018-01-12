# junction-tree
Implementation of discrete factor graph inference utilizing the Junction Tree algorithm

Requirements:
-------------

* NumPy (>= 1.12)

Factor graphs:
--------------

A factor graph is given as a list of keys that tell which variables are in the
factor. (A key corresponds to a variable.)

```[keys1, ..., keysN]  # a list of N factors```

The index in the list can be used as an ID for the factor, that is, the first
factor in the list has ID 0 and the last factor has ID N-1.

A companion list (of numpy arrays) of the same length as the factor list is
provided as a representation for the factor values

```[values1, ..., valuesN]```

Also, the size of each of the M variables is given as a dictionary:

```
{
    key1: size1,
    ...
    keyM: sizeM
}
```

Here, size is an integer representing the size of the variable. It is the same as
the length of the corresponding axis in the numeric array.


Generic trees (recursive definition):
-------------------------------------

```
[index, keys, child_tree1, ..., child_treeN]
```


Junction trees:
---------------

```
tree structure (composed of node indices found in node list):
[
    index,
    (
        separator1_index,
        child_tree1
    ),
    ...,
    (
        separatorN_index,
        child_treeN
    )
]
node list (elements are list of keys which define node):
[node0_keys, node1_keys,...,nodeN_keys]

maxcliques and separators are both types of nodes
```

Potentials in (junction) trees:
-------------------------------

A list of arrays. The node IDs in the tree graphs map
to the arrays in this data structure in order to get the numeric
arrays in the execution phase. The numeric arrays are not needed
in the compilation phase.




## Usage:

### Junction Tree construction

Starting with the definition of a factor graph
```
import bp
from junction_tree import JunctionTree
import numpy as np

key_sizes = {
              "A": 2,
              "B": 2,
              "C": 2,
              "D": 2,
              "E": 2,
              "F": 2,
              "G": 2,
              "H": 2
            }

factors = [
            ["A"],
            ["A", "B"],
            ["A", "C"],
            ["B", "D"],
            ["C", "E"],
            ["C", "G"],
            ["G", "E", "H"],
            ["D", "E", "F"]
]

values = [
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

fg = [key_sizes,factors,values]

```

A junction tree can be constructed with corresponding initialized clique potentials:

```
jt, init_potentials = JunctionTree.from_factor_graph(fg)
```

### Global Propagation

The initial clique potentials are inconsistent. The potentials are made consistent through global propagation on the junction tree

```
new_potentials = jt.propagate(init_potentials)
```

### Observing Data

Alternatively, clique potentials can be made consistent after observing data for the variables in the junction tree

```
data = {"A": 0, "F": 1, "H": 1}
new_potentials = jt.propagate(init_potentials, data=data)
```

### Marginalization

From a collection of consistent clique potentials, the marginal value of variables of interest can be calculated

```
value = jt.marginalize(new_potentials, ["D"])
```


References:

S. M. Aji and R. J. McEliece, "The generalized distributive law," in IEEE Transactions on Information Theory, vol. 46, no. 2, pp. 325-343, Mar 2000. doi: 10.1109/18.825794

Cecil Huang, Adnan Darwiche, Inference in belief networks: A procedural guide, International Journal of Approximate Reasoning, Volume 15, Issue 3, 1996, Pages 225-263, ISSN 0888-613X, http://dx.doi.org/10.1016/S0888-613X(96)00069-2.

F. R. Kschischang, B. J. Frey and H. A. Loeliger, "Factor graphs and the sum-product algorithm," in IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 498-519, Feb 2001. doi: 10.1109/18.910572
