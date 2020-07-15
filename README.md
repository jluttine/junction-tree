# junction-tree
Implementation of discrete factor graph inference utilizing the Junction Tree algorithm

## Requirements

* Python3 (>= 3.5.5), NumPy (>= 1.13.3), SciPy (>= 1.1), attrs (>=17.4)

## Factor graphs:

A factor graph is given as a list of variables that indicate which variables are in the
factor.

```
[vars1, ..., varsN]  # a list of N factors
```

The index in the list can be used as an ID for the factor, that is, the first
factor in the list has ID 0 and the last factor has ID N-1.

A companion list (of numpy arrays) of the same length as the factor list is
provided as a representation for the factor values

```
[values1, ..., valuesN]
```

Also, the size of each of the M variables is given as a dictionary:

```
{
    var1: size1,
    ...
    varM: sizeM
}
```

Here, size is an integer representing the size of the variable. It is the same as
the length of the corresponding axis in the numeric array.


## Generic trees (recursive definition)

```
[index, vars, child_tree1, ..., child_treeN]
```


## Junction trees

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
node list (elements are list of variables which define node):
[node0_vars, node1_vars,...,nodeN_vars]

maxcliques and separators are both types of nodes
```

## Potentials in (junction) trees

A list of arrays. The node IDs in the tree graphs map to the arrays in this data
structure in order to get the numeric arrays in the execution phase. The numeric
arrays are not needed in the compilation phase.


## Usage

### Junction Tree construction

Starting with the definition of a factor graph

(Example taken from http://mensxmachina.org/files/software/demos/jtreedemo.html)
```
import junctiontree.beliefpropagation as bp
import junctiontree.junctiontree as jt
import numpy as np

var_sizes = {
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

tree = jt.create_junction_tree(factors, var_sizes)

```


### Global Propagation

The initial clique potentials are inconsistent. The potentials are made consistent through global propagation on the junction tree

```
prop_values = tree.propagate(values)
```

### Observing Data

Alternatively, clique potentials can be made consistent after observing data for the variables in the junction tree

```
# Update the size of observed variable
cond_sizes = var_sizes.copy()
cond_sizes["wet_grass"] = 1
cond_tree = jt.create_junction_tree(factors, cond_sizes)

# Then, also similarly the values:
cond_values = values.copy()
# remove axis corresponding to "wet_grass" == 0
cond_values[3] = cond_values[3][:,:,1:2]

# Perform global propagation using conditioned values
prop_values =  tree.propagate(cond_values)
```

### Marginalization

From a collection of consistent clique potentials, the marginal value of variables of interest can be calculated

```
# Pr(sprinkler|wet_grass = 1)
marginal = np.sum(prop_values[1], axis=0)

# The probabilities are unnormalized but we can calculate the normalized values:
norm_marginal = marginal/np.sum(marginal)
```


## References

S. M. Aji and R. J. McEliece, "The generalized distributive law," in IEEE Transactions on Information Theory, vol. 46, no. 2, pp. 325-343, Mar 2000. doi: 10.1109/18.825794

Cecil Huang, Adnan Darwiche, Inference in belief networks: A procedural guide, International Journal of Approximate Reasoning, Volume 15, Issue 3, 1996, Pages 225-263, ISSN 0888-613X, http://dx.doi.org/10.1016/S0888-613X(96)00069-2.

F. R. Kschischang, B. J. Frey and H. A. Loeliger, "Factor graphs and the sum-product algorithm," in IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 498-519, Feb 2001. doi: 10.1109/18.910572

Kjærulff, Uffe. 1997. Nested junction trees. In Proceedings of the Thirteenth conference on Uncertainty in artificial intelligence (UAI’97). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 294–301.

## Other Python Junction Tree Implementations

[symfer](https://mbsd.cs.ru.nl/symfer/index.html) - a tool suite, written mostly in Python, for performing probabilistic inference
[bayesian-belief-networks](https://github.com/eBay/bayesian-belief-networks) - a library for the creation of and exact inference on Bayesian Belief Networks specified as pure Python functions
