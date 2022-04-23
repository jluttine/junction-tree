from junctiontree.sum_product import SumProduct
import numpy as np
import itertools

# setting optimize to true allows einsum to benefit from speed up due to
# contraction order optimization but at the cost of memory usage
# need to evaulate tradeoff within library
#sum_product = SumProduct(np.einsum,optimize=True)

sum_product = SumProduct(np.einsum)

def apply_evidence(potentials, variables, evidence):
    ''' Shrink potentials based on given evidence

    :param potentials: list of numpy arrays subject to evidence
    :param variables: list of variables in corresponding to potentials
    :param evidence: dictionary with variables as keys and assigned value as value
    :return: a new list of potentials after evidence applied
    '''

    return [
                [
                    # index array based on evidence value when evidence provided otherwise use full array
                    pot[
                        tuple(
                                [
                                    slice(evidence.get(var, 0), evidence.get(var, pot.shape[i]) + 1)
                                    for i, var in enumerate(vars)
                                ]
                        )
                    # workaround for scalar factors
                    ] if not np.isscalar(pot) else pot
                ]
                for pot, vars in zip(potentials, variables)
    ]


def compute_beliefs(tree, potentials, clique_vars, dl=sum_product):
    '''Computes beliefs for clique potentials in a junction tree
    using Shafer-Shenoy updates.

    :param tree: list representing the structure of the junction tree
    :param potentials: list of numpy arrays for cliques in junction tree
    :param clique_vars: list of variables included in each clique in potentials list
    :return: list of numpy arrays defining computed beliefs of each clique
    '''

    def get_message(sepset_ix, tree, beliefs, clique_vars):
        '''Computes message from root of tree with scope defined by sepset

        :param sepset_ix: index of sepset scope in which to return message
        (use slice(0) for no sepset)
        :param tree: list representation of tree rooted by cluster for which message
        will be computed
        :param beliefs: list of numpy arrays for cliques in junction tree
        :param clique_vars: list of variables included in each clique in potentials list
        :return: message: potential with scope defined by sepset (or None if tree includes root)
        '''

        cluster_ix = tree[0]
        messages = [
            comp # message or sepset variables in order processed
            for sepset_ix, subtree in tree[1:]
            for comp in [
                get_message(sepset_ix, subtree, beliefs, clique_vars),
                clique_vars[sepset_ix]
            ]
        ]

        # multiply neighbor messages
        #messages = messages if len(messages) else [1,[]]

        args = messages + [beliefs[cluster_ix], clique_vars[cluster_ix], clique_vars[cluster_ix]]

        # compute message as marginalization over non-sepset values
        # multiplied by product of messages with output being vars in input sepset

        #update clique belief
        beliefs[cluster_ix] = dl.einsum(*args)

        try:
            # attempt to update sepset belief
            args = [beliefs[cluster_ix], clique_vars[cluster_ix], clique_vars[sepset_ix]]
            beliefs[sepset_ix] = dl.einsum(*args)

            # send sepset belief as message
            return beliefs[sepset_ix]
        except TypeError:
            # function called on full tree so no message to send
            return None


    def send_message(message, sepset_ix, tree, beliefs, pots, clique_vars):
        '''Sends message from clique at root of tree

        :param message: message sent by neighbor
                (use np.array(1) for no message)
        :param sepset_ix: index of sepset scope in which message sent
                (use slice(0) for no sepset)
        :param tree: list representation of tree rooted by cluster receiving message
        :param beliefs: beliefs to update for cliques in junction tree
        :param pots: list of original numpy arrays for cliques in junction tree
        :param clique_vars: list of variables included in each clique in potentials list
        '''

        cluster_ix = tree[0]
        # computed messages stored in beliefs for neighbor sepsets
        messages = [
                comp
                for ss_ix, _ in tree[1:]
                for comp in [beliefs[ss_ix], clique_vars[ss_ix]]
        # adding message sent
        ] + [message, clique_vars[sepset_ix]]

        # send message to each neighbor
        for ss_ix, subtree in tree[1:]:

            # collect all messages (excluding those from ss_ix)
            # using id() as sepset variables can be same even when sepsets are unique
            mod_messages = [
                                comp
                                for i in range(1,len(messages), 2)
                                for comp in messages[i-1:i+1] if id(messages[i]) != id(clique_vars[ss_ix])
            ]

            # calculate message to be sent
            args = mod_messages + [pots[cluster_ix], clique_vars[cluster_ix], clique_vars[ss_ix]]
            msg = dl.einsum( *args )

            # update sepset belief
            args = [beliefs[ss_ix], clique_vars[ss_ix], msg, clique_vars[ss_ix], clique_vars[ss_ix]]
            beliefs[ss_ix] = dl.einsum(*args)

            # send message to neighbor (excludes message from subtree)
            send_message(msg, ss_ix, subtree, beliefs, pots, clique_vars)


        # update belief for clique
        args = [
            beliefs[cluster_ix],
            clique_vars[cluster_ix],
            message,
            clique_vars[sepset_ix],
            clique_vars[cluster_ix]
        ]

        beliefs[cluster_ix] = dl.einsum(*args)


    def __run(tree, potentials, clique_vars):
        '''Collect messages from neighbors recursively. Then, send messages
        recursively. Updated beliefs when this

        :param tree: list representing the structure of the junction tree
        :param potentials: list of numpy arrays for cliques in junction tree
        :param clique_vars: list of variables included in each clique in potentials list
        :return beliefs: consistent beliefs after Shafer-Shenoy updates applied
        '''

        beliefs = [np.copy(p) for p in potentials]

        # get messages from each neighbor
        get_message(slice(0), tree, beliefs, clique_vars)

        # send message to each neighbor
        send_message(np.array(1), slice(0), tree, beliefs, potentials, clique_vars)


        return beliefs


    return __run(tree, potentials, clique_vars)
