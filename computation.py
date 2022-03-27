from junctiontree.sum_product import SumProduct
import numpy as np

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

        messages = [
            comp # message or sepset variables in order processed
            for sepset_ix, subtree in tree[1:]
            for comp in [
                get_message(sepset_ix, subtree, beliefs, clique_vars),
                clique_vars[sepset_ix]
            ]
        ]

        neighbor_vars = [
            var
            for ss_ix, subtree in tree[1:]
            for var in clique_vars[ss_ix]
        ]

        neighbor_vars = np.unique(neighbor_vars)

        # multiply neighbor messages        
        msg_prod = dl.einsum(*messages, neighbor_vars)

        args = [msg_prod, neighbor_vars] + [beliefs[tree[0]], clique_vars[tree[0]], clique_vars[sepset_ix]]

        # compute message as marginalization over non-sepset values
        # multiplied by product of messages with output being vars in input sepset

        message = dl.einsum(*args)


        try:
            # attempt to update belief
            beliefs[sepset_ix] = message
            return message
        except TypeError:
            # function called on full tree so no message to send
            return None


    def send_message(message, sepset_ix, tree, beliefs, clique_vars):
        '''Sends message from clique at root of tree

        :param message: message sent by neighbor
                (use np.array(1) for no message)
        :param sepset_ix: index of sepset scope in which message sent
                (use slice(0) for no sepset)
        :param tree: list representation of tree rooted by cluster receiving message
        :param beliefs: list of numpy arrays for cliques in junction tree
        :param clique_vars: list of variables included in each clique in potentials list
        '''

        # computed messages stored in beliefs for neighbor sepsets
        messages = [
                comp
                for ss_ix, _ in tree[1:]
                for comp in [beliefs[ss_ix], clique_vars[ss_ix]]
        # adding message sent
        ] + [message, clique_vars[sepset_ix]]


        all_neighbor_vars = [
            var
            for vars in messages[1::2]
            for var in vars
        ]

        neighbor_vars = np.unique(all_neighbor_vars)

        # multiply neighbor messages
        msg_prod = dl.einsum(
                                *messages,
                                neighbor_vars
        )

        # send message to each neighbor
        ss_num = 0
        for ss_ix, subtree in tree[1:]:

            # remove sepset ix vars from neighbor vars
            mod_neighbor_vars = np.setdiff1d(neighbor_vars, clique_vars[ss_ix])




            # create product of messages that excludes the message from this sepset
            mod_messages = [
                                comp
                                for i in range(1,len(messages), 2)
                                for comp in messages[i-1:i+1] if messages[i] != clique_vars[ss_ix]
            ]
            args = [dl.einsum(*mod_messages, mod_neighbor_vars), mod_neighbor_vars] + [beliefs[tree[0]], clique_vars[tree[0]], clique_vars[ss_ix]]

            # calculate message to be sent
            message = dl.einsum( *args )

            # update sepset belief
            beliefs[ss_ix] *= message

            send_message(message, ss_ix, subtree, beliefs, clique_vars)
            ss_num += 1

        # update belief for clique
        args = [
            beliefs[tree[0]],
            clique_vars[tree[0]],
            msg_prod,
            neighbor_vars,
            clique_vars[tree[0]]
        ]


        beliefs[tree[0]] = dl.einsum(*args)


    def __run(tree, beliefs, clique_vars):
        '''Collect messages from neighbors recursively. Then, send messages
        recursively. Updated beliefs when this

        :param tree: list representing the structure of the junction tree
        :param briefs: list of numpy arrays for cliques in junction tree
        :param clique_vars: list of variables included in each clique in potentials list
        :return beliefs: consistent beliefs after Shafer-Shenoy updates applied
        '''

        # get messages from each neighbor
        get_message(slice(0), tree, beliefs, clique_vars)

        # send message to each neighbor
        send_message(np.array(1), slice(0), tree, beliefs, clique_vars)

        return beliefs


    beliefs = [np.copy(p) for p in potentials]
    return __run(tree, beliefs, clique_vars)
