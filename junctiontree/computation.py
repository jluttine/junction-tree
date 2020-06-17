from junctiontree.sum_product import SumProduct
import numpy as np

def apply_evidence(potentials, variables, evidence):
    ''' Shrink potentials based on given evidence

    :param potentials: list of numpy arrays subject to evidence
    :param variables: list of varialables corresponding to potentials
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


def compute_beliefs(tree, potentials, clique_vars):
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

        neighbor_vars = list(set(neighbor_vars))

        # multiply neighbor messages
        messages = messages if len(messages) else [1]

        msg_prod = sum_product.einsum(
                                *messages,
                                neighbor_vars
        )

        args = [msg_prod, neighbor_vars] + [beliefs[tree[0]], clique_vars[tree[0]], clique_vars[sepset_ix]]

        # compute message as marginalization over non-sepset values
        # multiplied by product of messages with output being vars in input sepset
        message = sum_product.einsum(*args)

        try:
            # attempt to update belief
            beliefs[sepset_ix] = message
            return message
        except TypeError:
            # function called on full tree so no message to send
            return None


    def remove_message(msg_prod, prod_ixs, msg, msg_ixs, out_ixs):
        '''Removes (divides out) sepset message from
        product of all neighbor sepset messages for a clique

        :param msg_prod: product of all messages for clique
        :param prod_ixs: variable indices in clique
        :param msg: sepset message to be removed from product
        :param msg_ixs: variable indices in sepset
        :param out_ixs: variables indices expected in result
        :return: the product of messages with sepset msg removed (divided out)
        '''

        exp_mask = np.in1d(prod_ixs, msg_ixs)

        # use mask to specify expanded dimensions in message
        exp_ixs = np.full(msg_prod.ndim, None)
        exp_ixs[exp_mask] = slice(None)

        # use mask to select slice dimensions
        slice_mask = np.in1d(prod_ixs, out_ixs)
        slice_ixs = np.full(msg_prod.ndim, slice(None))
        slice_ixs[~slice_mask] = 0

        if all(exp_mask) and msg_ixs != prod_ixs:
            # axis must be labeled starting at 0
            var_map = {var:i for i, var in enumerate(set(msg_ixs + prod_ixs))}

            # axis must be re-ordered if all variables shared but order is different
            msg = np.moveaxis(msg, [var_map[var] for var in prod_ixs], [var_map[var] for var in msg_ixs])

        # create dummy dimensions for performing division (with exp_ix)
        # slice out dimensions of sepset variables from division result (with slice_ixs)
        return np.divide( msg_prod, msg[ tuple(exp_ixs) ] )[ tuple(slice_ixs) ]


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

        neighbor_vars = list(set(all_neighbor_vars))

        # multiply neighbor messages
        msg_prod = sum_product.einsum(
                                *messages,
                                neighbor_vars
        )

        # send message to each neighbor
        ss_num = 0
        for ss_ix, subtree in tree[1:]:

            # divide product of messages by current sepset message for this neighbor
            output_vars = list(
                                set(
                                    [
                                        var
                                        for vars in messages[1::2][0:ss_num] + messages[1::2][ss_num+1:]
                                        for var in vars
                                    ]
                                )
            )

            mask = np.in1d(
                            neighbor_vars,
                            output_vars

            )

            mod_neighbor_vars = np.array(neighbor_vars)[mask].tolist()

            mod_msg_prod = remove_message(
                                    msg_prod,
                                    neighbor_vars,
                                    beliefs[ss_ix],
                                    clique_vars[ss_ix],
                                    mod_neighbor_vars
            )

            args = [mod_msg_prod, mod_neighbor_vars] + [beliefs[tree[0]], clique_vars[tree[0]], clique_vars[ss_ix]]
            # calculate message to be sent
            message = sum_product.einsum( *args )

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

        beliefs[tree[0]] = sum_product.einsum(*args)


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


sum_product = SumProduct(np.einsum)
