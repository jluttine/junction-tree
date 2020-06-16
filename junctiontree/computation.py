from junctiontree.sum_product import SumProduct
import numpy as np
import numbers

def eliminate_variables(factors, variables, order, evidence={}):
    '''Eliminate variables in the given variable order. Function assumes that the
    elimination order is valid for the specified factor graph.

    :param factors: list of factors
    :param variables: list of variables included in each factor in factors list
    :param order: elimination order for variables
    :param evidence: dictionary specifying (possible) observations for variables
    :param factors_and_vars: a list of interleaved factors and corresponding variable lists,
    when an elimination order is provided, this list will only include the marginalized factor
    and its variables

    '''

    def get_factors_and_vars(factors, variables, evidence):
        '''Creates a flat list of all factor arrays and lists of their variables
        Output: [factor1, vars1, ..., factorN, varsN]

        :param factors: list of factors
        :param variables: list of variables included in each factor in factors list
        :param evidence: dictionary specifying (possible) observations for variables
        :param factors_and_vars: list of interleaved factors and corresponding variable lists
        '''


        return sum(
                    [
                        [
                            # index array based on evidence value when evidence provided otherwise use full array
                            fac[
                                tuple(
                                        [
                                            slice(evidence.get(var, 0), evidence.get(var, fac.shape[i]) + 1)
                                            for i, var in enumerate(vars)
                                        ]
                                )
                            # workaround for scalar factors
                            ] if not (isinstance(fac, numbers.Number)) else fac,
                            vars
                        ]
                        for fac, vars in zip(factors, variables)
                    ],
                    []
        )


    def split_factors(factors_and_vars, var):
        '''Splits the factors and vars into two lists of indices. One list contains indices of factors
        including the variable and one with indices of factors not including the variable

        :param factors_and_vars: list of interleaved factors and corresponding variable lists
        :param var: variable for grouping factors into include and exclude sets
        :return include_factors: list of factor indices containing variable
        :return exclude_factors: list of factor indices not containing variable
        '''

        include_factors = [
                    fac_ix
                    for fac_ix in range(0, len(factors_and_vars), 2)
                    if var in factors_and_vars[fac_ix + 1]
        ]

        exclude_factors = list(set(range(0, len(factors_and_vars), 2)) - set(include_factors))

        return include_factors, exclude_factors


    def __run(factors_and_vars, order):
        ''' Recursive function for performing variable elimination

        :param factors_and_vars: interleaved list of factors and lists of their variables
        :param order: elimination order for variables
        :return: factors_and_vars: interleaved list of factors and lists of their variables with variables
        in elimination order removed
        '''

        if len(order) > 1:
            # call run on the factors with one variable removed from order
            # the return value will be a list of remaining factors and vars
            factors_and_vars =  __run(
                factors_and_vars,
                order[:-1]
            )

        # eliminate the variable at the end of order list
        elim_var = order[-1]
        include_factors, exclude_factors = split_factors(factors_and_vars, elim_var)

        # extract the variables that are in the included factors, removing the elimination variable from this set
        output_var_indices = list(
                                set(
                                    [
                                        # provides output indices for einsum
                                        var
                                        for fac_ix in include_factors
                                        for var in factors_and_vars[fac_ix + 1]
                                        if var != elim_var
                                    ]
                                )
        )


        args = sum(
                    [
                        [factors_and_vars[fac_ix], factors_and_vars[fac_ix + 1]]
                        for fac_ix in include_factors
                    ],
                    []
        ) + [output_var_indices]

        # perform einstein summation on the factors in the include group
        m_factor = sum_product.einsum(*args)


        # return a list with remaining factors(excluded and marginalized factor) and their variables
        # [ex_fac1, ex_fac1_vars, ex_fac2, ex_fac2_vars,..., new_fac, new_fac_vars]

        return(
                sum(
                    [
                        [factors_and_vars[fac_ix], factors_and_vars[fac_ix + 1]]
                        for fac_ix in exclude_factors
                    ],
                    [m_factor, output_var_indices] if m_factor.ndim  != 1 else [
                        # 1-dimensional factors require special treatment
                        np.expand_dims(m_factor, axis=1), output_var_indices + [...]
                    ]
                )
        )
        # end of __run function


    # create the interleaved factor and variable list structure
    factors_and_vars = get_factors_and_vars(factors, variables, evidence)

    if len(order) == 0:
        return factors_and_vars

    factors_and_vars = __run(factors_and_vars, order)
    marg_vars = list(
                    set(
                        [
                            var
                            for fac_ix in range(0, len(factors_and_vars), 2)
                            for var in factors_and_vars[fac_ix + 1]
                        ]
                    )
    )

    return (
            # squeeze to remove single dimensions entries
            np.squeeze(sum_product.einsum(*factors_and_vars, marg_vars)),
            [var for var in marg_vars if var != ...]
    )


def compute_beliefs(tree, potentials, clique_vars):
    ''' Computes beliefs for clique potentials in a junction tree
    using Shafer-Shenoy updates.

    :param tree: list representing the structure of the junction tree
    :param potentials: list of numpy arrays for cliques in junction tree
    :param clique_vars: list of variables included in each clique in potentials list
    :return: list of numpy arrays defining computed beliefs of each clique
    '''

    def get_message(sepset_ix, tree, beliefs, clique_vars):
        ''' Computes message with scope defined by sepset

        :param sepset_ix: index of sepset scope in which to return message
        (use slice(0) for no sepset)
        :param tree: list representation of tree rooted by cluster for which message
        will be computed
        :param beliefs: list of numpy arrays for cliques in junction tree
        :param clique_vars: list of variables included in each clique in potentials list
        :return: message: potential with scope defined by sepset (or None if tree includes root)
        '''

        clique = clique_vars[tree[0]]

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

        # reindex vars before doing einsum computation
        var_map = {
            var:i
            for i, var in enumerate(list(set(
                clique_vars[sepset_ix] + clique + neighbor_vars
            )))
        }

        # map sepset vars using new indices
        messages[1::2] = [
            [
                var_map[var]
                for var in ss
            ]
            for ss in messages[1::2]
        ]


        m_neighbor_vars = [var_map[var] for var in neighbor_vars]

        # multiply neighbor messages
        messages = messages if len(messages) else [1]

        msg_prod = sum_product.einsum(
                                *messages,
                                m_neighbor_vars
        )

        m_sepset = [var_map[var] for var in clique_vars[sepset_ix]]

        args = [msg_prod, m_neighbor_vars] + [beliefs[tree[0]], [var_map[var] for var in clique], m_sepset]

        # compute message as marginalization over non-sepset values
        # multiplied by product of messages
        # with output being vars in input sepset
        message = sum_product.einsum(*args)


        try:
            # attempt to update belief
            beliefs[sepset_ix] = message
            return message
        except TypeError:
            # function called on full tree so no message to send
            return None


    def remove_message(msg_prod, prod_ixs, msg, msg_ixs, out_ixs):
        ''' Removes (divides out) sepset message from
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

        # create dummy dimensions for performing division (with exp_ix)
        # slice out dimensions of sepset variables from division result (with slice_ixs)


        return np.divide(
                    msg_prod,
                    msg[ tuple(exp_ixs) ] \

                        # axis must be re-ordered if all variables shared but order is different
                        if not (all(exp_mask) and msg_ixs != prod_ixs) else \

                    np.moveaxis(msg, prod_ixs, msg_ixs)
        )[ tuple(slice_ixs) ]


    def send_message(message, sepset_ix, tree, beliefs, clique_vars):
        ''' Sends message to cluster at root of tree

        :param message: message sent by neighbor
                (use np.array(1) for no message)
        :param sepset_ix: index of sepset scope in which message sent
                (use slice(0) for no sepset)
        :param tree: list representation of tree rooted by cluster receiving message
        :param beliefs: list of numpy arrays for cliques in junction tree
        :param clique_vars: list of variables included in each clique in potentials list
        '''

        clique = clique_vars[tree[0]]

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

        # reindex vars before doing einsum computation
        var_map = {
            var:i
            for i, var in enumerate(list(set(
                clique_vars[sepset_ix] + clique + neighbor_vars
            )))
        }

        # map sepset vars using new indices
        messages[1::2] = [
            [
                var_map[var]
                for var in ss
            ]
            for ss in messages[1::2]
        ]

        m_neighbor_vars = [var_map[var] for var in neighbor_vars]

        # multiply neighbor messages
        msg_prod = sum_product.einsum(
                                *messages,
                                m_neighbor_vars
        )

        m_clique_vars = [var_map[var] for var in clique]

        # send message to each neighbor
        ss_num = 0
        for ss_ix, subtree in tree[1:]:

            # sum over vars not in sepset
            m_sepset = [var_map[var] for var in clique_vars[ss_ix]]

            # divide product of messages by message for this neighbor

            mask = np.in1d(
                            m_neighbor_vars,
                            list(
                                set(
                                    [
                                        var
                                        for vars in messages[1::2][0:ss_num] + messages[1::2][ss_num+1:]
                                        for var in vars
                                    ]
                                )
                            )
            )

            mod_neighbor_vars = np.array(m_neighbor_vars)[mask].tolist()

            mod_msg_prod = remove_message(
                                    msg_prod,
                                    m_neighbor_vars,
                                    beliefs[ss_ix],
                                    m_sepset,
                                    mod_neighbor_vars
            )

            args = [mod_msg_prod, mod_neighbor_vars] + [beliefs[tree[0]], m_clique_vars, m_sepset]
            # calculate message to be sent
            message = sum_product.einsum( *args )

            # update sepset belief
            beliefs[ss_ix] = beliefs[ss_ix] * message

            send_message(message, ss_ix, subtree, beliefs, clique_vars)
            ss_num += 1

        # update belief for clique
        args = [
            beliefs[tree[0]],
            m_clique_vars,
            msg_prod,
            m_neighbor_vars,
            m_clique_vars
        ]

        beliefs[tree[0]] = sum_product.einsum(*args)


    def __run(tree, beliefs, clique_vars):
        ''' Collect messages from neighbors recursively. Then, send messages
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
