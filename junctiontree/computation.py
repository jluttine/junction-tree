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

        if m_factor.ndim  == 1:
            # 1-dimensional factors require special treatment
            m_factor = m_factor[:, np.newaxis]
            output_var_indices += [...]


        # return a list with remaining factors(excluded and marginalized factor) and their variables
        # [ex_fac1, ex_fac1_vars, ex_fac2, ex_fac2_vars,..., new_fac, new_fac_vars]

        return(
                sum(
                    [
                        [factors_and_vars[fac_ix], factors_and_vars[fac_ix + 1]]
                        for fac_ix in exclude_factors
                    ],
                    [m_factor, output_var_indices]
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


sum_product = SumProduct(np.einsum)
