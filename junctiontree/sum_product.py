import numpy as np


class SumProduct():
    ''' Sum-product distributive law '''


    def __init__(self, einsum, *args, **kwargs):
        # Perhaps support for different frameworks (TensorFlow, Theano) could
        # be provided by giving the necessary functions.
        self.func = einsum
        self.args = args
        self.kwargs = kwargs
        return

    def einsum(self, *args, **kwargs):
        '''Performs Einstein summation based on input arguments

        :param args: the required positional arguments passed to underlying einsum function
        :param kwargs: provides ability to pass key-word args to underlying function
        :return: the resulting calculation based on the summation performed
        '''

        args_list = list(args)

        var_indices = args_list[1::2] + [args_list[-1]] if len(args_list) % 2 == 1 else []

        var_map = {var:i for i, var in enumerate(set([var for vars in var_indices for var in vars]))}

        args_list[1::2] = [
            [ var_map[var] for var in vars ] if len(vars) > 0 else [] for vars in args_list[1::2]
        ]

        # explicit output indices may be provided requiring one additional mapping
        args_list[-1] = [var_map[var] for var in args_list[-1]]

        return self.func(*args_list, *self.args, **kwargs, **self.kwargs)


    def project(self, clique_pot, clique_keys, sep_keys):
        ''' Compute sepset potential by summing over keys in clique not shared by separator

        :param clique_pot: clique potential
        :param clique_keys: keys (nodes) in clique
        :param sep_keys: keys (nodes) in separator
        :return: updated sepset potential after projection
        '''

        # map keys to get around variable count limitation in einsum
        mapped_keys = []
        m_keys = {}
        for i,k in enumerate(clique_keys):
            m_keys[k] = i
            mapped_keys.append(i)

        return self.einsum(
            clique_pot,
            mapped_keys,
            [m_keys[k] for k in sep_keys]
        )

    def absorb(self, clique_pot, clique_keys, sep_pot, new_sep_pot, sep_keys):
        '''Compute new clique potential as product of previous clique potential
        and quotient of new separator potential and previous separator potential

        :param clique_pot: clique potential to be updated
        :param clique_keys: clique keys (nodes)
        :param sep_pot: previous separator potential
        :param new_sep_pot: new separtor potential
        :param sep_keys: separtory keys (nodes
        :return: updated clique potential
        '''

        if np.all(sep_pot) == 0:
            return np.zeros_like(clique_pot)

        # map keys to get around variable count limitation in einsum
        mapped_keys = []
        m_keys = {}
        for i,k in enumerate(clique_keys):
            m_keys[k] = i
            mapped_keys.append(i)

        return self.einsum(
            new_sep_pot / sep_pot, [m_keys[k] for k in sep_keys],
            clique_pot, mapped_keys,
            mapped_keys
        )

    def update(self, clique1_pot, clique1_keys, clique2_pot, clique2_keys, sep_pot, sep_keys):
        '''A single update (message pass) from clique1 to clique2 through separator

        # See page 2:
        # http://compbio.fmph.uniba.sk/vyuka/gm/old/2010-02/handouts/junction-tree.pdf

        :param clique1_pot: clique1 potential
        :param clique1_keys: clique1 keys (nodes)
        :param clique2_pot: clique2 potential
        :param clique2_keys: clique2 keys (nodes)
        :param sep_pot: separator potential
        :param sep_keys: separator keys (nodes)
        :return new_clique2_pot: updated clique2 potential
        :return new_sep_pot: updated separator potential
        '''

        # Sum keys in clique 1 that are not in clique 2
        new_sep_pot = self.project(
                                clique1_pot,
                                clique1_keys,
                                sep_keys
        )

        # Compensate the updated separator in the clique
        new_clique2_pot = self.absorb(
                                clique2_pot,
                                clique2_keys,
                                sep_pot,
                                new_sep_pot,
                                sep_keys
        )

        return new_clique2_pot, new_sep_pot # may return unchanged clique_a
                                             # too if it helps elsewhere
