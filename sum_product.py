import numpy as np


class SumProduct():
    """ Sum-product distributive law """


    def __init__(self, einsum):
        # Perhaps support for different frameworks (TensorFlow, Theano) could
        # be provided by giving the necessary functions.
        self.einsum = einsum
        return

    def project(self, clique_pot, clique_keys, sep_keys):
        """
        Compute sepset potential by summing over keys
            in clique not shared by separator

        Input:
        ------

        Clique potential

        Clique keys

        Separator keys

        Output:
        -------

        Updated separator potential

        """
        # map keys to get around variable count limitation in einsum
        m_keys = {k:i for i,k in enumerate(set(clique_keys + sep_keys))}
        return self.einsum(
            clique_pot,
            [m_keys[k] for k in clique_keys],
            [m_keys[k] for k in sep_keys]
        )

    def absorb(self, clique_pot, clique_keys, sep_pot, new_sep_pot, sep_keys):
        """
        Compute new clique potential as product of old clique potential
            and quotient of new separator potential and old separator
            potential

        Input:
        ------

        Clique potential to be updated

        Clique keys

        Old separator potential

        New separator potential

        Separator keys

        Output:
        -------

        Updated clique potential

        """
        if np.all(sep_pot) == 0:
            return np.zeros_like(clique_pot)

        # map keys to get around variable count limitation in einsum
        m_keys = {k:i for i,k in enumerate(set(clique_keys + sep_keys))}
        return self.einsum(
            new_sep_pot / sep_pot, [m_keys[k] for k in sep_keys],
            clique_pot, [m_keys[k] for k in clique_keys],
            [m_keys[k] for k in clique_keys]
        )

    def update(self, clique1_pot, clique1_keys, clique2_pot, clique2_keys, sep_pot, sep_keys):
        """
        A single update (message pass) from clique1 to clique2
            through separator

        Input:
        ------

        Clique1 potential

        Clique1 keys

        Clique2 potential

        Clique2 keys

        Separator potential

        Separator keys

        Output:
        -------

        Updated clique2 potential and updated separator potential

        """
        # See page 2:
        # http://compbio.fmph.uniba.sk/vyuka/gm/old/2010-02/handouts/junction-tree.pdf


        # Sum keys in A that are not in B
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

        return (new_clique2_pot, new_sep_pot) # may return unchanged clique_a
                                             # too if it helps elsewhere
