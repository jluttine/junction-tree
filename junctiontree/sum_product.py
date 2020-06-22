
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
