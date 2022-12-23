from typing import List, Callable


class MIMODGFunctionInterface:
    """An Interface for functions. Sends only first variable to the function """
    def __init__(self, function: Callable, mapping_dict: dict, single_param_transferring: bool = True):
        self.function = function
        self.mapping_dict = mapping_dict
        self.single_param_transferring = single_param_transferring

    def __call__(self, values, col_names: List[str]):
        params_dict = dict(zip(col_names, values))
        if self.single_param_transferring:
            return self.function(**{k: params_dict.get(v) for k, v in self.mapping_dict.items()})
        else:
            return self.function(**{k: params_dict.get(v) for k, v in self.mapping_dict.items()})
