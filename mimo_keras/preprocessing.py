from __future__ import annotations

from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd

import multiprocessing

from tensorflow import keras


class MIMODataGenerator(keras.preprocessing.image.Iterator):
    """Keras DataGenerator for models with any number and any type of input/output"""

    def __init__(self, data_table: pd.DataFrame,
                 model_inputs: List[Tuple[str, List[str], Union[callable, str]]],
                 model_outputs: List[Tuple[str, List[str], Union[callable, str]]],
                 batch_size: int = 2,
                 shuffle: bool = False,
                 seed: Optional[int] = 42):
        """

        Parameters
        ----------
        data_table: pd.DataFrame, required
        model_inputs: list[tuple], required
            Each tuple belongs to one of the model inputs and contains N tuple for model has N inputs
             (input_name, columns, transformer)
            - input_name: str, name of the Nth input
            - columns: list[str], feature_names belonging to the Nth input
            - transformer: callable class or function that accepts two lists in it's input amd
                return ndarray with the same shape of the Nth input model. To send the raw values directly to
                the model without any additional processing, use word 'raw' instead of callable function.

        model_outputs: list[tuple], required
            like model_inputs but M tuple for model has M outputs
        batch_size: int, required
        shuffle: bool, required
            True: for training data generator
            False: for testing and validation
        seed: int, optional
            random seed
        """

        self.data_table = data_table
        self.batch_size = batch_size
        self.nb_samples = len(self.data_table)

        self.structure_inputs = self._get_cols_to_function_mapping_dict(model_inputs)
        self.structure_outputs = self._get_cols_to_function_mapping_dict(model_outputs)

        super().__init__(self.nb_samples, batch_size, shuffle, seed)

    def _get_cols_to_function_mapping_dict(self, mapping_list: List[Tuple[str, List[str], Union[callable, str]]]):

        if len(mapping_list) < 1:
            raise ValueError('The model must have at least one input and one output...! :D')

        structure = dict()
        for model_io_name, columns, function in mapping_list:
            structure[model_io_name] = dict()

            if callable(function):
                structure[model_io_name]['function'] = function
            elif function == 'raw':
                structure[model_io_name]['function'] = lambda x1, x2: x1
            else:
                raise ValueError("The transformer of the input should be callable or \'raw\' :D")

            if isinstance(columns, list):
                if len(columns) == 0:
                    raise ValueError('Each model io must have been mapped to at least one column ...! :D')
                structure[model_io_name]['columns'] = columns
                structure[model_io_name]['data_value'] = np.array(self.data_table[columns].to_dict('split')['data'])
            else:
                raise TypeError("\'columns\' must be a list of feature_names")

        return structure

    @staticmethod
    def _get_io_data_values(io_structure: dict, index_array) -> np.numarray:
        """Gathers data of an IO due to its structure

        Parameters
        ----------
        io_structure: dict, required
            subdict of self.structure_inputs or self.structure_outputs

        Returns
        -------
        io_batch_data: ndarray
            data belonging to an IO of inputs or outputs

        """
        io_batch_data = []
        results = []

        pool = multiprocessing.pool.ThreadPool(16)

        for i in index_array:
            col_names = io_structure['columns']
            col_values = io_structure['data_value'][i]
            results.append(pool.apply_async(io_structure['function'], (col_values, col_names,)))
        for res in results:
            x = res.get()
            if isinstance(x, (np.ndarray, np.integer, np.float)):
                if len(np.shape(x)) == 1 and np.shape(x)[0] == 1:
                    x = x[0]
                io_batch_data.append(x)
            else:
                raise TypeError(f"The type of the io data value must be ndarray or number, but received {type(x)}! :D")

        pool.close()
        pool.join()

        io_batch_data = np.array(io_batch_data)
        return io_batch_data

    def _structure_data_values(self, structure: dict, index_array):
        """Gathers data of all inputs or outputs

        Parameters
        ----------
        structure: dict, required
            Its value could self.structure_inputs or self.structure_outputs

        Returns
        -------
        batch_data_structure: ndarray
            Model input or output values belonging to the self.structure_inputs or self.structure_outputs

        """
        batch_data_structure = []
        for io_name, io_structure in structure.items():
            batch_data_structure.append(self._get_io_data_values(io_structure, index_array))
        if len(batch_data_structure) == 1:
            return batch_data_structure[0]
        else:
            return batch_data_structure

    def _get_batches_of_transformed_samples(self, index_array):

        batch_data_inputs = self._structure_data_values(self.structure_inputs, index_array)
        batch_data_outputs = self._structure_data_values(self.structure_outputs, index_array)

        return batch_data_inputs, batch_data_outputs

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    def get_io_data_values_by_name(self, io_name: str, indexes: str | List[int] = 'all'):
        if not isinstance(io_name, str):
            raise TypeError("io_name must be string...")
        if isinstance(indexes, str) and indexes == 'all':
            indexes = [i for i in range(self.n)]
        elif not isinstance(indexes, list):
            raise TypeError("indexes must be a list of int or 'all'...")

        if io_name in list(self.structure_inputs.keys()):
            return self._get_io_data_values(self.structure_inputs[io_name], indexes)
        elif io_name in list(self.structure_outputs.keys()):
            return self._get_io_data_values(self.structure_outputs[io_name], indexes)
        else:
            raise ValueError(f"Key error: {io_name} not founded in input and ouput names...")
