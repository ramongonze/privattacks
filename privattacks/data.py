import os
import pyreadr
import numpy as np
import pandas as pd

class Data:
    """
    A class for handling datasets. The supported formats are 'csv', 'rdata' and 'sas7bdat'.

    Parameters:
        - file_name (str, optional): Dataset file path.
        - cols (list, optional): Columns to be read from the file. When not given, all columns will be read.
        - sep_csv (str, optional): CSV delimiter, default is ",".
        - encoding: (str, optional, default 'utf-8'): Encoding to use for UTF when reading/writing (ex. 'utf-8', 'latin1'). [List of Python standard encodings](https://docs.python.org/3/library/codecs.html#standard-encodings).
        - dataframe (pandas.DataFrame, optional): Pandas dataframe containing the dataset.
        - na_values (int, optional): Value to fill missing data (NaN) with, default is -1.

    Attributes:
        - dataset (numpy.ndarray): Numpy matrix of integers.
        - n_rows (int): Number of rows (records) in the dataset.
        - n_cols (int): Number of columns (attributes) in the dataset.
        - cols (list): List of column names in the dataset. The same order as the dataset matrix.
        - int2value (list[list[any]]): Reference list for converting integer values in the new NumPy matrix back to the original column domains. Each element corresponds to a column in `cols` and contains the original values in the order they were mapped to integers (e.g., first value → 0, second → 1, etc.). Example: If columns A and B had values `['apple', 'banana', 'cherry']` and `['fig', 'grape', 'kiwi']`, respectively, and were converted as `apple -> 0, banana -> 1, cherry -> 2` and `fig -> 0, grape -> 1, kiwi -> 2`, then `int2value = [['apple', 'banana', 'cherry'], ['fig', 'grape', 'kiwi']]`.
    """

    def __init__(self, file_name=None, cols=None, sep_csv=None, encoding='utf-8', dataframe=None, na_values=-1):
        """
        Initializes a Dataset object.

        Parameters:
            - file_name (str, optional): Dataset file path.
            - cols (list, optional): Columns to be read from the file. When not given, all columns will be read.
            - sep_csv (str, optional): CSV delimiter, default is ",".
            - encoding: (str, optional, default 'utf-8'): Encoding to use for UTF when reading/writing (ex. 'utf-8', 'latin1'). [https://docs.python.org/3/library/codecs.html#standard-encodings](List of Python standard encodings).
            - dataframe (pandas.DataFrame, optional): Pandas dataframe containing the dataset.
            - na_values (int, optional): Value to fill missing data (NaN) with, default is -1.
        """
        if dataframe is not None:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("dataframe must be a pandas.DataFrame object")
        elif file_name is not None:
            file_type = self._file_extension(file_name)
            if file_type == ".csv":
                if sep_csv is None:
                    raise NameError("sep_csv must be provided for CSV files")
                dataframe = pd.read_csv(file_name, sep=sep_csv, usecols=cols, encoding=encoding)
            elif file_type == ".rdata":
                rdata = pyreadr.read_r(file_name)
                data = next(iter(rdata))
                dataframe = rdata[data]
            elif file_type == ".sas7bdat":
                dataframe = pd.read_sas(file_name)
            else:
                raise TypeError("The only supported files are 'csv', 'rdata' and 'sas7bdat'")
        else:
            raise TypeError("Either file_name or dataframe must be given")
        
        dataframe.replace(np.nan, na_values, inplace=True)
        self.n_rows = dataframe.shape[0]
        self.n_cols = dataframe.shape[1]
        self.cols = dataframe.columns.to_list()
        self.dataset, self.int2value = self.convert_df_to_np(dataframe)
        self._sort_dataset()

    def col2int(self, att):
        """Index of columns in the dataset numpy matrix."""
        return self.cols.index(att)

    def convert_df_to_np(self, dataframe:pd.DataFrame) -> np.ndarray:
        """Converts a pandas dataframe to a numpy.ndarray. The matrix contains integers in "standard" type, i.e., for all column c, the original values from the domain of c are converted to integers from 0 to |c|.
        The method generates a numpy.ndarray and a dictionary to convert integers to the original values.

        Parameters:
            dataframe (pandas.DataFrame): Dataset.
        
        Returns:
            dataset (numpy.ndarray): Dataset in standard type.
            int2value (list[list]): Reference list for converting integer values in the new NumPy matrix back to the original column domains. Each element corresponds to a column in `cols` and contains the original values in the order they were mapped to integers (e.g., first value → 0, second → 1, etc.). Example: If columns A and B had values `['apple', 'banana', 'cherry']` and `['fig', 'grape', 'kiwi']`, respectively, and were converted as `apple -> 0, banana -> 1, cherry -> 2` and `fig -> 0, grape -> 1, kiwi -> 2`, then `int2value = [['apple', 'banana', 'cherry'], ['fig', 'grape', 'kiwi']]`.
        """
        # Create a tranposed matrix because numpy is row-oriented
        dataset = np.empty(dataframe.shape[::-1], dtype=int)
        cols = dataframe.columns.tolist()
        int2value = []
        for i, col in enumerate(cols):
            # Define the order each value will be converted to integers
            domain = dataframe[col].unique().tolist()

            convert = lambda value : domain.index(value)
            new_col = dataframe[col].apply(convert)
            dataset[i, :] = new_col

            # Add refrence to original values
            int2value.append(domain)
        
        # Convert back the dataset to the correct orientation
        return dataset.T, int2value

    def _file_extension(self, file_name:str) -> str:
        """
        Infer the file extension from a given file path.

        Parameters:
            file_name (str): The path to the file.

        Returns:
            str: The file extension in lowercase.
        """
        _, extension = os.path.splitext(file_name)
        return extension.lower()

    def _sort_dataset(self):
        """Sorts the dataset by all columns. It's an assumption for attack methods in class Attack.

        Obs: Once the dataset is sorted for all columns, it doesn't need to be sorted anymore for subset of columns beacuse sorting is stable.
        """
        # Sort by all columns in ascending order (lexicographical sort)
        # Provide columns in reverse order of priority
        keys = tuple(self.dataset[:, i] for i in np.arange(self.n_cols - 1, -1, -1))
        sorted_indices = np.lexsort(keys)

        # Use the indices to sort the array
        self.dataset = self.dataset[sorted_indices]