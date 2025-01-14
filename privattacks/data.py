import os
import pyreadr
import copy
import numpy as np
import pandas as pd

class Data:
    """
    A class for handling datasets. The supported formats are 'csv', 'rdata' and 'sas7bdat'.

    Parameters:
        file_name (str, optional): Dataset file path.
        cols (list, optional): Columns to be read from the file. When not given, all columns will be read.
        sep_csv (str, optional): CSV delimiter, default is ",".
        encoding: (str, optional, default 'utf-8'): Encoding to use for UTF when reading/writing (ex. 'utf-8', 'latin1'). [List of Python standard encodings](https://docs.python.org/3/library/codecs.html#standard-encodings).
        dataframe (pandas.DataFrame, optional): Pandas dataframe containing the dataset.
        na_values (int, optional): Value to fill missing data (NaN) with, default is -1.

    Attributes:
        dataset (numpy.ndarray): Numpy matrix of integers.
        n_rows (int): Number of rows (records) in the dataset.
        n_cols (int): Number of columns (attributes) in the dataset.
        cols (list): List of column names in the dataset. The same order as the dataset matrix.
        domains (dict[str, list]): Column domains. Keys are column names and values are lists. To generate the numpy matrix each original value will be converted to its index in the domain's list.
    """

    def __init__(self, file_name=None, cols=None, sep_csv=",", encoding='utf-8', dataframe=None, matrix=None, domains=None, na_values=-1):
        """
        Initializes a Dataset object.

        Parameters:
            file_name (str, optional): Dataset file path.
            cols (list, optional): Dataset columns. If not given when given file_name, read all columns in the file.
            sep_csv (str, optional): CSV delimiter, default is ",".
            encoding: (str, optional, default 'utf-8'): Encoding to use for UTF when reading/writing (ex. 'utf-8', 'latin1'). [https://docs.python.org/3/library/codecs.html#standard-encodings](List of Python standard encodings).
            dataframe (pandas.DataFrame, optional): Pandas dataframe containing the dataset.
            matrix (numpy.ndarray, optional): Numpy 2d matrix containing the dataset.
            domains (dict[str, list], optional): Domain of columns. If not given, the domains will be taken from data. Keys are column names and values are lists.
            na_values (int, optional): Value to fill missing data (NaN) with, default is -1.
        """
        if dataframe is not None:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("dataframe must be a pandas.DataFrame object")  
        elif matrix is not None:
            if not isinstance(matrix, np.ndarray):
                raise TypeError("matrix must be a numpy.ndarray object")  
            
            if sep_csv is None:
                raise ValueError("cols argument not given")
            
            dataframe = pd.DataFrame(matrix, columns=cols)
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
        
        # If domains is not given, take the domains from the dataset
        if domains is not None:
            self.domains = domains
        else:
            self.domains = self._get_col_domains(dataframe)

        self.dataset = self.df2np(dataframe)

    def col2int(self, col) -> int:
        """Index of a column in the dataset numpy matrix."""
        return self.cols.index(col)

    def np2df(self) -> pd.DataFrame:
        """Convert the numpy matrix to the dataset original domains.
        
        Returns
            df (pandas.DataFrame): Dataset with original domains.
        """
        df = pd.DataFrame(self.dataset, columns=self.cols)
        for col in self.cols:
            df[col] = df[col].apply(lambda value : self.domains[col][value])
        
        return df

    def df2np(self, dataframe:pd.DataFrame) -> np.ndarray:
        """Converts a pandas dataframe to a numpy.ndarray. The matrix contains integers in "standard" type, i.e., for all column c, the original values from the domain of c are converted to integers from 0 to size(c). Each original value in a domain will be converted to the respective index the value is in the domain list.
        The method generates a numpy.ndarray.

        Parameters:
            dataframe (pandas.DataFrame): Dataset.
        
        Returns:
            dataset (numpy.ndarray): Dataset in standard type.
        """
        # Create a tranposed matrix because numpy is row-oriented
        dataset = np.empty(dataframe.shape[::-1], dtype=int)
        for i, col in enumerate(self.cols):
            convert = lambda value : self.domains[col].index(value)
            dataset[i, :] = dataframe[col].apply(convert)
        
        # Convert back the dataset to the correct orientation
        return dataset.T

    def _get_col_domains(self, dataframe):
        """Get columns domain from the dataset."""
        return {col:dataframe[col].unique().tolist() for col in self.cols}
            
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
