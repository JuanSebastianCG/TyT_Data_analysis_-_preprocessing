import pandas as pd
import pickle
import re
import csv
import numpy as np

class DataExtractor:
    
    @staticmethod
    def transform_json_keys(input_data: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        """
        Transforms the keys of input_data based on the provided mapping dictionary.
        Assumes input_data is a DataFrame with columns that need to be renamed.

        Parameters:
        input_data (pd.DataFrame): The DataFrame with columns that need to be renamed.
        mapping (dict): A dictionary that maps the original column names to the new column names expected by the model.

        Returns:
        pd.DataFrame: A new DataFrame with transformed column names.
        """
        # Verificar y mapear cada columna del DataFrame
        new_columns = []
        for col in input_data.columns:
            if col in mapping:
                new_columns.append(mapping[col])
            else:
                # Si no se encuentra mapeo, lanzar una advertencia o manejar el caso
                raise KeyError(f"La columna '{col}' no tiene un mapeo en el modelo.")

        # Renombrar las columnas del DataFrame según el mapeo
        input_data.columns = new_columns
        
        return input_data

    
    @staticmethod
    def apply_mapping(input_data, diccionary: dict) -> pd.DataFrame:
        """
        Applies a mapping dictionary to the data (X) by iterating through columns and applying
        the corresponding mapping manually.

        Parameters:
        input_data (pd.DataFrame): The input data as a pandas DataFrame. 
        diccionary (dict): A dictionary containing the mapping for each column.

        Returns:
        DataFrame: The transformed data with mapped features.
        """
        # Iterar sobre las columnas del diccionario de mapeo
        for column, mapping in diccionary.items():
            if column in input_data.columns:
                # Aplicar el mapeo de valores para cada columna de manera individual
                input_data[column] = input_data[column].map(mapping)
        
        return input_data

    @staticmethod
    def save_array_to_txt( array, filenamePath):
        """
        Save a given array to a text file in CSV format.

        Parameters:
        array (array-like): The array to be saved. It can be a list or a numpy array.
        filenamePath (str): The name of the file to save the array to.

        Returns:
        None
        """
        # Construct the full path for the output file
        path = filenamePath
        
        # Ensure the input is a numpy array
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        # Save the array to a text file with comma as the delimiter
        np.savetxt(path, array, delimiter=',', fmt='%s', newline='\n')

    @staticmethod
    def load_data_txt_to_dataframe(file_path, delimiter='¬', encoding='utf-8'):
        """
        Load a text file where the fields are separated by a custom delimiter (default '¬')
        and return it as a pandas DataFrame.

        Parameters:
        file_path (str): The path to the input text file.
        delimiter (str, optional): The delimiter used to separate fields in the text file. Default is '¬'.
        encoding (str, optional): The encoding of the text file. Default is 'utf-8'.

        Returns:
        pd.DataFrame: A pandas DataFrame with the content of the text file.
        """
        try:
            # Read the file directly into a pandas DataFrame using the custom delimiter
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
            return df
        except Exception as e:
            print(f"Error while reading the file: {e}")
            return None

    @staticmethod
    def load_data_csv(filenamePath, encoding='latin1', delimiter=';'):
        """
        Load data from a CSV file and attempt to convert columns to numeric types.

        Parameters:
        filenamePath (str): The name of the CSV file to be loaded.
        encoding (str, optional): The encoding of the CSV file. Default is 'latin1'.
        delimiter (str, optional): The delimiter used in the CSV file. Default is ';'.

        Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame. If there is a parsing error, returns None.
        """
        path = f"{filenamePath}"
        try:
            data = pd.read_csv(path, encoding=encoding, delimiter=delimiter, header=0, quoting=csv.QUOTE_NONE, decimal=',')
            for column in data.columns:
                original_data = data[column].copy()
                data[column] = pd.to_numeric(data[column], errors='coerce')
                if data[column].isna().any():
                    data[column] = original_data
        except pd.errors.ParserError as e:
            print(f"Error reading the file: {e}")
            return None
        return data

    @staticmethod
    def load_data_txt( filenamePath, delimiter='\n', encoding='utf-8'):
        """
        Load data from a text file and return it as a NumPy array of strings.

        Parameters:
        folder_name (str): The name of the folder containing the file.
        filenamePath (str): The name of the text file to be loaded.
        delimiter (str, optional): The delimiter used to separate values in the text file. Default is '\n'.
        encoding (str, optional): The encoding of the text file. Default is 'utf-8'.

        Returns:
        np.ndarray: A NumPy array containing the data from the text file.
        """
        path = f"{filenamePath}"
        return np.genfromtxt(path, delimiter=delimiter, dtype=str, encoding=encoding)
    
    @staticmethod
    def load_data_txt( filenamePath, delimiter='\n', encoding='utf-8'):
        """
        Load data from a text file and return it as a NumPy array of strings.

        Parameters:
        folder_name (str): The name of the folder containing the file.
        filenamePath (str): The name of the text file to be loaded.
        delimiter (str, optional): The delimiter used to separate values in the text file. Default is '\n'.
        encoding (str, optional): The encoding of the text file. Default is 'utf-8'.

        Returns:
        np.ndarray: A NumPy array containing the data from the text file.
        """
        path = f"{filenamePath}"
        return np.genfromtxt(path, delimiter=delimiter, dtype=str, encoding=encoding)

    @staticmethod
    def save_data_pickle( data, filenamePath, filter_condition=None):
        """
        Save data to a pickle file, optionally filtering it based on a condition.

        Parameters:
        data (DataFrame): The data to be saved.
        filenamePath (str): The name of the file to save the data in.
        filter_condition (str, optional): A condition to filter the data before saving. Defaults to None.

        Returns:
        None
        """
        # Construct the full path for the output file
        path = f"{filenamePath}"
        
        # If a filter condition is provided, filter the data
        if filter_condition:
            data = data.query(filter_condition)
        
        # Open the file in write-binary mode and save the data using pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_data_pickle(namefile_path):
        with open( namefile_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def remove_non_ascii(text):
        """
        Remove non-ASCII characters from a given text string.

        Parameters:
        text (str): The input string from which non-ASCII characters need to be removed.

        Returns:
        str: A new string with all non-ASCII characters removed.
        """
        return re.sub(r'[^\x00-\x7F]+', '', text)
    
        

