import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

class DataVisualization:
    def __init__(self):
        """
        Initialize the DataVisualization class. Currently, no special initialization is needed.
        """
        pass

    @staticmethod
    def visualize_numeric(numeric_data, kind='boxplot', cols_per_plot=10):
        """
        Visualizes numeric columns using either boxplot or histogram.

        Parameters:
        - numeric_data (pd.DataFrame): DataFrame containing numeric columns.
        - kind (str): The type of plot ('boxplot' or 'histogram').
        - cols_per_plot (int): Maximum number of columns to display per plot.

        This function detects numeric columns in the input DataFrame and divides the columns
        into chunks of size `cols_per_plot` for easier visualization. Depending on the `kind`
        parameter, it either generates boxplots or histograms.
        """
        # Select numeric columns from the DataFrame
        numeric_cols = numeric_data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            # Early return if no numeric columns are found
            print("No numeric columns to visualize.")
            return

        # Calculate the number of plots required based on the number of numeric columns
        num_plots = (len(numeric_cols) - 1) // cols_per_plot + 1
        for i in range(num_plots):
            # Define the range of columns to be visualized in the current plot
            start_col = i * cols_per_plot
            end_col = min((i + 1) * cols_per_plot, len(numeric_cols))
            subset = numeric_data.iloc[:, start_col:end_col]

            # Plot either boxplots or histograms based on the `kind` parameter
            if kind == 'boxplot':
                subset.plot(kind='box', figsize=(12, 8))  # Generate boxplot
                plt.title(f"Boxplot of Features: Plot {i + 1}")  # Add title
                plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
            elif kind == 'histogram':
                # Generate histograms with custom layout and binning
                subset.hist(figsize=(12, 8), bins=15, edgecolor='black', layout=(4, (end_col - start_col + 3) // 4))
                plt.suptitle(f"Histograms of Features: Plot {i + 1}", y=1.02)  # Add title
                plt.subplots_adjust(hspace=0.5)  # Adjust space between subplots
            plt.show()  # Display the plot

    @staticmethod
    def visualize_categorical(cat_data, cols_per_plot=10):
        """
        Visualizes categorical columns using bar plots.

        Parameters:
        - cat_data (pd.DataFrame or pd.Series): DataFrame or Series containing categorical columns to visualize.
        - cols_per_plot (int): Maximum number of columns to display per plot.

        This function identifies categorical columns and creates bar plots for the value counts
        of each category. The columns are divided into chunks for better visualization.
        """
        # Convert Series to DataFrame if needed for consistent handling
        if isinstance(cat_data, pd.Series):
            cat_data = cat_data.to_frame()

        # Select only columns of type 'object' (categorical)
        object_cols = cat_data.select_dtypes(include=['object']).columns
        if object_cols.empty:
            # Early return if no categorical columns are found
            print("No categorical columns to visualize.")
            return

        # Calculate the number of plots required based on the number of categorical columns
        num_plots = (len(object_cols) - 1) // cols_per_plot + 1
        for i in range(num_plots):
            # Define the range of columns to be visualized in the current plot
            start_col = i * cols_per_plot
            end_col = min((i + 1) * cols_per_plot, len(object_cols))
            subset = cat_data.iloc[:, start_col:end_col]

            # Create subplots for each categorical column in the subset
            fig, axes = plt.subplots(nrows=1, ncols=len(subset.columns), figsize=(15, 5))
            if len(subset.columns) == 1:
                axes = [axes]  # Ensure axes are treated as a list if only one plot is generated
            for j, col in enumerate(subset.columns):
                # Create a bar plot for the value counts of each categorical column
                subset[col].value_counts().plot(kind='bar', ax=axes[j], color='skyblue')
                axes[j].set_title(f"Bar plot for {col}")  # Set title
                axes[j].tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability
            plt.tight_layout()  # Adjust the layout to avoid overlap
            plt.show()  # Display the plot

    @staticmethod
    def describe_data(data):
        """
        Generates a detailed description of the DataFrame including general information,
        count of null values, and frequencies of unique values per column.

        Parameters:
        - data (pd.DataFrame or pd.Series): The data to describe.

        Returns:
        - str: Detailed description of the DataFrame or Series.

        This function provides an overview of the DataFrame's structure, including:
        - Data types of each column
        - Number of null values per column
        - Value counts for each column
        - DataFrame shape (rows, columns)
        """
        # Convert Series to DataFrame for consistent handling
        if isinstance(data, pd.Series):
            data = data.to_frame()

        # Capture DataFrame info (such as column data types) in a string buffer
        output = io.StringIO()
        data.info(buf=output)
        info = output.getvalue()

        # Append null value counts per column to the info
        null_counts = data.isnull().sum().to_string()
        info += "\n\nNull values per column:\n" + null_counts

        # Append data type information to the info
        dtypes_info = data.dtypes.to_string()
        info += "\n\nData Types:\n" + dtypes_info

        # Append value counts for each column to the info
        unique_counts = "\n\nUnique Values per Column:\n"
        for col in data.columns:
            unique_counts += f"{col}:\n{data[col].value_counts()}\n\n"

        # Append DataFrame shape information (rows, columns)
        info += unique_counts
        info += f"\nDataFrame Shape: {data.shape}"

        return info  # Return the complete detailed information
