import pandas as pd
import numpy as np


import os


def save_dataframe_with_embeddings(df, save_path, df_name='hybrid_chunks_df'):
    """
    Save a pandas DataFrame containing embeddings to a pickle file.
    Deletes existing file if present before saving.

    Args:
        df (pandas.DataFrame): DataFrame to save
        save_path (str): Full path where the DataFrame should be saved
        df_name (str, optional): Name of the DataFrame for logging purposes. Defaults to 'hybrid_chunks_df'.

    Returns:
        bool: True if save was successful, False otherwise
    """
    print(f"Attempting to save {df_name}...")
    try:
        if df is not None:
            # Delete existing file if it exists
            if os.path.exists(save_path):
                os.remove(save_path)
                print(f"Deleted existing file at '{save_path}'")

            df.to_pickle(save_path)
            print(f"DataFrame '{df_name}' saved to '{save_path}'")
            return True
        else:
            print(
                f"Error: {df_name} is None. Please ensure the DataFrame is properly defined.")
            return False
    except Exception as e:
        print(f"An error occurred while saving {df_name}: {e}")
        return False


def save_embeddings_matrix(arr, save_path, arr_name='full_embeddings_matrix'):
    """
    Save a NumPy array to a .npy file.
    Deletes existing file if present before saving.

    Args:
        arr (numpy.ndarray): NumPy array to save
        save_path (str): Full path where the array should be saved
        arr_name (str, optional): Name of the array for logging purposes. Defaults to 'full_embeddings_matrix'.

    Returns:
        bool: True if save was successful, False otherwise
    """
    print(f"Attempting to save {arr_name}...")
    try:
        if arr is not None:
            # Delete existing file if it exists
            if os.path.exists(save_path):
                os.remove(save_path)
                print(f"Deleted existing file at '{save_path}'")

            np.save(save_path, arr)
            print(f"NumPy array '{arr_name}' saved to '{save_path}'")
            return True
        else:
            print(
                f"Error: {arr_name} is None. Please ensure the NumPy array is properly defined.")
            return False
    except Exception as e:
        print(f"An error occurred while saving {arr_name}: {e}")
        return False
