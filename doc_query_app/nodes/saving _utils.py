import pandas as pd
import numpy as np


def save_dataframe_with_embeddings(df, save_path, df_name='hybrid_chunks_df'):
    """
    Save a pandas DataFrame containing embeddings to a pickle file.

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


def save_numpy_array(arr, save_path, arr_name='full_embeddings_matrix'):
    """
    Save a NumPy array to a .npy file.

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
