from sentence_transformers import SentenceTransformer
import torch
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from nodes.saving_utils import save_embeddings_matrix
from config import (
    BASE_DIR,
    EMBEDDING_MODEL_NAME
)

base_dir = BASE_DIR


def generate_optimized_embeddings(source_chunks_df, model_name_to_use=None):
    if model_name_to_use is None:
        print("Model name not provided. Using default model: all-MiniLM-L6-v2")
        model_name_to_use = EMBEDDING_MODEL_NAME
    print(f"Loading model: {model_name_to_use}")
    model = SentenceTransformer(model_name_to_use)

    batch_size = 32
    if torch.cuda.is_available():
        try:
            free_mem_bytes = torch.cuda.get_device_properties(
                0).total_memory - torch.cuda.memory_allocated(0)
            free_mem_gb = free_mem_bytes / (1024**3)
            if free_mem_gb > 12:
                batch_size = 128
            elif free_mem_gb > 8:
                batch_size = 64
            elif free_mem_gb > 4:
                batch_size = 32
            else:
                batch_size = 16
            print(
                f"CUDA available ({free_mem_gb:.2f}GB free). Using batch size: {batch_size}")
        except Exception as e:
            print(
                f"Could not query CUDA memory, defaulting batch size to 32. Error: {e}")
            batch_size = 32
    else:
        print(f"CUDA not available. Using CPU batch size: {batch_size}")

    print("Generating embeddings...")
    texts_to_embed = source_chunks_df['text'].tolist()
    num_texts = len(texts_to_embed)
    embedding_dim = model.get_sentence_embedding_dimension()
    all_embeddings = np.zeros((num_texts, embedding_dim), dtype=np.float32)

    if num_texts > 0:
        for i in tqdm(range(0, num_texts, batch_size), desc="Batch encoding"):
            batch_texts = texts_to_embed[i:i+batch_size]
            batch_embeddings = model.encode(
                batch_texts, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            all_embeddings[i:i+len(batch_texts)] = batch_embeddings

    # Make a copy to avoid SettingWithCopyWarning
    output_chunks_df = source_chunks_df.copy()
    output_chunks_df['embedding'] = list(all_embeddings)

    print(f"Generated embeddings with shape: {all_embeddings.shape}")
    return output_chunks_df, all_embeddings, model


def update_embedding_matrix(self, new_doc_df: pd.DataFrame, current_embeddings) -> np.ndarray:
    """
    Update the embedding matrix with new document embeddings.

    Args:
        new_doc_df (pd.DataFrame): DataFrame containing new document chunks
        current_embeddings (np.ndarray): Current embeddings matrix

    Returns:
        np.ndarray: Updated embeddings matrix
    """
    try:
        # Generate embeddings for new document chunks
        new_embeddings, _ = generate_optimized_embeddings(
            new_doc_df['text'].tolist(),
            EMBEDDING_MODEL_NAME
        )

        # Convert to numpy array if not already
        new_embeddings = np.array(new_embeddings)

        # Update the full embeddings matrix
        if current_embeddings is None:
            updated_embeddings = new_embeddings
        else:
            updated_embeddings = np.vstack([
                current_embeddings,
                new_embeddings
            ])

        # Save updated embeddings matrix
        save_embeddings_matrix(
            updated_embeddings,
            os.path.join(
                base_dir, "data/full_embeddings_matrix.npy"
            )
        )

        print(
            f"Successfully updated embeddings matrix with {len(new_embeddings)} new embeddings")

        return updated_embeddings

    except Exception as e:
        print(f"Error updating embedding matrix: {e}")
        raise
