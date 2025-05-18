from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from typing import Tuple


def calculate_document_similarity_by_mean(hybrid_chunks_df: pd.DataFrame, full_embeddings_matrix: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Calculate similarity between documents based on their chunk embeddings.

    Args:
        hybrid_chunks_df (pd.DataFrame): DataFrame containing document chunks with doc_id column
        full_embeddings_matrix (np.ndarray): Matrix of embeddings for all chunks

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: 
            - DataFrame with document similarities (doc_ids as index/columns)
            - Raw similarity matrix as numpy array
    """
    # Get unique document IDs
    doc_ids = hybrid_chunks_df['doc_id'].unique()

    # Initialize similarity matrix
    n_docs = len(doc_ids)
    doc_similarity_matrix = np.zeros((n_docs, n_docs))

    # Calculate mean similarity between documents
    for i, doc1_id in enumerate(doc_ids):
        # Get chunks and embeddings for first document
        doc1_chunks = hybrid_chunks_df[hybrid_chunks_df['doc_id'] == doc1_id]
        doc1_embeddings = full_embeddings_matrix[doc1_chunks.index]

        for j, doc2_id in enumerate(doc_ids):
            if i == j:
                # Same document - similarity is 1
                doc_similarity_matrix[i, j] = 1.0
                continue

            # Get chunks and embeddings for second document
            doc2_chunks = hybrid_chunks_df[hybrid_chunks_df['doc_id'] == doc2_id]
            doc2_embeddings = full_embeddings_matrix[doc2_chunks.index]

            # Calculate cosine similarity between all chunks
            similarity_matrix = cosine_similarity(
                doc1_embeddings, doc2_embeddings)

            # Take mean of all chunk similarities
            mean_similarity = np.mean(similarity_matrix)
            doc_similarity_matrix[i, j] = mean_similarity

    # Create DataFrame for better visualization using doc_ids as index/columns
    similarity_df = pd.DataFrame(
        doc_similarity_matrix,
        index=doc_ids,
        columns=doc_ids
    )

    return similarity_df, doc_similarity_matrix


def calculate_document_similarity_by_max(hybrid_chunks_df: pd.DataFrame, full_embeddings_matrix: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Calculate similarity between documents based on their most similar chunk embeddings.

    Args:
        hybrid_chunks_df (pd.DataFrame): DataFrame containing document chunks with doc_id column
        full_embeddings_matrix (np.ndarray): Matrix of embeddings for all chunks

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: 
            - DataFrame with document similarities (doc_ids as index/columns)
            - Raw similarity matrix as numpy array
    """
    # Get unique document IDs
    doc_ids = hybrid_chunks_df['doc_id'].unique()

    # Initialize similarity matrix
    n_docs = len(doc_ids)
    doc_similarity_matrix = np.zeros((n_docs, n_docs))

    # Calculate max similarity between documents
    for i, doc1_id in enumerate(doc_ids):
        # Get chunks and embeddings for first document
        doc1_chunks = hybrid_chunks_df[hybrid_chunks_df['doc_id'] == doc1_id]
        doc1_embeddings = full_embeddings_matrix[doc1_chunks.index]

        for j, doc2_id in enumerate(doc_ids):
            if i == j:
                # Same document - similarity is 1
                doc_similarity_matrix[i, j] = 1.0
                continue

            # Get chunks and embeddings for second document
            doc2_chunks = hybrid_chunks_df[hybrid_chunks_df['doc_id'] == doc2_id]
            doc2_embeddings = full_embeddings_matrix[doc2_chunks.index]

            # Calculate cosine similarity between all chunks
            similarity_matrix = cosine_similarity(
                doc1_embeddings, doc2_embeddings)

            # Take max of all chunk similarities
            max_similarity = np.max(similarity_matrix)
            doc_similarity_matrix[i, j] = max_similarity

    # Create DataFrame for better visualization using doc_ids as index/columns
    similarity_df = pd.DataFrame(
        doc_similarity_matrix,
        index=doc_ids,
        columns=doc_ids
    )

    return similarity_df, doc_similarity_matrix


def analyze_similarity_distribution(similarity_df: pd.DataFrame, n_baskets: int = 5) -> None:
    """
    Analyzes the distribution of similarities by putting them into baskets and printing statistics.

    Args:
        similarity_df (pd.DataFrame): DataFrame containing document similarities
        n_baskets (int): Number of baskets to divide the similarities into
    """
    # Get all similarities excluding self-similarities (diagonal)
    similarities = similarity_df.values[~np.eye(
        similarity_df.shape[0], dtype=bool)]

    # Create baskets
    basket_edges = np.linspace(0, 1, n_baskets + 1)
    basket_counts = np.zeros(n_baskets)

    # Count similarities in each basket
    for i in range(n_baskets):
        lower = basket_edges[i]
        upper = basket_edges[i + 1]
        basket_counts[i] = np.sum(
            (similarities >= lower) & (similarities < upper))

    # Print results
    total = len(similarities)
    print("\nSimilarity Distribution Analysis:")
    print("-" * 50)
    for i in range(n_baskets):
        lower = basket_edges[i]
        upper = basket_edges[i + 1]
        count = int(basket_counts[i])
        percentage = (count / total) * 100
        print(
            f"Basket {i+1} [{lower:.2f}-{upper:.2f}]: {count} pairs ({percentage:.1f}%)")
    print("-" * 50)
    print(f"Total document pairs analyzed: {total}")
