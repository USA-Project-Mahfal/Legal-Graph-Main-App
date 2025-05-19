# data_manager.py

import os
import pandas as pd
import numpy as np
import docx
import PyPDF2
import json
from typing import Dict
from nodes.chunking import optimized_hybrid_chunking
from nodes.generate_embeddings import generate_optimized_embeddings
from nodes.load_n_preprocess import load_documents_from_text_folder, load_random_documents
from nodes.saving_utils import save_dataframe_with_embeddings, save_embeddings_matrix
from nodes.similarity_check import calculate_document_similarity_by_max, analyze_similarity_distribution, calculate_document_similarity_by_mean
from displayed_graph_utils.displayed_graph_manager import Graph_visualizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DATA_DIR, USE_CACHED_DATA, USE_CACHED_3D_GRAPH
)
from GNN import gnn_manager


class DataManager:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.base_dir = "d:/PROJECTS/CLIENT/USA-Graph-ML/REPOS/Document_Fetch/doc_query_app"
        self.raw_files_dir = os.path.join(
            self.base_dir, "raw_files/chunk_input")
        self.hybrid_chunks_df_path = os.path.join(
            self.base_dir, "data/hybrid_chunks_df.pkl")
        self.full_embeddings_matrix_path = os.path.join(
            self.base_dir, "data/full_embeddings_matrix.npy")

        self.embedding_model = "all-MiniLM-L6-v2"

        self.graph_data_path = os.path.join(self.data_dir, "graph_data.json")
        self.file_data_path = os.path.join(self.data_dir, "file_data.json")

        self.field_to_group = {"file": 5, "new": 6}
        self.last_doc_id = None

        self.hybrid_chunks_df = None
        self.full_embeddings_matrix = None
        self.graph_visualize_data = None

        self.file_handlers = {
            '.pdf': lambda f: PyPDF2.PdfReader(f).pages[0].extract_text(),
            '.docx': lambda f: docx.Document(f).paragraphs[0].text if docx.Document(f).paragraphs else "",
            'default': lambda f: f.read()
        }
        self.init_embeddings_and_pilot_model()
        self.graph_visualizer = Graph_visualizer()
        self.build_3D_graph()

    def init_embeddings_and_pilot_model(self, force: bool = False) -> bool:
        if USE_CACHED_DATA and not force:  # Only check cached data if not forced
            self.hybrid_chunks_df = pd.read_pickle(self.hybrid_chunks_df_path)
            self.full_embeddings_matrix = np.load(
                self.full_embeddings_matrix_path)
            print("Using cached data. Not generating new embeddings.")
            return True

        print("Generating new embeddings.")
        docs_df = load_random_documents(self.raw_files_dir, 100)

        # docs_df = load_documents_from_text_folder(self.raw_files_dir, 50)

        if docs_df is not None and not docs_df.empty:
            self.last_doc_id = docs_df.iloc[-1]['doc_id']
            self.hybrid_chunks_df = optimized_hybrid_chunking(docs_df)
            save_dataframe_with_embeddings(
                self.hybrid_chunks_df, self.hybrid_chunks_df_path)

        else:
            print(
                "docs_df is empty. Please check the document loading and preprocessing steps.")

        # try:
        self.hybrid_chunks_df_with_embeddings, self.full_embeddings_matrix, self.final_model_object = generate_optimized_embeddings(
            self.hybrid_chunks_df, self.embedding_model)
        save_embeddings_matrix(self.full_embeddings_matrix,
                               self.full_embeddings_matrix_path)

    def build_3D_graph(self, force: bool = False):
        if USE_CACHED_3D_GRAPH and not force:  # Only check cached graph if not forced
            print("Using cached 3D graph. Not building new one.")
            return

        print("Building 3D graph.")

        # Calculate similarity matrices
        similarity_df_max, doc_similarity_matrix_max = calculate_document_similarity_by_max(
            self.hybrid_chunks_df, self.full_embeddings_matrix)
        analyze_similarity_distribution(similarity_df_max)
        similarity_df_mean, doc_similarity_matrix_mean = calculate_document_similarity_by_mean(
            self.hybrid_chunks_df, self.full_embeddings_matrix)
        analyze_similarity_distribution(similarity_df_mean)

        # Combine similarity matrices by taking element-wise maximum
        combined_similarity_matrix = np.maximum(
            doc_similarity_matrix_max,
            doc_similarity_matrix_mean
        )

        # Create DataFrame for the combined similarities
        combined_similarity_df = pd.DataFrame(
            combined_similarity_matrix,
            index=similarity_df_max.index,
            columns=similarity_df_max.columns
        )

        # Build and save the 3D force graph
        graph = self.graph_visualizer.build_3d_force_graph(
            combined_similarity_df,
            self.hybrid_chunks_df
        )
        print(
            f"Built 3D graph with {graph['metadata']['total_nodes']} nodes and {graph['metadata']['total_links']} links")

    # =====================
    # === Public Methods ==
    # =====================

    def add_document_to_graph(self, doc_id: str, text: str, category: str = "new") -> Dict:
        """
        Add a single document to the graph with minimal computation.
        Only calculates similarities for the new document.
        """
        print(f"Adding document {doc_id} to graph...")

        # Create a temporary DataFrame for the new document
        new_doc_df = pd.DataFrame({
            'doc_id': [doc_id],
            'text': [text],
            'category': [category]
        })

        # Generate chunks for the new document
        new_chunks_df = optimized_hybrid_chunking(new_doc_df)

        # Generate embeddings for the new chunks
        _, new_embeddings, _ = generate_optimized_embeddings(
            new_chunks_df, self.embedding_model
        )

        # Calculate similarities only between new document and existing documents
        similarities = {}
        if self.hybrid_chunks_df is not None and self.full_embeddings_matrix is not None:
            # Calculate max similarity
            sim_max = cosine_similarity(
                new_embeddings,
                self.full_embeddings_matrix
            ).max(axis=0)  # Take max similarity across chunks

            # Calculate mean similarity
            sim_mean = cosine_similarity(
                new_embeddings,
                self.full_embeddings_matrix
            ).mean(axis=0)  # Take mean similarity across chunks

            # Combine similarities
            combined_similarities = np.maximum(sim_max, sim_mean)

            # Create similarity dictionary
            for idx, sim_score in enumerate(combined_similarities):
                target_doc_id = self.hybrid_chunks_df.iloc[idx]['doc_id']
                similarities[str(target_doc_id)] = float(sim_score)

        # Add node to graph
        graph = self.graph_visualizer.add_node_to_graph(
            doc_id=doc_id,
            description=text,
            category=category,
            similarities=similarities
        )

        # Update our dataframes with the new document
        self.hybrid_chunks_df = pd.concat(
            [self.hybrid_chunks_df, new_chunks_df])
        self.full_embeddings_matrix = np.vstack(
            [self.full_embeddings_matrix, new_embeddings])

        return graph

    def get_graph_data(self) -> Dict:
        if self.graph_visualize_data:
            return self.graph_visualize_data
        else:
            self.graph_visualize_data = json.load(open(self.graph_data_path))
            if self.graph_visualize_data:
                return self.graph_visualize_data
            else:
                return {"error": "No graph data found. Please upload documents to build the graph."}

    def get_last_doc_id(self) -> str:
        return self.last_doc_id

    def update_chunks_df(self, new_chunks_df: pd.DataFrame):
        self.hybrid_chunks_df = pd.concat(
            [self.hybrid_chunks_df, new_chunks_df])

    def update_embeddings_matrix(self, new_embeddings: np.ndarray):
        self.full_embeddings_matrix = np.vstack(
            [self.full_embeddings_matrix, new_embeddings])


# Singleton
data_manager = DataManager()
