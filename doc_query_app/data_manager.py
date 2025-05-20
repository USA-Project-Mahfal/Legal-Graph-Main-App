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
            try:
                self.hybrid_chunks_df = pd.read_pickle(
                    self.hybrid_chunks_df_path)
                self.full_embeddings_matrix = np.load(
                    self.full_embeddings_matrix_path)
                self.last_doc_id = self.hybrid_chunks_df[
                    'doc_id'].iloc[-1] if 'doc_id' in self.hybrid_chunks_df.columns else 0
                print("Using cached data. Not generating new embeddings.")
                return True
            except (FileNotFoundError, EOFError):
                print("Cached data not found or corrupted. Generating new embeddings.")
                # Continue to generate new embeddings

        print("Generating new embeddings.")
        # docs_df = load_random_documents(self.raw_files_dir, 15)
        docs_df = load_documents_from_text_folder(self.raw_files_dir)

        if docs_df is not None and not docs_df.empty:
            self.hybrid_chunks_df = optimized_hybrid_chunking(docs_df)
            save_dataframe_with_embeddings(
                self.hybrid_chunks_df, self.hybrid_chunks_df_path)
            self.last_doc_id = self.hybrid_chunks_df[
                'doc_id'].iloc[-1] if 'doc_id' in self.hybrid_chunks_df.columns else 0

        else:
            print(
                "docs_df is empty. Please check the document loading and preprocessing steps.")

        # try:
        self.hybrid_chunks_df_with_embeddings, self.full_embeddings_matrix, self.final_model_object = generate_optimized_embeddings(
            self.hybrid_chunks_df, self.embedding_model)
        save_embeddings_matrix(self.full_embeddings_matrix,
                               self.full_embeddings_matrix_path)

    def build_3D_graph(self, force: bool = False):
        # Only check cached graph if not forced
        if USE_CACHED_3D_GRAPH and not force and self.graph_visualize_data is not None:
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
        self.graph_visualize_data = self.graph_visualizer.build_3d_force_graph(
            combined_similarity_df,
            self.hybrid_chunks_df
        )
        print(
            f"Built 3D graph with {self.graph_visualize_data['metadata']['total_nodes']} nodes and {self.graph_visualize_data['metadata']['total_links']} links")

    # =====================
    # === Public Methods ==
    # =====================
    def get_graph_data(self) -> Dict:
        if self.graph_visualize_data:
            return self.graph_visualize_data
        else:
            try:
                self.graph_visualize_data = json.load(
                    open(self.graph_data_path))
                if self.graph_visualize_data:
                    return self.graph_visualize_data
            except Exception as e:
                return {"error": f"Error loading graph data: {e}"}

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
