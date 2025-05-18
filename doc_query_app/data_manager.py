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

        self.hybrid_chunks_df = None
        self.full_embeddings_matrix = None

        self.file_handlers = {
            '.pdf': lambda f: PyPDF2.PdfReader(f).pages[0].extract_text(),
            '.docx': lambda f: docx.Document(f).paragraphs[0].text if docx.Document(f).paragraphs else "",
            'default': lambda f: f.read()
        }
        self.init_embeddings_and_pilot_model()
        self.graph_visualizer = Graph_visualizer()
        self.build_3D_graph()

    def init_embeddings_and_pilot_model(self) -> bool:
        if USE_CACHED_DATA:
            self.hybrid_chunks_df = pd.read_pickle(self.hybrid_chunks_df_path)
            self.full_embeddings_matrix = np.load(
                self.full_embeddings_matrix_path)
            print("Using cached data. Not generating new embeddings.")
            return True
        print("Generating new embeddings.")
        # docs_df = load_random_documents(self.raw_files_dir, 50)
        docs_df = load_documents_from_text_folder(self.raw_files_dir, 50)

        if docs_df is not None and not docs_df.empty:
            hybrid_chunks_df = optimized_hybrid_chunking(docs_df)
            save_dataframe_with_embeddings(
                hybrid_chunks_df, self.hybrid_chunks_df_path)
            self.hybrid_chunks_df = hybrid_chunks_df
        else:
            print(
                "docs_df is empty. Please check the document loading and preprocessing steps.")

        # try:
        hybrid_chunks_df_with_embeddings, full_embeddings_matrix, final_model_object = generate_optimized_embeddings(
            hybrid_chunks_df, self.embedding_model)
        save_embeddings_matrix(full_embeddings_matrix,
                               self.full_embeddings_matrix_path)
        self.full_embeddings_matrix = full_embeddings_matrix

    def build_3D_graph(self):
        if USE_CACHED_3D_GRAPH:
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

    def process_new_file(self, filename: str) -> Dict:
        return {"error": "Still working on it"}
        # file_path = os.path.join(self.raw_files_dir, 'uploads', filename)
        # description = self._get_file_description(file_path)
        # if "Error reading" in description:
        #     return {"error": description}

        # embedding = self.final_model_object.encode([description])[0]
        # gnn_manager.reload()

        # # Handle first node case or existing graph case
        # if gnn_manager.embeddings is None:
        #     # Create first node
        #     gnn_manager.add_node(embedding.reshape(1, -1), [])
        #     node = self._create_node(
        #         "0", filename, description, self.field_to_group["new"], 0)
        #     graph = self._update_graph_structure([node], [])
        # else:
        #     # Add to existing graph
        #     neighbors, sims = self._find_neighbors(
        #         embedding, gnn_manager.embeddings)
        #     new_node_id = gnn_manager.add_node(
        #         embedding.reshape(1, -1), neighbors)
        #     new_node = self._create_node(str(new_node_id), filename, description,
        #                                  self.field_to_group["new"], len(neighbors))
        #     new_links = [
        #         {"source": str(new_node_id), "target": str(n),
        #          "value": float(sims[n])}
        #         for n in neighbors
        #     ]
        #     graph = self._update_graph_structure([new_node], new_links)
        #     self.refine_embeddings()

        # self._save_json(self.graph_data_path, graph)
        # return graph

    def get_graph_data(self) -> Dict:
        graph = json.load(open(self.graph_data_path))
        if graph:
            return graph
        else:
            return {"error": "No graph data found. Please upload documents to build the graph."}

        # gnn_manager.reload()
        # if gnn_manager.embeddings is None:
        #     return {"error": "No embeddings found. Upload documents to build the graph."}

        # return json.load(open(self.graph_data_path))


# Singleton
data_manager = DataManager()
