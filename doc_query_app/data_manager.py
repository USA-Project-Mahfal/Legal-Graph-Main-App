import os
import pandas as pd
import numpy as np
import docx
import PyPDF2
import json
from typing import Dict, Tuple, Optional
import networkx as nx
from torch_geometric.data import Data

from nodes.chunking import optimized_hybrid_chunking
from nodes.generate_embeddings import generate_optimized_embeddings
from nodes.load_n_preprocess import (
    load_documents_from_text_folder,
    load_random_documents,
)
from nodes.saving_utils import save_dataframe_with_embeddings, save_embeddings_matrix
from nodes.similarity_check import (
    calculate_document_similarity_by_max,
    analyze_similarity_distribution,
    calculate_document_similarity_by_mean,
)
from displayed_graph_utils.displayed_graph_manager import (
    Graph_visualizer,
)  # Assuming this class is defined
from gnn.gnn_manager import GNNManager  # Corrected import

from nodes.graph_builder import build_document_graph
from nodes.pyg_converter import networkx_to_pyg_data

from config import (
    DATA_DIR,
    USE_CACHED_DATA,
    USE_CACHED_3D_GRAPH,
    GNN_MODEL_PATH,
    GNN_EMBEDDINGS_PATH,
)


class DataManager:
    def __init__(self):
        self.data_dir = DATA_DIR
        # Consider making this base_dir relative or from config for better portability
        self.base_dir = (
            "d:/PROJECTS/CLIENT/USA-Graph-ML/REPOS/Document_Fetch/doc_query_app"
        )
        self.raw_files_dir = os.path.join(
            self.base_dir, "raw_files/chunk_input")
        self.hybrid_chunks_df_path = os.path.join(
            self.base_dir, "data/hybrid_chunks_df.pkl"
        )
        self.full_embeddings_matrix_path = os.path.join(
            self.base_dir, "data/full_embeddings_matrix.npy"
        )

        self.embedding_model_name = "all-MiniLM-L6-v2"

        self.graph_data_path = os.path.join(
            self.data_dir, "graph_data.json"
        )  # For 3D graph visualization
        self.file_data_path = os.path.join(
            self.data_dir, "file_data.json"
        )  # Unused in current context, but kept

        self.field_to_group = {
            "file": 5,
            "new": 6,
        }  # Unused in current context, but kept
        self.last_doc_id: Optional[str] = "0"

        self.hybrid_chunks_df: Optional[pd.DataFrame] = None
        self.full_embeddings_matrix: Optional[np.ndarray] = None
        self.embedding_model_object = (
            None  # Stores the loaded sentence transformer model
        )

        self.graph_visualize_data: Optional[Dict] = None  # For 3D graph
        self.document_nx_graph: Optional[nx.MultiDiGraph] = None
        self.pyg_graph_data: Optional[Data] = None

        # GNN related attributes
        self.gnn_manager_instance: Optional[GNNManager] = None
        self.final_gnn_embeddings: Optional[np.ndarray] = None
        self.gnn_model_path = GNN_MODEL_PATH
        self.gnn_embeddings_path = GNN_EMBEDDINGS_PATH

        self.file_handlers = {
            ".pdf": lambda f: PyPDF2.PdfReader(f).pages[0].extract_text(),
            ".docx": lambda f: " ".join([p.text for p in docx.Document(f).paragraphs]),
            "default": lambda f: f.read(),  # Fallback for .txt or other plain text
        }

        # --- Initialization Sequence ---
        self.init_embeddings_and_pilot_model()

        self.graph_visualizer = Graph_visualizer()  # For 3D graph
        self.build_3D_graph()

        self.build_structural_document_graph()
        self.convert_nx_to_pyg()

        self.initialize_gnn_manager()
        self.train_or_load_gnn()

    def init_embeddings_and_pilot_model(self, force: bool = False) -> bool:
        if USE_CACHED_DATA and not force:
            try:
                self.hybrid_chunks_df = pd.read_pickle(
                    self.hybrid_chunks_df_path)
                if os.path.exists(self.full_embeddings_matrix_path):
                    self.full_embeddings_matrix = np.load(
                        self.full_embeddings_matrix_path
                    )
                else:  # If matrix file is missing but df is present, embeddings might be in df
                    if "embedding" in self.hybrid_chunks_df.columns:
                        print(
                            "Full embeddings matrix file not found, but 'embedding' column exists in DataFrame."
                        )
                        # Potentially reconstruct matrix from df if needed, or ensure generate_optimized_embeddings handles this
                    else:
                        print(
                            "Warning: Full embeddings matrix file not found and 'embedding' column missing in DataFrame."
                        )
                        # This might lead to issues if full_embeddings_matrix is strictly required later

                if (
                    self.hybrid_chunks_df is not None
                    and not self.hybrid_chunks_df.empty
                ):
                    self.last_doc_id = (
                        str(self.hybrid_chunks_df["doc_id"].iloc[-1])
                        if "doc_id" in self.hybrid_chunks_df.columns
                        else "0"
                    )
                    print(
                        "Using cached data for hybrid_chunks_df and/or full_embeddings_matrix."
                    )
                    # Assuming generate_optimized_embeddings also loads the model object if needed
                    # or that the embeddings themselves are the primary cache target.
                    # If the model object itself needs caching, that logic would be added here or in generate_optimized_embeddings
                    return True
            except (FileNotFoundError, EOFError, AttributeError, KeyError) as e:
                print(
                    f"Cache not found or error loading cache ({e}). Will generate new data."
                )

        print("Generating new document chunks and/or embeddings.")
        docs_df = load_documents_from_text_folder(self.raw_files_dir)

        if docs_df is None or docs_df.empty:
            print("No documents loaded. Aborting further processing.")
            self.hybrid_chunks_df = pd.DataFrame()
            self.full_embeddings_matrix = np.array([])
            return False

        initial_chunks_df = optimized_hybrid_chunking(docs_df)
        if initial_chunks_df is None or initial_chunks_df.empty:
            print("Chunking resulted in empty DataFrame. Aborting.")
            self.hybrid_chunks_df = pd.DataFrame()
            self.full_embeddings_matrix = np.array([])
            return False

        df_with_embeddings, matrix, model_obj = generate_optimized_embeddings(
            initial_chunks_df.copy(), self.embedding_model_name
        )

        self.hybrid_chunks_df = df_with_embeddings
        self.full_embeddings_matrix = matrix
        self.embedding_model_object = model_obj

        if self.hybrid_chunks_df is not None and not self.hybrid_chunks_df.empty:
            save_dataframe_with_embeddings(
                self.hybrid_chunks_df, self.hybrid_chunks_df_path
            )  # Saves df possibly with 'embedding' column
            self.last_doc_id = (
                str(self.hybrid_chunks_df["doc_id"].iloc[-1])
                if "doc_id" in self.hybrid_chunks_df.columns
                else "0"
            )

        if (
            self.full_embeddings_matrix is not None
            and self.full_embeddings_matrix.size > 0
        ):
            save_embeddings_matrix(
                self.full_embeddings_matrix, self.full_embeddings_matrix_path
            )

        print("Successfully generated and saved new chunks and embeddings.")
        return True

    def build_3D_graph(self, force: bool = False):
        if (
            USE_CACHED_3D_GRAPH
            and not force
            and os.path.exists(self.graph_data_path)
        ):  # Check if file exists
            print("Using cached 3D graph data.")
            try:
                with open(self.graph_data_path, "r") as f:
                    self.graph_visualize_data = json.load(f)
                print("Loaded 3D graph data from file.")
                return
            except Exception as e:
                print(
                    f"Could not load cached 3D graph data from file: {e}. Will rebuild."
                )

        if (
            self.hybrid_chunks_df is None
            or self.hybrid_chunks_df.empty
            or self.full_embeddings_matrix is None
            or self.full_embeddings_matrix.size == 0
        ):
            print(
                "Cannot build 3D graph: hybrid_chunks_df or full_embeddings_matrix is not available."
            )
            return

        print("Building 3D graph.")
        try:
            similarity_df_max, doc_similarity_matrix_max = (
                calculate_document_similarity_by_max(
                    self.hybrid_chunks_df, self.full_embeddings_matrix
                )
            )
            # analyze_similarity_distribution(similarity_df_max) # Optional
            similarity_df_mean, doc_similarity_matrix_mean = (
                calculate_document_similarity_by_mean(
                    self.hybrid_chunks_df, self.full_embeddings_matrix
                )
            )
            # analyze_similarity_distribution(similarity_df_mean) # Optional

            combined_similarity_matrix = np.maximum(
                doc_similarity_matrix_max, doc_similarity_matrix_mean
            )
            combined_similarity_df = pd.DataFrame(
                combined_similarity_matrix,
                index=similarity_df_max.index,
                columns=similarity_df_max.columns,
            )

            self.graph_visualize_data = self.graph_visualizer.build_3d_force_graph(
                combined_similarity_df,
                self.hybrid_chunks_df,  # Pass the df for node metadata
            )
            if self.graph_visualize_data:
                print(
                    f"Built 3D graph with {self.graph_visualize_data['metadata']['total_nodes']} nodes and {self.graph_visualize_data['metadata']['total_links']} links"
                )
                os.makedirs(os.path.dirname(
                    self.graph_data_path), exist_ok=True)
                with open(self.graph_data_path, "w") as f:
                    json.dump(self.graph_visualize_data, f)
        except Exception as e:
            print(f"Error building 3D graph: {e}")
            self.graph_visualize_data = None

    def build_structural_document_graph(self, force: bool = False):
        if self.document_nx_graph is not None and not force:
            print("Using cached structural document graph (NetworkX object).")
            return

        if self.hybrid_chunks_df is None or self.hybrid_chunks_df.empty:
            print(
                "Cannot build structural document graph: hybrid_chunks_df is not loaded."
            )
            return

        # Ensure embeddings are present if build_document_graph needs them directly from df
        # or if it uses self.full_embeddings_matrix
        if "embedding" not in self.hybrid_chunks_df.columns and (
            self.full_embeddings_matrix is None or self.full_embeddings_matrix.size == 0
        ):
            print(
                "Warning: Embeddings missing in hybrid_chunks_df and no full_embeddings_matrix. Graph construction might be limited if similarity is based on these."
            )
            # Proceeding, but build_document_graph should handle this gracefully

        print("Building structural and semantic NetworkX document graph...")
        self.document_nx_graph = build_document_graph(
            # This df must have features/embeddings
            hybrid_chunks_df=self.hybrid_chunks_df,
            # Pass full_embeddings_matrix if build_document_graph expects it separately
            # full_embeddings_matrix=self.full_embeddings_matrix,
            similarity_threshold=0.6,  # Example threshold
            num_l1_to_sample_for_similarity=100,  # Example
            random_state_for_sampling=42,  # Example
        )

        if self.document_nx_graph and self.document_nx_graph.number_of_nodes() > 0:
            print(
                f"Successfully built structural document graph with {self.document_nx_graph.number_of_nodes()} nodes and {self.document_nx_graph.number_of_edges()} edges."
            )
        elif self.document_nx_graph is not None:  # Graph object exists but is empty
            print("Structural document graph was built but is empty (0 nodes).")
        else:
            print("Failed to build structural document graph.")

    def convert_nx_to_pyg(self, force: bool = False):
        if self.pyg_graph_data is not None and not force:
            print("Using cached PyG graph data.")
            return

        if self.document_nx_graph is None:
            print(
                "NetworkX graph (self.document_nx_graph) is not built. Cannot convert to PyG."
            )
            return
        if self.hybrid_chunks_df is None or self.hybrid_chunks_df.empty:
            print(
                "hybrid_chunks_df is not available. Cannot get node features for PyG conversion."
            )
            return
        # Ensure that hybrid_chunks_df has the 'embedding' column if networkx_to_pyg_data expects it
        # or that self.full_embeddings_matrix is correctly aligned and passed if needed.

        print("Converting NetworkX graph to PyG Data object...")
        self.pyg_graph_data = networkx_to_pyg_data(
            G=self.document_nx_graph,
            # Pass df for node features and other attributes
            hybrid_chunks_df=self.hybrid_chunks_df,
            # Pass full_embeddings_matrix if pyg_converter needs it:
            # full_embeddings_matrix=self.full_embeddings_matrix
        )

        if self.pyg_graph_data:
            print("Successfully converted NetworkX graph to PyG Data object.")
            if (
                not hasattr(self.pyg_graph_data, "num_relations")
                or self.pyg_graph_data.num_relations is None
            ):
                print(
                    "Warning: num_relations not set on pyg_graph_data by converter. GNNManager might need this."
                )
            if (
                not hasattr(self.pyg_graph_data, "num_node_features")
                or self.pyg_graph_data.num_node_features == 0
            ):
                print(
                    "Warning: num_node_features not set or is 0 on pyg_graph_data. GNNManager will need this."
                )
        else:
            print("Failed to convert NetworkX graph to PyG Data object.")

    def initialize_gnn_manager(self):
        if (
            self.pyg_graph_data
            and hasattr(self.pyg_graph_data, "num_node_features")
            and self.pyg_graph_data.num_node_features > 0
            and hasattr(self.pyg_graph_data, "num_relations")
            and self.pyg_graph_data.num_relations is not None
        ):
            if self.pyg_graph_data.num_relations == 0:
                print(
                    "Warning: num_relations in pyg_graph_data is 0. RGCNConv might require at least 1 relation type."
                )
                # Consider how GNNManager handles this. For now, proceeding.

            self.gnn_manager_instance = GNNManager(
                num_node_features=self.pyg_graph_data.num_node_features,
                num_relations=self.pyg_graph_data.num_relations,
                model_path=self.gnn_model_path,
                embeddings_path=self.gnn_embeddings_path,
            )
            print("GNNManager initialized successfully.")
        else:
            self.gnn_manager_instance = None
            missing_attrs = []
            if not self.pyg_graph_data:
                missing_attrs.append("PyG data object (self.pyg_graph_data)")
            if not (
                self.pyg_graph_data
                and hasattr(self.pyg_graph_data, "num_node_features")
                and self.pyg_graph_data.num_node_features > 0
            ):
                missing_attrs.append("valid 'num_node_features' in PyG data")
            if not (
                self.pyg_graph_data
                and hasattr(self.pyg_graph_data, "num_relations")
                and self.pyg_graph_data.num_relations is not None
            ):
                missing_attrs.append("'num_relations' attribute in PyG data")

            print(
                f"GNNManager not initialized. Missing or invalid: {', '.join(missing_attrs)}."
            )

    def train_or_load_gnn(self, train_epochs: int = 50, force_train: bool = False):
        return False
        if not self.gnn_manager_instance:
            print("GNNManager not initialized. Cannot train or load GNN model.")
            return False

        loaded_successfully = False
        if not force_train:
            print(
                f"Attempting to load pre-trained GNN model from: {self.gnn_manager_instance.trained_model_path}"
            )
            if self.gnn_manager_instance.load_model():
                print("Successfully loaded pre-trained GNN model.")
                loaded_successfully = True
            else:
                print(
                    "Pre-trained GNN model not found or failed to load. Will attempt to train a new one."
                )

        if force_train or not loaded_successfully:
            if force_train:
                print("Forcing GNN model training.")
            else:
                print("Proceeding to train a new GNN model.")

            if self.pyg_graph_data:
                if self.pyg_graph_data.num_nodes == 0:
                    print("PyG data has 0 nodes. Skipping GNN training.")
                    return False
                if (
                    self.pyg_graph_data.num_edges == 0
                    and self.pyg_graph_data.num_nodes > 0
                ):
                    print(
                        "PyG data has 0 edges. GAE training for link prediction will be skipped or ineffective."
                    )

                print(
                    f"Starting GNN training pipeline for {train_epochs} epochs...")
                # Ensure train_pipeline in GNNManager can handle data with 0 edges if that's a valid scenario for your model type
                history, test_scores, embeddings = (
                    self.gnn_manager_instance.train_pipeline(
                        pyg_data=self.pyg_graph_data,
                        epochs=train_epochs,
                        plot_loss=True,
                    )
                )

                if embeddings is not None:
                    self.final_gnn_embeddings = embeddings
                    print(
                        f"GNN training completed. Final embeddings shape: {self.final_gnn_embeddings.shape}"
                    )
                    if history:
                        print(f"Test Scores (AUC, AP): {test_scores}")
                    # GNNManager's train_pipeline should save the model and embeddings
                    return True
                else:
                    print("GNN training pipeline did not produce embeddings.")
                    return False
            else:
                print(
                    "PyG data (self.pyg_graph_data) not available. Cannot train GNN model."
                )
                return False

        if loaded_successfully:  # Model was loaded, generate embeddings
            print("Generating embeddings using the loaded GNN model.")
            if self.pyg_graph_data:
                if self.pyg_graph_data.num_nodes == 0:
                    print("PyG data has 0 nodes. Cannot generate GNN embeddings.")
                    self.final_gnn_embeddings = None
                    return False

                self.final_gnn_embeddings = (
                    self.gnn_manager_instance.generate_and_save_embeddings(
                        pyg_data_to_encode=self.pyg_graph_data
                    )
                )
                if self.final_gnn_embeddings is not None:
                    print(
                        f"Generated GNN embeddings using loaded model. Shape: {self.final_gnn_embeddings.shape}"
                    )
                    return True
                else:
                    print("Failed to generate GNN embeddings even with a loaded model.")
                    return False
            else:
                print(
                    "PyG data not available. Cannot generate GNN embeddings with the loaded model."
                )
                return False
        return False

    # =====================
    # === Public Methods ==
    # =====================
    def get_graph_data(self) -> Optional[Dict]:  # For 3D graph visualization
        if self.graph_visualize_data:
            return self.graph_visualize_data
        elif os.path.exists(self.graph_data_path):
            try:
                with open(self.graph_data_path, "r") as f:
                    self.graph_visualize_data = json.load(f)
                print("Loaded 3D graph data from file for get_graph_data().")
                return self.graph_visualize_data
            except Exception as e:
                print(f"Error loading 3D graph data from file: {e}")
                # Return error dict
                return {"error": f"Error loading graph data: {e}"}
        print("3D graph visualization data not available.")
        return None

    def get_last_doc_id(self) -> str:
        return self.last_doc_id if self.last_doc_id is not None else "0"

    def update_chunks_df(self, new_chunks_df: pd.DataFrame):  # For dynamic updates
        if self.hybrid_chunks_df is None:
            self.hybrid_chunks_df = new_chunks_df
        else:
            self.hybrid_chunks_df = pd.concat(
                [self.hybrid_chunks_df, new_chunks_df]
            ).reset_index(drop=True)
        # Potentially re-run parts of the pipeline if df changes significantly

    def update_embeddings_matrix(
        self, new_embeddings: np.ndarray
    ):  # For dynamic updates
        if self.full_embeddings_matrix is None or self.full_embeddings_matrix.size == 0:
            self.full_embeddings_matrix = new_embeddings
        else:
            self.full_embeddings_matrix = np.vstack(
                [self.full_embeddings_matrix, new_embeddings]
            )
        # Potentially re-run parts of the pipeline

    def get_document_networkx_graph(self) -> Optional[nx.MultiDiGraph]:
        return self.document_nx_graph

    def get_pyg_data(self) -> Optional[Data]:
        return self.pyg_graph_data

    def get_gnn_embeddings(self) -> Optional[np.ndarray]:
        if self.final_gnn_embeddings is not None:
            return self.final_gnn_embeddings

        if self.gnn_manager_instance and os.path.exists(self.gnn_embeddings_path):
            print(
                f"Attempting to load GNN embeddings from file: {self.gnn_embeddings_path}"
            )
            try:
                # GNNManager might have a method to load embeddings, or load directly
                self.final_gnn_embeddings = (
                    self.gnn_manager_instance.load_embeddings()
                )  # Assumes GNNManager has this
                if self.final_gnn_embeddings is not None:
                    print("Loaded GNN embeddings successfully via GNNManager.")
                    return self.final_gnn_embeddings
                else:  # Fallback to direct load if GNNManager method fails or doesn't exist
                    self.final_gnn_embeddings = np.load(
                        self.gnn_embeddings_path)
                    print("Loaded GNN embeddings directly from file.")
                    return self.final_gnn_embeddings

            except AttributeError:  # If GNNManager has no load_embeddings()
                try:
                    self.final_gnn_embeddings = np.load(
                        self.gnn_embeddings_path)
                    print(
                        "Loaded GNN embeddings directly from file (GNNManager.load_embeddings not found)."
                    )
                    return self.final_gnn_embeddings
                except Exception as e:
                    print(
                        f"Error loading GNN embeddings directly from file: {e}")
            except Exception as e:
                print(f"Error loading GNN embeddings: {e}")

        print("Final GNN embeddings are not available.")
        return None


# Singleton instance
# Ensure all dependencies (like config.py, nodes/*, GNN/*) are correctly set up
# before instantiating.
# Comment out if you instantiate it elsewhere (e.g., in app.py)
data_manager = DataManager()

# # Example of how to run if this script is executed directly (for testing)
# if __name__ == "__main__":
#     print("Initializing DataManager...")

#     data_manager_instance = DataManager()
#     print("\nDataManager Initialization Complete.")
#     print(f"Last Doc ID: {data_manager_instance.get_last_doc_id()}")

#     if data_manager_instance.hybrid_chunks_df is not None:
#         print(
#             f"Hybrid Chunks DF shape: {data_manager_instance.hybrid_chunks_df.shape}")
#     if data_manager_instance.full_embeddings_matrix is not None:
#         print(
#             f"Full Embeddings Matrix shape: {data_manager_instance.full_embeddings_matrix.shape}"
#         )

#     nx_graph = data_manager_instance.get_document_networkx_graph()
#     if nx_graph:
#         print(
#             f"NetworkX Graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges"
#         )

#     pyg_data = data_manager_instance.get_pyg_data()
#     if pyg_data:
#         print(
#             f"PyG Data: Nodes={pyg_data.num_nodes}, Edges={pyg_data.num_edges}, Features={pyg_data.num_node_features}, Relations={pyg_data.num_relations if hasattr(pyg_data, 'num_relations') else 'N/A'}"
#         )

#     gnn_embeddings = data_manager_instance.get_gnn_embeddings()
#     if gnn_embeddings is not None:
#         print(f"GNN Embeddings shape: {gnn_embeddings.shape}")
#     else:
#         print("GNN Embeddings not available.")

#     graph_3d_data = data_manager_instance.get_graph_data()
#     if graph_3d_data and "error" not in graph_3d_data:
#         print(
#             f"3D Graph Visualization Data: {graph_3d_data['metadata']['total_nodes']} nodes, {graph_3d_data['metadata']['total_links']} links"
#         )
#     elif graph_3d_data:
#         print(f"3D Graph Data Error: {graph_3d_data.get('error')}")
