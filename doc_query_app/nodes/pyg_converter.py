import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data


def networkx_to_pyg_data(
    G: nx.MultiDiGraph,
    hybrid_chunks_df: pd.DataFrame = None,
    default_embedding_dim: int = 100,
) -> Data:
    """
    Converts a NetworkX MultiDiGraph to a PyTorch Geometric Data object.

    Args:
        G (nx.MultiDiGraph): The input NetworkX graph.
        hybrid_chunks_df (pd.DataFrame, optional): DataFrame containing chunk information,
                                                   used for inferring embedding dimensions
                                                   and adding optional attributes like category labels.
        default_embedding_dim (int): Default dimension to use for embeddings if it cannot be inferred.

    Returns:
        torch_geometric.data.Data: The converted PyG Data object, or None if G is None.
    """
    if G is None:
        print("Error: Input NetworkX graph G is None. Cannot convert.")
        return None

    print(
        f"Starting conversion of NetworkX graph G (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}) to PyG Data object."
    )

    # 1. Define edge type mapping to integers
    edge_type_to_int = {
        "hierarchical_child_to_parent": 0,
        "hierarchical_parent_to_child": 1,
        "same_document_l1": 2,
        "semantic_similarity": 3,
        # Add any other custom edge types if you have them
    }
    num_relations = len(edge_type_to_int)
    print(f"Defined {num_relations} edge types: {edge_type_to_int}")

    # 2. Create node mapping
    if G.number_of_nodes() > 0:
        sorted_nodes = sorted(list(G.nodes()))
        node_to_int_mapping = {node_id: i for i, node_id in enumerate(sorted_nodes)}
        int_to_node_id_mapping = {
            i: node_id for node_id, i in node_to_int_mapping.items()
        }
    else:
        sorted_nodes = []
        node_to_int_mapping = {}
        int_to_node_id_mapping = {}
    print(f"Created mapping for {len(sorted_nodes)} nodes.")

    # 3. Extract node features (x - embeddings)
    embedding_dim = None
    if (
        hybrid_chunks_df is not None
        and not hybrid_chunks_df.empty
        and "embedding" in hybrid_chunks_df.columns
    ):
        for emb_val in hybrid_chunks_df["embedding"]:
            if emb_val is not None:
                if isinstance(emb_val, (np.ndarray, list)):
                    embedding_dim = (
                        len(emb_val) if isinstance(emb_val, list) else emb_val.shape[0]
                    )
                    break

    if embedding_dim is None and G.number_of_nodes() > 0:
        for node_id_example in sorted_nodes:
            emb_attr = G.nodes[node_id_example].get("embedding")
            if emb_attr is not None:
                if isinstance(emb_attr, (np.ndarray, list)):
                    embedding_dim = (
                        len(emb_attr)
                        if isinstance(emb_attr, list)
                        else emb_attr.shape[0]
                    )
                    break

    if embedding_dim is None:
        if G.number_of_nodes() > 0:
            print(
                f"Warning: Could not auto-determine embedding dimension. Will try to infer or use default {default_embedding_dim}."
            )
        else:
            print(
                f"Graph has no nodes, embedding dimension not determined yet. Using default {default_embedding_dim}."
            )
        embedding_dim = default_embedding_dim  # Fallback to default

    node_features_list = []
    if G.number_of_nodes() > 0:
        first_valid_embedding_dim_found = False
        for i in range(len(sorted_nodes)):
            node_id = int_to_node_id_mapping[i]
            embedding = G.nodes[node_id].get("embedding")

            current_embedding_vector = None
            if embedding is not None:
                if isinstance(embedding, np.ndarray):
                    current_embedding_vector = embedding.flatten().tolist()
                elif isinstance(embedding, list):
                    current_embedding_vector = embedding
                else:
                    print(
                        f"Warning: Node {node_id} embedding is of unexpected type {type(embedding)}."
                    )

                if current_embedding_vector is not None:
                    if (
                        not first_valid_embedding_dim_found
                    ):  # Infer dim from first valid if not set
                        if (
                            embedding_dim != len(current_embedding_vector)
                            and embedding_dim == default_embedding_dim
                        ):
                            print(
                                f"Adjusting embedding dimension from default {embedding_dim} to inferred {len(current_embedding_vector)} from node {node_id}."
                            )
                            embedding_dim = len(current_embedding_vector)
                        first_valid_embedding_dim_found = True

                    if len(current_embedding_vector) != embedding_dim:
                        print(
                            f"Warning: Node {node_id} embedding dim {len(current_embedding_vector)} != expected {embedding_dim}. Using zero vector."
                        )
                        current_embedding_vector = [0.0] * embedding_dim

            if current_embedding_vector is None:
                current_embedding_vector = [0.0] * embedding_dim

            node_features_list.append(current_embedding_vector)

    if G.number_of_nodes() == 0:
        node_features_tensor = torch.empty((0, embedding_dim), dtype=torch.float)
    elif not node_features_list:
        print(
            "Error: node_features_list is empty despite graph having nodes. Creating zero tensor."
        )
        node_features_tensor = torch.zeros(
            (G.number_of_nodes(), embedding_dim), dtype=torch.float
        )
    else:
        try:
            node_features_tensor = torch.tensor(node_features_list, dtype=torch.float)
            if node_features_tensor.shape[0] != G.number_of_nodes():
                print(
                    f"Warning: Node features tensor rows ({node_features_tensor.shape[0]}) != num graph nodes ({G.number_of_nodes()})."
                )
            if (
                node_features_tensor.nelement() > 0
                and node_features_tensor.shape[1] != embedding_dim
            ):
                print(
                    f"Warning: Node features tensor dim ({node_features_tensor.shape[1]}) != determined embedding_dim ({embedding_dim}). This might be an issue."
                )
        except Exception as e:
            print(
                f"Error creating node_features_tensor: {e}. Check uniformity of embedding lengths. Fallback to zero tensor."
            )
            node_features_tensor = torch.zeros(
                (G.number_of_nodes(), embedding_dim), dtype=torch.float
            )

    print(f"Node features (x) tensor created with shape: {node_features_tensor.shape}")

    # 4. Extract edge_index, edge_type, and edge_attr
    edge_sources, edge_targets, edge_type_list, edge_attr_list = [], [], [], []

    if G.number_of_edges() > 0:
        for u, v, data in G.edges(data=True):
            if u not in node_to_int_mapping or v not in node_to_int_mapping:
                print(
                    f"Warning: Edge ({u}, {v}) involves node IDs not in mapping. Skipping."
                )
                continue

            edge_sources.append(node_to_int_mapping[u])
            edge_targets.append(node_to_int_mapping[v])

            edge_type_str = data.get("type")
            edge_type_list.append(
                edge_type_to_int.get(edge_type_str, -1)
            )  # Default to -1 if unknown
            if edge_type_str not in edge_type_to_int:
                print(
                    f"Warning: Unknown edge type '{edge_type_str}' for edge ({u}, {v}). Assigned -1."
                )

            edge_attr_list.append(
                [float(data.get("weight", 1.0))]
            )  # Default weight 1.0

    if not edge_sources:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_type_tensor = torch.empty((0,), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index_tensor = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_type_tensor = torch.tensor(edge_type_list, dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attr_list, dtype=torch.float)

    print(f"Edge index tensor created with shape: {edge_index_tensor.shape}")
    print(f"Edge type tensor created with shape: {edge_type_tensor.shape}")
    print(f"Edge attributes tensor created with shape: {edge_attr_tensor.shape}")

    # 5. Create PyG Data object
    pyg_data = Data(
        x=node_features_tensor,
        edge_index=edge_index_tensor,
        edge_type=edge_type_tensor,
        edge_attr=edge_attr_tensor,
    )

    pyg_data.num_relations = num_relations
    pyg_data.edge_type_mapping = edge_type_to_int
    pyg_data.node_id_mapping = node_to_int_mapping
    pyg_data.int_to_node_id_mapping = int_to_node_id_mapping
    if G.number_of_nodes() > 0:
        pyg_data.chunk_ids_ordered = sorted_nodes

    # Optional: Add other node attributes
    if (
        hybrid_chunks_df is not None
        and not hybrid_chunks_df.empty
        and G.number_of_nodes() > 0
    ):
        try:
            if "chunk_id" not in hybrid_chunks_df.columns:
                print(
                    "Warning: 'chunk_id' not in hybrid_chunks_df columns, cannot add category labels."
                )
            else:
                df_for_labels = hybrid_chunks_df.set_index("chunk_id")
                # Filter for nodes present in the graph and maintain their order
                # Use reindex to handle missing chunk_ids gracefully, then dropna if necessary
                aligned_labels_df = df_for_labels.reindex(sorted_nodes)

                if "category" in aligned_labels_df.columns:
                    # Handle potential NaN values in category before converting
                    valid_categories = aligned_labels_df["category"].dropna()
                    if not valid_categories.empty:
                        unique_categories = pd.Categorical(valid_categories)
                        # Create a full tensor of codes, using a placeholder for NaNs if needed
                        # Or, ensure your GNN can handle missing labels, or filter nodes without labels
                        labels = (
                            aligned_labels_df["category"]
                            .astype("category")
                            .cat.codes.values
                        )
                        pyg_data.y = torch.tensor(labels, dtype=torch.long)
                        pyg_data.category_mapping = dict(
                            enumerate(unique_categories.categories)
                        )
                        print(
                            f"Added 'y' (category labels) and 'category_mapping'. Shape: {pyg_data.y.shape}"
                        )
                    else:
                        print("No valid categories found to add as 'y'.")
        except KeyError as e:
            print(
                f"KeyError accessing hybrid_chunks_df for labels (some chunk_ids in graph might be missing in df): {e}"
            )
        except Exception as e:
            print(
                f"An error occurred while adding attributes from hybrid_chunks_df: {e}"
            )

    print("\n--- PyG Data object created ---")
    print(pyg_data)
    if pyg_data is not None:
        print(f"  Number of nodes: {pyg_data.num_nodes}")
        print(f"  Number of node features: {pyg_data.num_node_features}")
        print(f"  Number of edges: {pyg_data.num_edges}")
        if pyg_data.num_edges > 0:
            print(f"  Edge index shape: {pyg_data.edge_index.shape}")
            print(f"  Edge type shape: {pyg_data.edge_type.shape}")
            print(f"  Edge attr shape: {pyg_data.edge_attr.shape}")
        print(f"  Number of relations: {pyg_data.num_relations}")
        if hasattr(pyg_data, "y"):
            print(f"  Target 'y' shape: {pyg_data.y.shape}")
        print(
            f"  Contains isolated nodes: {pyg_data.has_isolated_nodes() if pyg_data.num_nodes > 0 else 'N/A'}"
        )
        print(
            f"  Contains self-loops: {pyg_data.has_self_loops() if pyg_data.num_nodes > 0 else 'N/A'}"
        )
        print(
            f"  Is directed: {pyg_data.is_directed() if pyg_data.num_nodes > 0 else 'N/A'}"
        )

    return pyg_data
