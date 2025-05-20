import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def build_document_graph(
    hybrid_chunks_df: pd.DataFrame,
    similarity_threshold: float = 0.6,
    num_l1_to_sample_for_similarity: int = 100,
    random_state_for_sampling: int = 42,
) -> nx.MultiDiGraph:
    """
    Builds a NetworkX MultiDiGraph from document chunks.

    The graph includes:
    - Nodes for each chunk with attributes.
    - Hierarchical edges between parent and child chunks.
    - Same-document edges between L1 chunks of the same document.
    - Semantic similarity edges between a sample of L1 chunks.

    Args:
        hybrid_chunks_df (pd.DataFrame): DataFrame containing chunk information and embeddings.
        similarity_threshold (float): Threshold for adding semantic similarity edges.
        num_l1_to_sample_for_similarity (int): Number of L1 chunks to sample for semantic similarity calculation.
        random_state_for_sampling (int): Random state for sampling L1 chunks.

    Returns:
        nx.MultiDiGraph: The constructed graph.
    """
    if hybrid_chunks_df is None or hybrid_chunks_df.empty:
        print("Error: hybrid_chunks_df is None or empty. Please load it first.")
        return nx.MultiDiGraph()  # Return an empty graph

    print(
        f"Using hybrid_chunks_df with {len(hybrid_chunks_df)} rows for graph construction."
    )

    # Initialize a new MultiDiGraph
    G = nx.MultiDiGraph()

    # Add nodes with attributes from hybrid_chunks_df
    for idx, row in hybrid_chunks_df.iterrows():
        embedding_value = row["embedding"]
        if isinstance(embedding_value, np.ndarray):
            embedding_value = (
                embedding_value.tolist()
            )  # Convert to list if it's a numpy array

        G.add_node(
            row["chunk_id"],
            doc_id=row["doc_id"],
            doc_name=row["doc_name"],
            category=row["category"],
            text=row["text"][:100] + "..." if len(row["text"]) > 100 else row["text"],
            level=row["level"],
            chunk_method=row["chunk_method"],
            start_idx=row["start_idx"],
            end_idx=row["end_idx"],
            is_special_section=row.get("is_special_section", False),
            section_type=row.get("section_type", None),
            embedding=embedding_value,
        )
    print(f"Added {G.number_of_nodes()} nodes to the graph.")

    # Add hierarchical and same-document edges
    edge_count = 0
    for idx, row in hybrid_chunks_df.iterrows():
        current_chunk_id = row["chunk_id"]
        relationships = row.get("chunk_relationships", {})

        if relationships and relationships.get("hierarchical_parents"):
            for parent_id in relationships["hierarchical_parents"]:
                if G.has_node(parent_id) and G.has_node(current_chunk_id):
                    G.add_edge(
                        current_chunk_id, parent_id, type="hierarchical_child_to_parent"
                    )
                    edge_count += 1

        if row.get("contained_chunks") and isinstance(row["contained_chunks"], list):
            for child_id in row["contained_chunks"]:
                if G.has_node(child_id) and G.has_node(current_chunk_id):
                    G.add_edge(
                        current_chunk_id, child_id, type="hierarchical_parent_to_child"
                    )
                    edge_count += 1

        if relationships and relationships.get("same_doc_l1_chunks"):
            for same_doc_l1_id in relationships["same_doc_l1_chunks"]:
                if G.has_node(same_doc_l1_id) and G.has_node(current_chunk_id):
                    G.add_edge(
                        current_chunk_id, same_doc_l1_id, type="same_document_l1"
                    )
                    edge_count += 1
    print(
        f"Added {edge_count} edges from hierarchical and same-document relationships."
    )

    # Add semantic similarity edges
    l1_chunks = hybrid_chunks_df[hybrid_chunks_df["level"] == "L1"]

    actual_num_l1_to_sample = min(num_l1_to_sample_for_similarity, len(l1_chunks))
    if actual_num_l1_to_sample > 0:
        sample_l1_chunks = l1_chunks.sample(
            n=actual_num_l1_to_sample, random_state=random_state_for_sampling
        )
    else:
        sample_l1_chunks = pd.DataFrame()

    semantic_edge_count = 0
    if not sample_l1_chunks.empty and "embedding" in sample_l1_chunks.columns:
        embeddings_list = [
            np.array(emb).flatten() for emb in sample_l1_chunks["embedding"].tolist()
        ]
        if embeddings_list:
            try:
                embeddings_sample = np.array(embeddings_list)
                if (
                    embeddings_sample.ndim == 2 and embeddings_sample.shape[0] > 0
                ):  # Ensure embeddings are correctly stacked and not empty
                    cosine_sim_matrix = cosine_similarity(embeddings_sample)
                    for i in range(len(sample_l1_chunks)):
                        for j in range(i + 1, len(sample_l1_chunks)):
                            if cosine_sim_matrix[i, j] > similarity_threshold:
                                node1_id = sample_l1_chunks.iloc[i]["chunk_id"]
                                node2_id = sample_l1_chunks.iloc[j]["chunk_id"]
                                if G.has_node(node1_id) and G.has_node(node2_id):
                                    G.add_edge(
                                        node1_id,
                                        node2_id,
                                        type="semantic_similarity",
                                        weight=float(cosine_sim_matrix[i, j]),
                                    )
                                    semantic_edge_count += 1
                    print(
                        f"Added {semantic_edge_count} semantic similarity edges (based on a sample of {actual_num_l1_to_sample} L1 chunks)."
                    )
                elif embeddings_sample.shape[0] == 0:
                    print("Sample L1 chunks embeddings array is empty.")
                else:
                    print(
                        "Embeddings for sample L1 chunks are not in the expected 2D array format after conversion."
                    )
            except Exception as e:
                print(f"Error processing embeddings for cosine similarity: {e}")
        else:
            print("Embedding list for sample L1 chunks is empty.")
    else:
        if sample_l1_chunks.empty:
            print("No L1 chunks found or sampled to calculate semantic similarity.")
        else:
            print(
                "No 'embedding' column found in the sample L1 chunks or sample is empty."
            )

    degrees = dict(G.degree())
    nx.set_node_attributes(G, degrees, "degree")
    print("Added 'degree' attribute to nodes.")

    print(f"Total nodes in graph: {G.number_of_nodes()}")
    print(f"Total edges in graph: {G.number_of_edges()}")

    if not hybrid_chunks_df.empty and G.number_of_nodes() > 0:
        unique_doc_ids = hybrid_chunks_df["doc_id"].unique()
        if len(unique_doc_ids) > 0:
            sample_doc_id = unique_doc_ids[0]
            nodes_of_sample_doc = [
                node
                for node, data in G.nodes(data=True)
                if data.get("doc_id") == sample_doc_id
            ]
            if nodes_of_sample_doc:
                subgraph_sample_doc = G.subgraph(nodes_of_sample_doc)
                print(
                    f"Subgraph for document ID {sample_doc_id} has {subgraph_sample_doc.number_of_nodes()} nodes and {subgraph_sample_doc.number_of_edges()} edges."
                )
            else:
                print(f"No nodes found for sample document ID {sample_doc_id}.")
        else:
            print(
                "No unique document IDs found in hybrid_chunks_df to create a subgraph sample."
            )
    elif G.number_of_nodes() == 0:
        print("Graph is empty, cannot create subgraph sample.")
    else:
        print(
            "hybrid_chunks_df is empty, cannot create document-specific subgraph sample."
        )

    return G
