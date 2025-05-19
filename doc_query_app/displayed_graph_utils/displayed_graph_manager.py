import os
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from config import (
    SIMILARITY_THRESHOLD,
    MAX_CHAR_LIMIT, FIELD_TO_GROUP
)


class Graph_visualizer:
    def __init__(self):
        self.base_dir = "d:/PROJECTS/CLIENT/USA-Graph-ML/REPOS/Document_Fetch/doc_query_app"
        self.graph_data_path = os.path.join(
            self.base_dir, "data/graph_data.json")
        self.file_data_path = os.path.join(
            self.base_dir, "data/file_data.json")

    def _create_node(self, node_id: str, name: str, desc: str, group: int, conn: int) -> Dict:
        return {
            "id": node_id,
            "name": name,
            "description": desc,
            "group": group,
            "connections": conn
        }

    def _update_graph_structure(self, nodes: List[Dict], links: List[Dict], is_initial: bool = False) -> Dict:
        if is_initial:
            # Building a new graph from scratch
            counts = {}
            for n in nodes:
                g = str(n["group"])
                counts[g] = counts.get(g, 0) + 1

            return {
                "nodes": nodes,
                "links": links,
                "metadata": {
                    "total_nodes": len(nodes),
                    "total_links": len(links),
                    "field_groups": self.field_to_group,
                    "field_group_counts": counts
                }
            }
        else:
            # Updating existing graph
            graph = self._load_json(self.graph_data_path)
            if not graph:
                return self._update_graph_structure(nodes, links, is_initial=True)

            # Update connection counts for existing nodes
            for l in links:
                idx = int(l["target"])
                if idx < len(graph["nodes"]):
                    graph["nodes"][idx]["connections"] += 1

            # Add new nodes and links
            graph["nodes"].extend(nodes)
            graph["links"].extend(links)

            # Update metadata
            graph["metadata"]["total_nodes"] += len(nodes)
            graph["metadata"]["total_links"] += len(links)

            # Update group counts
            for node in nodes:
                group = str(node["group"])
                graph["metadata"]["field_group_counts"][group] = graph["metadata"]["field_group_counts"].get(
                    group, 0) + 1

            return graph

    def _save_json(self, path: str, data: Any):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    def _load_json(self, path: str) -> Optional[Dict]:
        return json.load(open(path)) if os.path.exists(path) else None

    def build_3d_force_graph(self, similarity_df: pd.DataFrame, hybrid_chunks_df: pd.DataFrame) -> Dict:
        """
        Build a 3D force graph JSON with nodes and links based on similarity threshold.

        Args:
            similarity_df: DataFrame with document similarities
            hybrid_chunks_df: DataFrame containing document chunks and metadata

        Returns:
            Dict containing nodes and links for 3D force graph
        """
        # Get unique documents and their first chunks for descriptions
        doc_descriptions = {}
        for doc_id in similarity_df.index:
            doc_chunks = hybrid_chunks_df[hybrid_chunks_df['doc_id'] == doc_id]
            if not doc_chunks.empty:
                # Get the first chunk's text as description
                # First 100 characters
                description = doc_chunks.iloc[0]['text'][:MAX_CHAR_LIMIT]
                category = doc_chunks.iloc[0].get('category', 'unknown')
                doc_descriptions[doc_id] = {
                    'description': description,
                    'category': category
                }

        # Create nodes
        nodes = []
        for doc_id in similarity_df.index:
            if doc_id in doc_descriptions:
                node_info = doc_descriptions[doc_id]
                nodes.append({
                    "id": str(doc_id),
                    "name": f"Doc {doc_id}",
                    "description": node_info['description'],
                    # Using category as group for coloring
                    "group": node_info['category'],
                    "connections": 0  # Will be updated when processing links
                })

        # Create links based on similarity threshold (0.7 or 70%)
        links = []
        for i, doc1_id in enumerate(similarity_df.index):
            for j, doc2_id in enumerate(similarity_df.columns):
                if i < j:  # Only process upper triangle to avoid duplicates
                    similarity = similarity_df.loc[doc1_id, doc2_id]
                    if similarity >= 0.7:  # 70% similarity threshold
                        links.append({
                            "source": str(doc1_id),
                            "target": str(doc2_id),
                            "value": float(similarity)
                        })
                        # Update connection counts
                        nodes[i]["connections"] += 1
                        nodes[j]["connections"] += 1

        # Create the graph structure
        graph = {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "total_nodes": len(nodes),
                "total_links": len(links),
                "similarity_threshold": 0.7,
                "categories": list(set(node["group"] for node in nodes))
            }
        }

        # Save the graph
        self._save_json(self.graph_data_path, graph)
        return graph
