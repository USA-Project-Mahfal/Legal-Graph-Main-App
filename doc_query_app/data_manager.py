# data_manager.py

import os
import json
import numpy as np
import docx
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple, Any

from config import (
    DATA_DIR, RAW_FILES_DIR, MODEL_NAME, SIMILARITY_THRESHOLD,
    MAX_CHAR_LIMIT, RETRAIN_GNN, USE_CACHED_DATA
)
from GNN import gnn_manager


class DataManager:
    """Manager for document data, embeddings, and graph structure."""

    def __init__(self):
        """Initialize the DataManager."""
        # File paths
        self.data_dir = DATA_DIR
        self.raw_files_dir = RAW_FILES_DIR
        self.graph_data_path = os.path.join(self.data_dir, "graph_data.json")
        self.file_data_path = os.path.join(self.data_dir, "file_data.json")

        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.raw_files_dir, exist_ok=True)
        os.makedirs(os.path.join(self.raw_files_dir, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(self.raw_files_dir,
                    "full_contract_txt"), exist_ok=True)

        # Field to group mapping
        self.field_to_group = {
            "file": 5,  # Contract files
            "new": 6    # Newly uploaded files
        }

        # File type handlers
        self.file_handlers = {
            '.pdf': lambda f: PyPDF2.PdfReader(f).pages[0].extract_text(),
            '.docx': lambda f: docx.Document(f).paragraphs[0].text if docx.Document(f).paragraphs else "",
            'default': lambda f: f.read()
        }

        # Initialize embedding model
        self.model = SentenceTransformer(MODEL_NAME)

    def _save_json(self, path: str, data: Any) -> None:
        """Save data as JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    def _load_json(self, path: str) -> Optional[Dict]:
        """Load JSON data if file exists."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def save_graph_data(self, graph_data: Dict) -> None:
        """Save graph data to file."""
        print("Saving graph data...")
        self._save_json(self.graph_data_path, graph_data)

    def load_graph_data(self) -> Optional[Dict]:
        """Load graph data from file if it exists."""
        print("Loading graph data...")
        return self._load_json(self.graph_data_path)

    def save_file_data(self, file_data: List[Dict]) -> None:
        """Save file data to file."""
        print("Saving file data...")
        self._save_json(self.file_data_path, file_data)

    def load_file_data(self) -> Optional[List[Dict]]:
        """Load file data from file if it exists."""
        print("Loading file data...")
        return self._load_json(self.file_data_path)

    def get_file_description(self, file_path: str) -> str:
        """Extract content from file and return truncated description."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            handler = self.file_handlers.get(
                ext, self.file_handlers['default'])

            mode = 'rb' if ext == '.pdf' else 'r'
            encoding = None if ext == '.pdf' else 'utf-8'

            with open(file_path, mode, encoding=encoding) as f:
                content = handler(f)

            # Truncate and return
            return content[:MAX_CHAR_LIMIT] + "..." if len(content) > MAX_CHAR_LIMIT else content

        except Exception as e:
            return f"Error reading file {os.path.basename(file_path)}: {str(e)}"

    def process_new_file(self, filename: str) -> Dict:
        """Process a newly uploaded file and update the graph."""
        # Get file path and extract description
        file_path = os.path.join(self.raw_files_dir, 'uploads', filename)
        description = self.get_file_description(file_path)

        if "Error reading" in description:
            return {"error": description}

        # Generate embedding for the new file
        new_embedding = self.model.encode([description])[0]

        # Make sure GNN manager is synced with latest data
        gnn_manager.reload()
        current_embeddings = gnn_manager.embeddings

        if current_embeddings is None:
            # First file - initialize graph
            new_node_id = gnn_manager.add_node(
                new_embedding.reshape(1, -1), [])
            graph_data = {
                "nodes": [{
                    "id": "0",
                    "name": filename,
                    "group": self.field_to_group["new"],
                    "description": description,
                    "connections": 0
                }],
                "links": [],
                "metadata": {
                    "field_groups": self.field_to_group,
                    "field_group_counts": {self.field_to_group["new"]: 1},
                    "total_nodes": 1,
                    "total_links": 0
                }
            }
            self.save_graph_data(graph_data)
            return graph_data

        # Find neighbors using similarity
        similarities = cosine_similarity(
            [new_embedding], current_embeddings)[0]
        neighbors = [i for i, s in enumerate(
            similarities) if s > SIMILARITY_THRESHOLD]

        # Add node to GNN and get its ID
        new_node_id = gnn_manager.add_node(
            new_embedding.reshape(1, -1), neighbors)

        # Refine embeddings using the GNN
        gnn_manager.refine_embeddings(RETRAIN_GNN)

        # Update graph data
        graph_data = self.load_graph_data()
        if not graph_data:
            return {"error": "Failed to load graph data"}

        # Create new node entry
        new_node_str = str(new_node_id)
        new_node = {
            "id": new_node_str,
            "name": filename,
            "group": self.field_to_group["new"],
            "description": description,
            "connections": len(neighbors)
        }

        # Create links to neighbors
        new_links = [
            {"source": new_node_str, "target": str(
                n), "value": float(similarities[n])}
            for n in neighbors
        ]

        # Update connection counts for neighbors
        for link in new_links:
            target_idx = int(link['target'])
            if target_idx < len(graph_data['nodes']):
                graph_data['nodes'][target_idx]['connections'] += 1

        # Update graph data structure
        graph_data["nodes"].append(new_node)
        graph_data["links"].extend(new_links)
        graph_data["metadata"]["total_nodes"] += 1
        graph_data["metadata"]["total_links"] += len(new_links)

        # Update field group counts
        group = self.field_to_group["new"]
        if "field_group_counts" not in graph_data["metadata"]:
            graph_data["metadata"]["field_group_counts"] = {}
        graph_data["metadata"]["field_group_counts"][str(
            group)] = graph_data["metadata"]["field_group_counts"].get(str(group), 0) + 1

        # Save updated graph data
        self.save_graph_data(graph_data)
        return graph_data

    def get_graph_data(self) -> Dict:
        """Get the current graph data structure."""
        # Try to load existing graph data
        graph_data = self.load_graph_data()
        if graph_data and USE_CACHED_DATA:
            return graph_data

        # Get data from GNN manager
        gnn_manager.reload()
        embeddings = gnn_manager.embeddings

        if embeddings is None:
            return {"error": "No embeddings available"}

        # Build graph from GNN data
        links = gnn_manager.graph_links
        nodes = []

        # Create nodes
        for i in range(len(embeddings)):
            # Count connections for this node
            conn_count = sum(1 for link in links
                             if str(link.get("source")) == str(i) or str(link.get("target")) == str(i))

            # Create node entry
            nodes.append({
                "id": str(i),
                "name": f"Node {i}",
                "group": self.field_to_group["new"],
                "description": f"Auto-generated node {i}",
                "connections": conn_count
            })

        # Count nodes by group
        field_group_counts = {}
        for node in nodes:
            g_str = str(node["group"])
            field_group_counts[g_str] = field_group_counts.get(g_str, 0) + 1

        # Create graph data structure
        graph_data = {
            "nodes": nodes,
            "links": [
                {"source": str(link.get("source")), "target": str(
                    link.get("target")), "value": 1.0}
                for link in links
            ],
            "metadata": {
                "total_nodes": len(nodes),
                "total_links": len(links),
                "field_groups": self.field_to_group,
                "field_group_counts": field_group_counts
            }
        }

        # Save and return
        self.save_graph_data(graph_data)
        return graph_data


# Create singleton instance
data_manager = DataManager()
