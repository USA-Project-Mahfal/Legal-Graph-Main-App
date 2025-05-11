# GNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple
from config import (
    DEVICE, GNN_HIDDEN_DIM, GNN_OUTPUT_DIM, GNN_LAYERS,
    GNN_DROPOUT, GNN_VERSION, GNN_TRAIN_EPOCHS
)

# Paths and config
MODEL_PATH = "models/gnn_model.pt"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


class GraphSAGE(nn.Module):
    """GraphSAGE model for node embedding refinement."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.0):
        """Initialize the GraphSAGE model.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
        """
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Node features
            edge_index: Graph connectivity

        Returns:
            Refined node features
        """
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GNNManager:
    """Manager for GraphSAGE model and graph data operations."""

    def __init__(self):
        """Initialize GNNManager with paths and device configuration."""
        # Model setup
        self.device = torch.device(DEVICE)
        self.model = None
        self.version = GNN_VERSION

        # File paths
        self.data_dir = "data"
        self.model_dir = "models"
        self.embeddings_path = os.path.join(
            self.data_dir, "raw_embeddings.npy")
        self.refined_path = os.path.join(
            self.data_dir, "refined_embeddings.npy")
        self.graph_path = os.path.join(self.data_dir, "graph_links.json")
        self.model_path = os.path.join(self.model_dir, "gnn_model.pt")
        self.version_path = os.path.join(self.model_dir, "gnn_version.json")

        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Load cached data
        self.embeddings = self._load_npy(self.embeddings_path)
        self.graph_links = self._load_json(self.graph_path) or []
        self.input_dim = self.embeddings.shape[1] if self.embeddings is not None else None

    def _save_json(self, path: str, data: Dict) -> None:
        """Save data as JSON."""
        with open(path, 'w') as f:
            json.dump(data, f)

    def _load_json(self, path: str) -> Optional[Dict]:
        """Load JSON data if file exists."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def _save_npy(self, path: str, array: np.ndarray) -> None:
        """Save numpy array."""
        np.save(path, array)

    def _load_npy(self, path: str) -> Optional[np.ndarray]:
        """Load numpy array if file exists."""
        if os.path.exists(path):
            return np.load(path)
        return None

    def _save_version_info(self) -> None:
        """Save model version and architecture information."""
        version_info = {
            'version': self.version,
            'input_dim': self.input_dim,
            'hidden_dim': GNN_HIDDEN_DIM,
            'output_dim': GNN_OUTPUT_DIM,
            'num_layers': GNN_LAYERS
        }
        self._save_json(self.version_path, version_info)

    def _should_retrain(self) -> bool:
        """Check if model needs retraining based on version or architecture changes."""
        version_info = self._load_json(self.version_path)
        if not version_info:
            return True

        # Check if model architecture or version has changed
        return (
            version_info['version'] != self.version or
            version_info['input_dim'] != self.input_dim or
            version_info['hidden_dim'] != GNN_HIDDEN_DIM or
            version_info['output_dim'] != GNN_OUTPUT_DIM or
            version_info['num_layers'] != GNN_LAYERS
        )

    def reload(self) -> None:
        """Reload embeddings and graph data from disk."""
        self.embeddings = self._load_npy(self.embeddings_path)
        self.graph_links = self._load_json(self.graph_path) or []
        if self.embeddings is not None:
            self.input_dim = self.embeddings.shape[1]

    def prepare_graph_data(self, embeddings: np.ndarray, links: List[Dict]) -> Data:
        """Convert embeddings and links to PyTorch Geometric Data format.

        Args:
            embeddings: Node features matrix
            links: List of edge dictionaries with source and target

        Returns:
            PyTorch Geometric Data object
        """
        x = torch.tensor(embeddings, dtype=torch.float, device=self.device)

        # Create bidirectional edges
        edge_index = []
        for link in links:
            source = int(link['source'])
            target = int(link['target'])
            edge_index.append([source, target])
            edge_index.append([target, source])  # Add reverse edge

        if not edge_index:  # Handle case with no links
            edge_index = [[0, 0]]  # Self-loop as fallback

        edge_index = torch.tensor(
            edge_index, dtype=torch.long, device=self.device).t().contiguous()
        return Data(x=x, edge_index=edge_index)

    def load_model(self) -> bool:
        """Load trained model if it exists.

        Returns:
            True if model was loaded successfully, False otherwise
        """
        if self.model is None:
            if self.input_dim is None:
                self.reload()
                if self.input_dim is None:
                    return False

            # Initialize model architecture
            self.model = GraphSAGE(
                input_dim=self.input_dim,
                hidden_dim=GNN_HIDDEN_DIM,
                output_dim=GNN_OUTPUT_DIM,
                num_layers=GNN_LAYERS,
                dropout=GNN_DROPOUT
            ).to(self.device)

        # Load weights if available
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(
                self.model_path, map_location=self.device))
            self.model.eval()
            return True
        return False

    def train_model(self, force_retrain: bool = False) -> bool:
        """Train or retrain the GraphSAGE model.

        Args:
            force_retrain: Force retraining even if not needed

        Returns:
            True if training was performed, False otherwise
        """
        self.reload()

        if self.embeddings is None or len(self.graph_links) < 2:
            print("Insufficient data for training")
            return False

        if not force_retrain and not self._should_retrain() and os.path.exists(self.model_path):
            print("Using existing model (no retraining needed)")
            return self.load_model()

        print(f"Training new GNN model (input dim: {self.input_dim})")

        # Initialize new model
        self.model = GraphSAGE(
            input_dim=self.input_dim,
            hidden_dim=GNN_HIDDEN_DIM,
            output_dim=GNN_OUTPUT_DIM,
            num_layers=GNN_LAYERS,
            dropout=GNN_DROPOUT
        ).to(self.device)

        # Prepare training data
        data = self.prepare_graph_data(self.embeddings, self.graph_links)

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(GNN_TRAIN_EPOCHS):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = criterion(out, data.x)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch {epoch+1}/{GNN_TRAIN_EPOCHS}, Loss: {loss.item():.4f}')

        # Save model and metadata
        torch.save(self.model.state_dict(), self.model_path)
        self._save_version_info()
        print("GNN model saved successfully")
        return True

    def refine_embeddings(self, force_retrain: bool = False) -> np.ndarray:
        """Refine embeddings using the GNN model.

        Args:
            force_retrain: Force retraining before refinement

        Returns:
            Refined node embeddings
        """
        self.reload()

        if self.embeddings is None:
            raise ValueError("No embeddings available for refinement")

        # Train or load the model
        model_ready = self.train_model(
            force_retrain) if force_retrain else self.load_model()

        if not model_ready:
            print("Warning: Using original embeddings (no model available)")
            return self.embeddings

        # Prepare data and run inference
        data = self.prepare_graph_data(self.embeddings, self.graph_links)
        self.model.eval()

        with torch.no_grad():
            refined_embeddings = self.model(
                data.x, data.edge_index).cpu().numpy()

        # Save refined embeddings
        self._save_npy(self.refined_path, refined_embeddings)
        return refined_embeddings

    def add_node(self, new_embedding: np.ndarray, neighbors: List[int]) -> int:
        """Add a new node to the graph.

        Args:
            new_embedding: Embedding vector for the new node
            neighbors: List of neighbor node indices

        Returns:
            ID of the new node
        """
        self.reload()

        if self.embeddings is None:
            self.embeddings = new_embedding
            new_id = 0
        else:
            new_id = len(self.embeddings)
            self.embeddings = np.vstack([self.embeddings, new_embedding])

        # Add edges to neighbors
        for neighbor in neighbors:
            self.graph_links.append({'source': new_id, 'target': neighbor})

        # Save updated data
        self._save_npy(self.embeddings_path, self.embeddings)
        self._save_json(self.graph_path, self.graph_links)
        return new_id

    def get_node_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of a node.

        Args:
            node_id: ID of the node

        Returns:
            List of neighbor node IDs
        """
        neighbors = []
        for link in self.graph_links:
            if int(link['source']) == node_id:
                neighbors.append(int(link['target']))
            elif int(link['target']) == node_id:
                neighbors.append(int(link['source']))
        return list(set(neighbors))  # Remove duplicates

    def initialize_with_embeddings(self, embeddings: np.ndarray) -> None:
        """Initialize the GNN with new embeddings.

        Args:
            embeddings: Array of embeddings for nodes
        """
        print(f"Initializing GNN with {len(embeddings)} embeddings")

        # Save the raw embeddings
        self._save_npy(self.embeddings_path, embeddings)
        self.embeddings = embeddings
        self.input_dim = embeddings.shape[1]

        # Initialize an empty graph links list
        self.graph_links = []
        self._save_json(self.graph_path, self.graph_links)

        # Reset the refined embeddings
        if os.path.exists(self.refined_path):
            os.remove(self.refined_path)

        # Force retrain the model with the new embeddings
        self.model = None  # Reset the model to ensure it's re-initialized properly
        print("Model will be retrained with new dimensions on next refinement")


# Singleton instance
gnn_manager = GNNManager()
