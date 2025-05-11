import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from config import MODEL_NAME, DEVICE, GNN_HIDDEN_DIM, GNN_OUTPUT_DIM, GNN_LAYERS, GNN_DROPOUT


class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.0):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.convs.append(SAGEConv(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class GNNManager:
    def __init__(self):
        self.model = None
        self.device = torch.device(DEVICE)
        self.model_path = 'models/gnn_model.pt'
        os.makedirs('models', exist_ok=True)

    def prepare_graph_data(self, embeddings: np.ndarray, links: List[Dict]) -> Data:
        """Convert embeddings and links to PyTorch Geometric Data format"""
        # Convert embeddings to tensor
        x = torch.FloatTensor(embeddings).to(self.device)

        # Create edge index from links
        edge_index = []
        for link in links:
            edge_index.append([int(link['source']), int(link['target'])])
            # Add reverse edge
            edge_index.append([int(link['target']), int(link['source'])])

        edge_index = torch.LongTensor(
            edge_index).t().contiguous().to(self.device)

        return Data(x=x, edge_index=edge_index)

    def train_model(self, embeddings: np.ndarray, links: List[Dict], epochs: int = 100) -> None:
        """Train the GraphSAGE model"""
        # Initialize model if not exists
        if self.model is None:
            input_dim = embeddings.shape[1]
            self.model = GraphSAGE(
                input_dim=input_dim,
                hidden_dim=GNN_HIDDEN_DIM,
                output_dim=GNN_OUTPUT_DIM,
                num_layers=GNN_LAYERS,
                dropout=GNN_DROPOUT
            ).to(self.device)

        # Prepare data
        data = self.prepare_graph_data(embeddings, links)

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            out = self.model(data.x, data.edge_index)

            # Reconstruction loss (try to reconstruct original embeddings)
            loss = criterion(out, data.x)

            # Backward pass
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

        # Save model
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self, input_dim: int) -> None:
        """Load a trained model"""
        self.model = GraphSAGE(
            input_dim=input_dim,
            hidden_dim=GNN_HIDDEN_DIM,
            output_dim=GNN_OUTPUT_DIM,
            num_layers=GNN_LAYERS,
            dropout=GNN_DROPOUT
        ).to(self.device)

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()

    def refine_embeddings(self, embeddings: np.ndarray, links: List[Dict]) -> np.ndarray:
        """Refine embeddings using the trained GNN model"""
        if self.model is None:
            self.load_model(embeddings.shape[1])

        if self.model is None:
            return embeddings  # Return original embeddings if model loading fails

        # Prepare data
        data = self.prepare_graph_data(embeddings, links)

        # Get refined embeddings
        self.model.eval()
        with torch.no_grad():
            refined_embeddings = self.model(data.x, data.edge_index)

        return refined_embeddings.cpu().numpy()


# Create singleton instance
gnn_manager = GNNManager()
