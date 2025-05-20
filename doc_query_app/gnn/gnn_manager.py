import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GAE
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
import numpy as np
import os
import matplotlib.pyplot as plt  # For plotting loss if desired


# --- Model Definition ---
class RGCNEncoder(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_relations, num_bases=None
    ):
        super().__init__()
        # If num_bases is None, it defaults to num_relations (no basis decomposition)
        # Using num_bases can reduce parameters if num_relations is very high.
        if (
            num_bases is None
        ):  # Or set to a specific number like 30 if you have many relations
            num_bases = num_relations

        self.conv1 = RGCNConv(
            in_channels, hidden_channels, num_relations, num_bases=num_bases
        )
        self.conv2 = RGCNConv(
            hidden_channels, out_channels, num_relations, num_bases=num_bases
        )
        # self.dropout = torch.nn.Dropout(0.5) # Optional dropout

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        # x = self.dropout(x) # Optional
        x = self.conv2(x, edge_index, edge_type)
        return x


class GNNManager:
    def __init__(
        self,
        num_node_features: int,
        num_relations: int,
        model_path: str = "rgcn_gae_model.pth",
        embeddings_path: str = "final_node_embeddings.npy",
        hidden_channels: int = 128,
        out_channels: int = 64,
        learning_rate: float = 0.01,
        device=None,
    ):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"GNNManager using device: {self.device}")

        self.in_channels = num_node_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.lr = learning_rate

        self.trained_model_path = model_path
        self.final_embeddings_path = embeddings_path

        self.encoder = RGCNEncoder(
            self.in_channels,
            self.hidden_channels,
            self.out_channels,
            self.num_relations,
        ).to(self.device)
        self.model = GAE(self.encoder).to(
            self.device
        )  # GAE implicitly uses dot product decoder
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.data_for_gae: Data = None  # Will hold the data with train/val/test splits

        print("GNNManager initialized with RGCN-GAE model.")
        print(self.model)

    def _prepare_data_for_gae(self, pyg_data: Data, val_ratio=0.1, test_ratio=0.1):
        """Splits edges for GAE training. Modifies self.data_for_gae."""
        if pyg_data is None:
            raise ValueError("pyg_data cannot be None for GAE preparation.")
        if pyg_data.num_nodes == 0:
            print("Warning: pyg_data has 0 nodes. GAE preparation might be trivial.")

        print("Preparing data for GAE (splitting edges)...")
        self.data_for_gae = pyg_data.clone().to(self.device)  # Clone and move to device

        if self.data_for_gae.num_edges > 0:  # train_test_split_edges requires edges
            self.data_for_gae = train_test_split_edges(
                self.data_for_gae, val_ratio=val_ratio, test_ratio=test_ratio
            )
        else:
            # Handle case with no edges: create empty attributes to avoid errors later
            print(
                "No edges in data to split. Creating empty train/val/test edge indices."
            )
            self.data_for_gae.train_pos_edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )
            self.data_for_gae.val_pos_edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )
            self.data_for_gae.val_neg_edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )
            self.data_for_gae.test_pos_edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )
            self.data_for_gae.test_neg_edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )

        print("Data prepared for GAE training (edges split):")
        # Print all attributes except node_id_mapping
        data_attrs = {k: v for k, v in self.data_for_gae.items()
                      if k != 'node_id_mapping'}
        print(data_attrs)

    def _train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        # GAE's model.encode uses the graph structure passed to it.
        # For training, we want it to learn from the training edges.
        # The edge_type needs to correspond to the train_pos_edge_index.
        # This is a complex part: train_test_split_edges does not directly give edge_types for the split.
        # A common simplification for GAE is to encode using the full graph structure
        # or ensure edge_type is correctly subsetted if the encoder strictly needs it per edge.
        # For RGCNConv, edge_type is crucial.

        # Assuming self.data_for_gae.edge_type contains types for *all* original edges.
        # We need to be careful here. If train_pos_edge_index is a subset of original edges,
        # we need to pass the corresponding subset of edge_types.
        # For now, we pass the full edge_type tensor from data_for_gae.
        # The RGCNConv layer should internally select the types for the edges in train_pos_edge_index.

        z = self.model.encode(
            self.data_for_gae.x,
            # Edges for message passing during encoding
            self.data_for_gae.train_pos_edge_index,
            # Full edge types; RGCNConv handles selection based on edge_index
            self.data_for_gae.edge_type,
        )

        loss = self.model.recon_loss(z, self.data_for_gae.train_pos_edge_index)

        num_pos_edges = self.data_for_gae.train_pos_edge_index.size(1)
        if num_pos_edges > 0:
            loss = loss / num_pos_edges  # Normalize loss
        else:  # Avoid division by zero if no positive training edges
            loss = torch.tensor(
                0.0,
                device=self.device,
                requires_grad=True if self.model.training else False,
            )

        if self.model.training:  # Only backward and step if in training mode
            loss.backward()
            self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def _test_epoch(self, pos_edge_index, neg_edge_index):
        self.model.eval()
        # Encode using the training graph structure to get node embeddings for evaluation
        # This is a common practice: learn embeddings on training graph, then test link prediction.
        z = self.model.encode(
            self.data_for_gae.x,
            # Use training graph structure for encoding
            self.data_for_gae.train_pos_edge_index,
            self.data_for_gae.edge_type,  # Full edge types
        )
        return self.model.test(z, pos_edge_index, neg_edge_index)

    def train_pipeline(
        self, pyg_data: Data, epochs=200, val_ratio=0.1, test_ratio=0.1, plot_loss=False
    ):
        if pyg_data.num_nodes == 0:
            print("Skipping GNN training as there are no nodes in the graph.")
            # history, test_scores, final_embeddings
            return None, (0.0, 0.0), None

        self._prepare_data_for_gae(pyg_data, val_ratio, test_ratio)

        if self.data_for_gae.train_pos_edge_index.size(1) == 0:
            print("No positive training edges after split. Cannot train GAE model.")
            # Potentially load a pre-trained model or return
            if self.load_model():  # Try loading if no training edges
                print("Loaded pre-existing model as no training edges were available.")
                final_embeddings = self.generate_and_save_embeddings(pyg_data)
                return None, "Loaded Pretrained", final_embeddings
            return None, (0.0, 0.0), None

        print(f"Starting GNN training for {epochs} epochs...")
        history = {"loss": [], "val_auc": [], "val_ap": []}

        for epoch in range(1, epochs + 1):
            loss = self._train_epoch()
            # Validation
            if self.data_for_gae.val_pos_edge_index.size(1) > 0:
                val_auc, val_ap = self._test_epoch(
                    self.data_for_gae.val_pos_edge_index,
                    self.data_for_gae.val_neg_edge_index,
                )
            else:
                val_auc, val_ap = 0.0, 0.0  # No validation edges

            history["loss"].append(loss)
            history["val_auc"].append(val_auc)
            history["val_ap"].append(val_ap)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}"
                )

        # Test set evaluation
        if self.data_for_gae.test_pos_edge_index.size(1) > 0:
            test_auc, test_ap = self._test_epoch(
                self.data_for_gae.test_pos_edge_index,
                self.data_for_gae.test_neg_edge_index,
            )
            print(
                f"\nTraining complete. Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}"
            )
        else:
            test_auc, test_ap = 0.0, 0.0
            print(f"\nTraining complete. No test edges for evaluation.")

        self.save_model()
        # Generate embeddings using the original full pyg_data for a complete set of node embeddings
        final_embeddings = self.generate_and_save_embeddings(pyg_data)

        if plot_loss:
            self._plot_training_history(history)

        return history, (test_auc, test_ap), final_embeddings

    def _plot_training_history(self, history):
        if not history or not history["loss"]:
            print("No training history to plot.")
            return

        epochs_range = range(1, len(history["loss"]) + 1)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history["loss"], label="Training Loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        if history["val_auc"]:
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, history["val_auc"], label="Validation AUC")
            if history["val_ap"]:
                plt.plot(epochs_range,
                         history["val_ap"], label="Validation AP")
            plt.title("Validation Metrics")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, path=None):
        if path is None:
            path = self.trained_model_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"GNN Model saved to '{path}'")

    def load_model(self, path=None) -> bool:
        if path is None:
            path = self.trained_model_path
        if os.path.exists(path):
            try:
                self.model.load_state_dict(
                    torch.load(path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"GNN Model loaded from '{path}'")
                return True
            except Exception as e:
                print(f"Error loading GNN model from '{path}': {e}")
                return False
        else:
            print(f"No GNN model found at '{path}'")
            return False

    def generate_and_save_embeddings(self, pyg_data_to_encode: Data, save_path=None):
        if pyg_data_to_encode.num_nodes == 0:
            print("Cannot generate embeddings, no nodes in the provided graph data.")
            return None

        if save_path is None:
            save_path = self.final_embeddings_path

        self.model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            # Encode using the provided data, ensuring it's on the correct device
            data_on_device = pyg_data_to_encode.to(self.device)
            node_embeddings = self.model.encode(
                data_on_device.x, data_on_device.edge_index, data_on_device.edge_type
            )

        np_embeddings = node_embeddings.cpu().numpy()

        # Ensure directory for embeddings exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, np_embeddings)
        print(
            f"Final node embeddings saved to '{save_path}' (shape: {np_embeddings.shape})"
        )
        return np_embeddings
