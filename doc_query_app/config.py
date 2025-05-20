import torch

# Server settings
HOST = "0.0.0.0"
PORT = 8000
APP_TITLE = "Document Query API"
APP_DESCRIPTION = "API for document querying and graph visualization"
APP_VERSION = "1.0.0"

# Data settings
DATA_DIR = "data"
RAW_FILES_DIR = "raw_files"
USE_CACHED_DATA = True
USE_CACHED_3D_GRAPH = True
MAX_CHAR_LIMIT = 200

# Embedding model settings
MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.7

# GNN settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GNN_HIDDEN_DIM = 256
GNN_OUTPUT_DIM = 384  # Should match embeddings dimension
GNN_LAYERS = 3
GNN_DROPOUT = 0.2
GNN_TRAIN_EPOCHS = 100
GNN_VERSION = "1.0.0"  # Increment when model architecture changes
RETRAIN_GNN = False    # Set to True to force retraining
GNN_MODEL_PATH = "models/gnn_model.pth"
GNN_EMBEDDINGS_PATH = "models/gnn_embeddings.pth"

# Data caching settings
RETRAIN = True

FIELD_TO_GROUP = {"License_Agreements": 1, "Maintenance": 2,
                  "Service": 3, "Sponsorship": 4, "Strategic Alliance": 5}

GROUP_COLORS = {
    "License_Agreements": "#FFD700",
    "Maintenance": "#EF4444",
    "Service": "#008000",
    "Sponsorship": "#0000FF",
    "Strategic Alliance": "#800080",
    "unknown": "#808080"
}
