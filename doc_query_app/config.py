import torch

# Server configuration
HOST = "0.0.0.0"
PORT = 8000

# App settings
APP_TITLE = "Document Query API"
APP_DESCRIPTION = "API for document querying and graph visualization"
APP_VERSION = "1.0.0"

# Embedding model settings
MODEL_NAME = "all-MiniLM-L6-v2"
USE_CACHED_DATA = True
SIMILARITY_THRESHOLD = 0.7
MAX_CHAR_LIMIT = 200

# GNN settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GNN_HIDDEN_DIM = 256
GNN_OUTPUT_DIM = 384  # Same as input dimension for reconstruction
GNN_LAYERS = 3
GNN_DROPOUT = 0.2
GNN_TRAIN_EPOCHS = 100
RETRAIN_GNN = False  # Set to True to retrain the GNN model

# Data caching settings
RETRAIN = True
