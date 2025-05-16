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
USE_CACHED_DATA = False
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

# Data caching settings
RETRAIN = True
