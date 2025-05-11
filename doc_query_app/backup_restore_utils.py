import os
import json
import numpy as np

# Paths for data storage
GRAPH_DATA_PATH = 'data/graph_data.json'
RAW_EMBEDDINGS_PATH = 'data/raw_embeddings.npy'
REFINED_EMBEDDINGS_PATH = 'data/refined_embeddings.npy'
FILE_DATA_PATH = 'data/file_data.json'
DATA_DIR = 'data'
RAW_FILES_DIR = 'raw_files'


def make_data_dir():
    """Create data directory if it doesn't exist"""
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_FILES_DIR, exist_ok=True)


def save_raw_embeddings(embeddings):
    """Save embeddings to file"""
    print("Saving rawembeddings...")
    os.makedirs(os.path.dirname(RAW_EMBEDDINGS_PATH), exist_ok=True)
    np.save(RAW_EMBEDDINGS_PATH, embeddings)


def save_refined_embeddings(embeddings, replace=False):
    """Save embeddings to file"""
    print("Saving embeddings...")
    os.makedirs(os.path.dirname(REFINED_EMBEDDINGS_PATH), exist_ok=True)
    np.save(REFINED_EMBEDDINGS_PATH, embeddings)


def update_raw_embeddings(embeddings):
    embeddings = load_raw_embeddings()
    embeddings = np.concatenate((embeddings, embeddings), axis=0)
    save_raw_embeddings(embeddings)


def load_raw_embeddings():
    """Load embeddings from file if exists, otherwise return None"""
    print("Loading raw embeddings...")
    if os.path.exists(RAW_EMBEDDINGS_PATH):
        return np.load(RAW_EMBEDDINGS_PATH)
    return None


def load_refined_embeddings():
    """Load embeddings from file if exists, otherwise return None"""
    print("Loading refined embeddings...")
    if os.path.exists(REFINED_EMBEDDINGS_PATH):
        return np.load(REFINED_EMBEDDINGS_PATH)
    return None


def save_graph_data(graph_data):
    """Save graph data to file"""
    print("Saving graph data...")
    os.makedirs(os.path.dirname(GRAPH_DATA_PATH), exist_ok=True)
    with open(GRAPH_DATA_PATH, 'w') as f:
        json.dump(graph_data, f)


def load_graph_data():
    """Load graph data from file if exists, otherwise return None"""
    print("Loading graph data...")
    if os.path.exists(GRAPH_DATA_PATH):
        with open(GRAPH_DATA_PATH, 'r') as f:
            return json.load(f)
    return None


def save_file_data(file_data):
    """Save file data to file"""
    print("Saving file data...")
    with open(FILE_DATA_PATH, 'w') as f:
        json.dump(file_data, f)


def load_file_data():
    """Load file data from file if exists, otherwise return None"""
    print("Loading file data...")
    if os.path.exists(FILE_DATA_PATH):
        with open(FILE_DATA_PATH, 'r') as f:
            return json.load(f)
    return None
