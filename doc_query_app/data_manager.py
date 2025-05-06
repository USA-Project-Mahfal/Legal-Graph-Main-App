import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import USE_CACHED_DATA, MODEL_NAME, SIMILARITY_THRESHOLD
from data import project_data
# Paths for data storage
GRAPH_DATA_PATH = 'data/graph_data.json'
EMBEDDINGS_PATH = 'data/embeddings.npy'
DATA_DIR = 'data'

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Field to group mapping
field_to_group = {
    "web": 1,
    "mobile": 2,
    "cybersecurity": 3,
    "iot": 4
}


def save_embeddings(embeddings):
    """Save embeddings to file"""
    np.save(EMBEDDINGS_PATH, embeddings)


def load_embeddings():
    """Load embeddings from file if exists, otherwise return None"""
    if os.path.exists(EMBEDDINGS_PATH):
        return np.load(EMBEDDINGS_PATH)
    return None


def save_graph_data(graph_data):
    """Save graph data to file"""
    with open(GRAPH_DATA_PATH, 'w') as f:
        json.dump(graph_data, f)


def load_graph_data():
    """Load graph data from file if exists, otherwise return None"""
    if os.path.exists(GRAPH_DATA_PATH):
        with open(GRAPH_DATA_PATH, 'r') as f:
            return json.load(f)
    return None


def generate_embeddings():
    """Generate embeddings for project data"""
    # Get descriptions from project data
    descriptions = [project["description"] for project in project_data]

    # Load pre-trained sentence transformer model
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    embeddings = model.encode(descriptions)

    # Always save new embeddings when generating
    save_embeddings(embeddings)

    return embeddings


def create_graph_data(embeddings):
    """Create graph data from embeddings"""
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Create links between nodes based on similarity threshold
    links = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            if similarity_matrix[i][j] > SIMILARITY_THRESHOLD:
                links.append({
                    "source": str(i),
                    "target": str(j),
                    "value": float(similarity_matrix[i][j])
                })

    # Create nodes with embeddings and metadata
    nodes = []
    for i, project in enumerate(project_data):
        nodes.append({
            "id": str(i),
            "name": project["title"],
            "group": field_to_group[project["field"]],
            "description": project["description"],
            "val": 1 + len([link for link in links if str(i) in [link["source"], link["target"]]])
        })

    # Create graph data
    graph_data = {
        "nodes": nodes,
        "links": links
    }

    # Always save new graph data when creating
    print(
        f"Generated graph data with {len(nodes)} nodes and {len(links)} links")
    save_graph_data(graph_data)

    return graph_data


def get_graph_data():
    """Get graph data - handle caching based on USE_CACHED_DATA setting"""
    if USE_CACHED_DATA:
        # Try to load existing graph data
        graph_data = load_graph_data()
        if graph_data:
            return graph_data

        # If no graph data, try to use existing embeddings
        embeddings = load_embeddings()
        if embeddings is not None:
            return create_graph_data(embeddings)
    else:
        # If not using cached data, remove old files if they exist
        if os.path.exists(GRAPH_DATA_PATH):
            os.remove(GRAPH_DATA_PATH)
        if os.path.exists(EMBEDDINGS_PATH):
            os.remove(EMBEDDINGS_PATH)

    # Generate new embeddings and graph data
    embeddings = generate_embeddings()
    return create_graph_data(embeddings)
