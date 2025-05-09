import json
import os
import numpy as np
import docx
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import USE_CACHED_DATA, MODEL_NAME, SIMILARITY_THRESHOLD, MAX_CHAR_LIMIT
# Paths for data storage
GRAPH_DATA_PATH = 'data/graph_data.json'
EMBEDDINGS_PATH = 'data/embeddings.npy'
FILE_DATA_PATH = 'data/file_data.json'
DATA_DIR = 'data'
RAW_FILES_DIR = 'raw_files'

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_FILES_DIR, exist_ok=True)

# Field to group mapping
field_to_group = {
    "file": 5,  # New group for uploaded files
    "new": 6  # New group for uploaded files
}


def save_embeddings(embeddings):
    """Save embeddings to file"""
    print("Saving embeddings...")
    np.save(EMBEDDINGS_PATH, embeddings)


def load_embeddings():
    """Load embeddings from file if exists, otherwise return None"""
    print("Loading embeddings...")
    if os.path.exists(EMBEDDINGS_PATH):
        return np.load(EMBEDDINGS_PATH)
    return None


def save_graph_data(graph_data):
    """Save graph data to file"""
    print("Saving graph data...")
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


def generate_embeddings():
    """Generate embeddings for project data"""
    print("Generating embeddings...")
    # Get descriptions from project data
    # Get descriptions from project data and contract files
    descriptions = []
    file_data = []

    def process_directory(directory, field_type, limit=30):
        """Process files in a directory and add them to descriptions and file_data"""
        if not os.path.exists(directory):
            return

        for filename in os.listdir(directory)[:limit]:
            if not filename.endswith('.txt'):
                continue

            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    descriptions.append(content[:MAX_CHAR_LIMIT])
                    file_data.append({
                        "id": len(file_data),
                        "title": filename,
                        "field": field_type,
                        "description": content[:MAX_CHAR_LIMIT]
                    })
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")

    # Process both directories
    contract_dir = os.path.join(RAW_FILES_DIR, 'full_contract_txt')
    uploads_dir = os.path.join(RAW_FILES_DIR, 'uploads')

    process_directory(contract_dir, "file", 30)
    process_directory(uploads_dir, "new")

    print("Number of files added:", len(file_data))

    # Load pre-trained sentence transformer model
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    embeddings = model.encode(descriptions)

    # Always save new embeddings when generating
    save_embeddings(embeddings)
    # Always save new file data when generating
    save_file_data(file_data)

    return embeddings, file_data


def create_graph_data(embeddings, file_data):
    """Create graph data from embeddings"""
    print("Creating graph data...")
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

    # Create nodes with only essential data
    nodes = []
    for i, project in enumerate(file_data):
        # Calculate node size based on number of connections
        connection_count = len([link for link in links if str(i) in [
                               link["source"], link["target"]]])

        nodes.append({
            "id": str(i),
            "name": project["title"],
            "group": field_to_group[project["field"]],
            "description": project["description"],
            "connections": connection_count  # Let frontend calculate size based on this
        })

    # Create minimal graph data structure
    graph_data = {
        "nodes": nodes,
        "links": links,
        "metadata": {
            "field_groups": field_to_group,
            "total_nodes": len(nodes),
            "total_links": len(links)
        }
    }

    # Always save new graph data when creating
    print(
        f"Generated graph data with {len(nodes)} nodes and {len(links)} links")
    save_graph_data(graph_data)

    return graph_data


def get_graph_data():
    """Get graph data - handle caching based on USE_CACHED_DATA setting"""
    print("Getting graph data...")
    if USE_CACHED_DATA:
        print("Using cached data...")
        # Try to load existing graph data
        graph_data = load_graph_data()
        if graph_data:
            return graph_data

        # If no graph data, try to use existing embeddings
        embeddings = load_embeddings()
        file_data = load_file_data()
        if embeddings is not None:
            return create_graph_data(embeddings, file_data)
    else:
        # If not using cached data, remove old files if they exist
        if os.path.exists(GRAPH_DATA_PATH):
            os.remove(GRAPH_DATA_PATH)
        if os.path.exists(EMBEDDINGS_PATH):
            os.remove(EMBEDDINGS_PATH)

    # Generate new embeddings and graph data
    embeddings, file_data = generate_embeddings()
    return create_graph_data(embeddings, file_data)


def get_file_description(file_path):
    """Extract content from file and return truncated description"""
    file_handlers = {
        '.pdf': lambda f: PyPDF2.PdfReader(f).pages[0].extract_text(),
        '.docx': lambda f: docx.Document(f).paragraphs[0].text if docx.Document(f).paragraphs else "",
        'default': lambda f: f.read()
    }

    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        handler = file_handlers.get(file_ext, file_handlers['default'])

        open_mode = 'rb' if file_ext == '.pdf' else 'r'
        encoding = None if file_ext == '.pdf' else 'utf-8'

        with open(file_path, open_mode, encoding=encoding) as f:
            content = handler(f)
            return content[:MAX_CHAR_LIMIT] + "..." if len(content) > MAX_CHAR_LIMIT else content

    except Exception as e:
        return f"Error reading file {os.path.basename(file_path)}: {str(e)}"


def process_new_file(filename):
    """Process a newly uploaded file and update graph data"""
    # Get existing graph data
    graph_data = load_graph_data()
    # if not graph_data:
    #     # If no existing graph data, generate it from scratch
    #     return get_graph_data()

    # Get existing embeddings
    embeddings = load_embeddings()
    # if embeddings is None:
    #     # If no existing embeddings, generate them
    #     return get_graph_data()

    # Create new node data for the file
    file_path = os.path.join(RAW_FILES_DIR, 'uploads', filename)
    description = get_file_description(file_path)

    # Load the model
    model = SentenceTransformer(MODEL_NAME)

    # Generate embedding for new file
    new_embedding = model.encode([description])[0]

    # Add new embedding to existing embeddings
    updated_embeddings = np.vstack([embeddings, new_embedding])

    # Save updated embeddings
    save_embeddings(updated_embeddings)

    # Create new node
    new_node_id = str(len(graph_data['nodes']))
    new_node = {
        "id": new_node_id,
        "name": filename,
        "group": field_to_group["new"],
        "description": description,
        "connections": 0
    }

    # Calculate similarities with existing nodes
    similarities = cosine_similarity([new_embedding], embeddings)[0]

    # Create new links based on similarity threshold
    new_links = []
    for i, similarity in enumerate(similarities):
        if similarity > SIMILARITY_THRESHOLD:
            new_links.append({
                "source": new_node_id,
                "target": str(i),
                "value": float(similarity)
            })
            # Update connection count for existing node
            graph_data['nodes'][i]['connections'] += 1

    # Update new node's connection count
    new_node['connections'] = len(new_links)

    # Update graph data
    graph_data['nodes'].append(new_node)
    graph_data['links'].extend(new_links)
    graph_data['metadata']['total_nodes'] += 1
    graph_data['metadata']['total_links'] += len(new_links)

    # Save updated graph data
    save_graph_data(graph_data)

    return graph_data
