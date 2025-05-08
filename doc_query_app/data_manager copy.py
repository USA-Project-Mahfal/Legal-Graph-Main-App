# import json
# import os
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from config import USE_CACHED_DATA, MODEL_NAME, SIMILARITY_THRESHOLD
# from data import project_data
# # Paths for data storage
# GRAPH_DATA_PATH = 'data/graph_data.json'
# EMBEDDINGS_PATH = 'data/embeddings.npy'
# DATA_DIR = 'data'
# RAW_FILES_DIR = 'raw_files'

# # Ensure data directory exists
# os.makedirs(DATA_DIR, exist_ok=True)
# os.makedirs(RAW_FILES_DIR, exist_ok=True)

# # Field to group mapping
# field_to_group = {
#     "web": 1,
#     "mobile": 2,
#     "cybersecurity": 3,
#     "iot": 4,
#     "file": 5  # New group for uploaded files
# }


# def save_embeddings(embeddings):
#     """Save embeddings to file"""
#     np.save(EMBEDDINGS_PATH, embeddings)


# def load_embeddings():
#     """Load embeddings from file if exists, otherwise return None"""
#     if os.path.exists(EMBEDDINGS_PATH):
#         return np.load(EMBEDDINGS_PATH)
#     return None


# def save_graph_data(graph_data):
#     """Save graph data to file"""
#     with open(GRAPH_DATA_PATH, 'w') as f:
#         json.dump(graph_data, f)


# def load_graph_data():
#     """Load graph data from file if exists, otherwise return None"""
#     if os.path.exists(GRAPH_DATA_PATH):
#         with open(GRAPH_DATA_PATH, 'r') as f:
#             return json.load(f)
#     return None


# def generate_embeddings():
#     """Generate embeddings for project data"""
#     # Get descriptions from project data
#     descriptions = [project["description"] for project in project_data]

#     # Load pre-trained sentence transformer model
#     model = SentenceTransformer(MODEL_NAME)

#     # Generate embeddings
#     embeddings = model.encode(descriptions)

#     # Always save new embeddings when generating
#     save_embeddings(embeddings)

#     return embeddings


# def create_graph_data(embeddings):
#     """Create graph data from embeddings"""
#     # Compute similarity matrix
#     similarity_matrix = cosine_similarity(embeddings)

#     # Create links between nodes based on similarity threshold
#     links = []
#     for i in range(len(similarity_matrix)):
#         for j in range(i+1, len(similarity_matrix)):
#             if similarity_matrix[i][j] > SIMILARITY_THRESHOLD:
#                 links.append({
#                     "source": str(i),
#                     "target": str(j),
#                     "value": float(similarity_matrix[i][j])
#                 })

#     # Create nodes with only essential data
#     nodes = []
#     for i, project in enumerate(project_data):
#         # Calculate node size based on number of connections
#         connection_count = len([link for link in links if str(i) in [
#                                link["source"], link["target"]]])

#         nodes.append({
#             "id": str(i),
#             "name": project["title"],
#             "group": field_to_group[project["field"]],
#             "description": project["description"],
#             "connections": connection_count  # Let frontend calculate size based on this
#         })

#     # Create minimal graph data structure
#     graph_data = {
#         "nodes": nodes,
#         "links": links,
#         "metadata": {
#             "field_groups": field_to_group,
#             "total_nodes": len(nodes),
#             "total_links": len(links)
#         }
#     }

#     # Always save new graph data when creating
#     print(
#         f"Generated graph data with {len(nodes)} nodes and {len(links)} links")
#     save_graph_data(graph_data)

#     return graph_data


# def get_graph_data():
#     """Get graph data - handle caching based on USE_CACHED_DATA setting"""
#     if USE_CACHED_DATA:
#         # Try to load existing graph data
#         graph_data = load_graph_data()
#         if graph_data:
#             return graph_data

#         # If no graph data, try to use existing embeddings
#         embeddings = load_embeddings()
#         if embeddings is not None:
#             return create_graph_data(embeddings)
#     else:
#         # If not using cached data, remove old files if they exist
#         if os.path.exists(GRAPH_DATA_PATH):
#             os.remove(GRAPH_DATA_PATH)
#         if os.path.exists(EMBEDDINGS_PATH):
#             os.remove(EMBEDDINGS_PATH)

#     # Generate new embeddings and graph data
#     embeddings = generate_embeddings()
#     return create_graph_data(embeddings)


# def get_file_description(file_path):
#     """Extract first 200 characters from file content as description"""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#             return content[:200] + "..." if len(content) > 200 else content
#     except UnicodeDecodeError:
#         return f"Binary file: {os.path.basename(file_path)}"


# def process_new_file(filename):
#     """Process a newly uploaded file and update graph data"""
#     # Get existing graph data
#     graph_data = load_graph_data()
#     # if not graph_data:
#     #     # If no existing graph data, generate it from scratch
#     #     return get_graph_data()

#     # Get existing embeddings
#     embeddings = load_embeddings()
#     # if embeddings is None:
#     #     # If no existing embeddings, generate them
#     #     return get_graph_data()

#     # Create new node data for the file
#     file_path = os.path.join(RAW_FILES_DIR, filename)
#     description = get_file_description(file_path)

#     # Load the model
#     model = SentenceTransformer(MODEL_NAME)

#     # Generate embedding for new file
#     new_embedding = model.encode([description])[0]

#     # Add new embedding to existing embeddings
#     updated_embeddings = np.vstack([embeddings, new_embedding])

#     # Save updated embeddings
#     save_embeddings(updated_embeddings)

#     # Create new node
#     new_node_id = str(len(graph_data['nodes']))
#     new_node = {
#         "id": new_node_id,
#         "name": filename,
#         "group": field_to_group["file"],
#         "description": description,
#         "connections": 0
#     }

#     # Calculate similarities with existing nodes
#     similarities = cosine_similarity([new_embedding], embeddings)[0]

#     # Create new links based on similarity threshold
#     new_links = []
#     for i, similarity in enumerate(similarities):
#         if similarity > SIMILARITY_THRESHOLD:
#             new_links.append({
#                 "source": new_node_id,
#                 "target": str(i),
#                 "value": float(similarity)
#             })
#             # Update connection count for existing node
#             graph_data['nodes'][i]['connections'] += 1

#     # Update new node's connection count
#     new_node['connections'] = len(new_links)

#     # Update graph data
#     graph_data['nodes'].append(new_node)
#     graph_data['links'].extend(new_links)
#     graph_data['metadata']['total_nodes'] += 1
#     graph_data['metadata']['total_links'] += len(new_links)

#     # Save updated graph data
#     save_graph_data(graph_data)

#     return graph_data
