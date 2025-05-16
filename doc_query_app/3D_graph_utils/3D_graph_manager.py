import os
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    SIMILARITY_THRESHOLD,
    MAX_CHAR_LIMIT, FIELD_TO_GROUP
)
from data_manager import data_manager
from GNN import gnn_manager


class Graph_visualizer:
    def __init__(self):
        self.base_dir = "d:/PROJECTS/CLIENT/USA-Graph-ML/REPOS/Document_Fetch/doc_query_app"
        self.graph_data_path = os.path.join(
            self.base_dir, "data/graph_data.json")
        self.file_data_path = os.path.join(
            self.base_dir, "data/file_data.json")

    def _create_node(self, node_id: str, name: str, desc: str, group: int, conn: int) -> Dict:
        return {
            "id": node_id,
            "name": name,
            "description": desc,
            "group": group,
            "connections": conn
        }

    def _update_graph_structure(self, nodes: List[Dict], links: List[Dict], is_initial: bool = False) -> Dict:
        if is_initial:
            # Building a new graph from scratch
            counts = {}
            for n in nodes:
                g = str(n["group"])
                counts[g] = counts.get(g, 0) + 1

            return {
                "nodes": nodes,
                "links": links,
                "metadata": {
                    "total_nodes": len(nodes),
                    "total_links": len(links),
                    "field_groups": self.field_to_group,
                    "field_group_counts": counts
                }
            }
        else:
            # Updating existing graph
            graph = self._load_json(self.graph_data_path)
            if not graph:
                return self._update_graph_structure(nodes, links, is_initial=True)

            # Update connection counts for existing nodes
            for l in links:
                idx = int(l["target"])
                if idx < len(graph["nodes"]):
                    graph["nodes"][idx]["connections"] += 1

            # Add new nodes and links
            graph["nodes"].extend(nodes)
            graph["links"].extend(links)

            # Update metadata
            graph["metadata"]["total_nodes"] += len(nodes)
            graph["metadata"]["total_links"] += len(links)

            # Update group counts
            for node in nodes:
                group = str(node["group"])
                graph["metadata"]["field_group_counts"][group] = graph["metadata"]["field_group_counts"].get(
                    group, 0) + 1

            return graph

    def refine_embeddings(self):
        if gnn_manager.model is None:
            gnn_manager.train_model()
        gnn_manager.refine_embeddings()

    def _find_neighbors(self, emb: np.ndarray, ref: np.ndarray) -> Tuple[List[int], np.ndarray]:
        sims = cosine_similarity([emb], ref)[0]
        return [i for i, s in enumerate(sims) if s > SIMILARITY_THRESHOLD], sims

    def _compute_similar_links(self, embs: np.ndarray) -> List[Dict]:
        links = []
        for i in range(len(embs)):
            sims = cosine_similarity([embs[i]], embs)[0]
            for j in range(i + 1, len(embs)):
                if sims[j] > SIMILARITY_THRESHOLD:
                    links.append(
                        {"source": str(i), "target": str(j), "value": float(sims[j])})
        return links

    def _process_directory(self, dir_path: str, field: str,
                           descriptions: List[str], file_data: List[Dict], limit: int = 30):
        if not os.path.exists(dir_path):
            return
        count = 0
        for fname in os.listdir(dir_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in self.file_handlers and ext != '.txt':
                continue
            if count >= limit:
                break
            fpath = os.path.join(dir_path, fname)
            desc = self._get_file_description(fpath)
            if "Error reading" in desc:
                continue
            descriptions.append(desc)
            file_data.append({
                "id": len(file_data),
                "title": fname,
                "field": field,
                "description": desc[:MAX_CHAR_LIMIT]
            })
            count += 1

    def _get_file_description(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        handler = self.file_handlers.get(ext, self.file_handlers['default'])
        mode, encoding = ('rb', None) if ext == '.pdf' else ('r', 'utf-8')
        try:
            with open(path, mode, encoding=encoding) as f:
                text = handler(f)
            return text[:MAX_CHAR_LIMIT]
        except Exception as e:
            return f"Error reading {os.path.basename(path)}: {str(e)}"

    def _save_json(self, path: str, data: Any):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    def _load_json(self, path: str) -> Optional[Dict]:
        return json.load(open(path)) if os.path.exists(path) else None

     # try:
        #     embeddings = self.model.encode(descriptions)
        #     gnn_manager.initialize_with_embeddings(embeddings)
        #     links = self._compute_similar_links(embeddings)

        #     # Save links to GNN
        #     graph_links = [{"source": l["source"],
        #                     "target": l["target"]} for l in links]
        #     gnn_manager.graph_links = graph_links
        #     gnn_manager._save_json(gnn_manager.graph_path, graph_links)

        #     # Create nodes
        #     nodes = []
        #     for i, f in enumerate(file_data):
        #         g = self.field_to_group.get(
        #             f["field"], self.field_to_group["file"])
        #         nodes.append(self._create_node(
        #             str(i), f["title"], f["description"], g, 0))

        #     # Update connections count
        #     for link in links:
        #         nodes[int(link["source"])]["connections"] += 1
        #         nodes[int(link["target"])]["connections"] += 1

        #     # Build and save graph
        #     graph = self._update_graph_structure(nodes, links, is_initial=True)
        #     self._save_json(self.graph_data_path, graph)
        #     self._save_json(self.file_data_path, file_data)

        #     self.refine_embeddings()
        #     return True
        # except Exception as e:
        #     print(f"Error during embedding generation: {e}")
        #     return False
