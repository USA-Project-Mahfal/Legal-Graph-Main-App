# data_manager.py

import os
import json
import numpy as np
import docx
import PyPDF2
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DATA_DIR, RAW_FILES_DIR, MODEL_NAME, SIMILARITY_THRESHOLD,
    MAX_CHAR_LIMIT, RETRAIN_GNN, USE_CACHED_DATA
)
from GNN import gnn_manager


class DataManager:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.raw_files_dir = RAW_FILES_DIR
        self.model = SentenceTransformer(MODEL_NAME)
        self.graph_data_path = os.path.join(self.data_dir, "graph_data.json")
        self.file_data_path = os.path.join(self.data_dir, "file_data.json")

        self.field_to_group = {"file": 5, "new": 6}

        self.file_handlers = {
            '.pdf': lambda f: PyPDF2.PdfReader(f).pages[0].extract_text(),
            '.docx': lambda f: docx.Document(f).paragraphs[0].text if docx.Document(f).paragraphs else "",
            'default': lambda f: f.read()
        }
        self.init_embeddings_and_pilot_model()

    # =====================
    # === Public Methods ==
    # =====================

    def process_new_file(self, filename: str) -> Dict:
        file_path = os.path.join(self.raw_files_dir, 'uploads', filename)
        description = self._get_file_description(file_path)
        if "Error reading" in description:
            return {"error": description}

        embedding = self.model.encode([description])[0]
        gnn_manager.reload()

        if gnn_manager.embeddings is None:
            return self._init_first_node(filename, description, embedding)

        neighbors, sims = self._find_neighbors(
            embedding, gnn_manager.embeddings)
        new_node_id = gnn_manager.add_node(embedding.reshape(1, -1), neighbors)

        new_node = self._create_node(str(new_node_id), filename, description,
                                     self.field_to_group["new"], len(neighbors))
        new_links = [
            {"source": str(new_node_id), "target": str(n),
             "value": float(sims[n])}
            for n in neighbors
        ]

        self._maybe_refine_embeddings(is_initial=False, links=new_links)

        graph = self._update_graph_with_node(new_node, new_links)
        self._save_json(self.graph_data_path, graph)
        return graph

    def get_graph_data(self) -> Dict:
        graph = self._load_json(self.graph_data_path)
        if graph and USE_CACHED_DATA:
            return graph

        gnn_manager.reload()

        # if gnn_manager.embeddings is None:
        #     self.init_embeddings_and_pilot_model()
        if gnn_manager.embeddings is None:
            return {"error": "No embeddings found. Upload documents to build the graph."}
        graph = self._load_json(self.graph_data_path)
        return graph

    def init_embeddings_and_pilot_model(self) -> bool:
        descriptions, file_data = [], []
        contract_dir = os.path.join(self.raw_files_dir, 'full_contract_txt')
        uploads_dir = os.path.join(self.raw_files_dir, 'uploads')

        self._process_directory(contract_dir, "file", descriptions, file_data)
        self._process_directory(uploads_dir, "new", descriptions, file_data)

        if not descriptions:
            return False

        try:
            embeddings = self.model.encode(descriptions)
            gnn_manager.initialize_with_embeddings(embeddings)
            links = self._compute_similar_links(embeddings)
            self._save_links_to_gnn(links)

            nodes = []
            for i, f in enumerate(file_data):
                g = self.field_to_group.get(
                    f["field"], self.field_to_group["file"])
                nodes.append(self._create_node(
                    str(i), f["title"], f["description"], g, 0))

            for link in links:
                nodes[int(link["source"])]["connections"] += 1
                nodes[int(link["target"])]["connections"] += 1

            graph = self._build_graph_structure(nodes, links)
            self._save_json(self.graph_data_path, graph)
            self._save_json(self.file_data_path, file_data)

            self._maybe_refine_embeddings(is_initial=True, links=links)
            return True
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            return False

    # ========================
    # === Internal Helpers ===
    # ========================

    def _init_first_node(self, name, desc, emb) -> Dict:
        node = self._create_node(
            "0", name, desc, self.field_to_group["new"], 0)
        gnn_manager.add_node(emb.reshape(1, -1), [])
        graph = self._build_graph_structure([node], [])
        self._save_json(self.graph_data_path, graph)
        self._maybe_refine_embeddings(is_initial=True, links=[])
        return graph

    def _create_node(self, node_id: str, name: str, desc: str, group: int, conn: int) -> Dict:
        return {
            "id": node_id,
            "name": name,
            "description": desc,
            "group": group,
            "connections": conn
        }

    def _build_graph_structure(self, nodes: List[Dict], links: List[Dict]) -> Dict:
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

    def _update_graph_with_node(self, node: Dict, links: List[Dict]) -> Dict:
        graph = self._load_json(
            self.graph_data_path) or self._build_graph_structure([], [])
        for l in links:
            idx = int(l["target"])
            if idx < len(graph["nodes"]):
                graph["nodes"][idx]["connections"] += 1
        graph["nodes"].append(node)
        graph["links"].extend(links)
        graph["metadata"]["total_nodes"] += 1
        graph["metadata"]["total_links"] += len(links)
        group = str(node["group"])
        graph["metadata"]["field_group_counts"][group] = graph["metadata"]["field_group_counts"].get(
            group, 0) + 1
        return graph

    def _maybe_refine_embeddings(self, is_initial: bool, links: List[Dict]):
        if not links:
            return
        if is_initial:
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

    def _save_links_to_gnn(self, links: List[Dict]):
        graph_links = [{"source": l["source"], "target": l["target"]}
                       for l in links]
        gnn_manager.graph_links = graph_links
        gnn_manager._save_json(gnn_manager.graph_path, graph_links)

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


# Singleton
data_manager = DataManager()
