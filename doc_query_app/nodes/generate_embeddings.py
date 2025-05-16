from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm


def generate_optimized_embeddings(source_chunks_df, model_name_to_use):
    print(f"Loading model: {model_name_to_use}")
    model = SentenceTransformer(model_name_to_use)

    batch_size = 32
    if torch.cuda.is_available():
        try:
            free_mem_bytes = torch.cuda.get_device_properties(
                0).total_memory - torch.cuda.memory_allocated(0)
            free_mem_gb = free_mem_bytes / (1024**3)
            if free_mem_gb > 12:
                batch_size = 128
            elif free_mem_gb > 8:
                batch_size = 64
            elif free_mem_gb > 4:
                batch_size = 32
            else:
                batch_size = 16
            print(
                f"CUDA available ({free_mem_gb:.2f}GB free). Using batch size: {batch_size}")
        except Exception as e:
            print(
                f"Could not query CUDA memory, defaulting batch size to 32. Error: {e}")
            batch_size = 32
    else:
        print(f"CUDA not available. Using CPU batch size: {batch_size}")

    print("Generating embeddings...")
    texts_to_embed = source_chunks_df['text'].tolist()
    num_texts = len(texts_to_embed)
    embedding_dim = model.get_sentence_embedding_dimension()
    all_embeddings = np.zeros((num_texts, embedding_dim), dtype=np.float32)

    if num_texts > 0:
        for i in tqdm(range(0, num_texts, batch_size), desc="Batch encoding"):
            batch_texts = texts_to_embed[i:i+batch_size]
            batch_embeddings = model.encode(
                batch_texts, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            all_embeddings[i:i+len(batch_texts)] = batch_embeddings

    # Make a copy to avoid SettingWithCopyWarning
    output_chunks_df = source_chunks_df.copy()
    output_chunks_df['embedding'] = list(all_embeddings)

    print(f"Generated embeddings with shape: {all_embeddings.shape}")
    return output_chunks_df, all_embeddings, model
