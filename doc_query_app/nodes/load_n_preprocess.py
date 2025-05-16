import os
import re
import pandas as pd
from tqdm import tqdm


def preprocess_legal_text(text):
    """Clean and normalize legal text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove line numbers or article numbers like "1.1", "Section 2", etc. (optional)
    text = re.sub(
        r'\b(Section|Article)?\s?\d+(\.\d+)*[:.)]?\s+', '', text, flags=re.IGNORECASE)

    # Normalize special unicode quotes and dashes
    text = text.replace('“', '"').replace(
        '”', '"').replace('–', '-').replace('—', '-')

    # Remove page headers/footers if repeating
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    return ' '.join(lines).strip()


def load_documents_from_text_folder(base_dir, limit=None):
    """
    Load and preprocess text files from subfolders in base_dir. 
    Each subfolder is a document category.

    Args:
        base_dir (str): Base directory containing document categories
        limit (int, optional): Maximum number of documents to process. Defaults to None (process all documents).

    Returns a DataFrame with columns: id, name, category, text, file_path
    """
    documents = []
    doc_id = 0

    print(f"Loading documents from: {base_dir}\n")

    for category in sorted(os.listdir(base_dir)):
        if limit is not None and doc_id >= limit:
            break

        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue

        category_doc_count = 0

        for fname in os.listdir(category_path):
            if limit is not None and doc_id >= limit:
                break

            if not fname.endswith(".txt"):
                continue

            file_path = os.path.join(category_path, fname)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                    cleaned_text = preprocess_legal_text(raw_text)

                    if len(cleaned_text.split()) < 50:
                        print(f"Skipping too-short file: {fname}")
                        continue

                    documents.append({
                        'id': doc_id,
                        'name': os.path.splitext(fname)[0],
                        'category': category,
                        'text': cleaned_text,
                        'file_path': file_path
                    })
                    doc_id += 1
                    category_doc_count += 1

            except Exception as e:
                print(f" Error reading {file_path}: {e}")

        print(
            f" Loaded {category_doc_count} documents from category: {category}")

    docs_df = pd.DataFrame(documents)
    print(
        f"\n Total Documents Loaded: {len(docs_df)} across {docs_df['category'].nunique()} categories.")
    return docs_df


def load_random_documents(base_dir: str, max_docs: int = None, seed: int = 42) -> pd.DataFrame:
    """
    Load a random selection of documents up to max_docs limit.

    Args:
        base_dir (str): Base directory containing document categories
        max_docs (int, optional): Maximum number of documents to load. Defaults to None (load all).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: DataFrame containing randomly selected documents
    """
    import random
    random.seed(seed)

    print(f"Loading random documents from: {base_dir}\n")

    # First collect all valid file paths
    all_files = []
    for category in sorted(os.listdir(base_dir)):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue

        for fname in os.listdir(category_path):
            if not fname.endswith(".txt"):
                continue

            file_path = os.path.join(category_path, fname)
            all_files.append((file_path, category, fname))

    # Randomly select files up to max_docs
    if max_docs is not None:
        all_files = random.sample(all_files, min(max_docs, len(all_files)))
    else:
        random.shuffle(all_files)

    documents = []
    doc_id = 0
    category_counts = {}

    for file_path, category, fname in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
                cleaned_text = preprocess_legal_text(raw_text)

                if len(cleaned_text.split()) < 50:
                    print(f"Skipping too-short file: {fname}")
                    continue

                documents.append({
                    'id': doc_id,
                    'name': os.path.splitext(fname)[0],
                    'category': category,
                    'text': cleaned_text,
                    'file_path': file_path
                })
                doc_id += 1
                category_counts[category] = category_counts.get(
                    category, 0) + 1

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Print summary
    for category, count in category_counts.items():
        print(f"Loaded {count} documents from category: {category}")

    docs_df = pd.DataFrame(documents)
    print(
        f"\nTotal Documents Loaded: {len(docs_df)} across {docs_df['category'].nunique()} categories.")
    return docs_df
