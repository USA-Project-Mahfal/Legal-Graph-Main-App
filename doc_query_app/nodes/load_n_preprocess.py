import os
import re
import pandas as pd
from tqdm import tqdm
import random
import PyPDF2
from pathlib import Path
from config import (
    BASE_DIR,
)

base_dir = BASE_DIR
new_raw_files_dir = os.path.join(base_dir, "chunk_input")
temp_upload_dir = os.path.join(base_dir, "raw_files/temp_uploads")
categories = [
    # "License_Agreements",
    # "Maintenance",
    # "Service",
    # "Sponsorship",
    "Strategic Alliance"
]

# Ensure temp directory exists
os.makedirs(temp_upload_dir, exist_ok=True)


async def save_uploaded_file_temp(file_content: bytes, original_filename: str) -> str:
    """
    Save an uploaded file to a temporary location.

    Args:
        file_content (bytes): The binary content of the uploaded file
        original_filename (str): Original name of the uploaded file

    Returns:
        str: Path to the temporary file
    """
    try:
        # Create a safe filename
        safe_filename = Path(original_filename).name.replace(" ", "_")

        # Create temporary file path
        temp_file_path = os.path.join(temp_upload_dir, f"temp_{safe_filename}")

        # Write content to temporary file
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        return temp_file_path

    except Exception as e:
        print(f"Error saving temporary file: {e}")
        raise e


def cleanup_temp_file(temp_file_path: str) -> None:
    """
    Clean up a temporary file if it exists.

    Args:
        temp_file_path (str): Path to the temporary file to clean up
    """
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    except Exception as e:
        print(f"Error cleaning up temporary file {temp_file_path}: {e}")


def preprocess_legal_text(text):
    """Clean and normalize legal text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove line numbers or article numbers like "1.1", "Section 2", etc. (optional)
    text = re.sub(
        r'\b(Section|Article)?\s?\d+(\.\d+)*[:.)]?\s+', '', text, flags=re.IGNORECASE)

    # Normalize special unicode quotes and dashes
    text = text.replace('"', '"').replace(
        '"', '"').replace('-', '-').replace('-', '-')

    # Remove page headers/footers if repeating
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    return ' '.join(lines).strip()


def _process_document(file_path: str, category: str, fname: str, doc_id: int) -> dict:
    """Process a single document and return its metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
            cleaned_text = preprocess_legal_text(raw_text)

            if len(cleaned_text.split()) < 50:
                print(f"Skipping too-short file: {fname}")
                return None

            return {
                'id': doc_id,
                'name': os.path.splitext(fname)[0],
                'category': category,
                'text': cleaned_text,
                'file_path': file_path
            }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


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
    category_counts = {}

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
            doc_data = _process_document(file_path, category, fname, doc_id)

            if doc_data:
                documents.append(doc_data)
                doc_id += 1
                category_doc_count += 1
                category_counts[category] = category_counts.get(
                    category, 0) + 1

        print(
            f"Loaded {category_doc_count} documents from category: {category}")

    docs_df = pd.DataFrame(documents)
    print(
        f"\nTotal Documents Loaded: {len(docs_df)} across {docs_df['category'].nunique()} categories.")
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
        doc_data = _process_document(file_path, category, fname, doc_id)
        if doc_data:
            documents.append(doc_data)
            doc_id += 1
            category_counts[category] = category_counts.get(category, 0) + 1

    # Print summary
    for category, count in category_counts.items():
        print(f"Loaded {count} documents from category: {category}")

    docs_df = pd.DataFrame(documents)
    print(
        f"\nTotal Documents Loaded: {len(docs_df)} across {docs_df['category'].nunique()} categories.")
    return docs_df


def detect_category(text):
    """Mock function to detect document category. Would be replaced with actual ML model."""
    # In reality this would use NLP/ML to analyze the text and determine category
    # For now just return a random category
    return random.choice(categories)


def process_n_add_new_document(file_path, file_name, category=None, doc_id=None):
    """
    Process a new document and add it to the raw files directory.

    Args:
        file_path (str): Path to the temporary uploaded file
        file_name (str): Original filename
        category (str, optional): Document category. If None, will be auto-detected

    Returns:
        dict: Document metadata or None if processing failed
    """
    try:
        # Read file content based on extension
        file_ext = os.path.splitext(file_name)[1].lower()

        if file_ext == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = '\n'.join(page.extract_text()
                                 for page in pdf_reader.pages)
        else:  # Default to text file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()

        # Clean text
        cleaned_text = preprocess_legal_text(text)

        if len(cleaned_text.split()) < 50:
            raise ValueError("Document is too short")

        # Create category directory if it doesn't exist
        category_dir = os.path.join(new_raw_files_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        # Save as .txt for consistency
        new_file_name = os.path.splitext(file_name)[0] + '.txt'
        new_file_path = os.path.join(category_dir, new_file_name)

        # Save processed text
        with open(new_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        return {
            "id": doc_id,
            'text': cleaned_text,
            'category': category,
            'file_path': new_file_path,
            'name': os.path.splitext(file_name)[0]
        }

    except Exception as e:
        print(f"Error processing document {file_path}: {e}")
        return None
