import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

print("Loading CSV file...")
# Load the CSV
chunks_df = pd.read_csv("hybrid_chunks.csv").dropna(subset=['text'])
print(f"Loaded {len(chunks_df)} rows from CSV")

print("Initializing embedding model...")
# Embedding model
embedding_model_name = "msmarco-distilbert-base-tas-b"
embedding_function = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cuda"}  # Using GPU for faster processing
)
print(f"Initialized {embedding_model_name} embedding model")

print("Creating Document objects with metadata...")
# Create Document objects with metadata
documents = []
for idx, row in chunks_df.iterrows():
    if idx % 100 == 0:  # Print progress every 100 documents
        print(f"Processing document {idx}/{len(chunks_df)}")
    metadata = {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "doc_name": row["doc_name"],
        "category": row["category"]
        # "chunk_method": row["chunk_method"],
        # "level": row["level"],
        # "start_idx": row["start_idx"],
        # "end_idx": row["end_idx"],
        # "document_position": row["document_position"],
        # "position_score": row["position_score"],
        # "is_special_section": row["is_special_section"],
        # "section_type": row["section_type"],
        # "level_size": row["level_size"],
        # "contained_chunks": row["contained_chunks"],
        # "chunk_relationships": row["chunk_relationships"]
    }
    doc = Document(page_content=row["text"], metadata=metadata)
    documents.append(doc)
print(f"Created {len(documents)} Document objects")

print("Creating FAISS vector store...")
# Create FAISS vector store
vectorstore = FAISS.from_documents(documents, embedding_function)

print("Saving vector store to disk...")
# Save to disk
vectorstore.save_local("vector_store")

print(f"FAISS store created with {len(documents)} chunks and metadata.")
