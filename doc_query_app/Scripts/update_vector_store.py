import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# Load your new DataFrame with the same schema
new_hybrid_chunks = pd.read_csv("new_hybrid_chunks.csv").dropna(subset=["text"])
print(f"Loaded {len(new_hybrid_chunks)} new chunks.")

embedding_model_name = "msmarco-distilbert-base-tas-b"
embedding_function = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cuda"}  # or "cpu" if no GPU
)

vectorstore = FAISS.load_local(
    folder_path="vector_store",
    embeddings=embedding_function,
    allow_dangerous_deserialization=True
)
print("Loaded existing FAISS vector store.")

new_documents = []
for idx, row in new_hybrid_chunks.iterrows():
    metadata = {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "doc_name": row["doc_name"],
        "category": row["category"]
    }
    doc = Document(page_content=row["text"], metadata=metadata)
    new_documents.append(doc)
print(f"Prepared {len(new_documents)} new documents.")

vectorstore.add_documents(new_documents)
print("Added new documents to the FAISS vector store.")

vectorstore.save_local("vector_store")
print("Vector store updated and saved to disk.")

# Merge new chunks into hybrid_chunks.csv and delete new_hybrid_chunks.csv
if os.path.exists("hybrid_chunks.csv"):
    existing_chunks = pd.read_csv("hybrid_chunks.csv")
    combined_chunks = pd.concat([existing_chunks, new_hybrid_chunks], ignore_index=True)
else:
    combined_chunks = new_hybrid_chunks

combined_chunks.to_csv("hybrid_chunks.csv", index=False)
os.remove("new_hybrid_chunks.csv")
print("Merged new chunks into hybrid_chunks.csv and deleted new_hybrid_chunks.csv")
