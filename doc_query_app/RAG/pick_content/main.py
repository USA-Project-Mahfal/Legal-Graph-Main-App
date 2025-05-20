# app/main.py
import os
from fastapi import HTTPException, Request
from .services.document_service import DocumentService

def get_document(docs):
    """
    Get a document with highlighted text and aggregate page content
    
    Args:
        docs (list): List of documents with metadata and page content
    
    Returns:
        dict: Document with highlighted text
    """
    # Get first doc for metadata
    first_doc = docs[0]
    
    # Aggregate page content from all docs with same doc_name
    doc_name = first_doc.metadata['doc_name']
    aggregated_content = []
    
    for doc in docs:
        if doc.metadata['doc_name'] == doc_name:
            aggregated_content.append(doc.page_content)
    
    # Create DocumentService instance
    document_service = DocumentService(base_path=os.path.dirname(os.path.abspath(__file__)))
    
    result = document_service.get_highlighted_document(
        category=first_doc.metadata['category'],
        doc_name=doc_name,
        highlight_text_content=aggregated_content
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result["message"])
        
    return result
