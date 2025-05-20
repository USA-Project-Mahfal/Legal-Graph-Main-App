# app/main.py
import os
from fastapi import HTTPException, Request
from .services.document_service import DocumentService

def get_document(docs):
    """
    Get a document with highlighted text
    
    Args:
        document_request (DocumentRequest): Document request with category, name and text to highlight
    
    Returns:
        dict: Document with highlighted text
    """
    doc = docs[0]  # Take only the first document
    
    # Create DocumentService instance
    document_service = DocumentService(base_path=os.path.dirname(os.path.abspath(__file__)))
    
    result = document_service.get_highlighted_document(
        category=doc.metadata['category'],
        doc_name=doc.metadata['doc_name'],
        highlight_text_content=doc.page_content
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result["message"])
        
    return result
