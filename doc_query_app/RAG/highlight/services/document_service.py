# app/services/document_service.py
import os

class DocumentService:
    def __init__(self, base_path):
        """
        Initialize document service with base path to document directories
        
        Args:
            base_path (str): Base path to document directories
        """
        self.base_path = base_path
        
    def get_document_path(self, category, doc_name):
        """
        Get the full path to a document
        
        Args:
            category (str): Document category (directory name)
            doc_name (str): Document filename
            
        Returns:
            str: Full path to the document
        """
        # Get the root project directory by going up from current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))  # services dir
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # up to project root
        chunk_input_path = os.path.join(project_root, "chunk_input")
        
        doc_name = f"{doc_name}.txt"
            
        return os.path.join(chunk_input_path, category, doc_name)
    
    def get_document_content(self, category, doc_name):
        """
        Get the content of a document
        
        Args:
            category (str): Document category (directory name)
            doc_name (str): Document filename
            
        Returns:
            str: Document content or None if not found
        """
        doc_path = self.get_document_path(category, doc_name)
        print(f"doc_path: {doc_path}")
        try:
            with open(doc_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error reading document: {e}")
            return None
    
    def get_highlighted_document(self, category, doc_name, highlight_text_content):
        """
        Get document content with highlighted text
        
        Args:
            category (str): Document category (directory name)
            doc_name (str): Document filename
            highlight_text_content (str): Text to be highlighted
            
        Returns:
            dict: Dictionary with document info and content
        """
        content = self.get_document_content(category, doc_name)
        if content is None:
            return {
                "status": "error",
                "message": f"Document not found: {category}/{doc_name}"
            }
                    
        return {
            "category": category,
            "doc_name": doc_name,
            "highlight_text_content": highlight_text_content,
            "original_content": content
        }
