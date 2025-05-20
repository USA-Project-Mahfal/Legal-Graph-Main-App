# app/utils/text_highlighter.py
def highlight_text(text, search_term, highlight_start="<mark>", highlight_end="</mark>"):
    """
    Highlights search term in text by wrapping it with HTML mark tags.
    
    Args:
        text (str): Original text content
        search_term (str): Text to be highlighted
        highlight_start (str): Opening HTML tag for highlighting
        highlight_end (str): Closing HTML tag for highlighting
        
    Returns:
        str: Text with highlighted search term
    """
    if not search_term or not text:
        return text
        
    highlighted_text = text.replace(search_term, f"{highlight_start}{search_term}{highlight_end}")
    return highlighted_text