# api_server.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
from nodes.load_n_preprocess import (
    mock_detect_category,
    process_n_add_new_document,
    save_uploaded_file_temp,
    cleanup_temp_file,
)
from nodes.generate_embeddings import generate_optimized_embeddings
from nodes.chunking import optimized_hybrid_chunking
from data_manager import data_manager
from config import HOST, PORT, APP_TITLE, APP_DESCRIPTION, APP_VERSION, FIELD_TO_GROUP, GROUP_COLORS
import pandas as pd

# Initialize FastAPI application
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-multiple")
async def upload_multiple_docs(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Upload and process multiple document files.

    Args:
        files: List of files to upload

    Returns:
        Dictionary with upload results and graph data
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    graph_data = None
    temp_file_path = None

    for file in files:
        try:
            # Save file to temporary location
            contents = await file.read()
            temp_file_path = await save_uploaded_file_temp(contents, file.filename)

            # Get file content for category detection
            with open(temp_file_path, 'rb') as f:
                # Read first 10KB for category detection
                sample_content = f.read(10000)

            # Detect category from content
            detected_category = mock_detect_category(sample_content)
            print(
                f"Detected category for {file.filename}: {detected_category}")

            # Process the document
            doc_data = process_n_add_new_document(
                temp_file_path, file.filename, detected_category, data_manager.get_last_doc_id())

            if doc_data is None:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Failed to process document"
                })
                continue

            new_chunks_df = optimized_hybrid_chunking(pd.DataFrame([doc_data]))
            _, new_embeddings, _ = generate_optimized_embeddings(
                new_chunks_df, data_manager.embedding_model
            )
            data_manager.update_chunks_df(new_chunks_df)
            data_manager.update_embeddings_matrix(new_embeddings)
            data_manager.build_3D_graph(force=True)

            results.append({
                "filename": file.filename,
                "status": "success",
                "message": f"File processed and categorized as '{detected_category}'"
            })

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Error: {str(e)}"
            })
        finally:
            # Clean up temporary file
            if temp_file_path:
                cleanup_temp_file(temp_file_path)

    # Return results
    return {
        "results": results,
        "total_files": len(files),
        "successful_uploads": len([r for r in results if r["status"] == "success"]),
        "graph_data": graph_data  # Most recent graph state
    }


@app.get("/graph")
async def get_graph() -> Dict[str, Any]:
    """Get graph data for visualization.

    Returns:
        Graph data structure with nodes, links, and metadata
    """
    graph_data = data_manager.get_graph_data()

    # Check for errors
    if isinstance(graph_data, dict) and "error" in graph_data:
        error_message = graph_data["error"]
        print("Error:", error_message)

        # Raise HTTP exception with the error message
        raise HTTPException(status_code=500, detail=error_message)

    return graph_data


@app.get("/group-config")
async def get_group_config():
    """
    Returns a simple mapping of field names to their colors for graph visualization.
    """
    return GROUP_COLORS

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
