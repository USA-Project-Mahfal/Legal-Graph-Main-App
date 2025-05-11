# api_server.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os

from data_manager import data_manager
from config import HOST, PORT, APP_TITLE, APP_DESCRIPTION, APP_VERSION, RAW_FILES_DIR

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

# Ensure upload directory exists
UPLOAD_DIR = os.path.join(RAW_FILES_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


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

    for file in files:
        # Clean filename
        safe_filename = file.filename.replace(" ", "_")
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        try:
            # Save file to disk
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)

            # Process file using data manager
            graph_data = data_manager.process_new_file(safe_filename)

            # Check for errors
            if isinstance(graph_data, dict) and "error" in graph_data:
                print("Error:", graph_data["error"])
                results.append({
                    "filename": safe_filename,
                    "status": "error",
                    "message": graph_data["error"]
                })
            else:
                results.append({
                    "filename": safe_filename,
                    "status": "success",
                    "message": f"File '{safe_filename}' uploaded and processed."
                })

        except Exception as e:
            print("Error:", str(e))
            results.append({
                "filename": safe_filename,
                "status": "error",
                "message": f"Error: {str(e)}"
            })

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
