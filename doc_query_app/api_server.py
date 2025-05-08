from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
from fastapi import UploadFile, File, Form
from typing import List

# Import project modules
from data_manager import get_graph_data, process_new_file
from config import HOST, PORT, APP_TITLE, APP_DESCRIPTION, APP_VERSION

# Create FastAPI app
app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION,
              version=APP_VERSION)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
UPLOAD_DIR = "raw_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    try:
        # Create a safe filename
        safe_filename = file.filename.replace(" ", "_")
        filepath = os.path.join(UPLOAD_DIR, safe_filename)

        # Save the file
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)

        # Process the new file and update graph data
        graph_data = process_new_file(safe_filename)

        return {
            "message": f"File '{safe_filename}' uploaded and processed successfully!",
            "filename": safe_filename,
            "status": "success",
            "graph_data": graph_data
        }
    except Exception as e:
        return {
            "message": f"Error uploading file: {str(e)}",
            "status": "error"
        }


@app.post("/upload-multiple")
async def upload_multiple_docs(files: List[UploadFile] = File(...)):
    results = []
    graph_data = None

    for file in files:
        try:
            # Create a safe filename
            safe_filename = file.filename.replace(" ", "_")
            filepath = os.path.join(UPLOAD_DIR, safe_filename)

            # Save the file
            contents = await file.read()
            with open(filepath, "wb") as f:
                f.write(contents)

            # Process the new file and update graph data
            graph_data = process_new_file(safe_filename)

            results.append({
                "filename": safe_filename,
                "status": "success",
                "message": f"File '{safe_filename}' uploaded and processed successfully!"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Error uploading file: {str(e)}"
            })

    return {
        "results": results,
        "total_files": len(files),
        "successful_uploads": len([r for r in results if r["status"] == "success"]),
        "graph_data": graph_data
    }

# Endpoint 3: Question/Prompt Submission


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    print("Received question:", question)
    return {"response": f"You asked: {question}"}

# Endpoint 4: Get Graph Data


@app.get("/graph")
async def get_graph():
    """Return graph data for 3D visualization"""
    return get_graph_data()

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
