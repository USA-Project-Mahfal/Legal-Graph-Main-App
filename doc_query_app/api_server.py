from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
from fastapi import UploadFile, File, Form

# Import project modules
from data_manager import get_graph_data
from config import HOST, PORT, APP_TITLE, APP_DESCRIPTION, APP_VERSION

# Create FastAPI app
app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory


# Ensure upload directory exists
os.makedirs("uploaded_docs", exist_ok=True)


# Endpoint 1: File Upload
@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    contents = await file.read()
    filepath = os.path.join("uploaded_docs", file.filename)
    with open(filepath, "wb") as f:
        f.write(contents)
    return {"message": f"File '{file.filename}' uploaded successfully!"}


# Endpoint 2: Question/Prompt Submission
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    print("Received question:", question)
    return {"response": f"You asked: {question}"}


# Endpoint 3: Get Graph Data
@app.get("/graph")
async def get_graph():
    """Return graph data for 3D visualization"""
    return get_graph_data()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
