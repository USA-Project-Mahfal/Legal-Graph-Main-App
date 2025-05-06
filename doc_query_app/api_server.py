from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Import project modules
from data_manager import get_graph_data
from config import HOST, PORT, APP_TITLE, APP_DESCRIPTION, APP_VERSION

# Create FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Serve the main visualization page"""
    return FileResponse('static/index.html')


@app.get("/graph")
async def get_graph():
    """Return graph data for 3D visualization"""
    return get_graph_data()

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
