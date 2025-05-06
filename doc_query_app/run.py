import uvicorn
from config import HOST, PORT

if __name__ == "__main__":
    print(
        f"Starting server at http://{HOST if HOST != '0.0.0.0' else 'localhost'}:{PORT}")
    uvicorn.run("api_server:app", host=HOST, port=PORT, reload=True)
