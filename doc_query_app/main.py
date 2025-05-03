from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# CORS: Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
