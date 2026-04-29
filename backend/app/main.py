from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from backend.app.services.recognition_service import recognition_service

app = FastAPI(title="Face Recognition API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR_RELATIVE = os.getenv("UPLOAD_DIR", "backend/uploads")
UPLOAD_DIR = os.path.join(ROOT_DIR, UPLOAD_DIR_RELATIVE)
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.post("/recognize")
async def recognize_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save temporary file
    file_extension = os.path.splitext(file.filename)[1]
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_file_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process with recognition service
        result = recognition_service.recognize(temp_file_path)
        
        # Optionally delete the temp file after processing
        # os.remove(temp_file_path)

        if "error" in result:
            return {"success": False, "error": result["error"]}
        
        is_success = result["person"] != "Unknown"
        
        return {
            "success": is_success, 
            "prediction": result["person"], 
            "confidence": result["confidence"],
            "box": result["box"]
        }

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run(app, host=host, port=port)
