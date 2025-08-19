from fastapi import FastAPI, HTTPException
import video_tools as vt
from pydantic import BaseModel
# Assuming 'video_tools' is a custom module you have.
import video_tools as vt
import logging
import shutil
import os

# Configure a logger for this application
# It's good practice to set up proper logging for production services.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("video_backend")

class VideoAnalysisData(BaseModel):
    time: list
    volt: list
    vidpath: str

app = FastAPI(title='Backend Video Analysis API')

@app.post("/process-video/")
def process_video(request: VideoAnalysisData):
    try:
        stress, strain = vt.obtain_stress_strain(request.time, request.volt, request.vidpath)
        return {
            "stress": list(stress),
            "strain": list(strain),
            "Response":'Video analysed successfully' 
        }
    except Exception as e:
        # If anything goes wrong during processing, return a 500 error.
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    
    finally:
        # --- CRITICAL FIX 2: The cleanup logic ---
        # This 'finally' block will execute after the 'try' block, regardless of
        # whether an error occurred or not.
        parent_dir = os.path.dirname(request.vidpath)  # Get parent directory of the file
        try:
            shutil.rmtree(parent_dir)
            print(f"Deleted directory: {parent_dir}")
        except Exception as e:
            print(f"Failed to delete {parent_dir}. Reason: {e}")