from fastapi import FastAPI, HTTPException
import video_tools as vt
from pydantic import BaseModel
import os
import shutil

class VideoAnalysisData(BaseModel):
    time: list
    volt: list
    vidpath: str

app = FastAPI(title='Backend Video Analysis API')

@app.post("/process-video/")
def process_video(request: VideoAnalysisData):
    try:
        frames, strain = vt.generate_strain_graph(request.time, request.volt, request.vidpath)
        return {
            "frames": list(frames),
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