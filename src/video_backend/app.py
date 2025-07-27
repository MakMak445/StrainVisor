from fastapi import FastAPI, HTTPException
import video_tools as vt
from pydantic import BaseModel
import os

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
        if os.path.exists(request.vidpath):
            print(f"Cleaning up. Deleting file: {request.vidpath}")
            os.remove(request.vidpath)