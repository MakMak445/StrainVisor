#!/usr/bin/env python3
"""
FastAPI backend for video analysis.
Receives video path, time, and volt data, and returns stress/strain results.
"""

import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Assuming 'video_tools' is a custom module you have.
import video_tools as vt

# Configure a logger for this application
# It's good practice to set up proper logging for production services.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("video_backend")

class VideoAnalysisData(BaseModel):
    """Defines the expected structure of the incoming JSON request body."""
    time: list[float]
    volt: list[float]
    vidpath: str

app = FastAPI(title='Backend Video Analysis API')

@app.post("/process-video/")
def process_video(request: VideoAnalysisData):
    """
    Analyzes a video file based on time and voltage data.

    Receives a POST request with a JSON body containing:
    - time: A list of timestamps.
    - volt: A list of voltage readings.
    - vidpath: The absolute path to the video file to be analyzed.

    Returns a JSON response with 'stress' and 'strain' lists.
    """
    try:
        # Log the incoming request details for debugging purposes.
        logger.info(
            "Request received: len(time)=%d, len(volt)=%d, vidpath=%r",
            len(request.time), len(request.volt), request.vidpath
        )

        # --- Input Validation ---
        # Check if the video path actually exists before processing.
        if not os.path.exists(request.vidpath):
            logger.error("File not found at path: %s", request.vidpath)
            raise HTTPException(status_code=404, detail=f"File not found: {request.vidpath}")

        file_size = os.path.getsize(request.vidpath)
        logger.info("File exists: path=%r, size=%d bytes", request.vidpath, file_size)

        # Check for empty or mismatched lists, which would cause errors.
        if not request.time or not request.volt or len(request.time) != len(request.volt):
            detail = (
                "Input validation failed: Time and volt lists must not be empty "
                f"and must be the same length. Got time={len(request.time)}, volt={len(request.volt)}"
            )
            logger.warning(detail)
            raise HTTPException(status_code=422, detail=detail) # 422 Unprocessable Entity

        # --- Core Logic ---
        # Call the analysis function from your video_tools library.
        stress, strain = vt.obtain_stress_strain(request.time, request.volt, request.vidpath)

        # --- Success Response ---
        logger.info("Successfully processed video: %r", request.vidpath)
        return {
            "stress": list(stress),
            "strain": list(strain),
            "Response": "Video analysed successfully"
        }

    except HTTPException:
        # Re-raise HTTPExceptions directly so FastAPI handles them.
        raise
    except Exception as e:
        # Catch any other unexpected errors during processing.
        logger.exception("An internal error occurred while processing vidpath=%r", request.vidpath)
        # Return a generic 500 error to the client.
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# NOTE: The dangerous directory deletion code that was previously in a 'finally'
# block has been completely removed. This endpoint no longer performs any file
# system cleanup or deletion operations.